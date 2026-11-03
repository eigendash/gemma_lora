"""Custom GRPO training loop (PyTorch + PEFT).

Implements Group Relative Policy Optimization without TRL's GRPOTrainer (which
cannot pause generation for tool execution):

- rollouts via ``rollout.run_rollout`` (tool interleaving + env masking),
- group-normalized advantages ``(r - mean) / (std + eps)``; zero-variance groups
  skipped (fraction logged),
- clipped per-token ratio vs old-policy log-probs,
- KL to the reference (LoRA adapter disabled, no_grad) with adaptive coefficient,
- chunked log-prob computation (Gemma's 256k vocab is never materialized whole),
- tier curriculum driven by per-tier pass rate,
- LoRA-only trainable parameters.

Resumability: ``--resume auto`` restores LoRA weights, optimizer state, step,
RNG states (python/numpy/torch/cuda), sampler position, adaptive-KL coefficient,
curriculum state, and run id. Checkpoints are written at step boundaries on a
timer AND every N steps (temp dir + atomic rename, ``latest.json`` last, keep
last 3). Storage backends: ``local`` implemented; ``gcs``/``s3``/``hf-hub`` are
untested interfaces for Phase 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from rewards import RewardConfig, compute_reward
from rollout import RolloutResult, format_prompt, run_rollout
from sandbox import Sandbox

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class GRPOConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    train_file: str = "data/train.jsonl"
    n_problems: int = 50
    G: int = 4
    prompts_per_step: int = 4
    total_steps: int = 30
    max_new_tokens: int = 128
    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    target_kl: float = 0.1
    kl_adapt_rate: float = 2.0
    min_kl_coef: float = 1e-3   # floor so the KL penalty never underflows to 0
    max_kl_coef: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    device: str = "auto"
    seed: int = 0
    output_dir: str = "outputs/rl"
    checkpoint_every: int = 10
    checkpoint_timer: int = 600
    log_every: int = 1
    rollout_dump_every: int = 10
    resume: str = "auto"
    force_resume: bool = False
    backend: str = "local"
    max_rounds: int = 2
    temperature: float = 1.0
    init_adapter: str = ""
    dtype: str = "float32"       # float32 | float16 | bfloat16
    attn: str = "sdpa"           # sdpa | eager (use eager for gemma-2)
    chunked_vocab: bool = False  # use chunked-lm_head logits (256k vocab)
    grad_accum: int = 1         # gradient accumulation steps (keeps effective batch fixed)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
class Curriculum:
    def __init__(self, tiers=("easy", "medium", "hard")):
        self.tiers = list(tiers)
        self.passes = {t: 0 for t in self.tiers}
        self.counts = {t: 0 for t in self.tiers}

    def update(self, tier: str, correct: bool):
        self.counts[tier] += 1
        self.passes[tier] += int(correct)

    def pass_rate(self, tier: str) -> float:
        return self.passes[tier] / self.counts[tier] if self.counts[tier] else 0.5

    def sample_tier(self, rng: random.Random) -> str:
        ws = []
        for t in self.tiers:
            pr = self.pass_rate(t)
            if pr < 0.2:
                w = 1.0 + (0.2 - pr) * 4.0
            elif pr > 0.7:
                w = max(0.1, 1.0 - (pr - 0.7) * 4.0)
            else:
                w = 1.0
            ws.append(w)
        return rng.choices(self.tiers, weights=ws, k=1)[0]

    def state(self) -> dict:
        return {"passes": dict(self.passes), "counts": dict(self.counts)}

    def load_state(self, s: dict):
        self.passes.update(s.get("passes", {}))
        self.counts.update(s.get("counts", {}))


# ---------------------------------------------------------------------------
# Chunked log-probs (256k vocab safety)
# ---------------------------------------------------------------------------
def chunked_logprobs(logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """Log-softmax gathered at target positions, computed in fp32 precision but
    casting only one vocab chunk at a time (never a full fp32 copy of a 256k-vocab
    logits tensor — a full ``.float()`` doubles memory and OOMs on MPS)."""
    max_logits = logits.max(dim=-1).values.float()  # (B, T) fp32
    logsumexp = torch.full(max_logits.shape, float("-inf"), device=logits.device, dtype=torch.float32)
    for i in range(0, logits.shape[-1], chunk_size):
        chunk = logits[:, :, i:i + chunk_size].float()
        logsumexp = torch.logaddexp(logsumexp, (chunk - max_logits.unsqueeze(-1)).exp().sum(dim=-1).log())
    logz = logsumexp + max_logits
    target_logit = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float()
    return target_logit - logz


def group_advantage(rewards: list[float], eps: float = 1e-4) -> tuple[torch.Tensor, bool]:
    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std()
    if std < eps:
        return torch.zeros_like(r), True
    return (r - r.mean()) / (std + eps), False


def _causal_backbone_and_head(model):
    """Return (backbone, lm_head) for a CausalLM, unwrapping peft wrappers."""
    causal = model.base_model.model if hasattr(model, "base_model") else model
    return causal.model, causal.lm_head


def _logprobs_from_hidden(hidden, lm_head, target_ids, vocab_chunk: int = 8192) -> torch.Tensor:
    """Log-softmax gathered at target positions, computed in vocab chunks so the
    full (B, T, V) logits tensor is never materialized."""
    B, T, _ = hidden.shape
    V = lm_head.weight.shape[0]
    dev, dtype = hidden.device, hidden.dtype
    linear = torch.nn.functional.linear

    max_logit = torch.full((B, T), float("-inf"), device=dev, dtype=dtype)
    for i in range(0, V, vocab_chunk):
        w = lm_head.weight[i:i + vocab_chunk]
        b = lm_head.bias[i:i + vocab_chunk] if lm_head.bias is not None else None
        max_logit = torch.maximum(max_logit, linear(hidden, w, b).max(dim=-1).values)

    logsumexp = torch.zeros((B, T), device=dev, dtype=dtype)
    for i in range(0, V, vocab_chunk):
        w = lm_head.weight[i:i + vocab_chunk]
        b = lm_head.bias[i:i + vocab_chunk] if lm_head.bias is not None else None
        logsumexp = logsumexp + (linear(hidden, w, b) - max_logit.unsqueeze(-1)).exp().sum(dim=-1)

    logz = max_logit + logsumexp.log()
    target_logit = (hidden * lm_head.weight[target_ids]).sum(dim=-1)
    if lm_head.bias is not None:
        target_logit = target_logit + lm_head.bias[target_ids]
    return target_logit - logz


def forward_logprobs_chunked(model, input_ids, attention_mask, target_ids, vocab_chunk: int = 8192) -> torch.Tensor:
    """Per-token log-probs for a CausalLM without materializing full-vocab logits."""
    backbone, lm_head = _causal_backbone_and_head(model)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0]
    return _logprobs_from_hidden(hidden, lm_head, target_ids, vocab_chunk)


# ---------------------------------------------------------------------------
# Batch collation (aligns prediction position p -> token p+1)
# ---------------------------------------------------------------------------
def collate_rollouts(rollouts: list[RolloutResult], pad_token_id: int, device) -> dict:
    L = max(len(r.all_ids) for r in rollouts)
    B = len(rollouts)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    gen_mask = torch.zeros((B, L), dtype=torch.long, device=device)
    old_logp = torch.zeros((B, L), dtype=torch.float32, device=device)

    for b, r in enumerate(rollouts):
        ids = r.all_ids
        full_mask = [0] * len(r.prompt_ids) + r.loss_mask
        full_old = [0.0] * len(r.prompt_ids) + r.old_logprobs
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[b, :n] = 1
        gen_mask[b, :n] = torch.tensor(full_mask, dtype=torch.long, device=device)
        old_logp[b, :n] = torch.tensor(full_old, dtype=torch.float32, device=device)

    target_ids = torch.full((B, L), pad_token_id, dtype=torch.long, device=device)
    target_ids[:, :-1] = input_ids[:, 1:]
    pred_mask = torch.zeros((B, L), dtype=torch.long, device=device)
    pred_mask[:, :-1] = gen_mask[:, 1:]
    old_logp_pred = torch.zeros((B, L), dtype=torch.float32, device=device)
    old_logp_pred[:, :-1] = old_logp[:, 1:]
    return {"input_ids": input_ids, "attention_mask": attn, "target_ids": target_ids,
            "pred_mask": pred_mask, "old_logp": old_logp_pred}


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------
class LocalBackend:
    def sync(self, src_dir: str, dest_dir: str):
        os.makedirs(dest_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            s, d = os.path.join(src_dir, name), os.path.join(dest_dir, name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)


class _UnsupportedBackend:
    def __init__(self, name):
        self.name = name

    def sync(self, *a, **k):
        raise NotImplementedError(f"{self.name} backend is an untested interface (Phase 8)")


def get_backend(name: str):
    return LocalBackend() if name == "local" else _UnsupportedBackend(name)


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, run_dir: str, backend, keep: int = 3):
        self.run_dir = run_dir
        self.backend = backend
        self.keep = keep
        os.makedirs(run_dir, exist_ok=True)

    def _ckpt_dir(self, step: int) -> str:
        return os.path.join(self.run_dir, f"step-{step}")

    def save(self, step, model, optimizer, kl_coef, curriculum, sampler, config, run_id):
        tmp = os.path.join(self.run_dir, "ckpt.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        model.save_pretrained(os.path.join(tmp, "adapter"))
        torch.save({"optimizer": optimizer.state_dict(), "kl_coef": kl_coef, "sampler": sampler.getstate()},
                   os.path.join(tmp, "train_state.pt"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            np_state = np.random.get_state()
            json.dump({"step": step, "run_id": run_id, "config_hash": config.hash(),
                       "curriculum": curriculum.state(),
                       "rng_python": random.getstate(),
                       "rng_numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
                       "rng_torch": torch.get_rng_state().tolist(),
                       "rng_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
                       }, f)
        dest = self._ckpt_dir(step)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(tmp, dest)
        with open(os.path.join(self.run_dir, "latest.json"), "w") as f:
            json.dump({"step": step}, f)
        steps = sorted(int(d.split("-")[1]) for d in os.listdir(self.run_dir) if d.startswith("step-"))
        for s in steps[:-self.keep]:
            shutil.rmtree(self._ckpt_dir(s), ignore_errors=True)

    def latest_step(self) -> int | None:
        p = os.path.join(self.run_dir, "latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)["step"]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _load_base(model_id: str, device: str, dtype: str = "float32", attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation=attn)
    return base, tokenizer


def _apply_lora(base, r: int, alpha: int):
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, cfg)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: GRPOConfig) -> None:
    device = cfg.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(cfg.train_file) as f:
        all_problems = [json.loads(line) for line in f]
    by_tier_all = {}
    for p in all_problems:
        by_tier_all.setdefault(p["tier"], []).append(p)
    # balanced round-robin selection so the overfit set spans tiers
    problems = []
    tiers_all = list(by_tier_all.keys())
    while len(problems) < cfg.n_problems:
        for t in tiers_all:
            if by_tier_all[t]:
                problems.append(by_tier_all[t].pop(0))
            if len(problems) >= cfg.n_problems:
                break
    by_tier = {}
    for p in problems:
        by_tier.setdefault(p["tier"], []).append(p)

    base, tokenizer = _load_base(cfg.model, device, cfg.dtype, cfg.attn)
    ckpt = CheckpointManager(cfg.output_dir, get_backend(cfg.backend))
    curriculum = Curriculum(tiers=sorted(by_tier.keys()))
    kl_coef = cfg.kl_coef
    start_step = 0
    run_id = f"rl-{int(time.time())}"
    resume_dir = None

    if cfg.resume == "auto":
        latest = ckpt.latest_step()
        if latest is not None:
            resume_dir = ckpt._ckpt_dir(latest)
    elif cfg.resume and cfg.resume != "none":
        resume_dir = ckpt._ckpt_dir(int(cfg.resume))

    if resume_dir:
        with open(os.path.join(resume_dir, "state.json")) as f:
            state = json.load(f)
        if state["config_hash"] != cfg.hash() and not cfg.force_resume:
            raise SystemExit("config hash mismatch; pass --force-resume to override")
        train_state = torch.load(os.path.join(resume_dir, "train_state.pt"), map_location="cpu")
        model = PeftModel.from_pretrained(base, os.path.join(resume_dir, "adapter"), is_trainable=True)
        start_step = state["step"]
        run_id = state.get("run_id", run_id)
        kl_coef = max(train_state["kl_coef"], cfg.min_kl_coef)
        curriculum.load_state(state["curriculum"])
        rp = state["rng_python"]
        random.setstate((rp[0], tuple(rp[1]), rp[2]))
        np_s = state["rng_numpy"]
        np.random.set_state((np_s[0], np.array(np_s[1], dtype=np.uint32), np_s[2], np_s[3], np_s[4]))
        torch.set_rng_state(torch.tensor(state["rng_torch"], dtype=torch.uint8))
        if state["rng_cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["rng_cuda"]])
    else:
        if cfg.init_adapter:
            sft = PeftModel.from_pretrained(base, cfg.init_adapter)
            base = sft.merge_and_unload()
        model = _apply_lora(base, cfg.lora_r, cfg.lora_alpha)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if resume_dir:
        optimizer.load_state_dict(train_state["optimizer"])
        sampler = random.Random()
        sampler.setstate(train_state["sampler"])
    else:
        sampler = random.Random(cfg.seed + 12345)

    sandbox = Sandbox(n_workers=2, timeout=0.050)
    reward_cfg = RewardConfig()
    gen_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    log_rows = []
    last_ckpt = time.time()
    last_hb = time.time()
    stop_requested = {"flag": False}

    # incremental log: append (resume) or truncate (fresh) so a crash doesn't lose the log
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.output_dir, "train_log.jsonl"), "a" if start_step > 0 else "w")
    import checkpoint as _ck
    hb_bucket = _ck.LocalBucket(os.path.join(cfg.output_dir, "_remote")) if cfg.backend == "local" else None
    if hb_bucket is not None:
        _ck.write_segment(hb_bucket, run_id, 0, {"start_step": start_step, "device": device,
                                                 "provider": "local", "config_hash": cfg.hash()})

    def _sig(signum, frame):
        stop_requested["flag"] = True

    old_handler = signal.signal(signal.SIGTERM, _sig)

    for step in range(start_step, cfg.total_steps):
        if stop_requested["flag"]:
            ckpt.save(step, model, optimizer, kl_coef, curriculum, sampler, cfg, run_id)
            print(f"SIGTERM: saved checkpoint at step {step}")
            break

        chosen = []
        for _ in range(cfg.prompts_per_step):
            tier = curriculum.sample_tier(sampler)
            chosen.append(sampler.choice(by_tier[tier]))

        rollouts: list[RolloutResult] = []
        advs: list[float] = []
        rewards = []
        zero_var = 0
        model.eval()
        for p in chosen:
            group = [run_rollout(model, tokenizer, format_prompt(p["nums"], p["target"]), sandbox,
                                 max_rounds=cfg.max_rounds, max_new_tokens=cfg.max_new_tokens,
                                 temperature=cfg.temperature, rng=gen_rng, device=device)
                     for _ in range(cfg.G)]
            gr = [compute_reward(r, p["nums"], p["target"], reward_cfg) for r in group]
            a, zv = group_advantage([x.total for x in gr])
            zero_var += int(zv)
            for r, x, av in zip(group, gr, a.tolist()):
                rollouts.append(r)
                rewards.append(x)
                advs.append(av)
                curriculum.update(p["tier"], x.correct > 0)

        model.train()
        batch = collate_rollouts(rollouts, tokenizer.pad_token_id or tokenizer.eos_token_id, device)
        adv_t = torch.tensor(advs, dtype=torch.float32, device=device)
        if cfg.chunked_vocab:
            policy_lp = forward_logprobs_chunked(model, batch["input_ids"], batch["attention_mask"], batch["target_ids"])
            with torch.no_grad(), model.disable_adapter():
                ref_lp = forward_logprobs_chunked(model, batch["input_ids"], batch["attention_mask"], batch["target_ids"])
        else:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            policy_lp = chunked_logprobs(out.logits, batch["target_ids"])
            del out
            with torch.no_grad(), model.disable_adapter():
                ref_out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            ref_lp = chunked_logprobs(ref_out.logits, batch["target_ids"])
            del ref_out

        ratio = torch.exp(policy_lp - batch["old_logp"])
        A = adv_t.unsqueeze(1)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * A
        pmask = batch["pred_mask"]
        n_tok = pmask.sum().clamp(min=1)
        policy_loss = (-torch.min(surr1, surr2) * pmask).sum() / n_tok
        kl = ((policy_lp - ref_lp) * pmask).sum() / n_tok
        loss = policy_loss + kl_coef * kl

        kl_val = float(kl.detach())
        if kl_val > cfg.target_kl * 1.5:
            kl_coef = min(kl_coef * cfg.kl_adapt_rate, cfg.max_kl_coef)
        elif kl_val < cfg.target_kl / 1.5:
            kl_coef = max(kl_coef / cfg.kl_adapt_rate, cfg.min_kl_coef)

        # gradient accumulation: effective batch = prompts_per_step * G * grad_accum
        (loss / cfg.grad_accum).backward()
        if (step + 1) % cfg.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        if device == "mps":
            torch.mps.empty_cache()  # combat MPS fragmentation over long runs

        row = {"step": step, "reward": sum(x.total for x in rewards) / len(rewards),
               "correct": sum(x.correct for x in rewards) / len(rewards),
               "exec": sum(x.exec for x in rewards) / len(rewards),
               "format": sum(x.format for x in rewards) / len(rewards),
               "length_penalty": sum(x.length_penalty for x in rewards) / len(rewards),
               "kl": kl_val, "kl_coef": kl_coef, "zero_var_frac": zero_var / cfg.prompts_per_step}
        log_rows.append(row)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if (step + 1) % cfg.log_every == 0 or step == start_step:
            print(f"step {step} reward {row['reward']:.3f} correct {row['correct']:.3f} "
                  f"exec {row['exec']:.3f} format {row['format']:.3f} kl {row['kl']:.4f} "
                  f"zero_var {row['zero_var_frac']:.2f}")

        if hb_bucket is not None and time.time() - last_hb > 120:
            _ck.write_heartbeat(hb_bucket, run_id, {"step": step, "elapsed": int(time.time() - last_hb),
                                                    "reward": row["reward"], "correct": row["correct"]})
            last_hb = time.time()

        if (step + 1) % cfg.rollout_dump_every == 0:
            _dump_rollouts(cfg.output_dir, step, rollouts)

        if ((step + 1) % cfg.checkpoint_every == 0) or (time.time() - last_ckpt > cfg.checkpoint_timer):
            ckpt.save(step + 1, model, optimizer, kl_coef, curriculum, sampler, cfg, run_id)
            last_ckpt = time.time()

    ckpt.save(step + 1, model, optimizer, kl_coef, curriculum, sampler, cfg, run_id)
    signal.signal(signal.SIGTERM, old_handler)
    sandbox.close()
    log_file.close()
    print(f"run {run_id} done; {len(log_rows)} steps -> {cfg.output_dir}/train_log.jsonl")


def _dump_rollouts(output_dir, step, rollouts):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "rollouts.jsonl"), "a") as f:
        for r in rollouts[:4]:
            f.write(json.dumps({"step": step, "transcript": r.transcript,
                                "answer": r.answer_text, "has_answer": r.has_answer,
                                "code_rounds": r.code_rounds}) + "\n")


def _write_log(output_dir, rows, run_id):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_log.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"run {run_id} done; {len(rows)} steps -> {output_dir}/train_log.jsonl")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    skip = {"model", "train_file", "output_dir", "device", "resume", "backend", "init_adapter", "force_resume", "chunked_vocab"}
    for fname, f in GRPOConfig.__dataclass_fields__.items():
        if fname in skip:
            continue
        ap.add_argument(f"--{fname.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/rl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--force-resume", action="store_true")
    ap.add_argument("--chunked-vocab", action="store_true")
    ap.add_argument("--backend", default="local")
    ap.add_argument("--init-adapter", default="")
    args = ap.parse_args(argv)
    cfg = GRPOConfig(**{k: v for k, v in vars(args).items() if k in GRPOConfig.__dataclass_fields__})
    train(cfg)


if __name__ == "__main__":
    main()

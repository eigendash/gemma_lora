"""Custom GRPO training loop (PyTorch + PEFT).

Implements Group Relative Policy Optimization without TRL's GRPOTrainer (which
cannot pause generation for tool execution):

- rollouts via ``rollout.run_rollouts_batched`` by default: all
  ``prompts_per_step * G`` rollouts of a step decode together, each pausing
  for its own tool calls (``--generation sequential`` = the per-sequence
  reference loop; identical semantics). The model is fed exactly the ids we
  train on,
- group-normalized advantages ``(r - mean) / (std + eps)``; groups with zero
  reward variance carry no policy gradient and are dropped from the update
  (first-round fraction logged as ``zero_var_frac``); with ``dynamic_sampling``
  (DAPO) they are replaced by freshly sampled prompts for up to
  ``max_refill_rounds`` extra rounds, so the batch stays full of signal,
- clipped per-token surrogate. With one optimizer update per rollout batch the
  behaviour policy equals the current policy, so the "old" log-probs are the
  same training forward, detached (ratio exactly 1; the loss is the exact
  group-normalized policy gradient). The rollout-time log-probs are logged as
  ``gen_train_logp_dev`` = max |logp_train - logp_rollout|: ~1e-5 in fp32 and
  ~0.1 in bf16 (KV-cache vs full-forward rounding); a large value means the
  rollout context and the trained ids diverge. LoRA dropout is 0 so the loss
  forward is deterministic; it runs in train mode with gradient checkpointing,
- loss normalization: token-level over all trainable tokens of the step
  (sum over tokens / total trainable tokens), accumulated over micro-batches,
- KL to the reference (LoRA adapter disabled, no_grad) with the non-negative
  k3 estimator ``exp(r) - r - 1``, ``r = logp_ref - logp_policy``, and an
  adaptive coefficient (floored/capped),
- vocab-chunked log-probs: Gemma-2's final-logit softcap is applied exactly as
  the HF forward does, and each vocab chunk is activation-checkpointed so the
  (B, T, V) logits are never held in memory, not even for backward,
- curriculum: tiers are sampled in proportion to the probability that a group
  of G rollouts has non-zero reward variance, from a sliding window of recent
  per-tier pass rates,
- LoRA-only trainable parameters.

Resumability (``--resume auto``): LoRA weights, optimizer, step, RNG states
(python / numpy / torch / cuda / the sampling generator), problem sampler,
adaptive-KL coefficient, curriculum, run id. ``--init-adapter`` (SFT) is merged
into the base on every start, fresh or resumed. Checkpoints are written locally
(temp dir + atomic rename, atomic ``latest.json``) and committed to the bucket
(``checkpoint.py``: create-only objects + COMMIT with sha256s). On a fresh host
with no local checkpoint, the newest valid remote COMMIT for ``--run-id`` is
downloaded first. Refuses to resume on config or dataset hash mismatch unless
``--force-resume``.

Cost guard: ``--max-hours`` / ``--max-total-usd`` (with ``--cost-per-hour`` or
``COST_PER_HOUR``) stop gracefully with a final checkpoint; GPU-hours of earlier
segments of the same run are read from the bucket heartbeats.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import shutil
import signal
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

import attention_gqa
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint as _activation_checkpoint

import checkpoint as ck
from rewards import RewardConfig, compute_reward
from rollout import RolloutResult, format_prompt, run_rollout, run_rollouts_batched
from sandbox import Sandbox

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# fields that do not change what is trained (excluded from the resume hash)
_NON_HASHED = {"resume", "force_resume", "device", "output_dir", "run_id", "backend", "total_steps",
               "checkpoint_every", "checkpoint_timer", "log_every", "rollout_dump_every",
               "max_hours", "max_total_usd", "cost_per_hour", "micro_batch", "generation", "gen_batch",
               "grad_ckpt"}
# fields added after runs were launched: left out of the hash while at their default, so existing
# configs (and their resumable checkpoints) keep the same hash
_HASH_IF_NOT_DEFAULT = {"kl_ref": "base", "hint_schedule": "", "hint_max_groups": 16, "lr_warmup_steps": 0,
                        "train_nocorrect_groups": True, "max_grad_norm": 0.0, "shuffle_minibatches": False}


@dataclass
class GRPOConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    train_file: str = "data/train.jsonl"
    n_problems: int = 50          # 0 = all problems in train_file
    G: int = 4
    prompts_per_step: int = 4
    total_steps: int = 30
    max_new_tokens: int = 128
    lr: float = 1e-5
    lr_warmup_steps: int = 0      # linear LR warmup over the first N steps of THIS run (warm starts get a fresh AdamW); 0 = off
    # recipe v2 options (EXPERIMENTS.md §2.7p); all off by default, so existing configs keep their hashes
    train_nocorrect_groups: bool = True  # False: a group with no correct answer is not trained (its advantages would come
                                         # only from exec/format/length shaping, z-scored to full size); DAPO refills it
    max_grad_norm: float = 0.0    # > 0: clip the LoRA gradient norm before each optimizer step (always logged as grad_norm)
    shuffle_minibatches: bool = False  # True: shuffle groups before splitting into updates_per_batch mini-batches
                                       # (otherwise later mini-batches always hold the hinted/refill groups)
    clip_eps: float = 0.2
    clip_eps_high: float = 0.0    # DAPO clip-higher upper bound (0 = same as clip_eps); matters when updates_per_batch > 1
    updates_per_batch: int = 1    # optimizer steps per generated batch (verl ppo_mini_batch: batch split into this many mini-batches)
    tis_cap: float = 0.0          # truncated importance sampling for sampler/trainer mismatch: w = min(pi_old/pi_gen, cap); 0 = off
    length_threshold: int = 256   # reward length penalty applies beyond this many generated tokens
    length_lambda: float = 0.001
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
    backend: str = "local"       # local | gcs | none  (checkpoint bucket)
    run_id: str = ""             # stable id across preemptions; default env RUN_ID
    max_rounds: int = 2
    on_round_cap: str = "truncate"  # truncate | notice (cap enforced either way; notice lets the model answer)
    temperature: float = 1.0      # sampling AND log-prob temperature: the policy is softmax(logits / T)
    init_adapter: str = ""
    # warm start from an RL LoRA WITHOUT merging: its weights become the initial trainable adapter
    # (bf16 merge_and_unload rounds away part of the delta: P7 s40 scored 104/300 unmerged, 92 merged).
    # The KL reference stays the plain base model (adapter disabled). Fresh runs only.
    init_lora: str = ""
    # KL reference: "base" = plain base model (adapter disabled; default, all runs up to P9);
    # "init" = the frozen init_lora weights (ProRL-style reference reset at a stage boundary)
    kl_ref: str = "base"
    # QuestA-style hinted rollouts (P10): after the first sampling round, every ALL-FAIL group (no correct
    # answer, even if the shaping terms vary) is
    # resampled once with a solution sub-expression in the prompt (sft_coldstart.solution_hint), so
    # problems the policy cannot solve yet still give gradient. "frac@start_step,..." e.g.
    # "0.5@0,0.25@40,0@80"; "" = off. Validation/test prompts never carry hints. Hinted groups are not
    # counted in the curriculum statistics (those measure unhinted ability).
    hint_schedule: str = ""
    hint_max_groups: int = 16     # cap on hinted groups per step
    dtype: str = "float32"       # float32 | float16 | bfloat16
    attn: str = "sdpa"           # sdpa | eager (use eager for gemma-2)
    chunked_vocab: bool = False  # vocab-chunked log-probs (needed for the 256k vocab)
    grad_accum: int = 1          # gradient accumulation steps (keeps effective batch fixed)
    micro_batch: int = 8         # rollouts per forward/backward in the loss pass
    prompt_style: str = "zero"   # zero | fewshot_direct (format-only) | fewshot_direct_retry (+1 instruction) | fewshot
    generation: str = "batched"  # batched (default) | sequential (reference loop) | vllm (CUDA: rollout_vllm, colocated engine)
    vllm_gpu_util: float = 0.3     # generation=vllm: fraction of GPU memory for the vLLM engine (weights + KV cache)
    vllm_max_model_len: int = 4096
    # colocated vLLM sleep mode: after generation the engine offloads weights to CPU and drops its KV
    # cache, so the loss pass gets (almost) the whole GPU; it wakes before the next generation
    vllm_sleep: bool = False
    gen_batch: int = 0           # max sequences per batched decode (0 = all prompts_per_step * G)
    gen_impl: str = "continuous"  # batched engine: continuous (default) | compact | lockstep (rollout.run_rollouts_batched)
    guard_window: int = 5          # auto-stop: rolling window (steps) for the shortcut guards below; 0 = off
    guard_contradicted: float = 0.6  # stop if mean `contradicted` over the window exceeds this
    guard_max_length: float = 0.4    # stop if mean fraction of rollouts hitting the token budget exceeds this
    gen_merged: bool = False     # generate with a plain copy of the model whose weights = base + LoRA delta (no PEFT
                                 # per-layer overhead); refreshed from the pristine base before every generation phase
    grad_ckpt: bool = True       # gradient checkpointing in the loss pass (activation memory)
    dynamic_sampling: bool = False  # DAPO: refill zero-variance groups with fresh prompts
    max_refill_rounds: int = 2      # extra sampling rounds per step when dynamic_sampling is on
    curriculum_window: int = 256
    # "formula": tier weight 1 - p^G - (1-p)^G from the tier pass rate (saturates: any tier with
    # 0.2 < p < 0.8 gets ~1). "signal": weight = measured fraction of that tier's recent groups whose
    # correctness was mixed (0 < #correct < G), which also sees per-problem bimodality.
    curriculum_mode: str = "formula"
    curriculum_group_window: int = 64
    zero_shaping_invalid_literals: bool = False  # rewards.RewardConfig.zero_shaping_if_invalid_literals
    # validation: hold out val_n train problems (equal per tier, from the end of each tier) and, with
    # generation=vllm, evaluate them greedily every val_every steps with the current LoRA
    val_n: int = 0
    val_every: int = 0
    max_hours: float = 0.0       # 0 = no limit (all segments of this run)
    max_total_usd: float = 0.0   # 0 = no limit
    cost_per_hour: float = 0.0   # 0 = read COST_PER_HOUR from the environment

    def hash(self) -> str:
        d = {k: v for k, v in self.__dict__.items()
             if k not in _NON_HASHED and not (k in _HASH_IF_NOT_DEFAULT and v == _HASH_IF_NOT_DEFAULT[k])}
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
class Curriculum:
    """Sample tiers where GRPO gets a gradient.

    A group of G rollouts carries signal only if its rewards differ, i.e. with
    probability ``1 - p**G - (1-p)**G`` for per-rollout pass rate ``p``. Tiers are
    sampled proportionally to that (floored, so no tier is abandoned); ``p`` is
    estimated from a sliding window of recent outcomes (prior 0.5 when empty).
    Mastered and hopeless tiers both get little mass; tiers near p=0.5 get most.
    """

    def __init__(self, tiers=("easy", "medium", "hard"), G: int = 8, window: int = 256, floor: float = 0.05,
                 mode: str = "formula", group_window: int = 64):
        if mode not in ("formula", "signal"):
            raise ValueError(f"curriculum mode must be formula|signal, got {mode!r}")
        self.tiers = list(tiers)
        self.G = G
        self.window = window
        self.floor = floor
        self.mode = mode
        self.group_window = group_window
        self.hist = {t: deque(maxlen=window) for t in self.tiers}
        self.ghist = {t: deque(maxlen=group_window) for t in self.tiers}

    def update_group(self, tier: str, n_correct: int, n: int):
        """Record whether a group had mixed correctness (the GRPO signal for accuracy)."""
        self.ghist[tier].append(int(0 < n_correct < n))

    def signal_rate(self, tier: str) -> float:
        g = self.ghist[tier]
        return sum(g) / len(g) if g else 0.5

    def update(self, tier: str, correct: bool):
        self.hist[tier].append(int(correct))

    def pass_rate(self, tier: str) -> float:
        h = self.hist[tier]
        return sum(h) / len(h) if h else 0.5

    def weight(self, tier: str) -> float:
        if self.mode == "signal":
            return max(self.floor, self.signal_rate(tier))
        p = self.pass_rate(tier)
        return max(self.floor, 1.0 - p ** self.G - (1.0 - p) ** self.G)

    def sample_tier(self, rng: random.Random) -> str:
        return rng.choices(self.tiers, weights=[self.weight(t) for t in self.tiers], k=1)[0]

    def state(self) -> dict:
        return {"hist": {t: list(h) for t, h in self.hist.items()},
                "ghist": {t: list(h) for t, h in self.ghist.items()}}

    def load_state(self, s: dict):
        for t, h in s.get("ghist", {}).items():
            if t in self.ghist:
                self.ghist[t] = deque(h, maxlen=self.group_window)
        if "hist" in s:
            for t, h in s["hist"].items():
                if t in self.hist:
                    self.hist[t] = deque(h, maxlen=self.window)
        else:  # legacy cumulative {"passes", "counts"} format
            for t in self.tiers:
                n, k = s.get("counts", {}).get(t, 0), s.get("passes", {}).get(t, 0)
                self.hist[t] = deque(([1] * k + [0] * (n - k))[-self.window:], maxlen=self.window)


# ---------------------------------------------------------------------------
# Log-probs
# ---------------------------------------------------------------------------
def chunked_logprobs(logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """Log-softmax gathered at target positions from full logits, computed in fp32
    one vocab chunk at a time (never a full fp32 copy of the logits)."""
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


def kl_k3(policy_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
    """Per-token k3 KL estimate of KL(policy || ref) on policy samples: exp(r) - r - 1,
    r = ref - policy. Non-negative; its gradient pulls the policy toward the reference
    (the plain log-ratio k1 has zero expected gradient on-policy and can go negative)."""
    r = ref_lp - policy_lp
    return torch.exp(r) - r - 1.0


def clipped_surrogate(policy_lp, old_lp, adv, clip_eps: float, clip_eps_high: float | None = None):
    """PPO/GRPO clipped surrogate per token (to be MAXIMIZED) and the ratio. ``clip_eps_high``
    decouples the upper bound (DAPO clip-higher); None = symmetric."""
    hi = clip_eps if not clip_eps_high else clip_eps_high
    ratio = torch.exp(policy_lp - old_lp)
    return torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + hi) * adv), ratio


def tis_weights(old_lp: torch.Tensor, gen_lp: torch.Tensor, cap: float) -> torch.Tensor:
    """Truncated importance weights min(pi_old / pi_gen, cap), detached: corrects for the sampler
    (bf16 KV-cache decoding) and the trainer (full forward) assigning slightly different
    probabilities to the same tokens (TRL vllm_importance_sampling_correction, verl rollout_is)."""
    return torch.exp(old_lp - gen_lp).clamp(max=cap).detach()


def _causal_backbone_and_head(model):
    """Return (backbone, lm_head, final_logit_softcap) for a CausalLM, unwrapping PEFT."""
    causal = model.base_model.model if isinstance(model, PeftModel) else model
    softcap = getattr(causal.config, "final_logit_softcapping", None)
    return causal.model, causal.lm_head, softcap


def _chunk_stats(hidden, weight, bias, target_ids, start: int, softcap, temperature: float = 1.0):
    """(logsumexp over this vocab chunk, target logit if the target falls in it else 0).

    Logits are produced exactly like the HF forward (matmul in model dtype, then
    softcap ``x/c -> tanh -> *c`` in model dtype) and only then cast to fp32."""
    logits = torch.nn.functional.linear(hidden, weight, bias)
    if softcap:
        logits = logits / softcap
        logits = torch.tanh(logits)
        logits = logits * softcap
    logits = logits.float()
    if temperature != 1.0:  # the policy being optimized is the sampling distribution softmax(z / T)
        logits = logits / temperature
    lse = torch.logsumexp(logits, dim=-1)
    local = target_ids - start
    in_chunk = (local >= 0) & (local < logits.shape[-1])
    tgt = logits.gather(-1, local.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    return lse, torch.where(in_chunk, tgt, torch.zeros_like(tgt))


def _logprobs_from_hidden(hidden, lm_head, target_ids, vocab_chunk: int = 8192, softcap=None,
                          temperature: float = 1.0) -> torch.Tensor:
    """Log-softmax gathered at target positions, in vocab chunks. Under autograd each
    chunk is activation-checkpointed, so peak memory is one (B, T, chunk) block
    in forward AND backward (plain chunking would still save every chunk)."""
    V = lm_head.weight.shape[0]
    bias = lm_head.bias
    use_ckpt = torch.is_grad_enabled() and (hidden.requires_grad or lm_head.weight.requires_grad)
    lses, tgt = [], None
    for i in range(0, V, vocab_chunk):
        w = lm_head.weight[i:i + vocab_chunk]
        b = bias[i:i + vocab_chunk] if bias is not None else None
        if use_ckpt:
            lse, t = _activation_checkpoint(_chunk_stats, hidden, w, b, target_ids, i, softcap, temperature,
                                            use_reentrant=False)
        else:
            lse, t = _chunk_stats(hidden, w, b, target_ids, i, softcap, temperature)
        lses.append(lse)
        tgt = t if tgt is None else tgt + t
    logz = torch.logsumexp(torch.stack(lses, dim=-1), dim=-1)
    return tgt - logz


def forward_logprobs_chunked(model, input_ids, attention_mask, target_ids, vocab_chunk: int = 8192,
                             temperature: float = 1.0) -> torch.Tensor:
    """Per-token log-probs of softmax(logits / T) without materializing full-vocab logits."""
    backbone, lm_head, softcap = _causal_backbone_and_head(model)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)[0]
    return _logprobs_from_hidden(hidden, lm_head, target_ids, vocab_chunk, softcap, temperature)


def _policy_logprobs(model, batch, chunked: bool, temperature: float = 1.0) -> torch.Tensor:
    if chunked:
        return forward_logprobs_chunked(model, batch["input_ids"], batch["attention_mask"], batch["target_ids"],
                                        temperature=temperature)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits / temperature if temperature != 1.0 else out.logits
    lp = chunked_logprobs(logits, batch["target_ids"])
    del out
    return lp


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
# Local checkpoints
# ---------------------------------------------------------------------------
def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, run_dir: str, keep: int = 3):
        self.run_dir = run_dir
        self.keep = keep
        os.makedirs(run_dir, exist_ok=True)

    def _ckpt_dir(self, step: int) -> str:
        return os.path.join(self.run_dir, f"step-{step}")

    def save(self, step, model, optimizer, kl_coef, curriculum, sampler, gen_rng, config, run_id,
             dataset_hash) -> str:
        tmp = os.path.join(self.run_dir, "ckpt.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        model.save_pretrained(os.path.join(tmp, "adapter"), selected_adapters=["default"])
        torch.save({"optimizer": optimizer.state_dict(), "kl_coef": kl_coef, "sampler": sampler.getstate(),
                    "gen_rng": gen_rng.get_state()},
                   os.path.join(tmp, "train_state.pt"))
        np_state = np.random.get_state()
        _atomic_write_json(os.path.join(tmp, "state.json"), {
            "step": step, "run_id": run_id, "config_hash": config.hash(), "dataset_hash": dataset_hash,
            "curriculum": curriculum.state(),
            "rng_python": random.getstate(),
            "rng_numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
            "rng_torch": torch.get_rng_state().tolist(),
            "rng_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        })
        dest = self._ckpt_dir(step)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.rename(tmp, dest)
        _atomic_write_json(os.path.join(self.run_dir, "latest.json"), {"step": step})
        steps = sorted(int(d.split("-")[1]) for d in os.listdir(self.run_dir)
                       if d.startswith("step-") and d.split("-")[1].isdigit())
        for s in steps[:-self.keep]:
            shutil.rmtree(self._ckpt_dir(s), ignore_errors=True)
        return dest

    def latest_step(self) -> int | None:
        p = os.path.join(self.run_dir, "latest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            step = json.load(f)["step"]
        return step if os.path.isdir(self._ckpt_dir(step)) else None


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _load_base(model_id: str, device: str, dtype: str = "float32", attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation=attention_gqa.resolve(attn))
    return base, tokenizer


def _apply_lora(base, r: int, alpha: int):
    # dropout 0: the loss forward must be deterministic so the PPO ratio is exactly 1
    # (it runs in train mode so gradient checkpointing is active)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=LORA_TARGETS,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, cfg)


@torch.no_grad()
def sync_gen_model(model, gen_model) -> None:
    """Set every LoRA-targeted weight of ``gen_model`` (a plain copy of the base taken before
    LoRA injection) to base + delta, computed in fp32 from the untouched base weights, so
    nothing drifts across steps (unlike PEFT merge/unmerge in bf16)."""
    gen_mods = dict(gen_model.named_modules())
    n = 0
    for name, mod in model.named_modules():
        if hasattr(mod, "get_delta_weight") and hasattr(mod, "base_layer") and "." in name:
            key = name.split("base_model.model.", 1)[-1]
            tgt = gen_mods[key]
            w = mod.base_layer.weight
            delta = sum(mod.get_delta_weight(a) for a in mod.active_adapters)
            tgt.weight.copy_((w.float() + delta.float()).to(tgt.weight.dtype))
            n += 1
    if n == 0:
        raise RuntimeError("sync_gen_model: no LoRA layers found")


@contextlib.contextmanager
def _reference(model, kl_ref: str):
    """KL reference forward. "base": adapter disabled. "init": the frozen "ref" adapter is made the
    active one by setting each LoRA layer's ``_active_adapter`` directly; PEFT's ``set_adapter``
    would also flip requires_grad on the trainable adapter in the middle of its autograd graph."""
    if kl_ref == "base":
        with model.disable_adapter():
            yield
        return
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    prev = [m._active_adapter for m in layers]
    for m in layers:
        m._active_adapter = ["ref"]
    try:
        yield
    finally:
        for m, a in zip(layers, prev):
            m._active_adapter = a


def lr_at(cfg, step: int) -> float:
    """Linear warmup lr * (step+1)/N for step < N, then constant cfg.lr."""
    n = cfg.lr_warmup_steps
    return cfg.lr * min(1.0, (step + 1) / n) if n > 0 else cfg.lr


def parse_hint_schedule(spec: str) -> list[tuple[int, float]]:
    """"0.5@0,0.25@40,0@80" -> [(0, 0.5), (40, 0.25), (80, 0.0)] (sorted by start step)."""
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        frac, at = part.split("@")
        out.append((int(at), float(frac)))
    out.sort()
    if out[0][0] != 0 or any(not 0.0 <= f <= 1.0 for _, f in out):
        raise ValueError(f"hint_schedule must start at step 0 with fractions in [0, 1]: {spec!r}")
    return out


def hint_frac_at(schedule: list[tuple[int, float]], step: int) -> float:
    frac = 0.0
    for at, f in schedule:
        if step >= at:
            frac = f
    return frac


def split_validation(all_problems: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Hold out ``n`` problems (equal per tier, the LAST ones of each tier in file order) for
    validation; returns (train, val). Deterministic, and disjoint from training by construction."""
    if n <= 0:
        return all_problems, []
    by_tier: dict[str, list[int]] = {}
    for i, p in enumerate(all_problems):
        by_tier.setdefault(p["tier"], []).append(i)
    per = n // len(by_tier)
    held = {i for idx in by_tier.values() for i in idx[-per:]}
    return ([p for i, p in enumerate(all_problems) if i not in held],
            [p for i, p in enumerate(all_problems) if i in held])


def run_validation(llm, tokenizer, sandbox, val_problems, cfg, lora_req, reward_cfg, prefix, dump_path) -> dict:
    """Greedy pass over the validation problems with the current LoRA on the trainer's vLLM
    engine; writes an eval.py-compatible dump (verification/eval_compare.py works on it)."""
    import sft_coldstart
    from rollout_vllm import run_rollouts_vllm

    t0 = time.time()
    prompts = [sft_coldstart.with_prefix(prefix, sft_coldstart.problem_prompt(p["nums"], p["target"], cfg.prompt_style))
               for p in val_problems]
    flat = run_rollouts_vllm(llm, tokenizer, prompts, sandbox, max_rounds=cfg.max_rounds,
                             max_new_tokens=cfg.max_new_tokens, temperature=0.0,
                             on_round_cap=cfg.on_round_cap, lora_request=lora_req)
    tiers: dict[str, list[int]] = {}
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w") as f:
        for p, r in zip(val_problems, flat):
            c = compute_reward(r, p["nums"], p["target"], reward_cfg).correct
            tiers.setdefault(p["tier"], []).append(int(c > 0))
            f.write(json.dumps({"nums": p["nums"], "target": p["target"], "tier": p["tier"], "correct": c,
                                "completion": r.completion_text, "outputs": r.outputs,
                                "terminal_reason": r.terminal_reason}) + "\n")
    allc = [x for v in tiers.values() for x in v]
    return {"val_correct": sum(allc) / max(1, len(allc)), "val_n": len(allc),
            "val_tier": {t: round(sum(v) / len(v), 3) for t, v in sorted(tiers.items())},
            "val_s": round(time.time() - t0, 1)}


def _select_problems(all_problems: list[dict], n: int) -> list[dict]:
    """Balanced round-robin across tiers (file order within a tier); n=0 -> all."""
    by_tier: dict[str, list[dict]] = {}
    for p in all_problems:
        by_tier.setdefault(p["tier"], []).append(p)
    n = len(all_problems) if n <= 0 else min(n, len(all_problems))
    out, idx = [], {t: 0 for t in by_tier}
    while len(out) < n:
        for t in by_tier:
            if idx[t] < len(by_tier[t]) and len(out) < n:
                out.append(by_tier[t][idx[t]])
                idx[t] += 1
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: GRPOConfig) -> None:
    if not cfg.temperature > 0:
        raise SystemExit("temperature must be > 0 for GRPO sampling")
    device = cfg.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(cfg.train_file) as f:
        all_problems = [json.loads(line) for line in f]
    dataset_hash = ck.sha256_file(cfg.train_file)
    all_problems, val_problems = split_validation(all_problems, cfg.val_n)
    problems = _select_problems(all_problems, cfg.n_problems)
    by_tier: dict[str, list[dict]] = {}
    for p in problems:
        by_tier.setdefault(p["tier"], []).append(p)

    run_id = cfg.run_id or os.environ.get("RUN_ID", "")
    os.makedirs(cfg.output_dir, exist_ok=True)
    bucket = ck.get_bucket(cfg.backend, root=os.environ.get("CKPT_ROOT") or os.path.join(cfg.output_dir, "_remote"))
    ckpt = CheckpointManager(cfg.output_dir)
    cfg_hash = cfg.hash()

    # fresh host: restore the newest valid remote checkpoint of this run first
    if cfg.resume == "auto" and ckpt.latest_step() is None and bucket is not None and run_id:
        step = ck.find_latest_valid(bucket, run_id, None if cfg.force_resume else cfg_hash, dataset_hash)
        if step is not None and ck.download_checkpoint(bucket, run_id, step, ckpt._ckpt_dir(step), None,
                                                       dataset_hash) is not None:
            _atomic_write_json(os.path.join(cfg.output_dir, "latest.json"), {"step": step})
            print(f"restored remote checkpoint {run_id} step {step}")

    resume_dir = None
    if cfg.resume == "auto":
        latest = ckpt.latest_step()
        if latest is not None:
            resume_dir = ckpt._ckpt_dir(latest)
    elif cfg.resume and cfg.resume != "none":
        resume_dir = ckpt._ckpt_dir(int(cfg.resume))

    base, tokenizer = _load_base(cfg.model, device, cfg.dtype, cfg.attn)
    if cfg.init_adapter:  # SFT cold start: merged on EVERY start (fresh or resumed)
        base = PeftModel.from_pretrained(base, cfg.init_adapter).merge_and_unload()
    vllm_base = cfg.model
    if cfg.generation == "vllm" and cfg.init_adapter:  # vLLM loads weights from disk: write the merged SFT base once
        vllm_base = os.path.join(cfg.output_dir, "_vllm_base")
        if not os.path.exists(os.path.join(vllm_base, "config.json")):
            base.save_pretrained(vllm_base)
            tokenizer.save_pretrained(vllm_base)
    gen_model = None
    if cfg.gen_merged:  # pristine copy taken BEFORE LoRA injection (PEFT wraps base in place)
        import copy

        gen_model = copy.deepcopy(base).eval()
        for p_ in gen_model.parameters():
            p_.requires_grad_(False)

    curriculum = Curriculum(tiers=sorted(by_tier.keys()), G=cfg.G, window=cfg.curriculum_window,
                            mode=cfg.curriculum_mode, group_window=cfg.curriculum_group_window)
    kl_coef = cfg.kl_coef
    hint_sched = parse_hint_schedule(cfg.hint_schedule)
    start_step = 0
    train_state = None
    if resume_dir:
        with open(os.path.join(resume_dir, "state.json")) as f:
            state = json.load(f)
        if not cfg.force_resume:
            if state["config_hash"] != cfg_hash:
                raise SystemExit("config hash mismatch; pass --force-resume to override")
            if state.get("dataset_hash") not in (None, dataset_hash):
                raise SystemExit("dataset hash mismatch; pass --force-resume to override")
        train_state = torch.load(os.path.join(resume_dir, "train_state.pt"), map_location="cpu", weights_only=False)
        model = PeftModel.from_pretrained(base, os.path.join(resume_dir, "adapter"), is_trainable=True)
        start_step = state["step"]
        run_id = run_id or state.get("run_id", "")
        kl_coef = max(train_state["kl_coef"], cfg.min_kl_coef)
        curriculum.load_state(state["curriculum"])
        rp = state["rng_python"]
        random.setstate((rp[0], tuple(rp[1]), rp[2]))
        np_s = state["rng_numpy"]
        np.random.set_state((np_s[0], np.array(np_s[1], dtype=np.uint32), np_s[2], np_s[3], np_s[4]))
        torch.set_rng_state(torch.tensor(state["rng_torch"], dtype=torch.uint8))
        if state["rng_cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["rng_cuda"]])
    elif cfg.init_lora:
        model = PeftModel.from_pretrained(base, cfg.init_lora, is_trainable=True)
        lc = model.peft_config["default"]
        if (lc.r, lc.lora_alpha) != (cfg.lora_r, cfg.lora_alpha):
            raise SystemExit(f"init_lora has r={lc.r} alpha={lc.lora_alpha}; config says {cfg.lora_r}/{cfg.lora_alpha}")
    else:
        model = _apply_lora(base, cfg.lora_r, cfg.lora_alpha)
    run_id = run_id or f"rl-{int(time.time())}"
    if cfg.kl_ref == "init":  # frozen copy of the warm-start weights as the KL reference (fresh run AND resume)
        n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        model.load_adapter(cfg.init_lora, adapter_name="ref", is_trainable=False)
        assert sum(p_.numel() for p_ in model.parameters() if p_.requires_grad) == n_train
        assert all(not p_.requires_grad for n_, p_ in model.named_parameters() if ".ref." in n_)

    model.to(device)
    if gen_model is not None:
        gen_model.to(device)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    gen_rng = torch.Generator(device=device).manual_seed(cfg.seed)
    if train_state is not None:
        optimizer.load_state_dict(train_state["optimizer"])
        sampler = random.Random()
        sampler.setstate(train_state["sampler"])
        if "gen_rng" in train_state:
            gen_rng.set_state(train_state["gen_rng"])
    else:
        sampler = random.Random(cfg.seed + 12345)

    sandbox = Sandbox(n_workers=2, timeout=0.050)
    import sft_coldstart

    prefix = sft_coldstart.fewshot_prefix(sandbox, cfg.prompt_style)
    reward_cfg = RewardConfig(length_threshold=cfg.length_threshold, length_lambda=cfg.length_lambda,
                              zero_shaping_if_invalid_literals=cfg.zero_shaping_invalid_literals)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    log_rows = []
    stop_requested = {"flag": False}
    vllm_asleep = False
    llm = None  # generation=vllm: created lazily on the first step (after the training model is on the GPU)
    log_file = open(os.path.join(cfg.output_dir, "train_log.jsonl"), "a" if start_step > 0 else "w")

    # segment + cost bookkeeping
    seg_t0 = time.time()
    segment = ck.next_segment_index(bucket, run_id) if bucket is not None else 0
    prior_hours = ck.accumulated_hours(bucket, run_id)[0] if bucket is not None else 0.0
    cost_per_hour = cfg.cost_per_hour or float(os.environ.get("COST_PER_HOUR", "0") or 0)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
    if bucket is not None:
        ck.write_segment(bucket, run_id, segment, {
            "start_step": start_step, "device": device, "gpu": gpu, "started_at": int(seg_t0),
            "provider": os.environ.get("PROVIDER", "local"), "config_hash": cfg_hash,
            "cost_per_hour": cost_per_hour})
    last_ckpt = last_hb = time.time()

    def heartbeat(step, row=None):
        if bucket is None:
            return
        ck.write_heartbeat(bucket, run_id, {"step": step, "segment": segment,
                                            "segment_elapsed_s": int(time.time() - seg_t0),
                                            "reward": (row or {}).get("reward"),
                                            "correct": (row or {}).get("correct")})

    def save(step):
        d = ckpt.save(step, model, optimizer, kl_coef, curriculum, sampler, gen_rng, cfg, run_id, dataset_hash)
        if bucket is not None:
            ck.commit_dir(bucket, run_id, step, d, cfg_hash, dataset_hash)
            heartbeat(step)
        return step

    def _sig(signum, frame):
        stop_requested["flag"] = True

    old_handler = signal.signal(signal.SIGTERM, _sig)
    last_saved = None
    step = start_step
    for step in range(start_step, cfg.total_steps):
        for g in optimizer.param_groups:  # recomputed from step, so resumes get the same LR
            g["lr"] = lr_at(cfg, step)
        hours = prior_hours + (time.time() - seg_t0) / 3600
        over_hours = cfg.max_hours and hours >= cfg.max_hours
        over_usd = cfg.max_total_usd and cost_per_hour and hours * cost_per_hour >= cfg.max_total_usd
        if stop_requested["flag"] or over_hours or over_usd:
            why = "SIGTERM" if stop_requested["flag"] else ("max-hours" if over_hours else "max-total-usd")
            last_saved = save(step)
            print(f"{why}: saved checkpoint at step {step} ({hours:.2f} GPU-h)")
            break

        if cfg.generation == "vllm" and llm is None:
            from vllm import LLM

            llm = LLM(model=vllm_base, dtype="bfloat16", enable_lora=True, max_lora_rank=cfg.lora_r, max_loras=1,
                      gpu_memory_utilization=cfg.vllm_gpu_util, enable_prefix_caching=True,
                      max_model_len=cfg.vllm_max_model_len, logprobs_mode="processed_logprobs", seed=cfg.seed,
                      enable_sleep_mode=cfg.vllm_sleep)
            vllm_asleep = False
        if cfg.generation == "vllm" and cfg.vllm_sleep and vllm_asleep:
            # release the trainer's cached blocks first: vLLM re-maps its weights + KV cache on wake
            # and fails with CUDA OOM if PyTorch's allocator still holds the loss pass's memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            llm.wake_up()
            vllm_asleep = False
        lora_req = None
        if cfg.generation == "vllm":
            from vllm.lora.request import LoRARequest

            ad = os.path.join(cfg.output_dir, "_vllm_lora", f"s{step}")
            model.save_pretrained(ad, selected_adapters=["default"])
            lora_req = LoRARequest(f"s{step}", step + 1, ad)
            old = os.path.join(cfg.output_dir, "_vllm_lora", f"s{step - 2}")
            if os.path.isdir(old):
                shutil.rmtree(old, ignore_errors=True)
        val_row = {}
        if val_problems and cfg.val_every and lora_req is not None and step % cfg.val_every == 0:
            val_row = run_validation(llm, tokenizer, sandbox, val_problems, cfg, lora_req, reward_cfg, prefix,
                                     os.path.join(cfg.output_dir, "val", f"step{step}.jsonl"))
            print(f"val step {step}: correct {val_row['val_correct']:.3f} {val_row['val_tier']} "
                  f"({val_row['val_s']:.0f}s)", flush=True)
        model.eval()
        gen = model
        if gen_model is not None:
            sync_gen_model(model, gen_model)
            gen = gen_model
        t_gen = time.time()
        all_rewards, all_rollouts, train_rollouts, train_advs = [], [], [], []
        first_rewards = []  # first sampling round only: comparable across runs with/without refill
        first_rollouts = []
        gen_kw = dict(max_rounds=cfg.max_rounds, max_new_tokens=cfg.max_new_tokens, on_round_cap=cfg.on_round_cap,
                      temperature=cfg.temperature, rng=gen_rng, device=device)
        zero_var = n_signal = n_sampled = refills = 0
        hint_frac = hint_frac_at(hint_sched, step)
        n_hint_groups = n_hint_correct = n_hint_signal = n_hint_nocorrect = 0
        n_signal_nocorrect = 0
        hint_rollouts = []

        def _generate(prompts):
            if cfg.generation == "batched":
                return run_rollouts_batched(gen, tokenizer, prompts, sandbox, batch_size=cfg.gen_batch,
                                            impl=cfg.gen_impl, **gen_kw)
            if cfg.generation == "sequential":
                return [run_rollout(gen, tokenizer, pr, sandbox, **gen_kw) for pr in prompts]
            if cfg.generation == "vllm":
                from rollout_vllm import run_rollouts_vllm

                return run_rollouts_vllm(llm, tokenizer, prompts, sandbox, max_rounds=cfg.max_rounds,
                                         max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                                         on_round_cap=cfg.on_round_cap, lora_request=lora_req)
            raise SystemExit(f"generation must be batched|sequential|vllm, got {cfg.generation!r}")

        for rnd in range(1 + (cfg.max_refill_rounds if cfg.dynamic_sampling else 0)):
            k = cfg.prompts_per_step - (n_signal if rnd else 0)
            if k <= 0:
                break
            refills += int(rnd > 0)
            chosen = []
            for _ in range(k):
                tier = curriculum.sample_tier(sampler)
                chosen.append(sampler.choice(by_tier[tier]))
            n_sampled += k
            prompts = [sft_coldstart.with_prefix(prefix, sft_coldstart.problem_prompt(p["nums"], p["target"],
                                                                                     cfg.prompt_style))
                       for p in chosen for _ in range(cfg.G)]
            flat = _generate(prompts)
            all_fail = []
            for gi, p in enumerate(chosen):
                group = flat[gi * cfg.G:(gi + 1) * cfg.G]
                gr = [compute_reward(r, p["nums"], p["target"], reward_cfg) for r in group]
                a, zv = group_advantage([x.total for x in gr])
                if rnd == 0:
                    zero_var += int(zv)
                    first_rewards.extend(gr)
                    first_rollouts.extend(group)
                for r, x in zip(group, gr):
                    all_rewards.append(x)
                    all_rollouts.append(r)
                    curriculum.update(p["tier"], x.correct > 0)
                curriculum.update_group(p["tier"], sum(x.correct > 0 for x in gr), len(gr))
                # hint candidates: NO correct answer in the group. Not "zero variance": all-fail groups
                # usually still vary in the small exec/format shaping terms, so they would take a training
                # slot with shaping-only signal. With hints on, the hinted resample REPLACES such a group
                # (QuestA), up to hint_max_groups per step (smoke runs 2026-11-05).
                if (rnd == 0 and hint_frac > 0 and len(all_fail) < cfg.hint_max_groups
                        and not any(x.correct > 0 for x in gr)
                        and sft_coldstart.solution_hint(p.get("solution", ""), hint_frac)):
                    all_fail.append(p)
                    continue
                # zero-variance groups have zero advantage: no policy signal (DAPO dynamic
                # sampling refills them with fresh prompts instead of shrinking the batch)
                if not zv and n_signal < cfg.prompts_per_step and (cfg.train_nocorrect_groups or any(x.correct > 0 for x in gr)):
                    train_rollouts.extend(group)
                    train_advs.extend(a.tolist())
                    n_signal += 1
                    # trained group with NO correct answer: its advantages come only from the exec/format/
                    # length shaping (z-scored per group, so full-size); logged since 2026-11-05
                    n_signal_nocorrect += not any(x.correct > 0 for x in gr)
            # hinted round (once per step, right after the first round, ahead of the refills)
            if rnd == 0 and all_fail:
                hinted = [(p, sft_coldstart.solution_hint(p["solution"], hint_frac)) for p in all_fail]
                if hinted:
                    hp = [sft_coldstart.with_prefix(prefix, sft_coldstart.problem_prompt(p["nums"], p["target"],
                                                                                         cfg.prompt_style, solution_hint=h))
                          for p, h in hinted for _ in range(cfg.G)]
                    hflat = _generate(hp)
                    for gi, (p, _) in enumerate(hinted):
                        group = hflat[gi * cfg.G:(gi + 1) * cfg.G]
                        gr = [compute_reward(r, p["nums"], p["target"], reward_cfg) for r in group]
                        a, zv = group_advantage([x.total for x in gr])
                        n_hint_groups += 1
                        n_hint_correct += sum(x.correct > 0 for x in gr)
                        all_rewards.extend(gr)        # logged, but NOT fed to the curriculum
                        all_rollouts.extend(group)
                        hint_rollouts.extend(group)
                        if (not zv and n_signal < cfg.prompts_per_step
                                and (cfg.train_nocorrect_groups or any(x.correct > 0 for x in gr))):
                            train_rollouts.extend(group)
                            train_advs.extend(a.tolist())
                            n_signal += 1
                            n_hint_signal += 1
                            n_hint_nocorrect += not any(x.correct > 0 for x in gr)  # shaping-only, like n_signal_nocorrect
            if not cfg.dynamic_sampling:
                break

        gen_s = time.time() - t_gen
        if cfg.generation == "vllm" and cfg.vllm_sleep:
            llm.sleep(level=1)
            vllm_asleep = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        t_loss = time.time()
        kl_val, ratio_dev = float("nan"), float("nan")
        U = max(1, cfg.updates_per_batch)
        grad_norms = []
        upd = {"gen_logp_dev_mean": float("nan"), "gen_logp_dev_p99": float("nan"),
               "tis_w_mean": float("nan"), "clip_frac": float("nan"), "n_updates": 0}
        if train_rollouts:
            # train mode so gradient checkpointing applies; LoRA dropout is 0, so the forward is deterministic
            model.train()
            N = len(train_rollouts)
            # pi_old under training numerics. With one update per batch it is the same forward,
            # detached (ratio exactly 1). With several updates (verl ppo_mini_batch style) it must be
            # computed before the first optimizer step.
            old_lps = {}
            if cfg.shuffle_minibatches and U > 1 and N % cfg.G == 0:
                # whole groups, deterministic per step (so a resumed step shuffles the same way)
                order = list(range(N // cfg.G))
                random.Random(cfg.seed * 100_003 + step).shuffle(order)
                idx = [g * cfg.G + k for g in order for k in range(cfg.G)]
                train_rollouts = [train_rollouts[i] for i in idx]
                train_advs = [train_advs[i] for i in idx]
            mini = -(-N // U)
            mini = -(-mini // cfg.micro_batch) * cfg.micro_batch  # mini-batches hold whole micro-batches
            if U > 1:
                # the first mini-batch is updated before any optimizer step, so its own forward
                # (detached) is pi_old; only later mini-batches need a separate no-grad pass
                with torch.no_grad():
                    for i in range(mini, N, cfg.micro_batch):
                        batch = collate_rollouts(train_rollouts[i:i + cfg.micro_batch], pad_id, device)
                        old_lps[i] = _policy_logprobs(model, batch, cfg.chunked_vocab, cfg.temperature).detach()
                        del batch
            use_ref = kl_coef > 0 or cfg.max_kl_coef > 0
            kl_sum = kl_n = 0.0
            dev, devs, w_sum, clip_n, tok_n = 0.0, [], 0.0, 0.0, 0.0
            for m0 in range(0, N, mini):
                m_tok = max(1, sum(sum(r.loss_mask) for r in train_rollouts[m0:m0 + mini]))
                for i in range(m0, min(m0 + mini, N), cfg.micro_batch):
                    batch = collate_rollouts(train_rollouts[i:i + cfg.micro_batch], pad_id, device)
                    A = torch.tensor(train_advs[i:i + cfg.micro_batch], dtype=torch.float32, device=device).unsqueeze(1)
                    pmask = batch["pred_mask"].float()
                    policy_lp = _policy_logprobs(model, batch, cfg.chunked_vocab, cfg.temperature)
                    old_lp = old_lps.pop(i) if i in old_lps else policy_lp.detach()
                    surr, ratio = clipped_surrogate(policy_lp, old_lp, A, cfg.clip_eps, cfg.clip_eps_high)
                    per_tok = -surr
                    if cfg.tis_cap > 0:
                        w = tis_weights(old_lp, batch["old_logp"], cfg.tis_cap)
                        per_tok = per_tok * w
                        w_sum += float((w * pmask).sum())
                    # reference forward: always when the KL term is live; otherwise only on the
                    # first micro-batch of the step, as a cheap drift monitor
                    if use_ref or i == 0:
                        with torch.no_grad(), _reference(model, cfg.kl_ref):
                            ref_lp = _policy_logprobs(model, batch, cfg.chunked_vocab, cfg.temperature)
                        kl_tok = kl_k3(policy_lp, ref_lp)
                        kl_sum += float((kl_tok.detach() * pmask).sum())
                        kl_n += float(pmask.sum())
                        if use_ref:
                            per_tok = per_tok + kl_coef * kl_tok
                        del ref_lp
                    loss = (per_tok * pmask).sum() / m_tok
                    (loss / (cfg.grad_accum if U == 1 else 1)).backward()
                    m = pmask.bool()
                    gd = (old_lp - batch["old_logp"]).abs()[m]
                    if gd.numel():
                        dev = max(dev, float(gd.max()))
                        devs.append(gd.float().cpu())
                    hi = cfg.clip_eps_high or cfg.clip_eps
                    clip_n += float((((ratio < 1 - cfg.clip_eps) | (ratio > 1 + hi)).float() * pmask).sum())
                    tok_n += float(pmask.sum())
                    del policy_lp, loss, batch, per_tok, surr, ratio
                if U > 1:
                    grad_norms.append(_clip_and_norm(optimizer, cfg.max_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    upd["n_updates"] += 1
            kl_val, ratio_dev = (kl_sum / kl_n if kl_n else float("nan")), dev
            if devs:
                allv = torch.cat(devs)
                upd["gen_logp_dev_mean"] = float(allv.mean())
                upd["gen_logp_dev_p99"] = float(torch.quantile(allv[:1_000_000], 0.99))
            if cfg.tis_cap > 0 and tok_n:
                upd["tis_w_mean"] = w_sum / tok_n
            upd["clip_frac"] = clip_n / tok_n if tok_n else float("nan")
            if kl_val > cfg.target_kl * 1.5:
                kl_coef = min(kl_coef * cfg.kl_adapt_rate, cfg.max_kl_coef)
            elif kl_val < cfg.target_kl / 1.5:
                kl_coef = max(kl_coef / cfg.kl_adapt_rate, cfg.min_kl_coef)

        if U == 1 and (step + 1) % cfg.grad_accum == 0:
            grad_norms.append(_clip_and_norm(optimizer, cfg.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.empty_cache()  # combat MPS fragmentation over long runs

        n = len(first_rewards)
        row = {"step": step, "reward": sum(x.total for x in first_rewards) / n,
               "correct": sum(x.correct for x in first_rewards) / n,
               "exec": sum(x.exec for x in first_rewards) / n,
               "format": sum(x.format for x in first_rewards) / n,
               "length_penalty": sum(x.length_penalty for x in first_rewards) / n,
               "hallucinated": sum(x.hallucinated for x in first_rewards) / n,
               "contradicted": sum(x.contradicted for x in first_rewards) / n,
               **behaviour_metrics(first_rollouts),
               "kl": kl_val, "kl_coef": kl_coef, "gen_train_logp_dev": ratio_dev,
               "zero_var_frac": zero_var / cfg.prompts_per_step,
               "n_train_rollouts": len(train_rollouts),
               "n_signal_groups": n_signal, "n_sampled_prompts": n_sampled, "refill_rounds": refills,
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in upd.items()},
               "gen_s": round(gen_s, 1), "loss_s": round(time.time() - t_loss, 1),
               "gen_tokens": sum(sum(r.loss_mask) for r in all_rollouts),
               "tier_pass": {t: round(curriculum.pass_rate(t), 3) for t in curriculum.tiers},
               "tier_signal": {t: round(curriculum.signal_rate(t), 3) for t in curriculum.tiers},
               "tier_weight": {t: round(curriculum.weight(t), 3) for t in curriculum.tiers},
               "invalid_literals": sum(x.invalid_literals for x in first_rewards) / n,
               "n_signal_nocorrect": n_signal_nocorrect,
               "lr": optimizer.param_groups[0]["lr"], "hint_frac": hint_frac, "n_hint_groups": n_hint_groups, "n_hint_signal": n_hint_signal,
               "hint_correct": n_hint_correct / max(1, n_hint_groups * cfg.G),
               "n_hint_nocorrect": n_hint_nocorrect,
               "grad_norm": round(max(grad_norms), 4) if grad_norms else float("nan"),
               **sampled_nll(all_rollouts),
               **val_row}
        log_rows.append(row)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if (step + 1) % cfg.log_every == 0 or step == start_step:
            print(f"step {step} reward {row['reward']:.3f} correct {row['correct']:.3f} "
                  f"exec {row['exec']:.3f} format {row['format']:.3f} kl {kl_val:.4f} "
                  f"gen/train dlogp {ratio_dev:.3g} zero_var {row['zero_var_frac']:.2f} "
                  f"gen {row['gen_s']}s loss {row['loss_s']}s")

        if time.time() - last_hb > 120:
            heartbeat(step, row)
            last_hb = time.time()

        tripped = guard_tripped(log_rows, cfg)
        if tripped:
            last_saved = save(step + 1)
            print(f"GUARD: {tripped}; saved checkpoint at step {step + 1} and stopped")
            with open(os.path.join(cfg.output_dir, "GUARD_STOP"), "w") as f:
                f.write(tripped + "\n")
            break

        if (step + 1) % cfg.rollout_dump_every == 0:
            _dump_rollouts(cfg.output_dir, step, all_rollouts[:4] + hint_rollouts[:4])

        if ((step + 1) % cfg.checkpoint_every == 0) or (time.time() - last_ckpt > cfg.checkpoint_timer):
            last_saved = save(step + 1)
            last_ckpt = time.time()
    else:
        if start_step < cfg.total_steps and last_saved != cfg.total_steps:
            last_saved = save(cfg.total_steps)

    signal.signal(signal.SIGTERM, old_handler)
    sandbox.close()
    log_file.close()
    print(f"run {run_id} done; {len(log_rows)} steps -> {cfg.output_dir}/train_log.jsonl")


def behaviour_metrics(rollouts) -> dict:
    """Shortcut monitors: of the rollouts whose FIRST tool output printed False, the share
    that ran another check afterwards (retry_after_false), and the share of rollouts that
    hit the token budget (frac_max_length)."""
    if not rollouts:
        return {}
    ff = [r for r in rollouts if r.outputs and "False" in r.outputs[0].split()]
    # strict: the first check printed no True at all (with the multi-candidate prompt, a check where candidate 1
    # failed but a later one passed counts as "False" above but is not a failure; logged since 2026-11-06)
    af = [r for r in rollouts if r.outputs and "True" not in r.outputs[0].split()]
    return {"retry_after_false": (sum(len(r.outputs) > 1 for r in ff) / len(ff)) if ff else float("nan"),
            "n_first_false": len(ff),
            "retry_after_allfalse": (sum(len(r.outputs) > 1 for r in af) / len(af)) if af else float("nan"),
            "n_first_allfalse": len(af),
            "frac_max_length": sum(r.terminal_reason == "max_length" for r in rollouts) / len(rollouts)}


def _clip_and_norm(optimizer, max_norm: float) -> float:
    """Total gradient norm of the optimized parameters (pre-clip); clips to max_norm when it is > 0."""
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    if not params:
        return float("nan")
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm if max_norm > 0 else float("inf")))


def sampled_nll(rollouts) -> dict:
    """Mean negative log-prob of the model's own sampled tokens (sampler log-probs at the sampling temperature):
    a Monte Carlo estimate of the policy's per-token entropy. Env tokens (loss_mask 0) are excluded."""
    tot = cnt = 0.0
    for r in rollouts:
        for lp, m in zip(r.old_logprobs, r.loss_mask):
            if m:
                tot -= lp
                cnt += 1
    return {"entropy_est": round(tot / cnt, 4) if cnt else float("nan")}


def guard_tripped(log_rows: list[dict], cfg) -> str:
    """Return a reason string if a shortcut guard trips over the last ``guard_window`` steps."""
    w = cfg.guard_window
    if not w or len(log_rows) < w:
        return ""
    last = log_rows[-w:]
    c = sum(r.get("contradicted", 0.0) for r in last) / w
    m = sum(r.get("frac_max_length", 0.0) for r in last) / w
    if c > cfg.guard_contradicted:
        return f"contradicted {c:.2f} > {cfg.guard_contradicted} over the last {w} steps"
    if m > cfg.guard_max_length:
        return f"frac_max_length {m:.2f} > {cfg.guard_max_length} over the last {w} steps"
    return ""


def _dump_rollouts(output_dir, step, rollouts):
    with open(os.path.join(output_dir, "rollouts.jsonl"), "a") as f:
        for r in rollouts:
            f.write(json.dumps({"step": step, "transcript": r.transcript,
                                "answer": r.answer_text, "has_answer": r.has_answer,
                                "code_rounds": r.code_rounds, "outputs": r.outputs,
                                "terminal_reason": r.terminal_reason}) + "\n")


# ---------------------------------------------------------------------------
# CLI: defaults < --config YAML < explicitly passed flags
# ---------------------------------------------------------------------------
def build_config(argv=None) -> GRPOConfig:
    ap = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    ap.add_argument("--config", help="YAML preset (configs/preset-*.yaml); explicit flags override it")
    for f in dataclasses.fields(GRPOConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction)
        else:
            ap.add_argument(flag, type=type(f.default))
    args = vars(ap.parse_args(argv))
    kwargs = {}
    if "config" in args:
        import yaml

        with open(args.pop("config")) as fh:
            preset = yaml.safe_load(fh) or {}
        unknown = sorted(set(preset) - {f.name for f in dataclasses.fields(GRPOConfig)})
        if unknown:
            raise SystemExit(f"unknown keys in config: {unknown}")
        kwargs.update(preset)
    kwargs.update(args)
    cfg = GRPOConfig(**kwargs)
    if cfg.updates_per_batch > 1 and cfg.grad_accum != 1:
        raise SystemExit("updates_per_batch > 1 steps the optimizer per mini-batch; use grad_accum: 1")
    if cfg.tis_cap and cfg.tis_cap < 1:
        raise SystemExit("tis_cap must be 0 (off) or >= 1")
    try:
        parse_hint_schedule(cfg.hint_schedule)
    except ValueError as e:
        raise SystemExit(str(e))
    if cfg.kl_ref not in ("base", "init"):
        raise SystemExit(f"kl_ref must be base|init, got {cfg.kl_ref!r}")
    if cfg.kl_ref == "init" and not cfg.init_lora:
        raise SystemExit("kl_ref: init needs init_lora (the reference is the warm-start adapter)")
    if cfg.init_adapter and cfg.init_lora:
        raise SystemExit("use init_adapter (merge, e.g. SFT) OR init_lora (unmerged warm start), not both")
    if cfg.curriculum_mode not in ("formula", "signal"):
        raise SystemExit(f"curriculum_mode must be formula|signal, got {cfg.curriculum_mode!r}")
    if cfg.val_every and (cfg.val_n <= 0 or cfg.generation != "vllm"):
        raise SystemExit("val_every needs val_n > 0 and generation: vllm (it reuses the trainer's vLLM engine)")
    return cfg


def main(argv=None) -> None:
    train(build_config(argv))


if __name__ == "__main__":
    main()

"""Evaluation: Pass@1, format, length, self-correction, round scaling.

Compares base (optional few-shot) vs SFT vs RL by running this CLI once per
model configuration on the same eval subset:

    base:  --model google/gemma-2-2b [--fewshot]
    SFT:   --model google/gemma-2-2b --init-adapter outputs/sft_gemma
    RL:    --model google/gemma-2-2b --init-adapter outputs/sft_gemma --adapter outputs/rl/step-N/adapter

``--init-adapter`` is merged into the base first (the RL adapter was trained on
top of the merged SFT model), then ``--adapter`` is applied.

**Self-correction definition (written down):**
- A rollout has a *failing execution* if any REAL injected sandbox output
  (``RolloutResult.outputs``, after the control transform) begins with ``False``
  (the model's own check reported a wrong candidate).
- *self-correction rate* = among rollouts with >=1 failing execution:
  (a) ``self_correction_correct``: the fraction whose final ``<answer>`` verifies;
  (b) ``self_correction_backtrack``: the fraction whose model-generated text
      AFTER the first failing output contains an explicit backtracking cue
      (``BACKTRACK_CUES``).
- *Controls* (``--control blank|shuffle``): the same eval with outputs blanked or
  with digits shuffled. A policy that uses the feedback should self-correct more
  with real outputs than under the controls.

**Rounds sweep** (``--rounds-sweep``): Pass@1 with at most 0, 1, 2 code
executions. Past the cap the model receives an ``ExecutionLimit`` notice instead
of an output and can still answer (``on_round_cap="notice"``), so the sweep
measures what extra executions buy rather than truncation.

Decoding defaults to greedy (``--temperature 0``) for reproducible Pass@1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

import torch

from countdown_rl.rl import attention

from countdown_rl.rl.rewards import RewardConfig, check_format, compute_reward
from countdown_rl.rl.rollout import OUTPUT_CLOSE, OUTPUT_OPEN, RolloutResult, format_prompt, run_rollout, run_rollouts_batched
from countdown_rl.sandbox import Sandbox

BACKTRACK_CUES = ("wrong", "incorrect", "not correct", "mistake", "try again", "retry", "doesn't work",
                  "does not work", "too high", "too low", "adjust", "that is not", "instead")


def blank_output(s: str) -> str:
    return ""


def shuffle_output(s: str) -> str:
    def _perm(m):
        d = list(m.group())
        random.shuffle(d)
        return "".join(d)

    return re.sub(r"\d+", _perm, s)


def output_contents(text: str) -> list[str]:
    """``<output>`` bodies found in text (kept for transcript inspection)."""
    return [m.strip() for m in re.findall(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL)]


def has_failing_execution(rr: RolloutResult) -> bool:
    return any(o.strip().startswith("False") for o in rr.outputs)


def text_after_first_failure(rr: RolloutResult) -> str:
    """Model-relevant text following the first failing injected output ('' if none)."""
    for o in rr.outputs:
        if o.strip().startswith("False"):
            marker = f"{OUTPUT_OPEN}\n{o}\n{OUTPUT_CLOSE}"
            i = rr.completion_text.find(marker)
            return rr.completion_text[i + len(marker):] if i != -1 else ""
    return ""


def has_backtracking_text(rr: RolloutResult) -> bool:
    low = text_after_first_failure(rr).lower()
    return any(cue in low for cue in BACKTRACK_CUES)


def compute_metrics(rows: list[dict]) -> dict:
    """rows: list of {"nums","target","tier","rollout","reward"}."""
    n = len(rows)
    correct = sum(1 for r in rows if r["reward"].correct > 0)
    fmt_ok = sum(1 for r in rows if check_format(r["rollout"].completion_text))
    lengths = sorted(sum(r["rollout"].loss_mask) for r in rows)

    per_tier = {}
    for r in rows:
        t = r["tier"]
        per_tier.setdefault(t, {"n": 0, "correct": 0})
        per_tier[t]["n"] += 1
        per_tier[t]["correct"] += int(r["reward"].correct > 0)

    failing = [r for r in rows if has_failing_execution(r["rollout"])]
    sc_correct = sum(1 for r in failing if r["reward"].correct > 0) / len(failing) if failing else 0.0
    sc_backtrack = sum(1 for r in failing if has_backtracking_text(r["rollout"])) / len(failing) if failing else 0.0

    def q(p):
        if not lengths:
            return 0
        return lengths[min(int(p * n), n - 1)]

    return {
        "n": n,
        "pass@1": correct / n if n else 0.0,
        "format_compliance": fmt_ok / n if n else 0.0,
        "length_mean": sum(lengths) / n if n else 0.0,
        "length_p50": q(0.50),
        "length_p95": q(0.95),
        "length_max": lengths[-1] if lengths else 0,
        "per_tier_pass@1": {t: (v["correct"] / v["n"]) for t, v in sorted(per_tier.items())},
        "mean_code_rounds": sum(r["rollout"].code_rounds for r in rows) / n if n else 0.0,
        "hallucination_rate": sum(r["rollout"].hallucinated_output for r in rows) / n if n else 0.0,
        "n_failing": len(failing),
        "self_correction_correct": sc_correct,
        "self_correction_backtrack": sc_backtrack,
    }


def stratified_subset(problems: list[dict], n: int) -> list[dict]:
    """First ``n`` problems taken round-robin across tiers (eval.jsonl is tier-sorted)."""
    if n <= 0 or n >= len(problems):
        return problems
    by_tier: dict[str, list[dict]] = {}
    for p in problems:
        by_tier.setdefault(p["tier"], []).append(p)
    out, i = [], 0
    while len(out) < n:
        for t in by_tier:
            if i < len(by_tier[t]) and len(out) < n:
                out.append(by_tier[t][i])
        i += 1
    return out


def _load(model_id: str, init_adapter: str | None, adapter: str | None, device: str, dtype: str, attn: str,
          merge_adapter: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, attn_implementation=attention.resolve(attn))
    if init_adapter or adapter:
        from peft import PeftModel

        if init_adapter:
            model = PeftModel.from_pretrained(model, init_adapter).merge_and_unload()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
            if merge_adapter:  # ~1.27x faster decode on soos (no per-layer LoRA side matmuls); bf16 rounding differs
                model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer


def make_fewshot(sandbox: Sandbox, seed: int = 0, style: str = "fewshot") -> str:
    from countdown_rl.rl import prompting

    return prompting.fewshot_prefix(sandbox, style, seed)


def evaluate(model, tokenizer, problems, sandbox, *, fewshot=None, max_rounds=2, max_new_tokens=512,
             temperature=0.0, seed=0, device="cpu", control="real", on_round_cap="notice",
             generation="batched", batch_size=32, prompt_style="zero") -> list[dict]:
    transform = {"real": None, "blank": blank_output, "shuffle": shuffle_output}[control]
    random.seed(seed)
    gen_rng = torch.Generator(device=device).manual_seed(seed)
    from countdown_rl.rl import prompting

    prompts = []
    for p in problems:
        prompt = prompting.problem_prompt(p["nums"], p["target"], prompt_style)
        prompts.append(fewshot + "\n\n" + prompt if fewshot else prompt)
    kw = dict(max_rounds=max_rounds, max_new_tokens=max_new_tokens, temperature=temperature, rng=gen_rng,
              device=device, output_transform=transform, on_round_cap=on_round_cap)
    if generation == "batched":
        rolls = run_rollouts_batched(model, tokenizer, prompts, sandbox, batch_size=batch_size, **kw)
    else:
        rolls = [run_rollout(model, tokenizer, pr, sandbox, **kw) for pr in prompts]
    rows = []
    for p, rr in zip(problems, rolls):
        reward = compute_reward(rr, p["nums"], p["target"], RewardConfig())
        rows.append({"nums": p["nums"], "target": p["target"], "tier": p["tier"], "rollout": rr, "reward": reward})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--init-adapter", default="", help="SFT adapter, merged into the base first")
    ap.add_argument("--adapter", default="", help="LoRA adapter applied on top (e.g. the RL adapter)")
    ap.add_argument("--eval-file", default="data/eval.jsonl")
    ap.add_argument("--n-eval", type=int, default=0, help="0 = all; otherwise a tier-stratified subset")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--rounds-sweep", action="store_true", help="also report Pass@1 for 0/1/2 rounds")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dtype", default="auto", help="auto (cuda: bfloat16, else float32) | float32 | float16 | bfloat16")
    ap.add_argument("--attn", default="eager", help="eager (required for gemma-2 softcapping) | sdpa")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true", help="prepend few-shot traces (= --prompt-style fewshot)")
    from countdown_rl.rl import prompting

    ap.add_argument("--prompt-style", default="", choices=[""] + sorted(prompting.PROMPT_STYLES),
                    help="zero | fewshot_direct (format only) | fewshot (also shows a failed check + fix)")
    ap.add_argument("--generation", default="batched", choices=["batched", "sequential"])
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per batched decode")
    ap.add_argument("--control", default="real", choices=["real", "blank", "shuffle"])
    ap.add_argument("--merge-adapter", action="store_true",
                    help="merge --adapter into the weights (faster; not bit-identical to the unmerged runs, so keep "
                         "it off when comparing against earlier unmerged evaluations of the same problems)")
    ap.add_argument("--out", default="", help="write metrics JSON here")
    ap.add_argument("--dump", default="", help="write per-problem transcripts JSONL here")
    args = ap.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype if args.dtype != "auto" else ("bfloat16" if device == "cuda" else "float32")

    with open(args.eval_file) as f:
        problems = stratified_subset([json.loads(line) for line in f], args.n_eval)

    model, tokenizer = _load(args.model, args.init_adapter or None, args.adapter or None, device, dtype, args.attn,
                             merge_adapter=args.merge_adapter)
    sandbox = Sandbox(n_workers=2, timeout=0.050)
    result = {"config": {k: v for k, v in vars(args).items()} | {"dtype": dtype, "device": device}}
    try:
        style = args.prompt_style or ("fewshot" if args.fewshot else "zero")
        fewshot = make_fewshot(sandbox, style=style) or None
        kw = dict(fewshot=fewshot, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  seed=args.seed, device=device, control=args.control, generation=args.generation,
                  batch_size=args.batch_size, prompt_style=style)
        rows = evaluate(model, tokenizer, problems, sandbox, max_rounds=args.max_rounds, **kw)
        result["metrics"] = compute_metrics(rows)
        if args.rounds_sweep:
            result["rounds_sweep_pass@1"] = {}
            for k in (0, 1, 2):
                rk = rows if k == args.max_rounds else evaluate(model, tokenizer, problems, sandbox, max_rounds=k, **kw)
                result["rounds_sweep_pass@1"][k] = compute_metrics(rk)["pass@1"]
    finally:
        sandbox.close()

    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    if args.dump:
        with open(args.dump, "w") as f:
            for r in rows:
                rr = r["rollout"]
                f.write(json.dumps({"nums": r["nums"], "target": r["target"], "tier": r["tier"],
                                    "correct": r["reward"].correct, "completion": rr.completion_text,
                                    "outputs": rr.outputs, "terminal_reason": rr.terminal_reason}) + "\n")


if __name__ == "__main__":
    main()

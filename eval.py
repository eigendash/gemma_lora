"""Evaluation: Pass@1, format, length, self-correction, round scaling.

Compares base (optional few-shot) vs SFT vs RL by running this CLI once per
model configuration on the same eval subset.

**Self-correction definition (written down):**
- A rollout has a *failing execution* if any injected ``<output>`` content begins
  with ``False`` (the sandbox reported a wrong candidate result).
- *self-correction rate* = among rollouts with >=1 failing execution:
  (a) the fraction whose final ``<answer>`` verifies, and
  (b) the fraction whose completion contains explicit backtracking text.
- *Control:* the same eval is re-run with outputs blanked (empty) and shuffled
  (digits of every number permuted). A policy that actually uses the feedback
  must have a higher real-output self-correction rate than the controls.
"""

from __future__ import annotations

import argparse
import json
import random
import re

import torch

from rewards import RewardConfig, check_format, compute_reward
from rollout import RolloutResult, format_prompt, run_rollout
from sandbox import Sandbox
from verifier import verify_answer

BACKTRACK_CUES = ("wrong", "not correct", "incorrect", "mistake", "adjust", "retry", "instead",
                  "correct", "no,", "try again", "let me", "that is not", "close")


def blank_output(s: str) -> str:
    return ""


def shuffle_output(s: str) -> str:
    def _perm(m):
        d = list(m.group())
        random.shuffle(d)
        return "".join(d)

    return re.sub(r"\d+", _perm, s)


def output_contents(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL)]


def has_failing_execution(rr: RolloutResult) -> bool:
    return any(c.startswith("False") for c in output_contents(rr.completion_text))


def has_backtracking_text(rr: RolloutResult) -> bool:
    low = rr.completion_text.lower()
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
        i = min(int(p * n), n - 1)
        return lengths[i]

    return {
        "n": n,
        "pass@1": correct / n if n else 0.0,
        "format_compliance": fmt_ok / n if n else 0.0,
        "length_mean": sum(lengths) / n if n else 0.0,
        "length_p50": q(0.50),
        "length_p95": q(0.95),
        "length_max": lengths[-1] if lengths else 0,
        "per_tier_pass@1": {t: (v["correct"] / v["n"]) for t, v in sorted(per_tier.items())},
        "n_failing": len(failing),
        "self_correction_correct": sc_correct,
        "self_correction_backtrack": sc_backtrack,
    }


def _load(model_id: str, adapter: str | None, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, attn_implementation="sdpa")
    model = base
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter)
    model.to(device)
    model.eval()
    return model, tokenizer


def make_fewshot(sandbox: Sandbox, seed: int = 0, k: int = 2) -> str:
    import sft_coldstart

    problems = [
        {"nums": [25, 3, 1], "target": 76, "solution": "(25*3)+1"},
        {"nums": [100, 75, 3, 5], "target": 875, "solution": "(100+75)*5"},
    ]
    rng = random.Random(seed)
    parts = []
    for p in problems[:k]:
        segs = sft_coldstart.generate_trace(p, rng, sandbox)
        parts.append(format_prompt(p["nums"], p["target"]) + "".join(t for _, t in segs))
    return "\n\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--adapter", default="", help="optional LoRA adapter (SFT or RL)")
    ap.add_argument("--eval-file", default="data/eval.jsonl")
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", action="store_true", help="prepend few-shot traces (base model)")
    ap.add_argument("--control", default="real", choices=["real", "blank", "shuffle"])
    args = ap.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.eval_file) as f:
        problems = [json.loads(line) for line in f][: args.n_eval]

    model, tokenizer = _load(args.model, args.adapter or None, device)
    sandbox = Sandbox(n_workers=2, timeout=0.050)
    try:
        fewshot = make_fewshot(sandbox) if args.fewshot else None
        transform = {"real": None, "blank": blank_output, "shuffle": shuffle_output}[args.control]
        if args.control == "shuffle":
            random.seed(args.seed)
        gen_rng = torch.Generator(device=device).manual_seed(args.seed)
        rows = []
        for p in problems:
            prompt = format_prompt(p["nums"], p["target"])
            if fewshot:
                prompt = fewshot + "\n\n" + prompt
            rr = run_rollout(model, tokenizer, prompt, sandbox, max_rounds=args.max_rounds,
                             max_new_tokens=args.max_new_tokens, temperature=1.0, rng=gen_rng,
                             device=device, output_transform=transform)
            reward = compute_reward(rr, p["nums"], p["target"], RewardConfig())
            rows.append({"nums": p["nums"], "target": p["target"], "tier": p["tier"],
                         "rollout": rr, "reward": reward})
    finally:
        sandbox.close()

    m = compute_metrics(rows)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()

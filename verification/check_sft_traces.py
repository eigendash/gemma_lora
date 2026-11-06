"""Independent SFT cold-start checks.

- traces come from the solver/data (no human CoT); every <output> is real
  sandbox stdout with no error; the trace never answers right after a 'False';
  the last real output before the answer shows 'True' or the target value; the
  final answer passes the independent ref verifier;
- SFT labels: every token of the injected <output> segments is -100 and no
  trainable token decodes to anything inside an injected segment;
- problems come from the train split only (no eval key);
- prints 10 random traces for eyeballing.

Usage: python verification/check_sft_traces.py [--tokenizer Qwen/Qwen2.5-0.5B]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_countdown import ref_verify  # noqa: E402

import sft_coldstart
from rollout import format_prompt
from sandbox import Sandbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    train = [json.loads(l) for l in open("data/train.jsonl")]
    evalk = {(tuple(sorted(p["nums"])), p["target"]) for p in map(json.loads, open("data/eval.jsonl"))}
    fails = []
    from collections import Counter
    kinds = Counter()
    with Sandbox(n_workers=2) as sb:
        chosen, traces = sft_coldstart.make_traces(train, a.n, 0, sb)
    for p, segs in zip(chosen, traces):
        if (tuple(sorted(p["nums"])), p["target"]) in evalk:
            fails.append(("eval-leak", p))
        env = [t for e, t in segs if e]
        bodies = [t.strip().removeprefix("<output>").removesuffix("</output>").strip() for t in env]
        if not env or any(b.startswith(("SecurityError", "RuntimeError", "TimeoutError")) for b in bodies):
            fails.append(("sandbox-error", p, bodies))
        # never answer right after a failing check; the last real output must show the target
        last = bodies[-1] if bodies else ""
        def _nums(text):
            out = []
            for tok_ in text.split():
                try:
                    out.append(float(tok_))
                except ValueError:
                    pass
            return out
        if last.startswith("False") or (not last.startswith("True") and float(p["target"]) not in _nums(last)):
            fails.append(("answer-after-miss", p, bodies))
        ans = segs[-1][1].split("<answer>\n")[1].split("\n</answer>")[0]
        if not ref_verify(p["nums"], p["target"], ans):
            fails.append(("answer", p, ans))
        kinds[{(1, True): "direct", (2, True): "near_miss", (2, False): "resplit"}.get(
            (len(env), bodies[0].startswith(("True", "False"))), "split" if len(env) == 1 else "other")] += 1
        ex = sft_coldstart.build_example(tok, format_prompt(p["nums"], p["target"]), segs, max_length=10_000)
        ids, labels = ex["input_ids"], ex["labels"]
        train_text = tok.decode([i for i, l in zip(ids, labels) if l != -100])
        if "<output>" in train_text or "</output>" in train_text:
            fails.append(("mask-leak", train_text[:120]))
        # every injected token masked: the env segments' token ids appear as -100 runs
        masked = [i for i, l in zip(ids, labels) if l == -100]
        for e in env:
            eids = tok.encode(e, add_special_tokens=False)
            s = ",".join(map(str, eids))
            if s not in ",".join(map(str, masked)):
                fails.append(("env-not-masked", e))
    print(f"checked {len(chosen)} traces; failures: {len(fails)}")
    print(f"trace kind distribution: {dict(kinds)}")
    for f in fails[:5]:
        print("  FAIL", f)
    rng = random.Random(1)
    for i in rng.sample(range(len(traces)), 10):
        print(f"\n----- trace {i}: nums={chosen[i]['nums']} target={chosen[i]['target']}")
        print("".join(t for _, t in traces[i]))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

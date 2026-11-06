"""Side-by-side summary of GRPO pilot runs pulled from soos.

usage: python verification/compare_pilots.py <dir>  (dir holds train_log_<tag>.jsonl and <P>_eval_step*.json[l])
Runs: A = format-only (P1), B = +retry instruction (P2), C = literature-aligned (P3).
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

RUNS = [("A format-only", "train_log.jsonl", "P1"),
        ("B +retry hint", "train_log_retry.jsonl", "P2"),
        ("C literature-aligned", "train_log_lit.jsonl", "P3"),
        ("C' lit, 16 prompts/step, contradicted-answer rule", "train_log_lit2.jsonl", "P4"),
        ("C'' lit2 + 2 updates/batch, clip-high, TIS, 768 tokens", "train_log_lit3.jsonl", "P5"),
        ("D lit4 = C'' + sdpa_gqa + merged gen + guards (go/no-go)", "train_log_lit4.jsonl", "P6")]


def m(seg, k):
    v = [x[k] for x in seg if k in x and x[k] == x[k]]
    return st.mean(v) if v else float("nan")


def main(d):
    for name, log, tag in RUNS:
        p = os.path.join(d, log)
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        print(f"== {name}: {len(rows)} steps")
        w = 10 if len(rows) > 20 else 2
        for i in range(0, len(rows), w):
            s = rows[i:i + w]
            extra = ""
            if "n_sampled_prompts" in s[0]:
                extra = f" sampled/step {m(s, 'n_sampled_prompts'):.0f} signal groups {m(s, 'n_signal_groups'):.1f}"
            print(f"  steps {s[0]['step']:3d}-{s[-1]['step']:3d}: correct {m(s, 'correct'):.3f} "
                  f"zero_var {m(s, 'zero_var_frac'):.2f} kl {m(s, 'kl'):.4f} "
                  f"tok/rollout {m(s, 'gen_tokens') / max(1, 64 if 'n_sampled_prompts' not in s[0] else 8 * m(s, 'n_sampled_prompts')):.0f} "
                  f"step_s {m(s, 'gen_s') + m(s, 'loss_s'):.0f}{extra}")
        print("  tier_pass (window):", rows[-1].get("tier_pass"))
        out = []
        for f in glob.glob(os.path.join(d, f"{tag}_eval_step*.json")):
            n = int(re.search(r"step(\d+)", f).group(1))
            mm = json.load(open(f))["metrics"]
            jl = f.replace(".json", ".jsonl")
            rr = [json.loads(l) for l in open(jl)] if os.path.exists(jl) else []
            ff = [r for r in rr if r["outputs"] and r["outputs"][0].startswith("False")]
            recov = sum(1 for r in ff if r["correct"] > 0)
            n_eval = len(rr) or mm.get("n", 48)
            out.append((n, f'{round(mm["pass@1"] * n_eval)}/{n_eval}', sum(len(r["outputs"]) > 1 for r in ff), len(ff), recov,
                        {k: round(v, 2) for k, v in mm["per_tier_pass@1"].items()}))
        print(f"  held-out (step, correct/n, retries after first False / first-False, recovered-correct, tiers):")
        for o in sorted(out):
            print("    ", o)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "verification/logs/r6_pilot")

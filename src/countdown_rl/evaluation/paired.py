"""Paired comparison of two eval dumps on the SAME problems (evaluate.py --dump JSONL).

Reports correct counts, the discordant pairs (solved only by A / only by B) and an exact
two-sided McNemar (binomial) p-value. With a fixed eval set, a paired test is far more
sensitive than comparing two pass@1 numbers with independent-sample error bars.

usage: python countdown_rl.evaluation.paired base.jsonl ckpt.jsonl [more.jsonl ...]
       (each later file is compared against the first)
"""

from __future__ import annotations

import json
import math
import sys


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(tuple(r["nums"]), r["target"])] = (r["correct"] > 0, r.get("tier", "?"))
    return rows


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a_path, b_path):
    A, B = load(a_path), load(b_path)
    keys = sorted(set(A) & set(B))
    a = sum(A[k][0] for k in keys)
    b = sum(B[k][0] for k in keys)
    only_a = sum(A[k][0] and not B[k][0] for k in keys)
    only_b = sum(B[k][0] and not A[k][0] for k in keys)
    tiers = {}
    for k in keys:
        t = A[k][1]
        d = tiers.setdefault(t, [0, 0, 0])
        d[0] += 1
        d[1] += A[k][0]
        d[2] += B[k][0]
    tier_s = " ".join(f"{t}:{x[1]}->{x[2]}/{x[0]}" for t, x in sorted(tiers.items()))
    print(f"{b_path}: n={len(keys)}  {a} -> {b} correct ({100 * (b - a) / max(1, len(keys)):+.1f} pp)  "
          f"only_base={only_a} only_new={only_b}  McNemar p={mcnemar_p(only_a, only_b):.3f}  [{tier_s}]")


if __name__ == "__main__":
    for other in sys.argv[2:]:
        compare(sys.argv[1], other)

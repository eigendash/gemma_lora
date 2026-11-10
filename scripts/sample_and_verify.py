"""Test-time compute scaling from the sampled evaluations in results/evals/test_time.

For each experiment with dumps ``<name>__t09_s{1..K}.jsonl.gz`` (temperature 0.9, one
dump per sampling seed) this prints

- unbiased pass@k: the probability that at least one of k samples drawn without
  replacement from the K available is correct, averaged over problems;
- sample-and-verify: draw samples in seed order, return the first one the exact
  verifier accepts, stop after k. Its accuracy is the share of problems with a
  verified answer within the first k seeds; the mean number of samples drawn is
  the cost.

The verifier needs only the problem (numbers and target), not a reference
solution, so sample-and-verify is usable at test time.

usage: uv run python scripts/sample_and_verify.py [--dir results/evals/test_time]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KS = (1, 2, 4, 8)


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def experiments(directory: str) -> dict[str, list[str]]:
    found: dict[str, dict[int, str]] = {}
    for path in glob.glob(os.path.join(directory, "*__t09_s*.jsonl.gz")):
        m = re.match(r"(.+)__t09_s(\d+)\.jsonl\.gz$", os.path.basename(path))
        if m:
            found.setdefault(m.group(1), {})[int(m.group(2))] = path
    return {name: [seeds[s] for s in sorted(seeds)] for name, seeds in sorted(found.items())}


def correct_matrix(paths: list[str]) -> list[list[int]]:
    runs = [load(p) for p in paths]
    n = len(runs[0])
    for run in runs:
        if len(run) != n:
            raise ValueError("dumps differ in length")
    rows = []
    for i in range(n):
        first = runs[0][i]
        for run in runs:
            if run[i]["nums"] != first["nums"] or run[i]["target"] != first["target"]:
                raise ValueError(f"problem {i} differs between seeds")
        rows.append([int(run[i]["correct"] >= 1.0) for run in runs])
    return rows


def pass_at_k(rows: list[list[int]], k: int) -> float:
    K = len(rows[0])
    return sum(1 - comb(K - sum(r), k) / comb(K, k) for r in rows) / len(rows)


def sample_and_verify(rows: list[list[int]], k: int) -> tuple[float, float]:
    solved = sum(any(r[:k]) for r in rows) / len(rows)
    drawn = sum(r[:k].index(1) + 1 if any(r[:k]) else k for r in rows) / len(rows)
    return solved, drawn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "evals", "test_time"))
    args = ap.parse_args()
    for name, paths in experiments(args.dir).items():
        rows = correct_matrix(paths)
        print(f"{name}  ({len(paths)} seeds, {len(rows)} problems)")
        for k in KS:
            if k > len(paths):
                continue
            solved, drawn = sample_and_verify(rows, k)
            print(f"  k={k}: pass@k {100 * pass_at_k(rows, k):5.1f}%   "
                  f"sample-and-verify {100 * solved:5.1f}% with {drawn:.2f} samples drawn")


if __name__ == "__main__":
    main()

"""Generate a fresh holdout set for final, once-only evaluation (EXPERIMENTS.md §2.7k).

Same generator, tier weights and value pool as data/train.jsonl and data/eval.jsonl, a new seed, and every
(number multiset, target) already used in an --exclude file is skipped. Every solution is re-verified.
Deterministic: these commands reproduce the frozen files byte for byte (hashes in tests/test_holdout.py):
  python scripts/make_holdout.py --seed 20261105 --out data/holdout.jsonl
  python scripts/make_holdout.py --seed 20261106 --out data/holdout2.jsonl \
      --exclude data/train.jsonl data/eval.jsonl data/holdout.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_generator as dg  # noqa: E402
from verifier import integer_intermediates, verify_answer  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--exclude", nargs="+", default=["data/train.jsonl", "data/eval.jsonl"])
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; holdouts are frozen, never regenerated in place")
    exclude = set()
    for f in a.exclude:
        exclude |= {(tuple(sorted(p["nums"])), p["target"]) for p in map(json.loads, open(f))}
    rows = dg.generate_split(a.n, random.Random(a.seed), dg.DEFAULT_TIER_WEIGHTS, exclude=frozenset(exclude))
    for r in rows:
        ok, why = verify_answer(r["nums"], r["target"], r["solution"])
        assert ok and integer_intermediates(r["solution"]), (r, why)
    dg._write(a.out, rows)
    print(f"wrote {len(rows)} -> {a.out} (excluded {len(exclude)} keys from {a.exclude})")


if __name__ == "__main__":
    main()

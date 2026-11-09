"""Independent audit of data/train.jsonl and data/eval.jsonl.

Usage: python scripts/checks/audit_dataset.py <train.jsonl> <eval.jsonl> [--per-tier N]

Uses only tests/independent/ref_countdown.py (independent verifier + DFS solver).
The project's solver is called ONLY for a cross-agreement count, never as the
source of truth. Exits non-zero on any hard failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_countdown import ref_intermediates, ref_solve, ref_verify  # noqa: E402

TIER_SPEC = {  # from SPEC + build prompt
    "easy": ({3, 4}, (10, 99)),
    "medium": ({5}, (100, 999)),
    "hard": ({6}, (100, 999)),
}
POOL = Counter(list(range(1, 11)) * 2 + [25, 50, 75, 100])


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def key(p):
    return (tuple(sorted(p["nums"])), p["target"])


def one_op_solvable(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            if a + b == target or a - b == target or a * b == target:
                return True
            if b and a % b == 0 and a // b == target:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("eval")
    ap.add_argument("--per-tier", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    failures = []
    train, eval_ = load(args.train), load(args.eval)
    print(f"counts: train={len(train)} eval={len(eval_)}")
    if len(train) != 10000 or len(eval_) != 1000:
        failures.append("counts")

    for name, split in (("train", train), ("eval", eval_)):
        tiers = Counter(p["tier"] for p in split)
        ncount = Counter((p["tier"], len(p["nums"])) for p in split)
        print(f"[{name}] tiers={dict(tiers)}")
        print(f"[{name}] (tier, n_nums)={dict(sorted(ncount.items()))}")
        for t in ("easy", "medium", "hard"):
            ts = [p["target"] for p in split if p["tier"] == t]
            print(f"[{name}] {t}: target min={min(ts)} max={max(ts)} "
                  f"median={statistics.median(ts)}")
        keys = [key(p) for p in split]
        dup = len(keys) - len(set(keys))
        print(f"[{name}] duplicate (sorted nums,target) keys: {dup}")
        if dup:
            failures.append(f"{name}-dupes")
        # exact-nums duplicate (same multiset, different target) is fine; report
        nums_dup = len(keys) - len({k[0] for k in keys})
        print(f"[{name}] rows sharing a nums-multiset with another row: {nums_dup}")
        bad_spec = bad_pool = bad_target = 0
        target_in_nums = one_op = 0
        sol_fail = sol_not_all = sol_nonint = 0
        for p in split:
            counts, (lo, hi) = TIER_SPEC[p["tier"]]
            if len(p["nums"]) not in counts or not (lo <= p["target"] <= hi):
                bad_spec += 1
            c = Counter(p["nums"])
            if any(c[k] > POOL[k] for k in c):
                bad_pool += 1
            if not (isinstance(p["target"], int) and p["target"] > 0):
                bad_target += 1
            if p["target"] in p["nums"]:
                target_in_nums += 1
            if one_op_solvable(p["nums"], p["target"]):
                one_op += 1
            if not ref_verify(p["nums"], p["target"], p["solution"]):
                sol_fail += 1
            if any(v.denominator != 1 or v <= 0 for v in ref_intermediates(p["solution"])):
                sol_nonint += 1
            # by construction the reference solution uses every number once
            lits = sorted(int(x) for x in
                          p["solution"].replace("(", " ").replace(")", " ")
                          .replace("+", " ").replace("-", " ").replace("*", " ")
                          .replace("/", " ").split())
            if lits != sorted(p["nums"]):
                sol_not_all += 1
        print(f"[{name}] tier-spec violations={bad_spec} pool violations={bad_pool} "
              f"non-positive/non-int targets={bad_target}")
        print(f"[{name}] stored solution fails INDEPENDENT verifier: {sol_fail}")
        print(f"[{name}] stored solution not using exactly all nums: {sol_not_all}")
        print(f"[{name}] stored solution with non-integer or non-positive intermediate: {sol_nonint}")
        print(f"[{name}] DEGENERATE target==an input: {target_in_nums} "
              f"({target_in_nums/len(split):.2%})")
        print(f"[{name}] DEGENERATE solvable in ONE op (a op b == target): {one_op} "
              f"({one_op/len(split):.2%})")
        for t in ("easy", "medium", "hard"):
            sub = [p for p in split if p["tier"] == t]
            ti = sum(p["target"] in p["nums"] for p in sub)
            oo = sum(one_op_solvable(p["nums"], p["target"]) for p in sub)
            print(f"[{name}]   {t}: target==input {ti}/{len(sub)} ({ti/len(sub):.1%}), "
                  f"one-op {oo}/{len(sub)} ({oo/len(sub):.1%})")
        if bad_spec or bad_pool or bad_target or sol_fail or sol_nonint:
            failures.append(f"{name}-content")

    tk = {key(p) for p in train}
    ek = {key(p) for p in eval_}
    ov = tk & ek
    print(f"train/eval overlap on (sorted nums, target): {len(ov)}")
    if ov:
        failures.append("overlap")
    nums_ov = {k[0] for k in tk} & {k[0] for k in ek}
    print(f"train/eval share an identical nums-multiset (different target): "
          f"{len(nums_ov)} eval multisets")

    # Stratified independent solvability + difficulty
    rng = random.Random(args.seed)
    print(f"\nIndependent DFS solver, {args.per_tier} per tier from train + "
          f"{args.per_tier} per tier from eval")
    try:
        from countdown_rl.data.solver import solve as proj_solve  # cross-agreement only
    except Exception:  # pragma: no cover
        proj_solve = None
    for t in ("easy", "medium", "hard"):
        sample = (rng.sample([p for p in train if p["tier"] == t], args.per_tier)
                  + rng.sample([p for p in eval_ if p["tier"] == t], args.per_tier))
        t0 = time.time()
        unsolved = 0
        mins, nodes = [], []
        agree = 0
        for p in sample:
            found, mo, nd = ref_solve(p["nums"], p["target"])
            if not found:
                unsolved += 1
                print("  UNSOLVED by ref:", p)
            else:
                mins.append(mo)
                nodes.append(nd)
            if proj_solve is not None and t == "easy":
                pf, pe = proj_solve(p["nums"], p["target"])
                agree += int(pf == found and (not pf or ref_verify(p["nums"], p["target"], pe)))
        dt = time.time() - t0
        print(f"[{t}] n={len(sample)} unsolved={unsolved} time={dt:.1f}s")
        print(f"[{t}] min #ops distribution: {dict(sorted(Counter(mins).items()))}")
        print(f"[{t}] search nodes to first solution: median={statistics.median(nodes)} "
              f"mean={statistics.mean(nodes):.0f} p90="
              f"{sorted(nodes)[int(0.9*len(nodes))-1]}")
        if proj_solve is not None and t == "easy":
            print(f"[{t}] project solver agrees with ref (found + valid expr): "
                  f"{agree}/{len(sample)}")
        if unsolved:
            failures.append(f"unsolved-{t}")

    print("\nHARD FAILURES:", failures or "none")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

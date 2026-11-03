"""Cross-check our solver/verifier against two independent references.

1. ``Jiayi-Pan/Countdown-Tasks-3to4`` (HF): audit a sample for solvability,
   duplicate keys, and leakage against our eval split.
2. ``reasoning-gym`` countdown (dev-only dep): check our verifier accepts their
   solutions and measure how often our solver solves their problems.

Run from repo root (needs network + HF token-free public data).
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifier import verify_answer
from solver import solve

COUNTDOWN_TASKS = "Jiayi-Pan/Countdown-Tasks-3to4"


def audit_countdown_tasks(n_solve_sample: int = 2000) -> None:
    from datasets import load_dataset

    ds = load_dataset(COUNTDOWN_TASKS, split="train")
    print(f"[countdown-tasks] rows={len(ds)}")

    # number-count distribution
    n3 = n4 = 0
    for row in ds:
        k = len(row["nums"])
        if k == 3:
            n3 += 1
        elif k == 4:
            n4 += 1
    print(f"[countdown-tasks] num_counts: 3={n3} 4={n4}")

    # duplicates on (sorted nums, target)
    seen = set()
    dupes = 0
    for row in ds:
        key = (tuple(sorted(row["nums"])), int(row["target"]))
        if key in seen:
            dupes += 1
        seen.add(key)
    print(f"[countdown-tasks] unique keys={len(seen)} duplicate rows={dupes}")

    # leakage vs our eval split
    eval_keys = set()
    with open("data/eval.jsonl") as f:
        for line in f:
            p = json.loads(line)
            eval_keys.add((tuple(sorted(p["nums"])), p["target"]))
    overlap = seen & eval_keys
    print(f"[countdown-tasks] leakage overlap with our eval={len(overlap)}")

    # solvability (positive integer intermediates) on a random sample
    rows = list(ds.select(range(len(ds))))
    rng = random.Random(0)
    sample = rng.sample(rows, min(n_solve_sample, len(rows)))
    solvable = 0
    for row in sample:
        found, _ = solve(row["nums"], int(row["target"]))
        if found:
            solvable += 1
    print(
        f"[countdown-tasks] solvable (pos-int intermediates) "
        f"={solvable}/{len(sample)} = {solvable/len(sample):.3f}"
    )


def audit_reasoning_gym(n: int = 300) -> None:
    import importlib.metadata as md
    from reasoning_gym.games.countdown import CountdownConfig, CountdownDataset

    print(f"[reasoning-gym] installed version={md.version('reasoning-gym')}")
    ds = CountdownDataset(CountdownConfig(seed=42, size=n, min_numbers=3, max_numbers=4))

    verifier_accepts = 0
    solver_solves = 0
    for i in range(n):
        entry = ds[i]
        numbers = entry["metadata"]["numbers"]
        target = entry["metadata"]["target"]
        answer = entry["answer"]

        ok, detail = verify_answer(numbers, target, answer)
        if ok:
            verifier_accepts += 1
        else:
            if verifier_accepts == 0 and i < 5:
                print(f"  [reasoning-gym] verifier rejected {answer!r} -> {detail}")

        found, _ = solve(numbers, target)
        if found:
            solver_solves += 1

    print(
        f"[reasoning-gym] our verifier accepts their answer "
        f"={verifier_accepts}/{n} = {verifier_accepts/n:.3f}"
    )
    print(
        f"[reasoning-gym] our solver solves their problem "
        f"(pos-int intermediates) ={solver_solves}/{n} = {solver_solves/n:.3f}"
    )


if __name__ == "__main__":
    audit_countdown_tasks()
    print()
    audit_reasoning_gym()

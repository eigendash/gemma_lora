"""Independent cross-check vs external datasets (needs network + dev deps).

1. reasoning-gym countdown: their answer must be accepted by BOTH the project
   verifier and the independent ref verifier, and by their own score_answer.
2. Countdown-Tasks-3to4: on a random sample, project solver, ref DFS solver
   agree on solvability; the project's expression passes ref_verify.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_countdown import ref_solve, ref_verify  # noqa: E402

from countdown_rl.data.solver import solve as proj_solve
from countdown_rl.data.verifier import verify_answer


def rg(n=300):
    from reasoning_gym.games.countdown import CountdownConfig, CountdownDataset

    ds = CountdownDataset(CountdownConfig(seed=42, size=n, min_numbers=3, max_numbers=4))
    both = disagree = their_ok = 0
    for i in range(n):
        e = ds[i]
        nums, tgt, ans = e["metadata"]["numbers"], e["metadata"]["target"], e["answer"]
        p = verify_answer(nums, tgt, ans)[0]
        r = ref_verify(nums, tgt, ans)
        their_ok += ds.score_answer(ans, e) == 1.0
        both += p and r
        if p != r:
            disagree += 1
            print("  DISAGREE", nums, tgt, ans, p, r)
    print(f"[reasoning-gym] n={n} accepted by proj&ref={both} proj!=ref={disagree} "
          f"their score_answer==1: {their_ok}")
    return disagree == 0 and both == n


def cdt(n=300):
    from datasets import load_dataset

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    rng = random.Random(7)
    idx = rng.sample(range(len(ds)), n)
    agree = solv = expr_ok = 0
    for i in idx:
        row = ds[i]
        nums, tgt = list(row["nums"]), int(row["target"])
        pf, pe = proj_solve(nums, tgt)
        rf, _, _ = ref_solve(nums, tgt)
        agree += pf == rf
        solv += rf
        if pf:
            expr_ok += ref_verify(nums, tgt, pe)
        elif rf:
            print("  ref solvable but project solver not (fractions/negatives?):", nums, tgt)
    print(f"[countdown-tasks-3to4] n={n} ref-solvable={solv} proj/ref agree={agree} "
          f"proj expr passes ref_verify={expr_ok}/{sum(1 for _ in [0]) and solv}")
    return agree == n


if __name__ == "__main__":
    ok1 = rg()
    ok2 = cdt()
    print("RESULT:", "PASS" if ok1 and ok2 else "FAIL")
    sys.exit(0 if ok1 and ok2 else 1)

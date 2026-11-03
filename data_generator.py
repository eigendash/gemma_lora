"""Synthetic Countdown problem generator (guaranteed solvable by construction).

Builds the target by repeatedly combining the sampled numbers with ``+ - * /``
(positive integer intermediates), so every problem carries a reference solution
and is solvable by definition. Tiers:

- easy:   3-4 numbers, 2-digit target
- medium: 5 numbers, 3-digit target   (spec only said "5 numbers"; 3-digit is our
           interpretation to keep difficulty monotonic)
- hard:   6 numbers, 3-digit target

Numbers are sampled from the classic Countdown pool (1..10 twice each, plus
25/50/75/100 once each).
"""

from __future__ import annotations

import argparse
import json
import random

SMALL = list(range(1, 11))
LARGE = [25, 50, 75, 100]
POOL = SMALL * 2 + LARGE  # each small x2, each large x1

TIERS = {
    "easy": {"num_count": (3, 4), "target_digits": 2},
    "medium": {"num_count": (5, 5), "target_digits": 3},
    "hard": {"num_count": (6, 6), "target_digits": 3},
}
TIER_ORDER = ["easy", "medium", "hard"]
DEFAULT_TIER_WEIGHTS = {"easy": 0.4, "medium": 0.3, "hard": 0.3}

# Per-tier operation weights (steer target magnitude). 2-digit targets favor
# +/-, 3-digit targets favor *.
OP_WEIGHTS = {
    "easy": {"+": 3, "-": 3, "*": 2, "/": 1},
    "medium": {"+": 2, "-": 2, "*": 3, "/": 2},
    "hard": {"+": 2, "-": 2, "*": 3, "/": 2},
}

MAX_VALUE = 10**6
MAX_ATTEMPTS = 20_000


def _apply(a: int, b: int, op: str) -> int | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0 or a % b != 0:
            return None
        return a // b
    raise ValueError(op)


def sample_numbers(rng: random.Random, k: int) -> list[int]:
    return rng.sample(POOL, k)


def build_problem(
    nums: list[int], rng: random.Random, op_weights: dict[str, int], max_value: int = MAX_VALUE
) -> tuple[int, str] | None:
    """Combine ``nums`` into a single target, returning ``(target, solution)``.

    Returns None on a dead-end (no valid op), prompting a retry.
    """
    values = list(nums)
    exprs = [str(x) for x in nums]
    while len(values) > 1:
        i, j = sorted(rng.sample(range(len(values)), 2))
        a, b = values[i], values[j]
        ea, eb = exprs[i], exprs[j]

        cands: list[tuple[str, int]] = []
        for op in "+-*/":
            r = _apply(a, b, op)
            if r is not None and 1 <= r <= max_value:
                cands.append((op, r))
        if not cands:
            return None

        ops = [c[0] for c in cands]
        op = rng.choices(ops, weights=[op_weights[o] for o in ops], k=1)[0]
        r = _apply(a, b, op)

        del values[j]
        del exprs[j]
        del values[i]
        del exprs[i]
        values.append(r)
        exprs.append(f"({ea}{op}{eb})")

    return values[0], exprs[0]


def generate_one(
    rng: random.Random,
    tier: str,
    op_weights: dict[str, int],
    max_value: int = MAX_VALUE,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[list[int], int, str]:
    spec = TIERS[tier]
    lo, hi = spec["num_count"]
    tmin = 10 ** (spec["target_digits"] - 1)
    tmax = 10 ** spec["target_digits"] - 1

    for _ in range(max_attempts):
        k = rng.randint(lo, hi)
        nums = sample_numbers(rng, k)
        res = build_problem(nums, rng, op_weights, max_value)
        if res is None:
            continue
        target, sol = res
        # reject the degenerate "answer is one of the inputs" case (audit m2)
        if tmin <= target <= tmax and target not in nums:
            return nums, target, sol
    raise RuntimeError(f"failed to generate tier={tier} after {max_attempts} attempts")


def _allocate(n: int, tier_weights: dict[str, float]) -> dict[str, int]:
    total = sum(tier_weights.values())
    alloc: dict[str, int] = {}
    remaining = n
    for i, t in enumerate(TIER_ORDER):
        if i == len(TIER_ORDER) - 1:
            alloc[t] = remaining
        else:
            alloc[t] = round(n * tier_weights[t] / total)
            remaining -= alloc[t]
    return alloc


def generate_split(
    n: int,
    rng: random.Random,
    tier_weights: dict[str, float],
    max_value: int = MAX_VALUE,
    exclude: frozenset | set = frozenset(),
) -> list[dict]:
    alloc = _allocate(n, tier_weights)
    seen: set = set()
    problems: list[dict] = []
    for tier in TIER_ORDER:
        ow = OP_WEIGHTS[tier]
        for _ in range(alloc[tier]):
            for _attempt in range(MAX_ATTEMPTS):
                nums, target, sol = generate_one(rng, tier, ow, max_value)
                key = (tuple(sorted(nums)), target)
                if key in seen or key in exclude:
                    continue
                seen.add(key)
                problems.append(
                    {"nums": nums, "target": target, "tier": tier, "solution": sol}
                )
                break
            else:
                raise RuntimeError(f"could not find a unique {tier} problem")
    return problems


def generate_dataset(
    n_train: int,
    n_eval: int,
    seed: int,
    tier_weights: dict[str, float] | None = None,
    max_value: int = MAX_VALUE,
) -> tuple[list[dict], list[dict]]:
    weights = tier_weights or DEFAULT_TIER_WEIGHTS
    rng = random.Random(seed)
    train = generate_split(n_train, rng, weights, max_value)
    train_keys = {(tuple(sorted(p["nums"])), p["target"]) for p in train}
    eval_ = generate_split(n_eval, rng, weights, max_value, exclude=frozenset(train_keys))
    return train, eval_


def _write(path: str, problems: list[dict]) -> None:
    with open(path, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate Countdown train/eval JSONL.")
    ap.add_argument("--train-out", default="data/train.jsonl")
    ap.add_argument("--eval-out", default="data/eval.jsonl")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy-weight", type=float, default=0.4)
    ap.add_argument("--medium-weight", type=float, default=0.3)
    ap.add_argument("--hard-weight", type=float, default=0.3)
    ap.add_argument("--verify", action="store_true", help="verify every problem before writing")
    args = ap.parse_args(argv)

    weights = {
        "easy": args.easy_weight,
        "medium": args.medium_weight,
        "hard": args.hard_weight,
    }
    train, eval_ = generate_dataset(args.n_train, args.n_eval, args.seed, weights)

    if args.verify:
        from verifier import integer_intermediates, verify_answer

        for split_name, probs in (("train", train), ("eval", eval_)):
            for p in probs:
                ok, detail = verify_answer(p["nums"], p["target"], p["solution"])
                if not ok:
                    raise SystemExit(f"verify failed ({split_name}): {p} -> {detail}")
                if not integer_intermediates(p["solution"]):
                    raise SystemExit(f"non-integer intermediate ({split_name}): {p}")

    _write(args.train_out, train)
    _write(args.eval_out, eval_)
    print(f"wrote {len(train)} train -> {args.train_out}")
    print(f"wrote {len(eval_)} eval  -> {args.eval_out}")


if __name__ == "__main__":
    main()

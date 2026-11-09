"""Fuzz the project's verifier.verify_answer against the independent
ref_countdown.ref_verify (hand-written parser, no ast).

Run from the project root:  uv run pytest -q tests/independent/test_verifier_fuzz.py

Disagreements are bucketed. Some are KNOWN BENIGN (Python literal syntax the
reference intentionally rejects, e.g. "0x0A", "1_0"); everything else fails.
"""

from __future__ import annotations

import os
import random
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_countdown import ref_parse, ref_verify, RefError  # noqa: E402

from countdown_rl.data.verifier import verify_answer  # project under test

POOL = list(range(1, 11)) * 2 + [25, 50, 75, 100]


def proj(nums, target, expr) -> bool:
    return verify_answer(nums, target, expr)[0]


def rand_expr(rng, leaves):
    """Random binary tree over the given leaf strings."""
    items = list(leaves)
    while len(items) > 1:
        i, j = sorted(rng.sample(range(len(items)), 2))
        a, b = items[i], items[j]
        op = rng.choice("+-*/")
        sep = rng.choice(["", " "])
        node = f"({a}{sep}{op}{sep}{b})" if rng.random() < 0.8 else f"{a}{sep}{op}{sep}{b}"
        del items[j]
        del items[i]
        items.append(node)
    return items[0]


MUTATIONS = [
    "none", "reuse", "foreign", "unary", "extra_paren", "div0", "float",
    "times_x", "divide_sign", "fullwidth", "unicode_minus", "drop_paren",
    "long", "empty", "whitespace_inside", "double_op", "zero_lit",
]


def mutate(rng, nums, expr, kind):
    if kind == "none":
        return expr
    if kind == "reuse":
        return f"({expr})+{nums[0]}-{nums[0]}"
    if kind == "foreign":
        return f"({expr})*1" if 1 not in nums else f"({expr})+0"
    if kind == "unary":
        return f"-(-({expr}))"
    if kind == "extra_paren":
        return "((((" + expr + "))))"
    if kind == "div0":
        return f"({expr})+{nums[0]}/({nums[1]}-{nums[1]})" if len(nums) > 1 else expr + "/0"
    if kind == "float":
        return expr.replace(str(nums[0]), f"{nums[0]}.0", 1)
    if kind == "times_x":
        return expr.replace("*", "x")
    if kind == "divide_sign":
        return expr.replace("/", "÷")
    if kind == "fullwidth":
        return expr.translate({ord(c): ord(c) + 0xFEE0 for c in "0123456789"})
    if kind == "unicode_minus":
        return expr.replace("-", "−")
    if kind == "drop_paren":
        return expr.replace("(", "", 1)
    if kind == "long":
        return "(" * 120 + expr + ")" * 120
    if kind == "empty":
        return rng.choice(["", " ", "\n", "()"])
    if kind == "whitespace_inside":
        return expr.replace("+", "\n+", 1)
    if kind == "double_op":
        return expr.replace("+", "++", 1)
    if kind == "zero_lit":
        return f"({expr})+0*{nums[0]}"
    raise ValueError(kind)


def test_fuzz_agreement():
    rng = random.Random(20261104)
    n_cases = 6000
    disagreements = []
    stats = {"agree_true": 0, "agree_false": 0}
    for _ in range(n_cases):
        k = rng.randint(3, 6)
        nums = rng.sample(POOL, k)
        use = rng.sample(nums, rng.randint(1, k))
        expr = rand_expr(rng, [str(x) for x in use])
        kind = rng.choice(MUTATIONS)
        e2 = mutate(rng, nums, expr, kind)
        # target: either the true value of the un-mutated expr, or random
        try:
            v, _ = ref_parse(expr)
        except RefError:
            continue
        if v.denominator == 1 and v > 0 and rng.random() < 0.8:
            target = int(v)
        else:
            target = rng.randint(1, 999)
        p, r = proj(nums, target, e2), ref_verify(nums, target, e2)
        if p == r:
            stats["agree_true" if p else "agree_false"] += 1
        else:
            disagreements.append((kind, nums, target, e2, p, r))
    report = [d for d in disagreements]
    print(f"\nagreement stats: {stats}; disagreements: {len(report)}")
    by_kind = {}
    for d in report:
        by_kind.setdefault(d[0], []).append(d)
    for kind, ds in by_kind.items():
        print(f"  {kind}: {len(ds)}  e.g. nums={ds[0][1]} target={ds[0][2]} "
              f"expr={ds[0][3][:80]!r} proj={ds[0][4]} ref={ds[0][5]}")
    assert stats["agree_true"] > 500, "fuzz did not exercise enough accepted answers"
    # Only the deliberate whitespace difference is tolerated (see edge test).
    unexplained = [d for d in report if d[0] != "whitespace_inside"]
    assert not unexplained, unexplained[:5]


# ---------------------------------------------------------------------------
# Hand-picked edge cases with explicit expected verdicts (ground truth = spec).
# ---------------------------------------------------------------------------
EDGE = [
    # (nums, target, expr, expected, note)
    ([1, 2, 3], 6, "1+2+3", True, "basic"),
    ([1, 2, 3], 6, "((1))+((2+3))", True, "extra parens"),
    ([100, 75, 3, 5], 875, "(100+75)*5", True, "spec example, subset"),
    ([1, 1, 2, 2], 1, "1/2+1/2", True, "non-integer intermediate (spec-allowed)"),
    ([3, 3, 1], 1, "3/3", True, "division"),
    ([1, 2, 3], 7, "1+2+2+2", False, "reuse"),
    ([1, 2, 3], 6, "1+2+3+0", False, "foreign 0"),
    ([4, 2], 2, "4/2", True, ""),
    ([4, 0], 4, "4+0", True, "0 given in nums"),
    ([5, 3], 5, "5+3/(3-3)", False, "div by zero inside"),
    ([2, 3], 8, "2**3", False, "pow"),
    ([2, 3], 6, "2.0*3", False, "float literal"),
    ([2, 3], 6, "2 x 3", False, "x as times"),
    ([6, 3], 2, "6÷3", False, "division sign"),
    ([2, 3], 6, "２*３", False, "fullwidth digits"),
    ([5, 3], 2, "5−3", False, "unicode minus"),
    ([5, 3], 8, "", False, "empty"),
    ([5, 3], 8, "5+3=8", False, "equation, not expression"),
    ([5, 3], 8, "5+3\n5+3", False, "two lines"),
    ([5, 3], 8, "(5+3)" + "+0" * 0, True, ""),
    ([5, 3], -2, "3-5", True, "negative target is just arithmetic"),
    ([5, 3], 2, "-3+5", False, "unary minus (design: rejected)"),
    ([5, 3], 8, "(" * 300 + "5+3" + ")" * 300, False, "over length cap / too deep"),
    ([50, 3, 3, 1], 50, "50", True, "0-op answer accepted when target is an input"),
    ([10], 10, "1_0", False, "python underscore literal (rejected by F5 charset whitelist)"),
    ([10], 10, "0xA", False, "python hex literal (rejected by F5 charset whitelist)"),
    ([5, 3], 8, "5 +\n 3", None, "newline inside expression (record only)"),
]


@pytest.mark.parametrize("nums,target,expr,expected,note", EDGE)
def test_edge_cases(nums, target, expr, expected, note):
    got = proj(nums, target, expr)
    print(f"proj={got} ref={ref_verify(nums, target, expr)} :: {note} :: {expr[:40]!r}")
    if expected is None:
        pytest.skip(f"recorded only: proj={got}")
    assert got == expected, (nums, target, expr, note)


def test_large_literal_cheap():
    # 500-char cap: a product of huge literals must not hang.
    import time
    expr = "9" * 240 + "*" + "9" * 240
    t0 = time.time()
    proj([1], 1, expr)
    assert time.time() - t0 < 0.5


def test_deep_parens_no_crash():
    for depth in (100, 1000, 10000):
        assert proj([1], 1, "(" * depth + "1" + ")" * depth) in (True, False)


def test_non_string_inputs():
    for bad in (None, 5, b"1+2", ["1+2"]):
        assert proj([1, 2], 3, bad) is False

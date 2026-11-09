"""Prompt construction: the per-problem prompt for each prompt style and the worked-example prefix.

Prompt styles (``PROMPT_STYLES``):

- ``r1zero_tool``           no example. An R1-Zero style conversation frame plus a plain description of the tool:
                            code in ``<python>`` is executed and its printed output comes back in ``<output>``.
- ``zero``                  no example, tag skeleton only.
- ``fewshot_direct_retry``  one worked example (a single check that prints True) plus a one-line retry instruction.
- ``fewshot_multi_retry``   one worked example that checks several candidates in one ``<python>`` block, plus the
                            multi-candidate instruction.

The worked examples are generated, not written by hand: ``generate_trace`` builds a protocol trace for a fixed
training problem from the reference solver, and every ``<output>`` in it is real sandbox stdout. ``solution_hint``
returns a partial solution for the hinted-resampling experiments (training only; evaluation never passes hints).
"""

from __future__ import annotations

import ast
import random
from fractions import Fraction
from itertools import combinations


from countdown_rl.rl.rollout import format_prompt, format_prompt_r1zero_tool
from countdown_rl.data.verifier import verify_answer
from countdown_rl.sandbox import Sandbox
from countdown_rl.data.solver import solve

TRACE_KINDS = ("direct", "near_miss", "split", "resplit")
DEFAULT_MIX = {"direct": 0.35, "near_miss": 0.3, "split": 0.15, "resplit": 0.2}

_OPS = {
    ast.Add: ("+", lambda a, b: a + b),
    ast.Sub: ("-", lambda a, b: a - b),
    ast.Mult: ("*", lambda a, b: a * b),
    ast.Div: ("/", lambda a, b: a / b if b != 0 else None),
}
_SYM = {sym: fn for sym, fn in _OPS.values()}


def _value(node) -> Fraction:
    if isinstance(node, ast.Constant):
        return Fraction(node.value)
    sym, fn = _OPS[type(node.op)]
    return fn(_value(node.left), _value(node.right))


def _operand(node) -> str:
    text = ast.unparse(node)
    return f"({text})" if isinstance(node, ast.BinOp) else text


def minimal_solution(nums: list[int], target: int, fallback: str) -> str:
    """A solution over the fewest input numbers (solver search by subset size)."""
    for k in range(2, len(nums) + 1):
        for combo in combinations(nums, k):
            found, expr = solve(list(combo), target)
            if found:
                return expr
    return fallback


def split_solution(expr: str):
    """(tree, L, op, R) for the top-level ``L op R`` of an expression."""
    tree = ast.parse(expr, mode="eval").body
    if not isinstance(tree, ast.BinOp):
        raise ValueError(f"solution is not a binary expression: {expr!r}")
    return tree, _operand(tree.left), _OPS[type(tree.op)][0], _operand(tree.right)


def _check_code(expr: str, target: int) -> str:
    return f"res = {expr}\nprint(res == {target}, res)"


def _multi_code(cands: list[str], target: int) -> str:
    """One check over several candidate expressions (sandbox allows ``for`` over a tuple)."""
    return "for res in (" + ", ".join(cands) + "):\n    print(res == " + str(target) + ", res)"


# Demonstration for the "multi" kind: several candidates verified in ONE execution, so the
# 2-execution cap (SPEC) still allows ~10 checked candidates. Only the few-shot problem uses it.
MULTI_DEMO = {(100, 75, 3, 5, 875): ["(100 - 75) * 5 * 3", "100 * 5 + 75 * 3", "(100 + 75) * 5", "75 * 3 * 5 - 100"]}

# Demonstration for the "wide" kind: TWO checks of 8 candidates each. The first misses entirely; the
# second is built from the printed values closest to the target (P8 checked ~3 per call and its
# failures were almost all "every check False -> unchecked guess", EXPERIMENTS.md §2.7g).
WIDE_DEMO = {(100, 75, 3, 5, 875): [
    ["100 * 5 + 75 * 3", "(100 - 75) * 5 * 3", "100 * 3 * 5 - 75", "75 * 5 + 100 * 3",
     "(100 + 75 + 5) * 3", "100 * (5 + 3) - 75", "(75 - 5) * 3 + 100", "75 * 3 * 5 - 100"],
    ["100 * (5 + 3)", "75 * (5 + 3) + 100", "(100 + 3) * 5 + 75", "100 * (5 + 3) + 75",
     "(100 + 75) * 5", "(100 + 75) * 5 - 3", "(100 - 3) * 5 + 75", "(75 + 100) * 3 + 5"],
]}


def _sweep_code(left: str, right: str) -> str:
    return f"a = {left}\nb = {right}\nprint(a + b, a - b, a * b, a / b)"


def _restate(stdout: str, target: int) -> tuple[str, str | None]:
    """'a + b = 13, a - b = 3, ...' from the REAL printed sweep, plus the operator whose
    printed value equals the target (None if no value does)."""
    vals = stdout.split()
    parts, hit = [], None
    for sym, v in zip(("+", "-", "*", "/"), vals):
        parts.append(f"a {sym} b = {v}")
        try:
            if hit is None and float(v) == target:
                hit = sym
        except ValueError:
            pass
    return ", ".join(parts), hit


def wrong_split(tree, nums: list[int], target: int, rng: random.Random):
    """A plausible but wrong (a, b) split: perturb one operator inside L or R (or swap a
    leaf for an unused input number) so that NO operation between a and b hits the target."""
    def variants(node):
        if isinstance(node, ast.BinOp):
            for t in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                if not isinstance(node.op, t):
                    yield ast.BinOp(node.left, t(), node.right)
            for child in ("left", "right"):
                for v in variants(getattr(node, child)):
                    n2 = ast.BinOp(node.left, node.op, node.right)
                    setattr(n2, child, v)
                    yield n2
        else:
            used = [c.value for c in ast.walk(tree) if isinstance(c, ast.Constant)]
            spare = list(nums)
            for u in used:
                if u in spare:
                    spare.remove(u)
            for x in spare:
                if x != node.value:
                    yield ast.Constant(x)

    cands = []
    for side in ("left", "right"):
        for v in variants(getattr(tree, side)):
            a = v if side == "left" else tree.left
            b = tree.right if side == "left" else v
            try:
                va, vb = _value(a), _value(b)
            except (ZeroDivisionError, TypeError):
                continue
            if va is None or vb is None or va <= 0 or vb <= 0 or va.denominator != 1 or vb.denominator != 1:
                continue
            outs = [va + vb, va - vb, va * vb] + ([va / vb] if vb else [])
            if all(o != target for o in outs):
                cands.append((_operand(a), _operand(b)))
    return rng.choice(cands) if cands else None


_SYM_TO_AST = {"+": ast.Add, "-": ast.Sub, "*": ast.Mult, "/": ast.Div}


def _answer_with(tree, sym: str) -> str:
    """The solution with its top-level operator replaced by ``sym`` (the operator the
    printed sweep showed hitting the target)."""
    return ast.unparse(ast.BinOp(tree.left, _SYM_TO_AST[sym](), tree.right))


def _py(code: str) -> str:
    return f"<python>\n{code}\n</python>"


def _out(stdout: str) -> str:
    return f"\n<output>\n{stdout}\n</output>\n"


def generate_trace(problem: dict, rng: random.Random, sandbox: Sandbox, kind: str | None = None,
                   mix: dict | None = None) -> list[tuple[bool, str]]:
    """Return ``(is_env, text)`` segments for one trace; ``is_env`` segments hold real
    sandbox output and must be masked during SFT."""
    nums, target = problem["nums"], problem["target"]
    if kind == "wide":
        rounds = WIDE_DEMO[(*nums, target)]
        segs = []
        for i, cands in enumerate(rounds):
            _, out = sandbox.run(_multi_code(cands, target))
            lines = out.strip().splitlines()
            assert len(lines) == len(cands), out
            for c in cands:  # every demo candidate must obey the number rule (only its VALUE may be wrong)
                assert verify_answer(nums, target, c)[1] in ("ok", "wrong-value"), (c, verify_answer(nums, target, c))
            hit = [c for c, line in zip(cands, lines) if line.startswith("True")]
            if i == 0:
                segs.append((False, f"<think>\nTarget {target}. Check 8 candidates at once.\n</think>\n"))
            segs += [(False, _py(_multi_code(cands, target))), (True, _out(out))]
            if hit:
                k = cands.index(hit[0]) + 1
                segs += [(False, f"<think>\nCandidate {k}, {hit[0]}, prints True.\n</think>\n"),
                         (False, f"<answer>\n{hit[0]}\n</answer>\n")]
                return segs
            assert i + 1 < len(rounds), "wide demo must end with a True"
            vals = sorted(((abs(float(line.split()[1]) - target), c, line.split()[1]) for c, line in zip(cands, lines)))
            near = "; ".join(f"{c} = {v}" for _, c, v in vals[:2])
            segs.append((False, f"<think>\nNone prints True. Closest: {near}. Adjust those and check 8 new "
                                f"candidates.\n</think>\n"))
        raise AssertionError("unreachable")
    if kind == "multi":
        cands = MULTI_DEMO[(*nums, target)]
        _, out = sandbox.run(_multi_code(cands, target))
        hit = [c for c, line in zip(cands, out.strip().splitlines()) if line.startswith("True")]
        assert hit, out
        k = cands.index(hit[0]) + 1
        return [
            (False, f"<think>\nTarget {target}. Check several candidates at once: " + "; ".join(cands) + ".\n</think>\n"),
            (False, _py(_multi_code(cands, target))),
            (True, _out(out)),
            (False, f"<think>\nCandidate {k}, {hit[0]}, prints True.\n</think>\n"),
            (False, f"<answer>\n{hit[0]}\n</answer>\n"),
        ]
    expr = minimal_solution(nums, target, problem["solution"])
    tree, left, op, right = split_solution(expr)
    answer = ast.unparse(tree)
    if kind is None:
        mix = mix or DEFAULT_MIX
        kind = rng.choices(list(mix), weights=list(mix.values()), k=1)[0]
    lv, rv = _value(tree.left), _value(tree.right)

    if kind == "direct":
        ok, out = sandbox.run(_check_code(answer, target))
        return [
            (False, f"<think>\nTarget {target}. Try {answer}.\n</think>\n"),
            (False, _py(_check_code(answer, target))),
            (True, _out(out)),
            (False, f"<answer>\n{answer}\n</answer>\n"),
        ]

    if kind == "split":
        _, out = sandbox.run(_sweep_code(left, right))
        said, hit = _restate(out, target)
        assert hit is not None, (out, target)
        return [
            (False, f"<think>\nTarget {target}. Split into a = {left} and b = {right}, "
                    f"and try every operation.\n</think>\n"),
            (False, _py(_sweep_code(left, right))),
            (True, _out(out)),
            (False, f"<think>\n{said}. a {hit} b is {target}, the target.\n</think>\n"),
            (False, f"<answer>\n{_answer_with(tree, hit)}\n</answer>\n"),
        ]

    if kind == "resplit":
        ws = wrong_split(tree, nums, target, rng)
        if ws is None:
            return generate_trace(problem, rng, sandbox, kind="near_miss")
        wl, wr = ws
        _, out1 = sandbox.run(_sweep_code(wl, wr))
        said1, hit1 = _restate(out1, target)
        assert hit1 is None, (out1, target)
        _, out2 = sandbox.run(_sweep_code(left, right))
        said2, hit2 = _restate(out2, target)
        assert hit2 is not None, (out2, target)
        return [
            (False, f"<think>\nTarget {target}. Split into a = {wl} and b = {wr}, "
                    f"and try every operation.\n</think>\n"),
            (False, _py(_sweep_code(wl, wr))),
            (True, _out(out1)),
            (False, f"<think>\n{said1}. None is {target}. Split differently: a = {left} and b = {right}.\n</think>\n"),
            (False, _py(_sweep_code(left, right))),
            (True, _out(out2)),
            (False, f"<think>\n{said2}. a {hit2} b is {target}, the target.\n</think>\n"),
            (False, f"<answer>\n{_answer_with(tree, hit2)}\n</answer>\n"),
        ]

    if kind == "near_miss":
        cands = []
        for sym, fn in _SYM.items():
            v = fn(lv, rv)
            if sym != op and v is not None and v != target:
                cands.append((abs(v - target), sym))
        if not cands:  # every other operator also hits the target: nothing to miss
            return generate_trace(problem, rng, sandbox, kind="direct")
        wrong = min(cands)[1]
        attempt = f"{left} {wrong} {right}"
        _, out1 = sandbox.run(_check_code(attempt, target))
        printed = out1.split()[-1]
        direction = "too high" if _SYM[wrong](lv, rv) > target else "too low"
        _, out2 = sandbox.run(_sweep_code(left, right))
        said2, hit2 = _restate(out2, target)
        assert hit2 is not None, (out2, target)
        return [
            (False, f"<think>\nTarget {target}. Try {attempt}.\n</think>\n"),
            (False, _py(_check_code(attempt, target))),
            (True, _out(out1)),
            (False, f"<think>\n{printed} is {direction}. Keep a = {left} and b = {right}, "
                    f"and try every operation.\n</think>\n"),
            (False, _py(_sweep_code(left, right))),
            (True, _out(out2)),
            (False, f"<think>\n{said2}. a {hit2} b is {target}, the target.\n</think>\n"),
            (False, f"<answer>\n{_answer_with(tree, hit2)}\n</answer>\n"),
        ]

    raise ValueError(f"unknown trace kind {kind!r}")


FEWSHOT_PROBLEMS = [  # not in the eval split ([25, 3, 1] -> 76 is a train problem)
    {"nums": [100, 75, 3, 5], "target": 875, "solution": "(100+75)*5"},
    {"nums": [25, 3, 1], "target": 76, "solution": "(25*3)+1"},
]
PROMPT_STYLES = {
    "zero": (),                              # no examples
    "r1zero_tool": (),                       # no examples: R1-Zero conversation frame + tool description only
    "fewshot_direct": ("direct",),           # format only: one direct, correct check -> answer
    "fewshot_direct_retry": ("direct",),     # same + one instruction sentence (rollout.RETRY_HINT)
    "fewshot": ("near_miss", "direct"),      # also demonstrates a failed check + systematic fix
    "fewshot_multi_retry": ("multi",),       # several candidates per check (+ rollout.MULTI_HINT)
    "fewshot_wide_retry": ("wide",),         # 8 candidates per check, refine from closest (+ rollout.WIDE_HINT)
}


def problem_prompt(nums: list[int], target: int, style: str = "zero", solution_hint: str = "") -> str:
    """The per-problem prompt for a prompt style (the ``*_retry`` styles add RETRY_HINT)."""
    if style == "r1zero_tool":
        if solution_hint:
            raise ValueError("r1zero_tool takes no solution hint")
        return format_prompt_r1zero_tool(nums, target)
    return format_prompt(nums, target, retry_hint=style.endswith("_retry"), multi=style.startswith("fewshot_multi"),
                         wide=style.startswith("fewshot_wide"), solution_hint=solution_hint)


def solution_hint(solution: str, frac: float) -> str:
    """A proper sub-expression of ``solution`` with about ``frac`` of its operations (at least 1):
    the largest such subtree, first in pre-order on ties. "" if there is nothing to reveal
    (frac <= 0, or the solution has a single operation). Used for QuestA-style hinted rollouts."""
    if frac <= 0:
        return ""
    tree = ast.parse(solution, mode="eval").body

    def n_ops(t):
        return 1 + n_ops(t.left) + n_ops(t.right) if isinstance(t, ast.BinOp) else 0

    total = n_ops(tree)
    if total < 2:
        return ""
    k = min(total - 1, max(1, round(frac * total)))
    best, best_ops = None, 0

    def walk(t, is_root):
        nonlocal best, best_ops
        if not isinstance(t, ast.BinOp):
            return
        o = n_ops(t)
        if not is_root and best_ops < o <= k:
            best, best_ops = t, o
        walk(t.left, False)
        walk(t.right, False)

    walk(tree, True)
    return ast.unparse(best) if best is not None else ""


def fewshot_prefix(sandbox: Sandbox, style: str = "fewshot", seed: int = 0) -> str:
    """Worked examples (real sandbox output) to prepend to a problem prompt; '' for 'zero'."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"prompt style must be one of {sorted(PROMPT_STYLES)}, got {style!r}")
    kinds = PROMPT_STYLES[style]
    rng = random.Random(seed)
    parts = []
    for kind, p in zip(kinds, FEWSHOT_PROBLEMS if len(kinds) == 1 else FEWSHOT_PROBLEMS[::-1]):
        segs = generate_trace(p, rng, sandbox, kind=kind)
        parts.append(problem_prompt(p["nums"], p["target"], style) + "".join(t for _, t in segs))
    return "\n\n".join(parts)


def with_prefix(prefix: str, prompt: str) -> str:
    return f"{prefix}\n\n{prompt}" if prefix else prompt

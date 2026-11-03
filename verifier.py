"""Model-independent Countdown answer verifier.

Parses an answer expression with :mod:`ast`, allows only ``+ - * /`` over
integer literals, evaluates exactly with :class:`fractions.Fraction`, and
requires the literals to be a sub-multiset of the given inputs (each input used
at most once).

Per the task spec this does **not** require every intermediate to be an integer
(Fraction allows exact non-integer intermediates, e.g. ``(1/2)+(1/2)``).
``integer_intermediates`` is provided separately for audit/analysis; the
generator produces integer-intermediate solutions by construction.
"""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction

_MAX_EXPR_LEN = 500
_ALLOWED_CHARS = frozenset("0123456789+-*/() ")

_BINOP = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


class ExprError(Exception):
    """Raised when an answer expression is malformed or uses disallowed syntax."""


def _eval(node, literals: Counter) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ExprError("non-integer literal")
        literals[node.value] += 1
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        fn = _BINOP.get(type(node.op))
        if fn is None:
            raise ExprError("disallowed operator")
        left = _eval(node.left, literals)
        right = _eval(node.right, literals)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ExprError("division by zero")
        return fn(left, right)
    raise ExprError("disallowed syntax")


def _check_int(node) -> Fraction:
    """Like _eval but requires every sub-expression to be an integer."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ExprError("non-integer literal")
        return Fraction(node.value)
    if isinstance(node, ast.BinOp):
        fn = _BINOP.get(type(node.op))
        if fn is None:
            raise ExprError("disallowed operator")
        left = _check_int(node.left)
        right = _check_int(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ExprError("division by zero")
        v = fn(left, right)
        if v.denominator != 1:
            raise ExprError("non-integer intermediate")
        return v
    raise ExprError("disallowed syntax")


def verify_answer(nums, target, expr) -> tuple[bool, str]:
    """Return ``(ok, detail)``.

    ``ok`` is True only when ``expr`` is an arithmetic expression over a
    sub-multiset of ``nums`` (using only ``+ - * /``) that equals ``target``.
    """
    if not isinstance(expr, str):
        return False, "not-a-string"
    s = " ".join(expr.split())  # normalize all whitespace (incl. newlines/tabs)
    if not s:
        return False, "empty"
    if len(s) > _MAX_EXPR_LEN:
        return False, "too-long"
    if any(c not in _ALLOWED_CHARS for c in s):
        return False, "disallowed-character"
    try:
        tree = ast.parse(s, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False, "syntax"

    literals: Counter = Counter()
    try:
        value = _eval(tree.body, literals)
    except ExprError as e:
        return False, str(e)
    except ZeroDivisionError:
        return False, "division-by-zero"
    except RecursionError:
        return False, "too-deep"

    if value != Fraction(target):
        return False, "wrong-value"

    counts = Counter(nums)
    if any(literals[k] > counts[k] for k in literals):
        return False, "reused/foreign-literal"
    return True, "ok"


def integer_intermediates(expr) -> bool:
    """True iff every sub-expression (including the whole) evaluates to an integer."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    try:
        _check_int(tree.body)
        return True
    except (ExprError, ZeroDivisionError, RecursionError):
        return False

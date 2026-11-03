"""Independent reference implementations for the verification audit.

Deliberately does NOT import or reuse anything from the project (no `ast`,
no project helpers), so a bug in the project's verifier/solver cannot hide
itself.

- ref_verify(nums, target, expr): hand-written tokenizer + recursive-descent
  parser over ASCII digits, + - * /, parentheses and ASCII whitespace.
  Exact Fraction arithmetic. Literals must be a sub-multiset of nums.
  Unary minus is rejected by default (matches the project's documented design;
  pass allow_unary=True to see what changes).
- ref_solve(nums, target): exhaustive DFS over pairs with + - * / (exact
  Fractions, negatives and non-integers allowed), subset semantics (a value
  equal to target anywhere in the working list counts). Returns
  (found, min_ops, nodes_to_first_solution).
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

ASCII_WS = " \t\n\r"


class RefError(Exception):
    pass


def _tokenize(s: str):
    toks = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ASCII_WS:
            i += 1
            continue
        if c in "+-*/()":
            toks.append(c)
            i += 1
            continue
        if "0" <= c <= "9":
            j = i
            while j < len(s) and "0" <= s[j] <= "9":
                j += 1
            lit = s[i:j]
            if len(lit) > 1 and lit[0] == "0":
                raise RefError("leading-zero literal")
            toks.append(int(lit))
            i = j
            continue
        raise RefError(f"bad char {c!r}")
    return toks


class _Parser:
    def __init__(self, toks, allow_unary):
        self.t = toks
        self.i = 0
        self.lits: list[int] = []
        self.allow_unary = allow_unary
        self.depth = 0
        self.inter: list[Fraction] = []

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.eat()
            r = self.term()
            v = v + r if op == "+" else v - r
            self.inter.append(v)
        return v

    def term(self):
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.eat()
            r = self.factor()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise RefError("div0")
                v = v / r
            self.inter.append(v)
        return v

    def factor(self):
        self.depth += 1
        if self.depth > 400:
            raise RefError("too deep")
        try:
            tok = self.peek()
            if tok is None:
                raise RefError("unexpected end")
            if isinstance(tok, int):
                self.eat()
                self.lits.append(tok)
                return Fraction(tok)
            if tok == "(":
                self.eat()
                v = self.expr()
                if self.peek() != ")":
                    raise RefError("missing )")
                self.eat()
                return v
            if tok == "-" and self.allow_unary:
                self.eat()
                return -self.factor()
            raise RefError(f"unexpected {tok!r}")
        finally:
            self.depth -= 1


def ref_parse(expr: str, allow_unary: bool = False):
    """Return (value: Fraction, literals: list[int]) or raise RefError."""
    toks = _tokenize(expr)
    if not toks:
        raise RefError("empty")
    p = _Parser(toks, allow_unary)
    v = p.expr()
    if p.i != len(toks):
        raise RefError("trailing tokens")
    return v, p.lits


def ref_intermediates(expr: str):
    """All binary-op results in evaluation order (raises RefError)."""
    toks = _tokenize(expr)
    p = _Parser(toks, False)
    p.expr()
    if p.i != len(toks):
        raise RefError("trailing tokens")
    return p.inter


def ref_verify(nums, target, expr, allow_unary: bool = False) -> bool:
    if not isinstance(expr, str):
        return False
    try:
        v, lits = ref_parse(expr, allow_unary)
    except RefError:
        return False
    if v != Fraction(target):
        return False
    have = Counter(int(x) for x in nums)
    need = Counter(lits)
    return all(need[k] <= have[k] for k in need)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _children(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vals[i], vals[j]
            rest = vals[:i] + vals[i + 1:j] + vals[j + 1:]
            outs = {a + b, a - b, b - a, a * b}
            if b != 0:
                outs.add(a / b)
            if a != 0:
                outs.add(b / a)
            for r in outs:
                yield tuple(sorted(rest + (r,)))


def ref_solve(nums, target, max_ops: int | None = None):
    """Iterative deepening on #operations. Returns (found, min_ops, nodes).

    nodes = total states expanded across all depths until the first solution
    (a search-effort proxy for difficulty).
    """
    t = Fraction(target)
    start = tuple(sorted(Fraction(x) for x in nums))
    if t in start:
        return True, 0, 1
    limit = (len(nums) - 1) if max_ops is None else max_ops
    nodes = 0
    for depth in range(1, limit + 1):
        seen: set = set()
        stack = [(start, 0)]
        while stack:
            vals, d = stack.pop()
            nodes += 1
            if d == depth:
                continue
            for child in _children(vals):
                key = (child, d + 1)
                if key in seen:
                    continue
                seen.add(key)
                if t in child:
                    return True, d + 1, nodes
                if d + 1 < depth:
                    stack.append((child, d + 1))
    return False, None, nodes

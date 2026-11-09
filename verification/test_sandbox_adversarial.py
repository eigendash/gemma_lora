"""Independent adversarial battery for sandbox.Sandbox (fast part).

Run: python -m pytest -q -s verification/test_sandbox_adversarial.py
Heavier checks (latency, 10k leak, concurrency, torch isolation, memory bombs)
live in verification/stress_sandbox_independent.py.
"""

from __future__ import annotations

import re
import time

import pytest

from countdown_rl.sandbox import Sandbox


@pytest.fixture(scope="module")
def sb():
    s = Sandbox(n_workers=2, timeout=0.050)
    yield s
    s.close()


LEAK_RE = re.compile(r"(/Users/|/home/|/private/|/tmp/|Traceback|File \"|\.py\b|site-packages)")

ESCAPES = {
    "import": "import os\nprint(os.getcwd())",
    "from_import": "from os import path",
    "__import__": "__import__('os')",
    "dunder_bases": "print(().__class__.__bases__)",
    "getattr": "getattr(1, 'real')",
    "eval": "eval('1+1')",
    "exec": "exec('print(1)')",
    "open": "open('/etc/passwd')",
    "globals": "print(globals())",
    "vars": "print(vars())",
    "dir": "print(dir())",
    "lambda": "f = lambda: 1",
    "listcomp": "print([x for x in range(3)])",
    "genexp": "print(sum(x for x in range(3)))",
    "fstring": "print(f'{1}')",
    "walrus": "print((y := 3))",
    "star_args": "print(*range(3))",
    "kw_sep": "print(1, 2, sep=0)",
    "kw_file": "print(1, file=0)",
    "attribute": "x = 1\nprint(x.real)",
    "subscript": "x = [1, 2]\nprint(x[0])",
    "decorator": "@print\ndef f():\n    pass",
    "def": "def f():\n    return 1",
    "class": "class A:\n    pass",
    "async": "async def f():\n    pass",
    "try": "try:\n    x = 1\nexcept Exception:\n    pass",
    "with": "with 1 as x:\n    pass",
    "yield": "yield 1",
    "del": "x = 1\ndel x",
    "global": "global x",
    "assert": "assert 1",
    "raise": "raise 1",
    "pow": "print(2 ** 10)",
    "str_literal": "print('hi')",
    "str_mult_bomb": "x = 'a' * 999999999",
    "print_str_bomb": "print('a' * 100000000)",
    "bytes": "print(b'x')",
    "float": "print(1.5)",
    "huge_int_literal": "print(10000000000000000)",
    "dict": "x = {1: 2}",
    "set": "x = {1}",
    "builtins_name": "print(__builtins__)",
    "print_obj": "print(print)",
    "deep_parens_10k": "(" * 10000 + "1" + ")" * 10000,
    "deep_parens_2k": "(" * 2000 + "1" + ")" * 2000,
    "long_expr": "print(" + "+".join(["1"] * 3000) + ")",
    "one_mb_source": "x = 1\n" * 150000,
}


@pytest.mark.parametrize("name", sorted(ESCAPES))
def test_escape_rejected_and_no_leak(sb, name):
    t0 = time.perf_counter()
    ok, msg = sb.run(ESCAPES[name])
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n{name:18s} ok={ok!s:5} {dt:7.1f}ms  {msg[:90]!r}")
    assert not LEAK_RE.search(msg), f"host info leaked: {msg!r}"
    assert len(msg) <= 300
    if name in ("builtins_name", "print_obj"):
        # allowed by design (Name / call print); must not expose anything usable
        return
    assert ok is False, f"{name} executed: {msg!r}"


@pytest.mark.parametrize("code,expect_ok", [
    ("print(len(range(1000000000000)))", True),          # lazy range: bounded
    ("print(range(1000000000))", True),
    ("x = 1000000000000\nwhile True:\n    x = x * x", False),   # bignum growth -> timeout
    ("while True:\n    pass", False),
    ("for i in range(1000000000000):\n    pass", False),
])
def test_resource_abuse_bounded(sb, code, expect_ok):
    t0 = time.perf_counter()
    ok, msg = sb.run(code)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\n{code[:40]!r:45} ok={ok} {dt:.1f}ms {msg[:80]!r}")
    assert ok == expect_ok
    assert dt < 1500, "call did not return promptly"


def test_stdout_truncated_to_200(sb):
    ok, out = sb.run("for i in range(500):\n    print(1)")   # ~1000 chars
    assert ok and len(out) == 200, (ok, len(out))


def test_big_print_loop_bounded(sb):
    ok, out = sb.run("for i in range(100000):\n    print(i)")
    print(f"\nprint-loop 100k -> ok={ok} {out[:80]!r}")
    assert len(out) <= 200


def test_timeout_is_50ms_not_relaxed():
    s = Sandbox(n_workers=1)
    try:
        assert s.timeout == 0.050
        t0 = time.perf_counter()
        ok, msg = s.run("while True:\n    pass")
        dt = (time.perf_counter() - t0) * 1000
        print(f"\ninfinite loop returned after {dt:.1f} ms: {msg!r}")
        assert not ok and msg.startswith("TimeoutError") and dt < 200
    finally:
        s.close()


def test_worker_killed_and_respawned_after_timeout():
    s = Sandbox(n_workers=1)
    try:
        pids_before = {w.proc.pid for w in s._all}
        s.run("while True:\n    pass")
        assert all(not w.proc.is_alive() for w in s._all if w.proc.pid in pids_before) or not (
            pids_before & {w.proc.pid for w in s._all})
        ok, out = s.run("print(41+1)")
        pids_after = {w.proc.pid for w in s._all}
        assert ok and out == "42"
        assert not (pids_before & pids_after), "timed-out worker was reused"
    finally:
        s.close()


def test_result_integrity_alternating_timeouts():
    s = Sandbox(n_workers=2)
    try:
        bad = []
        for i in range(60):
            s.run("while True:\n    pass")
            ok, out = s.run(f"print({i} * 7 + 1)")
            if not ok or out != str(i * 7 + 1):
                bad.append((i, ok, out))
        assert not bad, f"stale/misrouted results: {bad[:5]}"
    finally:
        s.close()

"""Minimal sandbox execution logic for spawned worker processes.

This module is deliberately tiny: it imports only the standard library
(``ast``, ``contextlib``, ``json``, ``os``, ``sys``) so a worker process starts
in milliseconds and never pulls in torch/transformers.

Safety model (defense in depth):
1. **AST whitelist** — rejects forbidden syntax before execution (no attribute
   access, subscript, imports, f-strings, comprehensions, lambdas, function/
   class defs, ``**`` (Pow), dictionaries/sets, try/with, etc.). Only integer
   literals (bounded), ``+ - * / // %``, comparisons, ``and/or/not``, ``if``,
   ``while``/``for`` loops, and calls to ``print``/``range``/``len``.
2. **Restricted builtins** — ``exec`` runs with ``__builtins__`` limited to
   ``{print, range, len}``, so ``__import__``/``open``/``eval``/``exec``/etc.
   raise NameError.
3. **Process isolation** — the caller runs each execution in a separate worker
   process (see ``sandbox.py``) that is killed on timeout, so infinite loops and
   C-level hangs cannot leak or wedge the pool.

Output is captured and truncated to ``max_stdout`` characters. The capture
buffer stops storing after ``_BUFFER_LIMIT`` characters (later writes are
counted and discarded), so an unbounded ``print`` loop cannot grow memory; it is
stopped by the parent's timeout instead.

Run as a script (``python -I -S sandbox_worker.py <max_stdout>``) this module is
the worker process: it reads one JSON request per line on stdin and writes one
JSON reply per line on the original stdout. It imports only the standard
library, so a worker never loads torch/transformers regardless of what the
parent process imported (unlike ``multiprocessing`` spawn, which re-imports the
parent's ``__main__``).
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import sys

MAX_CODE_LEN = 4096
MAX_INT = 10**12
MAX_NODES = 2000
MAX_DEPTH = 200
_BUFFER_LIMIT = 8192
DEFAULT_MAX_STDOUT = 200

ALLOWED_CALLS = frozenset({"print", "range", "len"})

_ALLOWED_TYPES = frozenset(
    {
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.If,
        ast.IfExp,
        ast.While,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Call,
        ast.Tuple,
        ast.List,
    }
)


class _BoundedWriter:
    """File-like capture buffer that keeps at most ``limit`` characters.

    Writes beyond the limit are discarded (memory stays bounded); execution is
    not interrupted, so the caller still gets the first characters of output.
    """

    def __init__(self, limit: int):
        self._limit = limit
        self._buf: list[str] = []
        self._size = 0

    def write(self, s) -> int:
        if not isinstance(s, str):
            s = str(s)
        room = self._limit - self._size
        if room > 0:
            self._buf.append(s[:room])
            self._size += min(len(s), room)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def getvalue(self) -> str:
        return "".join(self._buf)


def _check_node(node) -> str | None:
    if type(node) not in _ALLOWED_TYPES:
        return f"forbidden syntax: {type(node).__name__}"
    if isinstance(node, ast.Constant):
        v = node.value
        # allow int (and bool, which subclasses int: `while True` is idiomatic);
        # reject str/float/complex/bytes/None/Ellipsis (blocks string-multiply bombs)
        if not isinstance(v, int):
            return "only integer/bool literals are allowed"
        if abs(v) > MAX_INT:
            return "integer literal too large"
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id in ALLOWED_CALLS):
            return f"forbidden call: only {sorted(ALLOWED_CALLS)} are allowed"
    return None


def validate_ast(code: str) -> tuple[bool, str]:
    """Return ``(ok, message)`` for whether ``code`` passes the whitelist."""
    if not isinstance(code, str):
        return False, "code must be a string"
    if len(code) > MAX_CODE_LEN:
        return False, f"code too long (>{MAX_CODE_LEN} chars)"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return False, f"syntax error: {e.msg}"
    except (RecursionError, MemoryError):
        return False, "nesting too deep"

    stack = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES:
            return False, "too many AST nodes"
        if depth > MAX_DEPTH:
            return False, "nesting too deep"
        err = _check_node(node)
        if err:
            return False, err
        stack.extend((c, depth + 1) for c in ast.iter_child_nodes(node))
    return True, "ok"


def execute(code: str, max_stdout: int = DEFAULT_MAX_STDOUT) -> tuple[bool, str]:
    """Run validated code with restricted builtins. Returns ``(ok, text)``."""
    buf = _BoundedWriter(_BUFFER_LIMIT)
    safe_globals = {"__builtins__": {"print": print, "range": range, "len": len}}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<countdown-sandbox>", "exec"), safe_globals, {})
        return True, buf.getvalue().strip()[:max_stdout]
    except Exception as e:  # noqa: BLE001 — sandbox boundary; report, don't leak
        return False, f"RuntimeError: {type(e).__name__}: {e}"[:max_stdout]


def serve_stdio(max_stdout: int = DEFAULT_MAX_STDOUT) -> None:
    """Worker loop: one JSON request per stdin line -> one JSON reply per line.

    The protocol channel is a private dup of fd 1; fd 1 itself is pointed at
    /dev/null so nothing else in the process can corrupt the channel.
    """
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, "w")
    for line in sys.stdin:
        req = json.loads(line)
        if req is None:  # shutdown sentinel
            break
        code = req.get("code")
        ok, msg = validate_ast(code)
        reply = execute(code, max_stdout) if ok else (False, f"SecurityError: {msg}")
        proto.write(json.dumps(list(reply)) + "\n")


if __name__ == "__main__":
    serve_stdio(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_STDOUT)

"""Safe in-process execution sandbox with a persistent worker pool.

Each execution runs in a dedicated worker process (macOS ``spawn``) so a hard
timeout can kill and respawn the worker, and C-level hangs / infinite loops
cannot wedge the pool. Workers are reused across executions (no per-call spawn
cost) and recycled every ``recycle_every`` executions to stop slow leaks.

The AST whitelist + restricted builtins live in ``sandbox_worker`` (kept minimal
so worker import cost stays low). Validation is done in the parent first (fast
rejection) and repeated in the worker (defense in depth).
"""

from __future__ import annotations

import multiprocessing
import os
import threading
from collections import deque

import sandbox_worker


class _Worker:
    __slots__ = ("conn", "proc", "execs")

    def __init__(self, conn, proc):
        self.conn = conn
        self.proc = proc
        self.execs = 0

    def alive(self) -> bool:
        return self.proc.is_alive()


class Sandbox:
    def __init__(
        self,
        n_workers: int | None = None,
        timeout: float = 0.050,
        recycle_every: int = 1000,
        max_stdout: int = 200,
    ):
        self.timeout = float(timeout)
        self.recycle_every = int(recycle_every)
        self.max_stdout = int(max_stdout)
        self.n_workers = int(n_workers) if n_workers else max(1, os.cpu_count() or 4)
        self._ctx = multiprocessing.get_context("spawn")
        self._idle: deque[_Worker] = deque()
        self._all: set[_Worker] = set()
        self._lock = threading.Lock()

    # -- pool plumbing -------------------------------------------------------
    def _spawn(self) -> _Worker:
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=sandbox_worker.worker_main, args=(child, self.max_stdout), daemon=True
        )
        proc.start()
        child.close()
        w = _Worker(parent, proc)
        self._all.add(w)
        return w

    def _get_worker(self) -> _Worker:
        with self._lock:
            while self._idle:
                w = self._idle.popleft()
                if w.alive():
                    return w
                self._all.discard(w)
            if len(self._all) < self.n_workers:
                return self._spawn()
            return self._spawn()  # caller drives one execution at a time

    def _release(self, w: _Worker) -> None:
        should_terminate = False
        with self._lock:
            if w.execs >= self.recycle_every:
                self._all.discard(w)
                should_terminate = True
            elif w.alive():
                self._idle.append(w)
            else:
                self._all.discard(w)
        if should_terminate:
            self._terminate(w)

    def _kill(self, w: _Worker) -> None:
        with self._lock:
            self._all.discard(w)
        self._terminate(w)

    def _terminate(self, w: _Worker) -> None:
        """Lock-free teardown of a worker (must NOT be called while holding _lock)."""
        try:
            w.conn.close()
        except Exception:
            pass
        if w.proc.is_alive():
            w.proc.terminate()
            w.proc.join(timeout=1.0)
            if w.proc.is_alive():
                w.proc.kill()
                w.proc.join(timeout=1.0)

    # -- public API ----------------------------------------------------------
    def run(self, code: str) -> tuple[bool, str]:
        """Execute ``code`` and return ``(ok, stdout_or_error)``.

        Returns ``(False, "SecurityError: ...")`` for forbidden code,
        ``(False, "TimeoutError: ...")`` if execution exceeds ``self.timeout``,
        and ``(True, truncated_stdout)`` on success.
        """
        ok, msg = sandbox_worker.validate_ast(code)
        if not ok:
            return False, f"SecurityError: {msg}"

        w = self._get_worker()
        try:
            w.conn.send(code)
            if w.conn.poll(self.timeout):
                result = w.conn.recv()
            else:
                self._kill(w)
                return False, f"TimeoutError: execution exceeded {int(self.timeout * 1000)}ms"
        except (EOFError, BrokenPipeError, OSError, ConnectionError):
            self._kill(w)
            return False, "ExecutionError: worker died unexpectedly"

        w.execs += 1
        self._release(w)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return False, "ExecutionError: malformed worker response"

    def run_many(self, codes: list[str]) -> list[tuple[bool, str]]:
        """Convenience sequential runner over a list of code strings."""
        return [self.run(c) for c in codes]

    def close(self) -> None:
        for w in list(self._all):
            self._kill(w)
        self._idle.clear()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

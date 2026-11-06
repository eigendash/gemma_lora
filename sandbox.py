"""Safe execution sandbox with a persistent, capped worker pool.

Each execution runs in a dedicated worker *process* so a hard timeout can kill
it, and infinite loops or C-level hangs cannot wedge the pool. Workers are
reused across executions and recycled every ``recycle_every`` executions.

Workers are plain ``python -I -S sandbox_worker.py`` subprocesses speaking
line-delimited JSON over pipes (not ``multiprocessing``): under the ``spawn``
start method, multiprocessing re-imports the parent's ``__main__`` in every
worker, so a training script that imports torch made each worker load torch too
(~360 MB RSS, ~1.4 s per respawn). A subprocess worker imports only the stdlib,
and behaves identically on macOS and Linux.

Result integrity: each worker has its own pipe and serves one request at a
time. A worker that misses its deadline (execution or warm-up) is killed and
never reused, so a late reply can never be read by a later call.

The AST whitelist + restricted builtins live in ``sandbox_worker``. Validation
is done in the parent first (fast rejection) and repeated in the worker.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from collections import deque

import sandbox_worker

_WORKER_PATH = os.path.abspath(sandbox_worker.__file__)
_WARMUP_TIMEOUT = 10.0


class _Proc:
    """Small adapter over ``subprocess.Popen`` (``pid`` / ``is_alive``)."""

    __slots__ = ("p", "pid")

    def __init__(self, popen: subprocess.Popen):
        self.p = popen
        self.pid = popen.pid

    def is_alive(self) -> bool:
        return self.p.poll() is None


class _Worker:
    __slots__ = ("proc", "execs", "_buf")

    def __init__(self, proc: _Proc):
        self.proc = proc
        self.execs = 0
        self._buf = b""

    def alive(self) -> bool:
        return self.proc.is_alive()

    def send(self, code: str) -> None:
        self.proc.p.stdin.write((json.dumps({"code": code}) + "\n").encode())
        self.proc.p.stdin.flush()

    def recv(self, timeout: float):
        """Return the decoded reply, or None if ``timeout`` elapses first."""
        fd = self.proc.p.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError("worker closed its pipe")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def kill(self) -> None:
        p = self.proc.p
        if p.poll() is None:
            p.kill()
        try:
            p.wait(timeout=2.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill is not ignorable
            pass
        for f in (p.stdin, p.stdout):
            try:
                f.close()
            except OSError:
                pass


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
        self._idle: deque[_Worker] = deque()
        self._all: set[_Worker] = set()
        self._pending = 0  # workers being spawned (count toward the cap)
        self._cond = threading.Condition()
        self._closed = False
        # eagerly spawn + warm up so start-up cost is not paid inside an execution budget
        for _ in range(self.n_workers):
            w = self._spawn()
            with self._cond:
                self._idle.append(w)

    # -- pool plumbing -------------------------------------------------------
    def _spawn(self) -> _Worker:
        last_err = None
        for _ in range(2):
            popen = subprocess.Popen(
                [sys.executable, "-I", "-S", _WORKER_PATH, str(self.max_stdout)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            w = _Worker(_Proc(popen))
            try:
                w.send("print(0)")
                reply = w.recv(_WARMUP_TIMEOUT)
            except (OSError, EOFError, ValueError) as e:
                reply, last_err = None, e
            if reply == [True, "0"]:
                with self._cond:
                    self._all.add(w)
                return w
            # never keep a worker whose warm-up reply is missing: it could arrive later
            w.kill()
        raise RuntimeError(f"sandbox worker failed to start: {last_err!r}")

    def _get_worker(self) -> _Worker:
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("sandbox is closed")
                while self._idle:
                    w = self._idle.popleft()
                    if w.alive():
                        return w
                    self._all.discard(w)
                if len(self._all) + self._pending < self.n_workers:
                    self._pending += 1
                    break
                self._cond.wait()
        try:
            return self._spawn()
        finally:
            with self._cond:
                self._pending -= 1
                self._cond.notify()

    def _release(self, w: _Worker) -> None:
        retire = False
        with self._cond:
            if w.execs >= self.recycle_every or not w.alive() or self._closed:
                self._all.discard(w)
                retire = True
            else:
                self._idle.append(w)
            self._cond.notify()
        if retire:
            w.kill()

    def _kill(self, w: _Worker) -> None:
        with self._cond:
            self._all.discard(w)
            self._cond.notify()
        w.kill()

    # -- public API ----------------------------------------------------------
    def run(self, code: str) -> tuple[bool, str]:
        """Execute ``code`` and return ``(ok, stdout_or_error)``.

        ``(False, "SecurityError: ...")`` for forbidden code,
        ``(False, "TimeoutError: ...")`` if execution exceeds ``self.timeout``,
        ``(False, "RuntimeError: ...")`` if the code raised, and
        ``(True, stdout)`` (stripped, truncated to ``max_stdout``) on success.
        """
        ok, msg = sandbox_worker.validate_ast(code)
        if not ok:
            return False, f"SecurityError: {msg}"

        w = self._get_worker()
        try:
            w.send(code)
            result = w.recv(self.timeout)
        except (EOFError, BrokenPipeError, OSError, ValueError):
            self._kill(w)
            return False, "ExecutionError: worker died unexpectedly"
        if result is None:
            self._kill(w)
            return False, f"TimeoutError: execution exceeded {int(self.timeout * 1000)}ms"

        w.execs += 1
        self._release(w)
        if isinstance(result, list) and len(result) == 2:
            return bool(result[0]), str(result[1])
        return False, "ExecutionError: malformed worker response"

    def run_many(self, codes: list[str]) -> list[tuple[bool, str]]:
        """Convenience sequential runner over a list of code strings."""
        return [self.run(c) for c in codes]

    def close(self) -> None:
        with self._cond:
            self._closed = True
            workers = list(self._all)
            self._all.clear()
            self._idle.clear()
            self._cond.notify_all()
        for w in workers:
            w.kill()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

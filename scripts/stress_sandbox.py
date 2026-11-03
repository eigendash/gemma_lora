"""10k-execution sandbox stress test: latency distribution + RSS leak check.

Run from repo root:
    .venv/bin/python scripts/stress_sandbox.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402

from sandbox import Sandbox  # noqa: E402


def _total_rss() -> int:
    p = psutil.Process()
    rss = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            rss += c.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def main() -> None:
    n = 10_000
    n_timeout = 100
    sb = Sandbox(n_workers=4, timeout=0.050, recycle_every=1000)

    # warm up (spawn workers, settle memory)
    for _ in range(200):
        assert sb.run("print(1 + 2)") == (True, "3")
    base_rss = _total_rss()

    # latency over 10k normal executions
    lat: list[float] = []
    t_start = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        ok, _ = sb.run("print(7 * 3 + 1)")
        lat.append((time.perf_counter() - t0) * 1000.0)
        assert ok
    elapsed = time.perf_counter() - t_start
    mid_rss = _total_rss()

    # 100 timeout executions to exercise kill + respawn + recycle
    for _ in range(n_timeout):
        ok, text = sb.run("while True:\n    pass")
        assert not ok and "TimeoutError" in text
    final_rss = _total_rss()

    lat.sort()
    p50 = lat[int(0.50 * n)]
    p95 = lat[int(0.95 * n)]
    p99 = lat[int(0.99 * n)]

    print(f"executions: {n} normal + {n_timeout} timeout")
    print(f"workers: 4, timeout: 50ms, recycle_every: 1000")
    print(f"latency_ms: p50={p50:.3f} p95={p95:.3f} p99={p99:.3f} max={lat[-1]:.3f} mean={statistics.mean(lat):.3f}")
    print(f"throughput: {n / elapsed:.0f} exec/s")
    print(
        f"rss: base={base_rss / 1e6:.1f}MB mid={mid_rss / 1e6:.1f}MB "
        f"final={final_rss / 1e6:.1f}MB growth={(final_rss - base_rss) / 1e6:.1f}MB"
    )
    sb.close()


if __name__ == "__main__":
    main()

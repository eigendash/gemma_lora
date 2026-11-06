"""Independent sandbox stress: latency, concurrency, 10k leak, memory bombs,
and torch-in-worker isolation under macOS spawn.

Usage: python verification/stress_sandbox_independent.py [--n 10000]
Exit code = number of failed checks.
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

import psutil

from sandbox import Sandbox

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def latency():
    with Sandbox(n_workers=2) as s:
        lat = []
        for i in range(1000):
            t0 = time.perf_counter()
            ok, out = s.run(f"print({i}+1)")
            lat.append((time.perf_counter() - t0) * 1000)
            assert ok and out == str(i + 1)
    print(f"latency over 1000 trivial runs: p50={pct(lat,.5):.3f}ms p95={pct(lat,.95):.3f}ms "
          f"p99={pct(lat,.99):.3f}ms max={max(lat):.3f}ms")
    check("latency p99 < 5ms", pct(lat, .99) < 5)


def concurrency(threads=8, per=150):
    errs = []
    with Sandbox(n_workers=4) as s:
        def worker(t):
            for i in range(per):
                if i % 25 == 0:
                    s.run("while True:\n    pass")
                v = t * 100000 + i
                ok, out = s.run(f"print({v})")
                if not ok or out != str(v):
                    errs.append((t, i, ok, out))
        ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
        t0 = time.time()
        [t.start() for t in ts]
        [t.join() for t in ts]
        n_workers_alive = len([w for w in s._all if w.alive()])
        print(f"concurrency: {threads} threads x {per} calls (+timeouts) in {time.time()-t0:.1f}s; "
              f"errors={len(errs)}; workers in pool after={len(s._all)} (configured 4, alive {n_workers_alive})")
    check("no cross-talk under threads", not errs, str(errs[:3]))


def leak(n):
    me = psutil.Process()
    codes = ["print(3*7)", "x = 1\nprint(x + 2)", "print(1/0)", "print(y)", "print(",
             "for i in range(200):\n    print(i)", "x = [0] * 1000\nprint(len(x))"]
    rss_parent, rss_workers, n_children = [], [], []
    t0 = time.time()
    with Sandbox(n_workers=2, recycle_every=1000) as s:
        for i in range(n):
            code = "while True:\n    pass" if i % 100 == 99 else codes[i % len(codes)]
            s.run(code)
            if i % 500 == 0:
                rss_parent.append(me.memory_info().rss / 2**20)
                ch = me.children(recursive=True)
                n_children.append(len(ch))
                rss_workers.append(sum(c.memory_info().rss for c in ch if c.is_running()) / 2**20)
    print(f"leak: {n} mixed execs ({n//100} timeouts) in {time.time()-t0:.1f}s")
    print(f"  parent RSS MB first/last/max: {rss_parent[0]:.1f}/{rss_parent[-1]:.1f}/{max(rss_parent):.1f}")
    print(f"  workers RSS MB first/last/max: {rss_workers[0]:.1f}/{rss_workers[-1]:.1f}/{max(rss_workers):.1f}")
    print(f"  live child processes sampled: min={min(n_children)} max={max(n_children)}")
    check("parent RSS flat (<+30MB)", rss_parent[-1] - rss_parent[0] < 30)
    check("worker count bounded", max(n_children) <= 4, f"max children {max(n_children)}")
    # multiprocessing's resource_tracker is a legitimate long-lived child under spawn
    after = [c for c in me.children(recursive=True)
             if "resource_tracker" not in " ".join(c.cmdline())]
    check("no orphan workers after close", not after, f"{len(after)} left")


def memory_bomb():
    code = textwrap.dedent("""
        import resource, sys, time
        sys.path.insert(0, %r)
        from sandbox import Sandbox
        s = Sandbox(n_workers=1)
        t0 = time.perf_counter()
        r = s.run(%r)
        dt = (time.perf_counter() - t0) * 1000
        s.close()
        print(r[0], r[1][:60], f"{dt:.0f}ms", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    """)
    for bomb in ["x = [0] * 30000000\nprint(len(x))", "x = [0] * 1000000000", "x = (0,) * 300000000"]:
        out = subprocess.run([sys.executable, "-c", code % (os.getcwd(), bomb)], capture_output=True,
                             text=True, timeout=120).stdout.strip()
        *head, maxrss = out.rsplit(" ", 1)
        mb = int(maxrss) / 2**20  # macOS: bytes
        print(f"memory bomb {bomb!r:42} -> {' '.join(head)} | peak child RSS {mb:,.0f} MB")
        check(f"memory bomb bounded <1GB: {bomb.splitlines()[0]}", mb < 1024, f"{mb:.0f}MB")


def torch_isolation():
    """A training script imports torch at top-level; under spawn the worker
    re-imports the parent's __main__. Does the worker end up importing torch?"""
    d = tempfile.mkdtemp()
    main = os.path.join(d, "fake_train_main.py")
    with open(main, "w") as f:
        f.write(textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {os.getcwd()!r})
            import torch, transformers, peft  # like train_rl.py
            from sandbox import Sandbox
            if __name__ == "__main__":
                import psutil, subprocess
                t0 = time.perf_counter()
                s = Sandbox(n_workers=2)
                spawn_s = time.perf_counter() - t0
                for w in s._all:
                    p = psutil.Process(w.proc.pid)
                    libs = subprocess.run(["lsof", "-p", str(w.proc.pid)], capture_output=True, text=True).stdout
                    print("WORKER", w.proc.pid, "rss_mb=%.0f" % (p.memory_info().rss / 2**20),
                          "libtorch_mapped=%s" % ("libtorch" in libs))
                print("POOL_INIT_S %.2f" % spawn_s)
                print("RUN", s.run("print(7)"))
                s.close()
        """))
    out = subprocess.run([sys.executable, main], capture_output=True, text=True, timeout=300)
    print(out.stdout.strip())
    if out.returncode:
        print(out.stderr[-500:])
    mapped = "libtorch_mapped=True" in out.stdout
    check("worker does not load torch when parent __main__ imports torch", not mapped)
    check("first call after pool init returns its own result", "RUN (True, '7')" in out.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    latency()
    concurrency()
    leak(a.n)
    memory_bomb()
    torch_isolation()
    print("FAILED:", FAILS or "none")
    sys.exit(len(FAILS))


if __name__ == "__main__":
    main()

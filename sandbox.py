"""SPEC deliverable B under its spec name. The implementation is countdown_rl.sandbox (executor.py, worker.py).

Runs one snippet through the sandbox exactly as the training loop does and prints (ok, stdout):

    uv run python sandbox.py 'print(25 * 3 + 1)'
    echo 'import os' | uv run python sandbox.py -
"""
import sys

from countdown_rl.sandbox.executor import Sandbox

if __name__ == "__main__":
    code = sys.stdin.read() if len(sys.argv) < 2 or sys.argv[1] == "-" else sys.argv[1]
    with Sandbox(n_workers=1) as sb:
        print(sb.run(code))

#!/usr/bin/env bash
# Create .venv (Python 3.14) with the exact Linux/CUDA lockfile, independent of
# the image's system Python. uv downloads a standalone CPython 3.14 if needed.
# NOT run against a CUDA machine here (Mac only); the lockfile is verified by
# `uv pip compile` for linux x86_64 + Python 3.14.
set -euo pipefail
cd "$(dirname "$0")/.."
# system Pythons with PEP 668 (e.g. Ubuntu 24.04) refuse plain installs: fall back to --user
python3 -m pip install -q "uv==0.12.18" 2>/dev/null || python3 -m pip install -q --user --break-system-packages "uv==0.12.18"
python3 -m uv venv --python 3.14 .venv
python3 -m uv pip install --python .venv/bin/python -r lockfiles/cuda-linux-py314.lock
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

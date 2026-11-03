#!/usr/bin/env bash
# Install pinned CUDA dependencies on a stock PyTorch/CUDA Linux image.
# NOT run against a CUDA machine here (Mac only) — verified only by `uv pip compile`
# for linux x86_64 + Python 3.14 (see lockfiles/cuda-linux-py314.lock).
set -euo pipefail
python -m pip install --upgrade pip
# use the exact lockfile to reproduce the tested resolution
pip install -r lockfiles/cuda-linux-py314.lock
echo "CUDA deps installed (pinned). Verify with: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

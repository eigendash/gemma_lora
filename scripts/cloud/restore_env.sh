#!/usr/bin/env bash
# Restore the prebuilt vLLM venv from the bucket (saved from the L4 smoke VM, DLVM ubuntu-2404 image,
# user home /home/$USER/run) instead of a ~15 min PyPI install. Falls back silently: if the restored
# interpreter link is broken, scripts/cloud/train.sh rebuilds it with uv sync --extra cuda.
# usage (in the repo dir): GCS_BUCKET=gs://... bash scripts/cloud/restore_env.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
[ -x .venv-vllm/bin/python ] && { echo "venv present"; exit 0; }
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.14 >/dev/null 2>&1
gcloud storage cp "${GCS_BUCKET%/}/envs/venv-vllm-py314-cu-ubuntu2404.tar.zst" - | zstd -d -q | tar xf - \
  && .venv-vllm/bin/python -c "import vllm, torch; print('restored venv: vllm', vllm.__version__, 'torch', torch.__version__)" \
  || { echo "restore failed; will rebuild"; rm -rf .venv-vllm; }

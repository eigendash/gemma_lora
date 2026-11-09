#!/usr/bin/env bash
# One-command setup and check of the repository on a Linux machine with an NVIDIA (CUDA) or AMD (ROCm) GPU.
#
#   bash scripts/reproduce.sh            # environment, data, holdouts, tests, figures; then prints the run commands
#   bash scripts/reproduce.sh --smoke    # also a 30-problem evaluation of the main adapter (downloads ~9 GB)
#
# Training itself is not started here: a lineage takes about 12 GPU-hours on a 96 GB card. The commands to run one,
# and to evaluate any adapter on holdout3, are printed at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1

if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then EXTRA=cuda
elif command -v rocminfo >/dev/null || [ -e /dev/kfd ]; then EXTRA=rocm
else echo "no NVIDIA or AMD GPU found; the environment is defined for linux x86_64 with one of the two" >&2; exit 1; fi
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

echo "== environment (uv sync --extra $EXTRA)"
uv sync --extra "$EXTRA"

echo "== data: training and test sets, then the three holdouts, each disjoint from everything before it"
uv run countdown-data --n-train 10000 --n-eval 1000 --seed 0 --verify
uv run python -m countdown_rl.data.holdout --seed 20261105 --out data/holdout.jsonl
uv run python -m countdown_rl.data.holdout --seed 20261106 --out data/holdout2.jsonl \
    --exclude data/train.jsonl data/eval.jsonl data/holdout.jsonl
uv run python -m countdown_rl.data.holdout --seed 20261107 --out data/holdout3.jsonl \
    --exclude data/train.jsonl data/eval.jsonl data/holdout.jsonl data/holdout2.jsonl

echo "== tests (the data hashes are checked here; model-dependent tests skip without RUN_MODEL_TESTS=1)"
uv run pytest -q

echo "== figures from results/ into docs/figures/"
uv run python scripts/figures.py

if [ "$SMOKE" = 1 ]; then
  echo "== smoke evaluation: the zero-shot adapter on 30 holdout3 problems"
  ATTN=sdpa_gqa
  uv run hf download prada5023/countdown-rl-qwen3-4b-v2-zeroshot-s2 --local-dir adapters/qwen3-4b-v2-zeroshot-s2
  uv run countdown-eval --model Qwen/Qwen3-4B-Base --adapter adapters/qwen3-4b-v2-zeroshot-s2 \
      --eval-file data/holdout3.jsonl --n-eval 30 --prompt-style r1zero_tool --max-new-tokens 768 --max-rounds 2 \
      --attn "$ATTN" --batch-size 30
fi

cat <<'MSG'

Setup complete. Next steps:

  # one training phase (CUDA, >= 48 GB); later phases warm-start from the previous one via init_lora
  uv run countdown-train --config configs/experiments/qwen3-4b-v2-zeroshot-s2/phase_a.yaml --output_dir runs/a

  # a whole lineage on a GCP spot VM, with checkpoints in a bucket and auto-resume
  EXPERIMENT=qwen3-4b-v2-zeroshot-s2 BUCKET=gs://<bucket> CHAIN_ENV="EVAL_PROMPT=r1zero_tool" \
      bash scripts/cloud/create_gpu_vm.sh

  # any published adapter on holdout3, as for every number in REPORT.md (Gemma: --attn eager)
  uv run hf download prada5023/countdown-rl-<experiment> --local-dir adapters/<experiment>
  uv run countdown-eval --model Qwen/Qwen3-4B-Base --adapter adapters/<experiment> --eval-file data/holdout3.jsonl \
      --prompt-style r1zero_tool --max-new-tokens 768 --max-rounds 2 --attn sdpa_gqa --batch-size 128
MSG

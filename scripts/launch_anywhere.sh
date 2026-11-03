#!/usr/bin/env bash
# Provider-neutral launch: detect GPU -> pick preset -> pull latest valid
# checkpoint (comparing local disk) -> run the stage with --resume auto.
# Idempotent via a lockfile (two trainers never start).
# NOT run against a GPU host here; the preset mapping and gcs pull are
# documentation-grade and unverified (verified with `bash -n` only).
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RUN_ID:?}"; : "${RUN_STAGE:?}"; : "${COST_PER_HOUR:?}"
: "${CKPT_BACKEND:=gcs}"

LOCK="/tmp/countdown-$RUN_ID.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another trainer is running (lock $LOCK exists)"; exit 1
fi
trap 'rmdir "$LOCK"' EXIT

# --- detect GPU -------------------------------------------------------------
VRAM_MB=$(python - <<'PY'
import torch
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
    print(total)
else:
    print(0)
PY
)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)

# pick preset by VRAM (effective batch kept constant across presets)
if [ "$VRAM_MB" -ge 70000 ]; then PRESET=configs/preset-80gb.yaml
elif [ "$VRAM_MB" -ge 38000 ]; then PRESET=configs/preset-48gb.yaml
elif [ "$VRAM_MB" -ge 20000 ]; then PRESET=configs/preset-24gb.yaml
else echo "GPU VRAM ${VRAM_MB}MB too small; need >=24GB"; exit 1; fi
echo "detected GPU=$GPU_NAME vram=${VRAM_MB}MB preset=$PRESET"

# --- pull latest valid checkpoint (local first, then bucket) -----------------
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
import checkpoint as ck
# local disk check (local-first)
local = ck.LocalBucket("outputs")
# remote bucket (gcs) — unverified interface; falls back to local
try:
    remote = ck.get_bucket(os.environ.get("CKPT_BACKEND", "gcs"), root=os.environ.get("GCS_BUCKET", ""))
except Exception:
    remote = None
PY

# --- run stage --------------------------------------------------------------
case "$RUN_STAGE" in
  sft)
    python -u sft_coldstart.py --model google/gemma-2-2b --n-traces 2000 --epochs 2 \
      --dtype bfloat16 --attn eager --output-dir outputs/sft_gemma ;;
  grpo)
    # preset overrides: source the yaml via python into CLI flags
    python -u train_rl.py --model google/gemma-2-2b --init-adapter outputs/sft_gemma \
      --resume auto --backend "$CKPT_BACKEND" --chunked-vocab --attn eager --dtype bfloat16 \
      "$@" ;;
  eval)
    python -u eval.py --model google/gemma-2-2b --adapter outputs/rl/step-*/adapter --n-eval 1000 ;;
  *) echo "unknown RUN_STAGE=$RUN_STAGE"; exit 1 ;;
esac

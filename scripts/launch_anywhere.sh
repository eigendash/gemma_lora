#!/usr/bin/env bash
# Provider-neutral launch: detect GPU -> pick preset -> run the stage. Every
# stage is restartable on a NEW box with the same RUN_ID:
#   smoke : 10-step GRPO on configs/smoke_cuda.yaml (records step time / memory)
#   sft   : pull the SFT adapter from the bucket if it exists, else train + push it
#   grpo  : pull SFT adapter; train_rl.py restores the newest valid remote COMMIT
#           by itself (--resume auto + --run-id + --backend) and commits every save
#   eval  : pull SFT + latest RL checkpoint, run base / SFT / RL (+ controls,
#           rounds sweep) on all 1000 eval problems, push the results
# Idempotent via a lockfile (two trainers never start on one box).
# NOT run against a GPU host here (bash -n + local dry runs of the python CLIs).
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RUN_ID:?}"; : "${RUN_STAGE:?}"; : "${COST_PER_HOUR:?}"
: "${CKPT_BACKEND:=gcs}"
export CKPT_ROOT="${CKPT_ROOT:-$PWD/outputs/_remote}"   # only used by CKPT_BACKEND=local (dry runs)
PY="${PY:-.venv/bin/python}"
MODEL="${MODEL:-google/gemma-2-2b}"

LOCK="/tmp/countdown-$RUN_ID.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another trainer is running (lock $LOCK exists)"; exit 1
fi
trap 'rmdir "$LOCK"' EXIT

# --- detect GPU -> preset (override with PRESET=...) -------------------------
VRAM_MB=$("$PY" -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory // 2**20 if torch.cuda.is_available() else 0)')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
if [ -z "${PRESET:-}" ]; then
  if [ "$VRAM_MB" -ge 70000 ]; then PRESET=configs/preset-80gb.yaml
  elif [ "$VRAM_MB" -ge 38000 ]; then PRESET=configs/preset-48gb.yaml
  elif [ "$VRAM_MB" -ge 20000 ]; then PRESET=configs/preset-24gb.yaml
  else echo "GPU VRAM ${VRAM_MB}MB too small; need >=24GB"; exit 1; fi
fi
echo "detected GPU=$GPU_NAME vram=${VRAM_MB}MB preset=$PRESET stage=$RUN_STAGE run=$RUN_ID"

CK=("$PY" checkpoint.py --run-id "$RUN_ID" --backend "$CKPT_BACKEND")
pull_sft() { [ -f outputs/sft_gemma/adapter_config.json ] || "${CK[@]}" pull --kind sft --dir outputs/sft_gemma; }
COST=(--max-hours "${MAX_HOURS:-0}" --max-total-usd "${MAX_TOTAL_USD:-0}")

case "$RUN_STAGE" in
  smoke)
    "$PY" -u train_rl.py --config configs/smoke_cuda.yaml --model "$MODEL" --run-id "${RUN_ID}-smoke" \
      --backend "$CKPT_BACKEND" --output-dir outputs/smoke "${COST[@]}" "$@" ;;
  sft)
    if ! pull_sft; then
      "$PY" -u sft_coldstart.py --model "$MODEL" --n-traces 2000 --epochs 2 \
        --dtype bfloat16 --attn eager --output-dir outputs/sft_gemma
      "${CK[@]}" push --kind sft --dir outputs/sft_gemma
    fi ;;
  grpo)
    pull_sft
    "$PY" -u train_rl.py --config "$PRESET" --model "$MODEL" --init-adapter outputs/sft_gemma \
      --run-id "$RUN_ID" --backend "$CKPT_BACKEND" --resume auto --output-dir outputs/rl "${COST[@]}" "$@" ;;
  eval)
    pull_sft
    "${CK[@]}" pull --kind ckpt --dir outputs/rl_latest
    E=("$PY" -u eval.py --model "$MODEL" --n-eval 0 --max-new-tokens 512 --attn eager --dtype bfloat16)
    mkdir -p outputs/eval
    "${E[@]}" --fewshot --out outputs/eval/base_fewshot.json
    "${E[@]}" --init-adapter outputs/sft_gemma --out outputs/eval/sft.json
    "${E[@]}" --init-adapter outputs/sft_gemma --adapter outputs/rl_latest/adapter --rounds-sweep \
      --out outputs/eval/rl.json --dump outputs/eval/rl_transcripts.jsonl
    for c in blank shuffle; do
      "${E[@]}" --init-adapter outputs/sft_gemma --adapter outputs/rl_latest/adapter --control "$c" \
        --out "outputs/eval/rl_control_$c.json"
    done
    "${CK[@]}" push --kind eval --dir outputs/eval ;;
  *) echo "unknown RUN_STAGE=$RUN_STAGE (smoke|sft|grpo|eval)"; exit 1 ;;
esac

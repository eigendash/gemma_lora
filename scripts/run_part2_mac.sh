#!/usr/bin/env bash
# Part 2 on the Mac (MPS): gemma-2-2b SFT cold start -> GRPO. Resumable.
# eval.py is run separately afterwards (see HANDOFF.md / README).
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
if [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$(security find-generic-password -s HF_TOKEN -w 2>/dev/null)"
fi
PY="$PWD/.venv/bin/python"

SFT_DIR="outputs/sft_gemma"
RL_DIR="outputs/rl_gemma_part2"

echo "=== 1/2 SFT cold start ==="
if [ ! -d "$SFT_DIR" ]; then
  "$PY" -u sft_coldstart.py --model google/gemma-2-2b --n-traces 500 --epochs 1 \
    --device mps --dtype float16 --attn eager --output-dir "$SFT_DIR" 2>&1 | tee logs/sft_gemma.txt
else
  echo "SFT adapter already exists at $SFT_DIR (skipping)"
fi

echo "=== 2/2 GRPO (resume auto) ==="
"$PY" -u train_rl.py --model google/gemma-2-2b --init-adapter "$SFT_DIR" \
  --n-problems 500 --G 4 --prompts-per-step 1 --total-steps 500 --max-new-tokens 256 \
  --chunked-vocab --attn eager --dtype float16 \
  --checkpoint-every 5 --checkpoint-timer 300 \
  --output-dir "$RL_DIR" --device mps 2>&1 | tee logs/rl_gemma_part2.txt

echo "=== done (or killed) ==="
echo "latest checkpoint:"; cat "$RL_DIR/latest.json" 2>/dev/null
echo "log:"; tail -5 "$RL_DIR/train_log.jsonl" 2>/dev/null

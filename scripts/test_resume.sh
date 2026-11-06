#!/usr/bin/env bash
# kill -9 resume test: run a tiny GRPO, SIGKILL it mid-training, resume, and
# show the step counter continues. Local checkpoint backend only.
set -u
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
PY="$PWD/.venv/bin/python"
OUT="outputs/rl_resume_test"
rm -rf "$OUT"
ARGS="--model Qwen/Qwen2.5-0.5B --init-adapter outputs/sft_coldstart --n-problems 10 --G 4 --prompts-per-step 1 --total-steps 6 --max-new-tokens 96 --checkpoint-every 2 --dtype float16 --output-dir $OUT --device mps"

"$PY" -u train_rl.py $ARGS > logs/rl_resume_run1.txt 2>&1 &
PID=$!
for _ in $(seq 1 120); do
    [ -d "$OUT/step-4" ] && break
    sleep 2
done
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
sleep 2
echo "=== run1 (killed) steps ==="
grep -E "^step" logs/rl_resume_run1.txt

echo "=== resuming with --resume auto ==="
"$PY" -u train_rl.py $ARGS --resume auto > logs/rl_resume_run2.txt 2>&1
echo "=== run2 (resumed) steps ==="
grep -E "^step" logs/rl_resume_run2.txt
echo "=== latest.json ==="
cat "$OUT/latest.json"

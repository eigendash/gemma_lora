#!/usr/bin/env bash
# Run a GRPO config on a cloud GPU box with bucket checkpoints, auto-resuming after a crash.
# Each attempt runs in its own process group and is killed as a group (vLLM's EngineCore child
# would otherwise survive and hold GPU memory; see VERIFICATION.md §15). Spot preemption deletes
# the VM; relaunch on a new VM with the same RUN_ID and it resumes from the bucket.
# usage: RUN_ID=p7-... GCS_BUCKET=gs://... COST_PER_HOUR=2.0 CONFIG=configs/p7_rtx6000.yaml \
#        [MAX_TOTAL_USD=40] bash scripts/cloud/train.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0  # the FlashInfer sampler JIT-builds with ninja (not on PATH) and was never tested; vLLM's torch sampler was
: "${RUN_ID:?}"; : "${GCS_BUCKET:?}"; : "${COST_PER_HOUR:?}"; : "${CONFIG:?}"
export CKPT_BACKEND=gcs GCS_BUCKET PYTHONUNBUFFERED=1
PY="${PY:-.venv-vllm/bin/python}"
OUT="outputs/$RUN_ID"; mkdir -p "$OUT"
bash scripts/cloud/restore_env.sh
if [ ! -x "$PY" ]; then  # restore failed: build it (uv sync --extra cuda)
  UV_PROJECT_ENVIRONMENT=.venv-vllm ~/.local/bin/uv sync --frozen --extra cuda --no-dev
fi
EXTRA=(); [ -n "${MAX_TOTAL_USD:-}" ] && EXTRA+=(--max-total-usd "$MAX_TOTAL_USD")
sync_logs() {  # logs live on the VM disk, which spot preemption / max-run-duration deletes; mirror them to the bucket
  gcloud storage cp "$OUT/train.log" "${GCS_BUCKET%/}/runs/$RUN_ID/logs/" >/dev/null 2>&1 || true
  [ -f "$OUT/run/train_log.jsonl" ] && gcloud storage cp "$OUT/run/train_log.jsonl" "${GCS_BUCKET%/}/runs/$RUN_ID/logs/" >/dev/null 2>&1
  [ -d "$OUT/run/val" ] && gcloud storage rsync -r "$OUT/run/val" "${GCS_BUCKET%/}/runs/$RUN_ID/logs/val" >/dev/null 2>&1
  [ -f "$OUT/run/rollouts.jsonl" ] && gcloud storage cp "$OUT/run/rollouts.jsonl" "${GCS_BUCKET%/}/runs/$RUN_ID/logs/" >/dev/null 2>&1
  true
}
# provenance for later analysis: the exact config, code snapshot hash and environment of this attempt
{ echo "config: $CONFIG"; echo "host: $(hostname) $(date -u +%FT%TZ)"; sha256sum "$HOME/code.tgz" 2>/dev/null
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; { "$PY" -m pip freeze 2>/dev/null || ~/.local/bin/uv pip freeze --python "$PY" 2>/dev/null; }; } > "$OUT/provenance.txt"
gcloud storage cp "$CONFIG" "$OUT/provenance.txt" "${GCS_BUCKET%/}/runs/$RUN_ID/logs/" >/dev/null 2>&1 || true
# a resumed run on a fresh VM starts from the bucket's log, so the sync below never overwrites earlier steps
mkdir -p "$OUT/run"
[ -f "$OUT/run/train_log.jsonl" ] || gcloud storage cp "${GCS_BUCKET%/}/runs/$RUN_ID/logs/train_log.jsonl" "$OUT/run/" >/dev/null 2>&1 || true
( while sleep "${LOG_SYNC_S:-300}"; do sync_logs; done ) &
SYNC_PID=$!
for attempt in 1 2 3 4; do
  setsid "$PY" -u -m countdown_rl.rl.grpo --config "$CONFIG" --run-id "$RUN_ID" --backend gcs --output-dir "$OUT/run" \
    --device cuda --resume auto --cost-per-hour "$COST_PER_HOUR" "${EXTRA[@]}" >> "$OUT/train.log" 2>&1 &
  P=$!; wait $P; rc=$?
  kill -9 -- -$P 2>/dev/null || true
  echo "ATTEMPT $attempt exit $rc at $(date -u +%FT%TZ)" | tee -a "$OUT/train.log"
  [ $rc = 0 ] && break
  grep -q "GUARD:" "$OUT/train.log" && break
  sleep 20
done
kill "$SYNC_PID" 2>/dev/null || true
sync_logs

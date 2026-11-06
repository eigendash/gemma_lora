#!/usr/bin/env bash
# Run several GRPO configs back to back on one cloud GPU VM, then evaluate, then power off.
#   1. each RUN (id=config) through scripts/cloud_train.sh; a run whose final COMMIT is already in the bucket is
#      skipped, so relaunching this script on a fresh VM after a spot preemption resumes where it stopped;
#   2. 3-pass validation (separate vLLM processes, shuffled adapter order; EXPERIMENTS.md §2.7j) of VAL_ADAPTERS;
#   3. one test-set pass of TEST_ADAPTERS (all in one process, so the comparison is paired);
#   4. writes <bucket>/chain/<CHAIN_ID>/DONE and powers the VM off (POWEROFF=1), so an idle GPU is never billed.
# Adapters are given as name=src, src = a local dir or runs/<id>/ckpt/step_<n> (fetched from the bucket).
# Never reads the final-evaluation sets; a test greps this file to keep it that way.
# usage: CHAIN_ID=x GCS_BUCKET=gs://... COST_PER_HOUR=2.0 RUNS="id1=configs/a.yaml id2=configs/b.yaml" \
#        VAL_ADAPTERS="ref=baselines/p10b_s80 a80=runs/id1/ckpt/step_80" TEST_ADAPTERS="ref a80" \
#        [POWEROFF=1] bash scripts/cloud_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
: "${CHAIN_ID:?}"; : "${GCS_BUCKET:?}"; : "${COST_PER_HOUR:?}"; : "${RUNS:?}"; : "${VAL_ADAPTERS:?}"
B="${GCS_BUCKET%/}"; PY="${PY:-.venv-vllm/bin/python}"; LOG="outputs/chain_$CHAIN_ID"; mkdir -p "$LOG"
export VLLM_USE_FLASHINFER_SAMPLER=0
mark() { echo "$* $(date -u +%FT%TZ)" | tee -a "$LOG/chain.log"; gcloud storage cp "$LOG/chain.log" "$B/chain/$CHAIN_ID/" >/dev/null 2>&1 || true; }

for rc in $RUNS; do
  id="${rc%%=*}"; cfg="${rc#*=}"
  steps=$("$PY" -c "import train_rl as t; print(t.build_config(['--config', '$cfg']).total_steps)")
  if gcloud storage ls "$B/runs/$id/ckpt/step_$steps/COMMIT.json" >/dev/null 2>&1; then mark "SKIP $id (step_$steps committed)"; continue; fi
  mark "TRAIN $id $cfg"
  RUN_ID="$id" CONFIG="$cfg" GCS_BUCKET="$B" COST_PER_HOUR="$COST_PER_HOUR" bash scripts/cloud_train.sh > "$LOG/$id.wrapper.log" 2>&1
  gcloud storage ls "$B/runs/$id/ckpt/step_$steps/COMMIT.json" >/dev/null 2>&1 || { mark "FAILED $id"; exit 1; }
  mark "TRAINED $id"
done

NAMES=()
for a in $VAL_ADAPTERS; do
  n="${a%%=*}"; src="${a#*=}"
  if [ ! -d "baselines/$n" ]; then
    if [ -d "$src" ]; then cp -r "$src" "baselines/$n"; else gcloud storage cp -r "$B/$src/adapter" "baselines/$n.tmp" >/dev/null && mv "baselines/$n.tmp" "baselines/$n"; fi
  fi
  [ -f "baselines/$n/adapter_model.safetensors" ] || { mark "MISSING adapter $n ($src)"; exit 1; }
  NAMES+=("$n")
done
mark "VAL3 ${NAMES[*]}"
for pass in 1 2 3; do
  ORD=$(printf "%s\n" "${NAMES[@]}" | shuf --random-source=<(yes $pass))
  ARGS=(); for n in $ORD; do ARGS+=(--adapter "$n=baselines/$n"); done
  echo "=== pass $pass order: $(echo $ORD)" >> "$LOG/val3.log"
  "$PY" verification/eval_vllm.py --set val --val-n 300 --prompt-style fewshot_multi_retry --gpu-util 0.85 \
    --out-dir "$LOG/val3/pass$pass" "${ARGS[@]}" 2>&1 | grep -E ": correct |Traceback|Error" >> "$LOG/val3.log"
  gcloud storage rsync -r "$LOG" "$B/chain/$CHAIN_ID" >/dev/null 2>&1
done
if [ -n "${TEST_ADAPTERS:-}" ]; then
  ARGS=(); for n in $TEST_ADAPTERS; do ARGS+=(--adapter "$n=baselines/$n"); done
  mark "TEST $TEST_ADAPTERS"
  "$PY" verification/eval_vllm.py --set test --prompt-style fewshot_multi_retry --gpu-util 0.85 \
    --out-dir "$LOG/test" "${ARGS[@]}" 2>&1 | grep -E ": correct |Traceback|Error" >> "$LOG/test.log"
fi
gcloud storage rsync -r "$LOG" "$B/chain/$CHAIN_ID" >/dev/null 2>&1
mark DONE
gcloud storage cp "$LOG/chain.log" "$B/chain/$CHAIN_ID/DONE" >/dev/null 2>&1
[ "${POWEROFF:-0}" = 1 ] && sudo poweroff
exit 0

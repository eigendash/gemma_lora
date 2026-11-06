#!/usr/bin/env bash
# Cloud smoke test (~20-30 min on one >=24 GB NVIDIA GPU). UNTESTED until run on a CUDA box.
# Proves, before any long run: (1) vLLM generation works with our tool loop and LoRA sync,
# (2) checkpoints commit to the bucket and a killed run resumes from it, (3) the cost guard
# stops the run, (4) real step time on this GPU.
# usage: RUN_ID=smoke-$(date +%s) GCS_BUCKET=... COST_PER_HOUR=0.9 bash scripts/cloud_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # the FlashInfer sampler JIT-builds with ninja (not on PATH) and was never tested; vLLM's torch sampler was
: "${RUN_ID:?}"; : "${COST_PER_HOUR:?}"
export CKPT_BACKEND="${CKPT_BACKEND:-gcs}" PYTHONUNBUFFERED=1
PY="${PY:-.venv-vllm/bin/python}"
OUT=outputs/cloud_smoke; REP=$OUT/SMOKE_REPORT.txt
mkdir -p "$OUT"; : > "$REP"
say() { echo "$*" | tee -a "$REP"; }

if [ ! -x "$PY" ]; then   # one venv for trainer + vLLM (vLLM pins torch/transformers)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv .venv-vllm --python 3.14
  ~/.local/bin/uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt google-cloud-storage==3.17.1 pyyaml pytest
fi
say "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"$PY" -m pytest -q tests/test_rollout_vllm.py tests/test_train_rl.py 2>&1 | tail -1 | sed 's/^/unit tests: /' | tee -a "$REP"

T=("$PY" -u train_rl.py --config "${SMOKE_CONFIG:-configs/smoke_cloud_qwen3.yaml}" --run-id "$RUN_ID" --backend "$CKPT_BACKEND"
   --output-dir "$OUT/run" --device cuda --resume auto --cost-per-hour "$COST_PER_HOUR")

# 1) train until the first remote commit (step 2), then kill -9 (simulated preemption)
# setsid: own process group, so kill -9 also takes vLLM's EngineCore child (a real preemption kills the
# whole VM; killing only the trainer PID leaves an orphan EngineCore holding GPU memory)
setsid "${T[@]}" > "$OUT/phase1.log" 2>&1 & P=$!
for _ in $(seq 1 360); do
  rm -rf "$OUT/probe"; "$PY" checkpoint.py --run-id "$RUN_ID" --backend "$CKPT_BACKEND" pull --kind ckpt --dir "$OUT/probe" >/dev/null 2>&1 && break
  kill -0 $P 2>/dev/null || break; sleep 10
done
kill -9 -- -$P 2>/dev/null || kill -9 $P 2>/dev/null || true; wait $P 2>/dev/null || true
for _ in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
say "after kill: GPU procs left: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" " ")"
say "phase1 steps: $(grep -c '^step' "$OUT/phase1.log")  last: $(grep '^step' "$OUT/phase1.log" | tail -1 | cut -c1-160)"
rm -rf "$OUT/run"   # fresh box: nothing local survives

# 2) resume on a "new box": must restore the remote commit and finish
"${T[@]}" > "$OUT/phase2.log" 2>&1 || say "phase2 FAILED (exit $?)"
say "phase2 restore: $(grep -m1 -i 'restored remote checkpoint' "$OUT/phase2.log" || echo 'NO RESTORE LINE')"
say "phase2 steps: $(grep -c '^step' "$OUT/phase2.log")  done: $(grep -c 'done;' "$OUT/phase2.log")"

# 3) cost guard: a tiny budget must stop immediately with a checkpoint
"$PY" -u train_rl.py --config "${SMOKE_CONFIG:-configs/smoke_cloud_qwen3.yaml}" --run-id "${RUN_ID}-guard" --backend none \
  --output-dir "$OUT/guard" --device cuda --cost-per-hour "$COST_PER_HOUR" --max-total-usd 0.0001 > "$OUT/guard.log" 2>&1 || true
say "cost guard: $(grep -m1 'max-total-usd' "$OUT/guard.log" || echo 'DID NOT TRIP')"

# 4) throughput + vLLM/trainer agreement
"$PY" - <<'PY' | tee -a "$REP"
import json
rows=[json.loads(l) for l in open("outputs/cloud_smoke/run/train_log.jsonl")]
for r in rows:
    print(f"step {r['step']}: gen {r['gen_s']}s loss {r['loss_s']}s tokens {r['gen_tokens']} "
          f"logp_dev mean {r.get('gen_logp_dev_mean')} p99 {r.get('gen_logp_dev_p99')} tis_w {r.get('tis_w_mean')} "
          f"correct {r['correct']:.3f} contradicted {r.get('contradicted')}")
s=[r['gen_s']+r['loss_s'] for r in rows[1:]] or [rows[0]['gen_s']+rows[0]['loss_s']]
print(f"steady step time {sum(s)/len(s):.0f}s -> {3600*len(s)/sum(s):.1f} steps/hour (8 prompts x 8)")
PY
"$PY" checkpoint.py --run-id "$RUN_ID" --backend "$CKPT_BACKEND" push --kind eval --dir "$OUT" >/dev/null 2>&1 || true
say "report: $REP"

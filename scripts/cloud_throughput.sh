#!/usr/bin/env bash
# Short throughput check of the real GRPO recipe on one cloud GPU (no checkpoints to the bucket).
# usage: CONFIG=configs/throughput_rtx6000_qwen3.yaml COST_PER_HOUR=4.5 bash scripts/cloud_throughput.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # the FlashInfer sampler JIT-builds with ninja (not on PATH) and was never tested; vLLM's torch sampler was
: "${COST_PER_HOUR:?}"; CONFIG="${CONFIG:-configs/throughput_rtx6000_qwen3.yaml}"
export PYTHONUNBUFFERED=1
PY="${PY:-.venv-vllm/bin/python}"
OUT=outputs/cloud_throughput; REP=$OUT/THROUGHPUT_REPORT.txt
mkdir -p "$OUT"; : > "$REP"
say() { echo "$*" | tee -a "$REP"; }
if [ ! -x "$PY" ]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  ~/.local/bin/uv venv .venv-vllm --python 3.14
  ~/.local/bin/uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt google-cloud-storage==3.17.1 pyyaml pytest
fi
say "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
say "config: $CONFIG"
t0=$(date +%s)
"$PY" -u train_rl.py --config "$CONFIG" --run-id "tp-$(date +%s)" --backend none --output-dir "$OUT/run" \
  --device cuda --resume none --cost-per-hour "$COST_PER_HOUR" > "$OUT/train.log" 2>&1 || say "train FAILED (exit $?)"
say "wall: $(( $(date +%s) - t0 ))s"
say "peak mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1) (at end)"
"$PY" - <<'PY' | tee -a "$REP"
import json, os
p = "outputs/cloud_throughput/run/train_log.jsonl"
rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for r in rows:
    print(f"step {r['step']}: gen {r['gen_s']}s loss {r['loss_s']}s tokens {r.get('gen_tokens')} sampled {r.get('n_sampled_prompts')} "
          f"logp_dev mean {r.get('gen_logp_dev_mean')} p99 {r.get('gen_logp_dev_p99')} tis_w {r.get('tis_w_mean')} correct {r['correct']:.3f}")
s = [r['gen_s'] + r['loss_s'] for r in rows[1:]] or [r['gen_s'] + r['loss_s'] for r in rows]
if s: print(f"steady step time {sum(s)/len(s):.0f}s -> {3600*len(s)/sum(s):.1f} steps/hour (16 prompts x 8)")
PY
say "report: $REP"

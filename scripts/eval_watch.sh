#!/usr/bin/env bash
# Evaluate every new GRPO checkpoint of a run (plus the base model as step 0) on a fixed
# stratified eval subset, greedy, same prompt style as training. Stops when the run's
# exit-code file appears and all checkpoints are evaluated.
# usage: [MAXTOK=384] [BS=48] [ATTN=sdpa] scripts/eval_watch.sh <run_dir> <model> <prompt_style> <out_prefix> <done_file> [n_eval]
set -u
RUN="$1"; MODEL="$2"; STYLE="$3"; OUT="$4"; DONE="$5"; N="${6:-48}"
PY="${PY:-.venv/bin/python}"
EV=("$PY" -u eval.py --model "$MODEL" --n-eval "$N" --max-new-tokens "${MAXTOK:-384}" --temperature 0 --attn "${ATTN:-sdpa}"
    --prompt-style "$STYLE" --batch-size "${BS:-48}")
[ -f "${OUT}_step0.json" ] || "${EV[@]}" --out "${OUT}_step0.json" --dump "${OUT}_step0.jsonl" > "${OUT}_step0.log" 2>&1
while true; do
  for d in $(ls -d "$RUN"/step-* 2>/dev/null | sort -t- -k2 -n); do
    n=${d##*-}
    [ -f "${OUT}_step${n}.json" ] && continue
    [ -f "$d/adapter/adapter_config.json" ] || continue
    "${EV[@]}" --adapter "$d/adapter" --out "${OUT}_step${n}.json" --dump "${OUT}_step${n}.jsonl" > "${OUT}_step${n}.log" 2>&1
  done
  if [ -f "$DONE" ]; then
    left=0
    for d in "$RUN"/step-*; do [ -f "${OUT}_step${d##*-}.json" ] || left=1; done
    [ "$left" = 0 ] && break
  fi
  sleep 120
done

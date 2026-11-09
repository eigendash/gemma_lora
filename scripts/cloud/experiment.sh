#!/usr/bin/env bash
# Run one experiment (all phase configs of configs/experiments/NAME, in order) through chain.sh on this VM, then run
# 3-pass validation of every phase's final step and power off.
# usage: GCS_BUCKET=gs://... [GATE=<base val % | step0>:<min gain pp>] [EVAL_PROMPT=...] [EVAL_MODEL=...] \
#        bash scripts/cloud/experiment.sh NAME
# The gate, if given, is checked after phase a (or after the only config). Run ids are NAME-<phase>.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; DIR="configs/experiments/$NAME"
[ -d "$DIR" ] || { echo "no such experiment: $DIR"; exit 1; }
RUNS=""; VAL=""; FIRST=""
for cfg in "$DIR"/phase_*.yaml "$DIR"/cont.yaml; do
  [ -f "$cfg" ] || continue
  ph=$(basename "$cfg" .yaml); ph="${ph#phase_}"; id="$NAME-$ph"
  RUNS+="$id=$cfg "; VAL+="$ph=runs/$id/ckpt/latest "; FIRST="${FIRST:-$id}"
done
export CHAIN_ID="$NAME" RUNS VAL_ADAPTERS="$VAL" COST_PER_HOUR="${COST_PER_HOUR:-2.0}" POWEROFF="${POWEROFF:-1}"
[ -n "${GATE:-}" ] && export GATE="$FIRST:$GATE"
bash scripts/cloud/chain.sh

#!/usr/bin/env bash
# Gemma-4-E2B zero-shot, phase a ONLY with the gate (EXPERIMENTS.md 2.7w): a fail-fast test of whether zero-shot RL starts
# at ~2B (probe: 1/256 problems with signal, greedy val 0/300). Phases b-d need a separate decision. Powers off at the end.
cd ~/run
CHAIN_ID=e2v2z0a GCS_BUCKET=gs://countdown-rl-16875-ckpt COST_PER_HOUR=2.0 \
  RUNS="e2v2z0-a=configs/lineage/e2v2z0-a.yaml" \
  GATE="e2v2z0-a:0.0:3" EVAL_PROMPT=r1zero_tool EVAL_MODEL=models/gemma-4-E2B-text \
  VAL_ADAPTERS="e2z_a=runs/e2v2z0-a/ckpt/latest" \
  POWEROFF=1 bash scripts/cloud_chain.sh
sudo poweroff

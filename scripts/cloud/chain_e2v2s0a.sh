#!/usr/bin/env bash
# Gemma-4-E2B WITH the demo prompt (v2 recipe), phase a only, 50 steps (EXPERIMENTS.md 2.7w): the matched pair of e2v2z0-a.
cd ~/run
CHAIN_ID=e2v2s0a GCS_BUCKET=gs://countdown-rl-16875-ckpt COST_PER_HOUR=2.0 \
  RUNS="e2v2s0-a=configs/lineage/e2v2s0-a.yaml" \
  GATE="e2v2s0-a:step0:3" EVAL_PROMPT=fewshot_direct_retry EVAL_MODEL=models/gemma-4-E2B-text \
  VAL_ADAPTERS="e2s_a=runs/e2v2s0-a/ckpt/latest" \
  POWEROFF=1 bash scripts/cloud/chain.sh
sudo poweroff

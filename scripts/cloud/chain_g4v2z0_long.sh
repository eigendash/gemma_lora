#!/usr/bin/env bash
# Gemma zero-shot long continuation (EXPERIMENTS.md 2.7u), run on the restarted tp-rtx6000-g4 after chain g4v2z0 is DONE.
cd ~/run
CHAIN_ID=g4v2z0long GCS_BUCKET=gs://gdrive-cli-513614-g4ckpt COST_PER_HOUR=2.0 \
  RUNS="g4v2z0long=configs/g4v2z0_long.yaml" EVAL_PROMPT=r1zero_tool EVAL_MODEL=models/gemma-4-E4B-text \
  VAL_ADAPTERS="g4z_long=runs/g4v2z0long/ckpt/latest" \
  POWEROFF=1 bash scripts/cloud/chain.sh
sudo poweroff

#!/usr/bin/env bash
# Gemma-4-E4B zero-shot arm chain (EXPERIMENTS.md 2.7t). Runs on its own VM (project gdrive-cli-513614); committed phases
# are skipped and an interrupted phase resumes from its last bucket checkpoint. Powers off at the end or on a gate fail.
cd ~/run
CHAIN_ID=g4v2z0 GCS_BUCKET=gs://countdown-rl-16875-ckpt COST_PER_HOUR=2.0 \
  RUNS="g4v2z0-a=configs/lineage/g4v2z0-a.yaml g4v2z0-b=configs/lineage/g4v2z0-b.yaml g4v2z0-c=configs/lineage/g4v2z0-c.yaml g4v2z0-d=configs/lineage/g4v2z0-d.yaml" \
  GATE="g4v2z0-a:1.3:3" EVAL_PROMPT=r1zero_tool EVAL_MODEL=models/gemma-4-E4B-text \
  VAL_ADAPTERS="g4z_a=runs/g4v2z0-a/ckpt/latest g4z_b=runs/g4v2z0-b/ckpt/latest g4z_c=runs/g4v2z0-c/ckpt/latest g4z_d=runs/g4v2z0-d/ckpt/latest" \
  POWEROFF=1 bash scripts/cloud_chain.sh
sudo poweroff

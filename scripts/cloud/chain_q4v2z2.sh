#!/usr/bin/env bash
# Zero-shot arm chain (EXPERIMENTS.md 2.7s), for a fresh VM: committed phases are skipped and an interrupted phase
# resumes from its last bucket checkpoint, so relaunching this after a VM loss continues where it stopped.
cd ~/run
CHAIN_ID=q4v2z2 GCS_BUCKET=gs://countdown-rl-16875-ckpt COST_PER_HOUR=2.0 \
  RUNS="q4v2z2-a=configs/lineage/q4v2z2-a.yaml q4v2z2-b=configs/lineage/q4v2z2-b.yaml q4v2z2-c=configs/lineage/q4v2z2-c.yaml q4v2z2-d=configs/lineage/q4v2z2-d.yaml" \
  GATE="q4v2z2-a:4.3:3" EVAL_PROMPT=r1zero_tool \
  VAL_ADAPTERS="z2_a=runs/q4v2z2-a/ckpt/latest z2_b=runs/q4v2z2-b/ckpt/latest z2_c=runs/q4v2z2-c/ckpt/latest z2_d=runs/q4v2z2-d/ckpt/latest" \
  POWEROFF=1 bash scripts/cloud/chain.sh
sudo poweroff

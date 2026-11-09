#!/usr/bin/env bash
# Gemma-4-E2B zero-shot probe (r1zero_tool): learning signal at T=0.9 (256 train x 8), then greedy val300 baseline
cd ~/gemma_lora_run; export HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1
.venv/bin/python eval.py --model models/gemma-4-E2B-text --eval-file outputs/probe_train256x8.jsonl --n-eval 0 --max-new-tokens 768 \
  --max-rounds 2 --temperature 0.9 --seed 0 --dtype bfloat16 --attn eager --device cuda --prompt-style r1zero_tool \
  --generation batched --batch-size 128 --out outputs/zeroprobe/e2_r1zero_tool_t09.json --dump outputs/zeroprobe/e2_r1zero_tool_t09.jsonl \
  > outputs/zeroprobe/e2_r1zero_tool_t09.log 2>&1
.venv/bin/python eval.py --model models/gemma-4-E2B-text --eval-file outputs/zeroprobe/val300.jsonl --n-eval 0 --max-new-tokens 768 \
  --max-rounds 2 --temperature 0 --seed 0 --dtype bfloat16 --attn eager --device cuda --prompt-style r1zero_tool \
  --generation batched --batch-size 128 --out outputs/zeroprobe/e2_r1zero_tool_val300_greedy.json --dump outputs/zeroprobe/e2_r1zero_tool_val300_greedy.jsonl \
  > outputs/zeroprobe/e2_r1zero_tool_val300_greedy.log 2>&1

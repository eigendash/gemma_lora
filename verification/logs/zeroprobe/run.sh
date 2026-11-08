#!/usr/bin/env bash
# zero-shot learnability probe: base Qwen3-4B, T=0.9, 256 train problems x 8 samples, two no-demo prompts
cd ~/gemma_lora_run; export HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1
for st in r1zero_tool zero; do
  .venv/bin/python eval.py --model Qwen/Qwen3-4B-Base --eval-file outputs/probe_train256x8.jsonl --n-eval 0 --max-new-tokens 768 \
    --max-rounds 2 --temperature 0.9 --seed 0 --dtype bfloat16 --attn sdpa_gqa --device cuda --prompt-style $st \
    --generation batched --batch-size 128 --out outputs/zeroprobe/base_${st}_t09.json --dump outputs/zeroprobe/base_${st}_t09.jsonl \
    > outputs/zeroprobe/base_${st}_t09.log 2>&1
done

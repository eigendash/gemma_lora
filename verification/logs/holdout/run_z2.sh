#!/usr/bin/env bash
# Pre-registered (EXPERIMENTS 2.7s): q4v2z2-d s80 on holdout3, once. Primary r1zero_tool, secondary fewshot_multi_retry.
cd ~/gemma_lora_run; export HF_HOME=$PWD/.hf-cache HF_HUB_OFFLINE=1
for st in r1zero_tool fewshot_multi_retry; do
  tag=h3_q4v2z2_d80; [ $st = r1zero_tool ] || tag=h3_q4v2z2_d80_fewshot
  .venv/bin/python eval.py --model Qwen/Qwen3-4B-Base --adapter baselines/q4v2z2_d80 --eval-file data/holdout3.jsonl --n-eval 0 \
    --max-new-tokens 768 --max-rounds 2 --temperature 0 --seed 0 --dtype bfloat16 --attn sdpa_gqa --device cuda \
    --prompt-style $st --generation batched --batch-size 128 --out outputs/holdout/$tag.json --dump outputs/holdout/$tag.jsonl \
    > outputs/holdout/$tag.log 2>&1
done

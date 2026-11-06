"""Greedy vLLM evaluation of LoRA adapters (and/or the base model) on a fixed problem set.

Same prompt, tool loop and verifier as training (train_rl.run_validation). Writes one
eval.py-compatible dump per model, so verification/eval_compare.py gives paired McNemar tests.

usage (CUDA box, .venv-vllm):
  python verification/eval_vllm.py --set val --val-n 300 --out-dir outputs/val_eval \
      --model base --adapter p6_s60=/path/to/adapter --adapter p6_s99=/path/to/adapter
  python verification/eval_vllm.py --set test --n 1000 --out-dir outputs/test_eval --adapter p7=/path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sft_coldstart  # noqa: E402
import train_rl  # noqa: E402
from rewards import RewardConfig  # noqa: E402
from sandbox import Sandbox  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test", "holdout"], required=True,
                    help="holdout = data/holdout.jsonl: generated 2026-11-05 AFTER all model selection; final report only")
    ap.add_argument("--val-n", type=int, default=300, help="--set val: same split as train_rl val_n")
    ap.add_argument("--n", type=int, default=0, help="--set test: first n problems (0 = all)")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--test-file", default="data/eval.jsonl")
    ap.add_argument("--holdout-file", default="data/holdout.jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--prompt-style", default="fewshot_direct_retry")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--base", action="store_true", help="also evaluate the base model (no adapter)")
    ap.add_argument("--adapter", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    if a.set == "val":
        allp = [json.loads(line) for line in open(a.train_file)]
        _, probs = train_rl.split_validation(allp, a.val_n)
    else:
        probs = [json.loads(line) for line in open(a.test_file if a.set == "test" else a.holdout_file)]
        probs = probs[: a.n] if a.n else probs

    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(a.model_name)
    llm = LLM(model=a.model_name, dtype="bfloat16", enable_lora=True, max_lora_rank=a.lora_r, max_loras=1,
              gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True, max_model_len=4096, seed=0)
    cfg = train_rl.GRPOConfig(prompt_style=a.prompt_style, max_new_tokens=a.max_new_tokens,
                              max_rounds=a.max_rounds, on_round_cap="notice")
    sandbox = Sandbox(n_workers=2, timeout=0.050)  # same as train_rl
    prefix = sft_coldstart.fewshot_prefix(sandbox, cfg.prompt_style)
    models = ([("base", None)] if a.base else []) + [tuple(x.split("=", 1)) for x in a.adapter]
    for name, path in models:  # an empty/missing path would silently evaluate the BASE model
        if path is not None and not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise SystemExit(f"adapter {name!r}: no adapter_config.json under {path!r}")
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    for i, (name, path) in enumerate(models):
        req = LoRARequest(name, i + 1, path) if path else None
        t0 = time.time()
        res = train_rl.run_validation(llm, tok, sandbox, probs, cfg, req, RewardConfig(), prefix,
                                      os.path.join(a.out_dir, f"{name}.jsonl"))
        summary[name] = res
        print(f"{name}: correct {res['val_correct']:.3f} ({round(res['val_correct'] * res['val_n'])}/{res['val_n']}) "
              f"{res['val_tier']} {time.time() - t0:.0f}s", flush=True)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

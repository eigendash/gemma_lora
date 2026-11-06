"""Generation throughput benchmark (soos / any CUDA-ROCm box).

micro: per-decode-step cost vs batch size and context length (HF DynamicCache).
e2e:   one real GRPO generation batch (prompts_per_step x G rollouts, training prompt style,
       sampling temperature) through rollout.run_rollouts_batched with a chosen implementation;
       reports wall time, useful tokens/s and the length distribution.

usage:
  python verification/bench_generation.py micro
  python verification/bench_generation.py e2e --impl lockstep|compact|continuous|vllm --max-new-tokens 768
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def load(model_name, dtype=torch.bfloat16, attn="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import attention_gqa

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype,
                                                 attn_implementation=attention_gqa.resolve(attn)).to("cuda").eval()
    return tok, model


def prompts_for(n_prompts, G, style, seed=0):
    import random

    import sft_coldstart
    from sandbox import Sandbox

    sb = Sandbox(n_workers=4, timeout=0.05)
    prefix = sft_coldstart.fewshot_prefix(sb, style)
    rows = [json.loads(l) for l in open("data/train.jsonl")]
    random.Random(seed).shuffle(rows)
    chosen = rows[:n_prompts]
    return sb, [sft_coldstart.with_prefix(prefix, sft_coldstart.problem_prompt(p["nums"], p["target"], style))
                for p in chosen for _ in range(G)]


def micro(args):
    from rollout import _model_call

    tok, model = load(args.model, attn=args.attn)
    if args.compile:
        model.forward = torch.compile(model.forward, dynamic=True)
    sync = torch.cuda.synchronize
    Ls = [int(x) for x in args.Ls.split(",")]
    Bs = [int(x) for x in args.Bs.split(",")]
    for L in Ls:
        for B in Bs:
            ids = torch.randint(1000, 20000, (B, L), device="cuda")
            attn = torch.ones_like(ids)
            pos = attn.cumsum(-1) - 1
            with torch.no_grad():
                sync(); t = time.time()
                out = _model_call(model, input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=True)
                sync(); tp = time.time() - t
                past = out.past_key_values
                nxt = pos[:, -1] + 1
                steps, tm = 32, 0.0
                for k in range(steps + (8 if args.compile else 0)):
                    if args.compile and k == 8:
                        tm = 0.0  # discard warm-up / recompiles
                    attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device="cuda")], 1)
                    x = torch.randint(1000, 20000, (B, 1), device="cuda")
                    t = time.time()
                    out = _model_call(model, input_ids=x, attention_mask=attn, position_ids=nxt.unsqueeze(1),
                                      past_key_values=past, use_cache=True)
                    sync(); tm += time.time() - t
                    past = out.past_key_values
                    nxt = nxt + 1
            print(f"L={L:5d} B={B:4d}  prefill {tp:6.2f}s  decode {1000 * tm / steps:7.1f} ms/step  "
                  f"{B * steps / tm:7.0f} tok/s", flush=True)
            del past, out
            torch.cuda.empty_cache()


def e2e(args):
    import rollout

    tok, model = load(args.model, attn=args.attn)
    if args.lora:  # wrap exactly like train_rl (LoRA r=16 on all linear targets, grad ckpt on, eval mode)
        import train_rl

        model = train_rl._apply_lora(model, 16, 32)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.eval()
    sb, prompts = prompts_for(args.prompts, args.G, args.prompt_style, args.seed)
    rng = torch.Generator(device="cuda").manual_seed(args.seed)
    kw = dict(max_rounds=2, max_new_tokens=args.max_new_tokens, temperature=args.temperature, rng=rng,
              device="cuda", on_round_cap="notice")
    torch.cuda.synchronize()
    t = time.time()
    if args.impl == "lockstep":
        res = rollout.run_rollouts_batched(model, tok, prompts, sb, batch_size=args.batch, **kw)
    else:
        res = rollout.run_rollouts_batched(model, tok, prompts, sb, batch_size=args.batch, impl=args.impl, **kw)
    torch.cuda.synchronize()
    dt = time.time() - t
    gen = [sum(r.loss_mask) for r in res]
    reasons = {}
    for r in res:
        reasons[r.terminal_reason] = reasons.get(r.terminal_reason, 0) + 1
    from rewards import compute_reward

    rows = [json.loads(l) for l in open("data/train.jsonl")]
    import random
    random.Random(args.seed).shuffle(rows)
    chosen = [p for p in rows[:args.prompts] for _ in range(args.G)]
    correct = sum(compute_reward(r, p["nums"], p["target"]).correct > 0 for r, p in zip(res, chosen))
    out = {"impl": args.impl, "lora": args.lora, "grad_ckpt": args.grad_ckpt, "n": len(res), "max_new_tokens": args.max_new_tokens, "wall_s": round(dt, 1),
           "gen_tokens": sum(gen), "useful_tok_per_s": round(sum(gen) / dt, 1),
           "len_mean": round(st.mean(gen), 1), "len_p90": sorted(gen)[int(0.9 * len(gen))], "len_max": max(gen),
           "reasons": reasons, "correct": correct}
    print(json.dumps(out), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["micro", "e2e"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--impl", default="lockstep")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--prompt-style", default="fewshot_direct_retry")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--lora", action="store_true", help="wrap in the trainer's PEFT LoRA")
    ap.add_argument("--compile", action="store_true", help="micro: torch.compile(model.forward, dynamic=True)")
    ap.add_argument("--Ls", default="256,1024")
    ap.add_argument("--attn", default="sdpa", help="sdpa | sdpa_gqa (copy-free GQA decode)")
    ap.add_argument("--Bs", default="128,64,16,4")
    ap.add_argument("--grad-ckpt", action="store_true", help="with --lora: enable gradient checkpointing like the trainer")
    a = ap.parse_args()
    micro(a) if a.mode == "micro" else e2e(a)

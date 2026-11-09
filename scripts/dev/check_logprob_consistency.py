"""Generation vs training log-prob consistency at step 0 (real model).

A scripted rollout (fixed token choices, REAL model logits for the log-probs)
through rollout.run_rollout, then the exact training path:
grpo.collate_rollouts + (a) forward_logprobs_chunked (presets use
--chunked-vocab) and (b) chunked_logprobs(model(...).logits), compared against
the generation-time old_logprobs and against the ground truth
log_softmax(model(ids).logits) (HF forward, includes any final-logit softcap).

At step 0 the policy equals the behaviour policy, so ratio = exp(new - old)
must be ~1 everywhere the loss is applied. Reported separately for tokens
before and after the first injected <output>.

Usage: python scripts/dev/check_logprob_consistency.py --model Qwen/Qwen2.5-0.5B
       python scripts/dev/check_logprob_consistency.py --model google/gemma-2-2b --attn eager
"""

from __future__ import annotations

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from countdown_rl.rl import rollout
    from countdown_rl.rl import grpo
    from countdown_rl.sandbox import Sandbox

    dt = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, attn_implementation=args.attn)
    print(f"model={args.model} dtype={args.dtype} attn={args.attn} device={args.device} "
          f"final_logit_softcapping={getattr(base.config, 'final_logit_softcapping', None)}")
    model = grpo._apply_lora(base, 16, 32).to(args.device)
    model.eval()

    chunks = ["<think>\nTry 25*3+1.\n</think>\n", "<python>\nres = 25*3+1\nprint(res == 76, res)\n</python>",
              "Good.\n<answer>\n25*3+1\n</answer>"]
    script = []
    for c in chunks:
        script += tok.encode(c, add_special_tokens=False)
    queue = list(script)

    def scripted_sample(logits, temperature, rng):
        nxt = queue.pop(0)
        lp = torch.log_softmax(logits.float(), dim=-1)[0, nxt].item()
        return nxt, lp

    rollout._sample = scripted_sample
    with Sandbox(n_workers=1) as sb, torch.no_grad():
        res = rollout.run_rollout(model, tok, rollout.format_prompt([25, 3, 1], 76), sb,
                                  max_new_tokens=200, temperature=1.0, device=args.device)
    print(f"rollout: code_rounds={res.code_rounds} has_answer={res.has_answer} "
          f"n_completion={len(res.completion_ids)} n_env={res.loss_mask.count(0)}")

    batch = grpo.collate_rollouts([res], tok.pad_token_id or tok.eos_token_id, args.device)
    with torch.no_grad():
        lp_chunked = grpo.forward_logprobs_chunked(model, batch["input_ids"], batch["attention_mask"],
                                                       batch["target_ids"])
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        lp_full = grpo.chunked_logprobs(logits, batch["target_ids"])
        lp_truth = torch.log_softmax(logits.float(), dim=-1).gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)

    m = batch["pred_mask"][0].bool()
    old = batch["old_logp"][0]
    P = len(res.prompt_ids)
    first_env = P + res.loss_mask.index(0) if 0 in res.loss_mask else 10**9
    pos = torch.arange(m.shape[0], device=m.device)
    pre = m & (pos + 1 < first_env)
    post = m & (pos + 1 > first_env)

    def stats(name, new, sel):
        d = (new[0] - old)[sel].float()
        if d.numel() == 0:
            return f"{name}: n=0"
        r = d.exp()
        return (f"{name}: n={d.numel():3d} max|new-old|={d.abs().max().item():.4g} "
                f"ratio min={r.min().item():.4f} max={r.max().item():.4f}")

    ok = True
    for label, sel in (("BEFORE first <output>", pre), ("AFTER  first <output>", post)):
        print(f"--- {label}")
        for name, new in (("ground-truth HF log_softmax", lp_truth), ("train_rl chunked_logprobs  ", lp_full),
                          ("forward_logprobs_chunked   ", lp_chunked)):
            line = stats(name, new, sel)
            print("   ", line)
    tol = 1e-3 if args.dtype == "float32" else 5e-2
    for name, new in (("chunked_logprobs", lp_full), ("forward_logprobs_chunked", lp_chunked)):
        d = (new[0] - old)[m].abs().max().item()
        ok &= d < tol
        print(f"RATIO≈1 CHECK {name}: max|Δlogp|={d:.4g} (tol {tol}) -> {'PASS' if d < tol else 'FAIL'}")
    d_impl = (lp_chunked[0] - lp_truth[0])[m].abs().max().item()
    print(f"forward_logprobs_chunked vs HF ground truth (same ids, same context): max|Δ|={d_impl:.4g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

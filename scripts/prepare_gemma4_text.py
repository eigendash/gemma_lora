"""Gemma 4 (multimodal checkpoint) -> text-only model for GRPO: load, LoRA-target and vLLM checks.

google/gemma-4-E4B is published as Gemma4ForConditionalGeneration (vision + audio towers + language model), and
AutoModelForCausalLM maps model_type gemma4 to that multimodal class. Our LoRA targets are generic projection names
(q_proj, ..., down_proj), which the towers use too. This script:
  1. loads the text-only Gemma4ForCausalLM from the checkpoint and reports missing / unexpected weights;
  2. counts LoRA-target modules in the multimodal vs the text-only model (text-only must have none in towers);
  3. saves the text-only model + tokenizer to --out (what training and vLLM then both load);
  4. loads --out in vLLM with LoRA enabled and checks greedy completions agree with HF on a few-shot val prompt;
The few-shot base accuracy (the go/no-go read) then comes from python -m countdown_rl.evaluation.evaluate_vllm --base --model-name <out>.
usage: python scripts/prepare_gemma4_text.py --model google/gemma-4-E4B --out /path/gemma-4-E4B-text
"""
import argparse
import os
import sys
import time

import torch


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_targets(model):
    names = [n for n, _ in model.named_modules() if n.split(".")[-1] in TARGETS]
    towers = [n for n in names if any(t in n for t in ("vision", "audio", "multi_modal", "embed_vision", "embed_audio"))]
    return len(names), len(towers), towers[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4 import Gemma4ForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model)
    t0 = time.time()
    mm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    print(f"[1] AutoModelForCausalLM -> {type(mm).__name__}, params {sum(p.numel() for p in mm.parameters())/1e9:.2f}B,"
          f" LoRA targets (total, in towers, examples) = {lora_targets(mm)}  ({time.time()-t0:.0f}s)")
    # Gemma4ForCausalLM.from_pretrained cannot read the multimodal checkpoint (its keys live under
    # model.language_model.*): every language weight would be missing. Build the text model from text_config
    # and copy the language-model weights across, then check the two give identical logits.
    text = Gemma4ForCausalLM(mm.config.text_config).to(torch.bfloat16)
    src = mm.state_dict()
    pre = next(p for p in ("model.language_model.", "language_model.model.", "language_model.") if any(k.startswith(p) for k in src))
    sd = {"model." + k[len(pre):]: v for k, v in src.items() if k.startswith(pre)}
    res = text.load_state_dict(sd, strict=False)
    if getattr(text.config, "tie_word_embeddings", False) or "lm_head.weight" in res.missing_keys:
        text.tie_weights()
    missing = [k for k in res.missing_keys if k != "lm_head.weight"]
    print(f"[1] text-only Gemma4ForCausalLM from prefix {pre!r}: copied {len(sd)} tensors, missing={missing[:5]} "
          f"({len(missing)}), unexpected={list(res.unexpected_keys)[:5]} ({len(res.unexpected_keys)}), "
          f"params {sum(p.numel() for p in text.parameters())/1e9:.2f}B")
    assert not missing and not res.unexpected_keys, "language weights did not map 1:1"
    n, nt, ex = lora_targets(text)
    print(f"[2] text-only LoRA targets {n}, in towers {nt} {ex}")
    tc = text.config
    print(f"    final_logit_softcapping={getattr(tc, 'final_logit_softcapping', None)} sliding_window={getattr(tc, 'sliding_window', None)}"
          f" tie_word_embeddings={getattr(tc, 'tie_word_embeddings', None)}")
    probe = tok("Target 132. Check several candidates at once: 10 * 9 * (8 - 6);", return_tensors="pt").input_ids.cuda()
    mm, text = mm.cuda().eval(), text.cuda().eval()
    with torch.no_grad():
        lm, lt = mm(input_ids=probe).logits.float(), text(input_ids=probe).logits.float()
    print(f"[2] logits multimodal vs text-only: max |diff| {(lm - lt).abs().max().item():.3g}, "
          f"argmax agree {(lm.argmax(-1) == lt.argmax(-1)).float().mean().item():.3f}")
    del mm
    torch.cuda.empty_cache()
    text.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"[3] saved text-only model to {a.out}: {sorted(os.listdir(a.out))[:8]}")

    import json

    from countdown_rl.rl import prompting
    from countdown_rl.sandbox import Sandbox

    p0 = json.loads(open("data/train.jsonl").readlines()[-1])  # a val problem (val = tail of train.jsonl)
    sb = Sandbox(n_workers=1, timeout=0.050)
    prefix = prompting.fewshot_prefix(sb, "fewshot_multi_retry")
    prompt = prompting.with_prefix(prefix, prompting.problem_prompt(p0["nums"], p0["target"], "fewshot_multi_retry"))
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        hf = text.generate(ids, max_new_tokens=64, do_sample=False)[0, ids.shape[1]:].tolist()
    del text
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.out, dtype="bfloat16", enable_lora=True, max_lora_rank=16, max_loras=1,
              gpu_memory_utilization=0.6, max_model_len=4096)
    out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64))[0].outputs[0].token_ids
    agree = next((i for i, (x, y) in enumerate(zip(hf, out)) if x != y), min(len(hf), len(out)))
    print(f"[4] vLLM loaded {a.out} with LoRA; greedy agreement with HF: first {agree}/64 tokens identical")
    print("    HF  :", repr(tok.decode(hf)[:160]))
    print("    vLLM:", repr(tok.decode(list(out))[:160]))
    sb.close()


if __name__ == "__main__":
    main()

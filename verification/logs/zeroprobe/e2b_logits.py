import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/gemma-4-E2B-text")
ids = tok("Target 132. Check several candidates at once: 10 * 9 * (8 - 6);", return_tensors="pt").input_ids
for dev, dt in [("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16)]:
    out = {}
    for name in ["google/gemma-4-E2B", "models/gemma-4-E2B-text"]:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=dt, attn_implementation="eager").to(dev).eval()
        with torch.no_grad(): out[name] = m(input_ids=ids.to(dev)).logits.float().cpu()
        print(dev, dt, name, type(m).__name__, "nan:", out[name].isnan().any().item(), flush=True)
        del m; torch.cuda.empty_cache()
    a, b = out.values()
    print(dev, dt, "max|diff|", (a - b).abs().max().item(), "argmax agree", (a.argmax(-1) == b.argmax(-1)).float().mean().item(), flush=True)

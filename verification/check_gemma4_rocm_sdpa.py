"""Gemma-4-E4B on soos (ROCm): NaN logits from SDPA attention, eager is fine (2026-11-07). Run on soos from the repo root.
Prints, per attention implementation, whether left-padded / unpadded logits contain NaN."""
import sys, os, torch
sys.path.insert(0, os.getcwd())
import attention_gqa
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
M = "models/gemma-4-E4B-text"
tok = AutoTokenizer.from_pretrained(M)
A = "Target 15. Check several candidates at once: 100 - 10 * 5; 100 / 10 + 5."
Bp = "Numbers: 3 5 25. Target 17. " * 6 + "Check several candidates at once:"
for attn in ["eager", "sdpa", "sdpa_gqa"]:
    for adapter in [False, True]:
        m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation=attention_gqa.resolve(attn)).cuda().eval()
        if adapter: m = PeftModel.from_pretrained(m, "baselines/g4e4b_d80").eval()
        ids = [tok(A)["input_ids"], tok(Bp)["input_ids"]]
        L = max(map(len, ids)); pad = tok.pad_token_id
        x = torch.tensor([[pad] * (L - len(i)) + i for i in ids]).cuda()
        am = torch.tensor([[0] * (L - len(i)) + [1] * len(i) for i in ids]).cuda()
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = m(input_ids=x, attention_mask=am, position_ids=pos).logits.float()
            single = m(input_ids=torch.tensor([ids[0]]).cuda()).logits.float()
        nan_rows = [bool(torch.isnan(lg[b]).any()) for b in range(2)]
        nan_real = bool(torch.isnan(lg[0, L - len(ids[0]):]).any())
        d = (lg[0, -1] - single[0, -1]).abs().max().item()
        print(f"attn={attn:9s} adapter={adapter}: NaN per row {nan_rows}, NaN on real tokens of padded row {nan_real}, last-logit |padded - single| {d:.3g}, argmax {lg[0,-1].argmax().item()} vs {single[0,-1].argmax().item()}", flush=True)
        del m; torch.cuda.empty_cache()

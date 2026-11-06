import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import train_rl
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32, attn_implementation="eager")
model = train_rl._apply_lora(base, 16, 32).to("mps").eval()
ids = tok.encode("Using the numbers 25, 3, 1, make 76.\n<think>\nTry 25*3+1.\n</think>", add_special_tokens=True)
x = torch.tensor([ids], device="mps"); tgt = torch.tensor([ids[1:] + [0]], device="mps")
with torch.no_grad():
    truth = torch.log_softmax(model(input_ids=x).logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, :-1]
    chunked = train_rl.forward_logprobs_chunked(model, x, torch.ones_like(x), tgt)[0, :-1]
    backbone, head = train_rl._causal_backbone_and_head(model)
    h = backbone(input_ids=x, use_cache=False)[0]
    raw = head(h)
    cap = base.config.final_logit_softcapping
    capped = torch.tanh(raw / cap) * cap
    lp_cap = torch.log_softmax(capped.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, :-1]
print("max |raw logit| =", raw.abs().max().item())
print("chunked (no softcap) vs truth  max|Δ| =", (chunked - truth).abs().max().item())
print("manual softcap        vs truth max|Δ| =", (lp_cap - truth).abs().max().item())

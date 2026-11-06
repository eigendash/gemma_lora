import time, torch
from transformers import Qwen2Config, Qwen2ForCausalLM
import train_rl
print("device:", torch.cuda.get_device_name(0), "| bf16 supported:", torch.cuda.is_bf16_supported())
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=32000, hidden_size=512, intermediate_size=1408, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=2048)
m = Qwen2ForCausalLM(cfg).to("cuda", torch.bfloat16)
m.config._attn_implementation = "eager"
x = torch.randint(0, 32000, (8, 256), device="cuda")
t0 = time.time()
out = m(input_ids=x, labels=x); out.loss.backward(); torch.cuda.synchronize()
print(f"bf16 fwd+bwd ok: loss {out.loss.item():.3f} in {time.time()-t0:.2f}s, grad finite:",
      all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))
# chunked log-probs (+softcap path) vs full on GPU
h = torch.randn(2, 16, 512, device="cuda"); head = torch.nn.Linear(512, 32000, bias=False).cuda()
tgt = torch.randint(0, 32000, (2, 16), device="cuda")
ref = torch.log_softmax(torch.tanh(head(h) / 30) * 30, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
got = train_rl._logprobs_from_hidden(h, head, tgt, vocab_chunk=4096, softcap=30.0)
print("chunked+softcap vs full max|d|:", (got - ref).abs().max().item())
# batched generation with KV cache, left padding, position ids
from transformers import AutoTokenizer
class Tok:  # minimal byte-level stand-in so no tokenizer download is needed
    eos_token_id = 1; pad_token_id = 0
    def encode(self, s, add_special_tokens=False): return [2 + (b % 30000) for b in s.encode()]
    def decode(self, ids, skip_special_tokens=False): return "".join(chr(min(127, 32 + i % 90)) for i in ids)
import rollout
from sandbox import Sandbox
m.eval()
with Sandbox(n_workers=2) as sb:
    t0 = time.time()
    rs = rollout.run_rollouts_batched(m, Tok(), ["a" * n for n in (10, 40, 25, 7)], sb, max_new_tokens=32,
                                      temperature=1.0, device="cuda", rng=torch.Generator(device="cuda").manual_seed(0))
    print(f"batched rollout ok: {[len(r.completion_ids) for r in rs]} tokens in {time.time()-t0:.2f}s")

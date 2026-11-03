"""Tag tokenization + segment-boundary probe (Qwen2.5-0.5B, gemma-2-2b).

Run: zsh -c "source ~/.zshrc; python verification/check_tokenizer_tags.py"  (HF_TOKEN needed for gated gemma; never printed)
"""
import os
from transformers import AutoTokenizer
CASES = ["<think>", "</think>", "<python>", "</python>", "<output>", "</output>", "<answer>", "</answer>",
         "</python>\n", "\n</python>", "</python>\n<output>\n", "\n</output>\n", ")\n</python>", "</output><answer>1</answer>", "print(x)</python>"]
for name in ["Qwen/Qwen2.5-0.5B", "google/gemma-2-2b"]:
    try:
        tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print(f"== {name}: LOAD FAILED: {type(e).__name__}: {str(e)[:200]}")
        continue
    print(f"== {name} vocab={len(tok)} bos={tok.bos_token!r} eos={tok.eos_token!r} pad={tok.pad_token!r} padding_side={tok.padding_side}")
    for s in CASES:
        ids = tok.encode(s, add_special_tokens=False)
        print(f"  {s!r:40} -> {len(ids)} {tok.convert_ids_to_tokens(ids)}")
    # does "</python>" as a token-suffix survive concatenation? boundary merge test
    a = tok.encode("x = 1\nprint(x)\n</python>", add_special_tokens=False)
    b = tok.encode("\n<output>\n1\n</output>\n", add_special_tokens=False)
    joint = tok.encode("x = 1\nprint(x)\n</python>\n<output>\n1\n</output>\n", add_special_tokens=False)
    print("  concat(segment ids) == tokenize(joined text):", a + b == joint, len(a+b), len(joint))

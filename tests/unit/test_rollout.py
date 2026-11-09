import os

# keep HF cache inside the workspace so tokenizers can be fetched regardless of
# how pytest is invoked (avoids writing to ~/.cache/huggingface). Must run BEFORE
# importing transformers, which reads HF_HOME at import time.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf-cache"))

import subprocess  # noqa: E402

import pytest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from countdown_rl.rl import rollout  # noqa: E402

QWEN = "Qwen/Qwen2.5-0.5B"
GEMMA = "google/gemma-2-2b"


def _hf_token() -> str:
    try:
        return subprocess.run(
            ["security", "find-generic-password", "-s", "HF_TOKEN", "-w"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""


@pytest.fixture(scope="module")
def qwen_tok():
    return AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture(scope="module")
def gemma_tok():
    tok = _hf_token()
    if not tok:
        pytest.skip("HF_TOKEN unavailable from keychain")
    try:
        return AutoTokenizer.from_pretrained(GEMMA, token=tok)
    except Exception as e:
        pytest.skip(f"gemma-2-2b unavailable: {type(e).__name__}")


def test_assemble_masking(qwen_tok):
    prompt = qwen_tok.encode("PROMPT", add_special_tokens=True)
    gen = qwen_tok.encode("1+2", add_special_tokens=False)
    env = "\n<output>\nFalse 875\n</output>\n"
    env_ids = qwen_tok.encode(env, add_special_tokens=False)
    ids, mask = rollout.assemble(qwen_tok, prompt, [(False, gen), (True, env), (False, gen)])
    assert ids == prompt + gen + env_ids + gen
    assert mask == [0] * len(prompt) + [1] * len(gen) + [0] * len(env_ids) + [1] * len(gen)


def test_assemble_env_never_trained(qwen_tok, gemma_tok):
    # the injected <output> segment must be label -100 (loss_mask 0) on both tokenizers
    for tok in (qwen_tok, gemma_tok):
        env = "\n<output>\n42\n</output>\n"
        ids, mask = rollout.assemble(tok, [], [(True, env)])
        assert len(ids) == len(mask) and all(m == 0 for m in mask)


def test_find_stop_clean_and_merged(qwen_tok, gemma_tok):
    for tok in (qwen_tok, gemma_tok):
        assert rollout.find_stop("x</python>", "</python>")
        assert rollout.find_stop("x</python>\n", "</python>")  # merged >\n (qwen) or \n (gemma)
        assert rollout.find_stop("x</answer>", "</answer>")
        assert not rollout.find_stop("x</pytho", "</python>")
        assert not rollout.find_stop("x</python>more", "</python>")


def test_stop_detection_on_decoded_stream(qwen_tok, gemma_tok):
    # even though )</ and >< merge into single tokens, the decoded stream still
    # contains the exact stop string, so detection on decoded text is correct.
    for tok in (qwen_tok, gemma_tok):
        ids = tok.encode("print(x)</python>\n", add_special_tokens=False)
        text = tok.decode(ids, skip_special_tokens=False)
        assert "</python>" in text
        assert rollout.find_stop(text, "</python>")


def test_segment_vs_joint_pinned(qwen_tok):
    # Pin the audit M2 fact that motivates "never re-tokenize" on Qwen.
    a = qwen_tok.encode("print(x)</python>", add_special_tokens=False)
    b = qwen_tok.encode("\n<output>\n42\n</output>\n", add_special_tokens=False)
    joint = qwen_tok.encode("print(x)</python>\n<output>\n42\n</output>\n", add_special_tokens=False)
    assert a + b != joint


def test_extract_between():
    text = (
        "<think>\nx\n</think>\n<python>\ntarget=952\nprint(1)\n</python>\n"
        "<output>\n1\n</output>\n<answer>\n(100-3)*8\n</answer>"
    )
    assert rollout.extract_between(text, "<python>", "</python>") == "target=952\nprint(1)"
    assert rollout.extract_between(text, "<answer>", "</answer>") == "(100-3)*8"


def test_format_prompt_contains_tags():
    p = rollout.format_prompt([100, 75, 3, 8], 952)
    for t in ("<think>", "</think>", "<python>", "</python>", "<answer>", "</answer>"):
        assert t in p
    assert "952" in p and "100" in p


QWEN3 = "Qwen/Qwen3-4B-Base"
_TAG_TEXT = ("<think>\nTarget 15.\n</think>\n<python>\nres = 5 + 10\nprint(res == 15, res)\n</python>"
             "\n<output>\nTrue 15\n</output>\n<answer>\n5 + 10\n</answer>\n")


def test_encode_text_unchanged_for_tokenizers_without_tag_tokens(qwen_tok, gemma_tok):
    for tok in (qwen_tok, gemma_tok):
        for add in (False, True):
            assert rollout.encode_text(tok, _TAG_TEXT, add) == tok.encode(_TAG_TEXT, add_special_tokens=add)


def test_protocol_tags_never_become_reserved_tokens():
    """Qwen3 registers <think>/</think> as added tokens its base model never emits;
    protocol text must encode them as ordinary pieces (audit R4)."""
    try:
        tok = AutoTokenizer.from_pretrained(QWEN3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"{QWEN3} tokenizer unavailable: {type(e).__name__}")
    plain = tok.encode(_TAG_TEXT, add_special_tokens=False)
    assert tok.convert_tokens_to_ids("<think>") in plain  # the problem exists with plain encode
    ids = rollout.encode_text(tok, _TAG_TEXT)
    added_ids = set(tok.get_added_vocab().values())
    assert not (set(ids) & added_ids), tok.convert_ids_to_tokens([i for i in ids if i in added_ids])
    assert tok.decode(ids) == _TAG_TEXT

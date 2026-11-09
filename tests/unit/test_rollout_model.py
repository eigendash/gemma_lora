"""Model-dependent rollout tests (gated: RUN_MODEL_TESTS=1).

These load Qwen2.5-0.5B and run real forward passes, so they are excluded from
the fast suite. Run once:

    RUN_MODEL_TESTS=1 .venv/bin/python -m pytest tests/test_rollout_model.py -q
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_MODEL_TESTS"),
    reason="set RUN_MODEL_TESTS=1 to run model-dependent tests",
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown_rl.rl import rollout
from countdown_rl.sandbox import Sandbox

QWEN = "Qwen/Qwen2.5-0.5B"


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_tokenizer():
    tok = AutoTokenizer.from_pretrained(QWEN)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.to(_device())
    model.eval()
    return model, tok


def test_generation_logprob_matches_forward(model_tokenizer):
    """Generation-time log-prob of the first sampled token must equal a
    training-style forward pass at step 0 (ratio ~ 1), proving the log-prob
    bookkeeping does not re-tokenize or shift positions."""
    model, tok = model_tokenizer
    device = _device()
    prompt = rollout.format_prompt([100, 75, 3, 8], 952)
    prompt_ids = tok.encode(prompt, add_special_tokens=True)

    with torch.no_grad():
        out = model(input_ids=torch.tensor([prompt_ids], device=device), use_cache=True)
    logits = out.logits[:, -1, :]

    rng = torch.Generator(device=device).manual_seed(1)
    next_id, logp_gen = rollout._sample(logits, 1.0, rng)

    with torch.no_grad():
        out2 = model(input_ids=torch.tensor([prompt_ids + [next_id]], device=device))
    logp_train = torch.log_softmax(out2.logits[:, len(prompt_ids) - 1, :], dim=-1)[
        0, next_id
    ].item()

    assert abs(logp_gen - logp_train) < 1e-4, (logp_gen, logp_train)


def test_full_rollout_structure(model_tokenizer):
    model, tok = model_tokenizer
    device = _device()
    sb = Sandbox(n_workers=1, timeout=0.050)
    try:
        rng = torch.Generator(device=device).manual_seed(2)
        res = rollout.run_rollout(
            model,
            tok,
            rollout.format_prompt([25, 3, 1], 76),
            sb,
            max_new_tokens=64,
            temperature=1.0,
            rng=rng,
        )
        assert len(res.loss_mask) == len(res.completion_ids)
        assert len(res.old_logprobs) == len(res.completion_ids)
        assert set(res.loss_mask) <= {0, 1}
        assert res.code_rounds <= 2
        if res.code_rounds > 0:
            assert "<output>" in res.transcript
            assert 0 in res.loss_mask  # injected output tokens are env-masked
        if res.has_answer:
            assert res.answer_text
    finally:
        sb.close()

"""sdpa_gqa decode attention == stock SDPA (with repeat_kv), incl. padded and fully masked rows."""

from __future__ import annotations

import os
import types

import pytest
import torch

import attention_gqa


def test_decode_matches_stock_sdpa():
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(0)
    B, H, kvh, L, D = 3, 8, 2, 11, 16
    mod = types.SimpleNamespace(training=False, num_key_value_groups=H // kvh, is_causal=True)
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, kvh, L, D)
    v = torch.randn(B, kvh, L, D)
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool)
    mask[0, ..., :4] = False          # left padding
    mask[2] = False                   # idle slot: fully masked
    ours, _ = attention_gqa.gqa_decode_attention(mod, q, k, v, mask, scaling=D ** -0.5)
    ref, _ = sdpa_attention_forward(mod, q, k, v, mask, scaling=D ** -0.5)
    assert ours.shape == ref.shape == (B, 1, H, D)
    assert torch.allclose(ours[:2], ref[:2], atol=1e-5)
    assert torch.isfinite(ours[2]).all()


def test_prefill_and_training_delegate_to_sdpa(monkeypatch):
    import transformers.integrations.sdpa_attention as sa

    called = []
    monkeypatch.setattr(sa, "sdpa_attention_forward", lambda *a, **k: called.append(1) or ("x", None))
    mod = types.SimpleNamespace(training=False, num_key_value_groups=2, is_causal=True)
    attention_gqa.gqa_decode_attention(mod, torch.randn(1, 4, 5, 8), torch.randn(1, 2, 5, 8),
                                       torch.randn(1, 2, 5, 8), None)
    mod.training = True
    attention_gqa.gqa_decode_attention(mod, torch.randn(1, 4, 1, 8), torch.randn(1, 2, 5, 8),
                                       torch.randn(1, 2, 5, 8), None)
    assert called == [1, 1]


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="needs Qwen weights (RUN_MODEL_TESTS=1)")
@pytest.mark.parametrize("impl,bs", [("continuous", 0), ("continuous", 2), ("lockstep", 0)])
def test_real_model_sdpa_gqa_greedy_matches_sequential(impl, bs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import rollout
    from sandbox import Sandbox

    name = "Qwen/Qwen2.5-0.5B"
    tk = AutoTokenizer.from_pretrained(name)
    ref = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
    gqa = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
                                               attn_implementation=attention_gqa.register()).eval()
    prompts = [rollout.format_prompt(n, t) for n, t in
               [([25, 3, 1], 76), ([100, 75, 3, 5, 8], 875), ([6, 2], 12), ([9, 8, 7, 6, 5, 4], 300)]]
    with Sandbox(n_workers=2) as sb:
        seq = [rollout.run_rollout(ref, tk, p, sb, max_new_tokens=48, temperature=0.0, device="cpu") for p in prompts]
        bat = rollout.run_rollouts_batched(gqa, tk, prompts, sb, max_new_tokens=48, temperature=0.0, device="cpu",
                                           impl=impl, batch_size=bs)
    for a, b in zip(seq, bat):
        assert a.completion_ids == b.completion_ids
        assert max(abs(x - y) for x, y in zip(a.old_logprobs, b.old_logprobs)) < 1e-3

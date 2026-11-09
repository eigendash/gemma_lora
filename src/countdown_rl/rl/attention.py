"""Copy-free grouped-query attention for single-token decode steps.

With a padding mask, Hugging Face's SDPA path cannot use ``enable_gqa`` on CUDA/ROCm, so
``repeat_kv`` materialises the whole KV cache ``num_key_value_groups`` times per layer per
decode step (Qwen3-4B: 32 query / 8 KV heads -> 4x the cache copied every step). On soos
that copy took ~43% of the trainer's generation wall time (py-spy, 2026-11-04).

``sdpa_gqa`` is registered as an attention implementation. For decode (query length 1,
eval mode, no softcap / position bias) it groups the query heads per KV head and computes
exact attention with broadcasting matmuls (no KV copy). Everything else (prefill, the
training forward) delegates to the stock SDPA path, so training numerics are unchanged.
"""

from __future__ import annotations

import torch

NAME = "sdpa_gqa"
_registered = False


def gqa_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None,
                         **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    B, H, Tq, D = query.shape
    kvh = key.shape[1]
    if (Tq != 1 or module.training or H == kvh or kwargs.get("softcap") is not None
            or kwargs.get("position_bias") is not None or kwargs.get("cache") is not None):
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    rep = H // kvh
    L = key.shape[2]
    q = query.reshape(B, kvh, rep, D)                                  # heads h = g*rep + r
    scale = scaling if scaling is not None else D ** -0.5
    scores = torch.matmul(q, key.transpose(-1, -2)).float() * scale    # [B, kvh, rep, L]
    if attention_mask is not None:
        m = attention_mask[..., :L]                                    # [B, 1, 1, L]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0).to(value.dtype)          # fully masked (idle) rows -> 0
    out = torch.matmul(probs, value)                                   # [B, kvh, rep, D]
    return out.reshape(B, H, 1, D).transpose(1, 2).contiguous(), None  # [B, 1, H, D]


def register() -> str:
    """Register the implementation (idempotent) and return its name for ``attn_implementation``."""
    global _registered
    if not _registered:
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.masking_utils import sdpa_mask

        AttentionInterface.register(NAME, gqa_decode_attention)
        AttentionMaskInterface.register(NAME, sdpa_mask)
        _registered = True
    return NAME


def resolve(attn: str) -> str:
    """Map a CLI/config ``attn`` value to an ``attn_implementation`` (registers sdpa_gqa on demand)."""
    return register() if attn == NAME else attn

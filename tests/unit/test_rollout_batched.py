"""Batched rollout engine == sequential reference (same state machine), with a fake
batched LM that records, per row, exactly which real tokens and positions it was fed.

The fake LM marks each row's identity (which sequence) and liveness in its logits;
a scripted sampler pops that sequence's next scripted token, so batched and
sequential runs consume identical scripts.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

from countdown_rl.rl import rollout
from test_rollout_engine import RecordingSandbox, script_ids, tok  # noqa: F401  (tok = tokenizer fixture)

KEY = 1000  # logits index KEY + seq_idx marks "row belongs to sequence seq_idx and is live"
IMPLS = ["lockstep", "compact", "continuous"]


class FakePast:
    """Per-row records standing in for a KV cache; implements the engine's cache surgery."""

    def __init__(self, rows):
        self.rows = rows

    def select(self, keep):
        self.rows = [self.rows[k] for k in keep]

    def crop_left(self, c):
        pass

    def pad_left(self, c):
        pass

    def merge_rows(self, slots, new, pad):
        for s, row in zip(slots, new.rows):
            self.rows[s] = row


class FakeBatchLM(torch.nn.Module):
    def __init__(self, vocab: int, prompts: list[list[int]]):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab
        self.prompts = [tuple(p) for p in prompts]
        self.streams: list[dict] = []   # one per (round, row): {"seq", "ids", "pos"}

    def _seq_of(self, real_ids):
        for i, p in enumerate(self.prompts):
            if tuple(real_ids[:len(p)]) == p:
                return i
        raise AssertionError("row does not start with a known prompt")

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True, **kw):
        n, T = input_ids.shape
        if past_key_values is None:  # prefill of a new round
            rows = []
            for r in range(n):
                m = attention_mask[r].tolist()
                ids = [t for t, k in zip(input_ids[r].tolist(), m) if k]
                pos = [p for p, k in zip(position_ids[r].tolist(), m) if k]
                st = {"seq": self._seq_of(ids), "ids": ids, "pos": pos}
                self.streams.append(st)
                rows.append(st)
            past = FakePast(rows)
            live = [True] * n
        else:
            past = past_key_values
            live = []
            for r in range(n):
                alive = bool(attention_mask[r, -1])
                if alive:
                    past.rows[r]["ids"].append(int(input_ids[r, 0]))
                    past.rows[r]["pos"].append(int(position_ids[r, 0]))
                live.append(alive)
        logits = torch.zeros((n, T, self.vocab))
        for r in range(n):
            logits[r, -1, KEY + past.rows[r]["seq"] if live[r] else 0] = 50.0
        return types.SimpleNamespace(logits=logits, past_key_values=past)


def scripted_batch_sampler(queues):
    def _s(logits, temperature, rng):
        ids, lps = [], []
        for j in logits.argmax(-1).tolist():
            ids.append(queues[j - KEY].pop(0) if j >= KEY else 0)
            lps.append(-0.5)
        return ids, lps
    return _s


CODE = ["<think>\nplan\n</think>\n", "<python>\nprint(1+2)\n</python>"]
ANS = ["ok\n<answer>\n1+2+3\n</answer>"]
SCRIPTS = [
    CODE + ANS,                                                         # tool call then answer
    CODE + ["\n"] + ANS,                                                # whitespace right after injection
    ["<think>\nx\n</think>\n<python>\nprint(1)\n<output>\n1\n</output>"] + ANS,   # hallucination
    CODE + ["hm\n<python>\nprint(2)\n</python>", "hm\n<python>\nprint(3)\n</python>"] + ANS,  # round cap
    ["<think>\nshort\n</think>\n"],                                     # EOS
    ["<answer>\n1+2+3\n</answer>\n"],                                   # immediate answer
]
PROMPT_NUMS = [[1, 2, 3], [4, 5, 6, 7], [10, 20], [3, 3, 3, 3, 3], [8, 9, 10, 11, 12, 13], [2, 2]]


def _prompts():
    return [rollout.format_prompt(n, 6) for n in PROMPT_NUMS]


def _scripts(tok):
    return [script_ids(tok, s, eos=(i == 4)) for i, s in enumerate(SCRIPTS)]


def _run_batched(monkeypatch, tok, batch_size=0, impl="lockstep", **kw):
    prompts = _prompts()
    pids = [tok.encode(p, add_special_tokens=True) for p in prompts]
    queues = [list(s) for s in _scripts(tok)]
    monkeypatch.setattr(rollout, "_sample_batch", scripted_batch_sampler(queues))
    lm = FakeBatchLM(len(tok), pids)
    sb = RecordingSandbox()
    res = rollout.run_rollouts_batched(lm, tok, prompts, sb, max_new_tokens=400, device="cpu",
                                       batch_size=batch_size, impl=impl, **kw)
    return res, lm, sb


def _run_sequential(monkeypatch, tok, **kw):
    from test_rollout_engine import FakeLM

    out = []
    for prompt, script in zip(_prompts(), _scripts(tok)):
        q = list(script)
        monkeypatch.setattr(rollout, "_sample", lambda *a, q=q: (q.pop(0), -0.5))
        out.append(rollout.run_rollout(FakeLM(len(tok)), tok, prompt, RecordingSandbox(), max_new_tokens=400,
                                       device="cpu", **kw))
    return out


FIELDS = ["prompt_ids", "completion_ids", "loss_mask", "old_logprobs", "completion_text", "codes", "code_oks",
          "outputs", "generated_text", "code_rounds", "hallucinated_output", "has_answer", "answer_text",
          "truncated", "terminal_reason"]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch_size", [0, 2, 4])
@pytest.mark.parametrize("on_round_cap", ["truncate", "notice"])
def test_batched_equals_sequential(monkeypatch, tok, on_round_cap, batch_size, impl):
    seq = _run_sequential(monkeypatch, tok, on_round_cap=on_round_cap)
    bat, _, _ = _run_batched(monkeypatch, tok, on_round_cap=on_round_cap, batch_size=batch_size, impl=impl)
    for i, (a, b) in enumerate(zip(seq, bat)):
        for f in FIELDS:
            assert getattr(a, f) == getattr(b, f), (i, f, getattr(a, f), getattr(b, f))
    reasons = [r.terminal_reason for r in bat]
    assert reasons[:3] == ["answer", "answer", "hallucinated_output"] and reasons[4] == "eos"


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("batch_size", [0, 2])
def test_batched_fed_context_and_positions(monkeypatch, tok, impl, batch_size):
    res, lm, _ = _run_batched(monkeypatch, tok, impl=impl, batch_size=batch_size)
    for i, r in enumerate(res):
        last = [s for s in lm.streams if s["seq"] == i][-1]   # the final round of sequence i
        assert last["ids"] == r.all_ids[:-1], f"seq {i}: fed context != trained ids"
        assert last["pos"] == list(range(len(last["ids"]))), f"seq {i}: positions not contiguous"
        assert r.code_rounds == len(r.codes)


def test_batch_size_chunking_is_invariant(monkeypatch, tok):
    full, _, _ = _run_batched(monkeypatch, tok, batch_size=0)
    chunk, _, _ = _run_batched(monkeypatch, tok, batch_size=2)
    for a, b in zip(full, chunk):
        assert all(getattr(a, f) == getattr(b, f) for f in FIELDS)


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="needs Qwen weights (RUN_MODEL_TESTS=1)")
@pytest.mark.parametrize("impl,bs", [("lockstep", 0), ("compact", 0), ("continuous", 0), ("continuous", 2)])
def test_real_model_greedy_batched_matches_sequential(impl, bs):
    """Real Qwen2.5-0.5B, fp32, greedy: left-padded batched decoding must produce the
    same tokens as the per-sequence reference loop, and its rollout-time log-probs
    must match a training-style forward."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from countdown_rl.sandbox import Sandbox

    name = "Qwen/Qwen2.5-0.5B"
    tk = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="eager").eval()
    prompts = [rollout.format_prompt(n, t) for n, t in
               [([25, 3, 1], 76), ([100, 75, 3, 5, 8], 875), ([6, 2], 12), ([9, 8, 7, 6, 5, 4], 300)]]
    with Sandbox(n_workers=2) as sb:
        seq = [rollout.run_rollout(model, tk, p, sb, max_new_tokens=48, temperature=0.0, device="cpu") for p in prompts]
        bat = rollout.run_rollouts_batched(model, tk, prompts, sb, max_new_tokens=48, temperature=0.0, device="cpu",
                                           impl=impl, batch_size=bs)
    for a, b in zip(seq, bat):
        assert a.completion_ids == b.completion_ids
        assert a.loss_mask == b.loss_mask
        assert max(abs(x - y) for x, y in zip(a.old_logprobs, b.old_logprobs)) < 1e-3

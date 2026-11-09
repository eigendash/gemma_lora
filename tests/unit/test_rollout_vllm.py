"""vLLM backend == sequential reference, with a fake vLLM engine that replays scripted
tokens and applies vLLM's stop-string semantics (stop checked only in text generated in
the current request; stop string included in the output; may over-generate)."""

from __future__ import annotations

import types

import pytest

from countdown_rl.rl import rollout
from countdown_rl.rl import rollout_vllm
from test_rollout_batched import FIELDS, SCRIPTS, _prompts, _run_sequential, _scripts
from test_rollout_engine import RecordingSandbox, tok  # noqa: F401


class FakeLLM:
    def __init__(self, tok, prompts_ids, scripts, overrun=0):
        self.tok = tok
        self.prompts = [tuple(p) for p in prompts_ids]
        self.queues = [list(s) for s in scripts]
        self.overrun = overrun  # extra tokens returned after the stop (must be ignored)
        self.calls = []

    def _seq(self, ids):
        for i, p in enumerate(self.prompts):
            if tuple(ids[:len(p)]) == p:
                return i
        raise AssertionError("unknown prompt")

    def generate(self, prompts, params, use_tqdm=False, lora_request=None):
        outs = []
        for pr, sp in zip(prompts, params):
            ids = pr["prompt_token_ids"]
            q = self.queues[self._seq(ids)]
            self.calls.append(len(ids))
            got, text = [], ""
            while q and len(got) < sp.max_tokens:
                t = q.pop(0)
                got.append(t)
                text += self.tok.decode([t], skip_special_tokens=False)
                if t == self.tok.eos_token_id or any(s in text for s in rollout_vllm.STOPS):
                    for _ in range(self.overrun):   # simulate over-generation past the stop
                        if q:
                            got.append(q.pop(0))
                    # put over-generated tokens back: the engine "never produced" them for the
                    # next request, but the backend must not consume them in this one either
                    for t2 in reversed(got[len(got) - self.overrun:] if self.overrun else []):
                        q.insert(0, t2)
                    break
            outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(
                token_ids=got, logprobs=[{t: types.SimpleNamespace(logprob=-0.5)} for t in got])]))
        return outs


def _params(temperature, max_tokens):
    return types.SimpleNamespace(temperature=temperature, max_tokens=max_tokens)


@pytest.mark.parametrize("overrun", [0, 2])
@pytest.mark.parametrize("on_round_cap", ["truncate", "notice"])
def test_vllm_backend_equals_sequential(monkeypatch, tok, on_round_cap, overrun):
    seq = _run_sequential(monkeypatch, tok, on_round_cap=on_round_cap)
    prompts = _prompts()
    pids = [tok.encode(p, add_special_tokens=True) for p in prompts]
    llm = FakeLLM(tok, pids, _scripts(tok), overrun=overrun)
    res = rollout_vllm.run_rollouts_vllm(llm, tok, prompts, RecordingSandbox(), max_new_tokens=400,
                                         on_round_cap=on_round_cap, params_factory=_params)
    for i, (a, b) in enumerate(zip(seq, res)):
        for f in FIELDS:
            assert getattr(a, f) == getattr(b, f), (i, f, getattr(a, f), getattr(b, f))
    assert len(res) == len(SCRIPTS)

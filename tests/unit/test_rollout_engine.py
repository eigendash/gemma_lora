"""Rollout-engine behaviour tests with a FAKE language model (promoted from
verification/test_rollout_independent.py, audit round 2).

The fake LM records exactly which token ids it was fed (prompt seed, incremental
KV steps, and full rebuilds), so we can check — without trusting any project
helper — that:

  * the context the model conditioned on == the token ids we later train on,
  * masks cover exactly the injected text and nothing the model generated,
  * stop / round-cap / hallucination / EOS / max-length behave as documented,
  * eval's self-correction metric can see failing executions.

Sampling is scripted by monkeypatching ``rollout._sample`` (except the greedy
infinite-loop test, which uses the real ``_sample`` at temperature 0).

Needs the Qwen tokenizer (and gemma-2-2b's if HF_TOKEN / keychain is available).
"""

from __future__ import annotations

import os
import subprocess
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf-cache"))

import pytest
import torch

from countdown_rl.rl import rollout

TOKENIZERS = ["Qwen/Qwen2.5-0.5B", "google/gemma-2-2b"]


@pytest.fixture(scope="module", params=TOKENIZERS)
def tok(request):
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = subprocess.run(["security", "find-generic-password", "-s", "HF_TOKEN", "-w"],
                                   capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            token = None
    try:
        return AutoTokenizer.from_pretrained(request.param, token=token)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer {request.param} unavailable: {e}")


class FakePast:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeLM(torch.nn.Module):
    """Records fed context; logits argmax chosen by ``policy(fed_ids)`` (or 0)."""

    def __init__(self, vocab: int, policy=None, max_calls: int = 100_000):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab
        self.policy = policy
        self.calls = 0
        self.max_calls = max_calls
        self.last_fed: list[int] = []
        self.rebuilds = 0

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kw):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError(f"FakeLM: exceeded {self.max_calls} forward calls (non-terminating loop)")
        new = input_ids[0].tolist()
        if past_key_values is None:
            if self.last_fed:
                self.rebuilds += 1
            fed = new
        else:
            fed = past_key_values.ids + new
        self.last_fed = fed
        logits = torch.zeros((1, len(new), self.vocab))
        if self.policy is not None:
            logits[0, -1, self.policy(fed)] = 50.0
        return types.SimpleNamespace(logits=logits, past_key_values=FakePast(fed))


class RecordingSandbox:
    def __init__(self, real=None):
        self.codes = []
        self.real = real

    def run(self, code):
        self.codes.append(code)
        if self.real is not None:
            return self.real.run(code)
        return True, "3"


def script_ids(tok, chunks, eos=False):
    ids = []
    for c in chunks:
        ids += tok.encode(c, add_special_tokens=False)
    if eos:
        ids.append(tok.eos_token_id)
    return ids


def run_scripted(monkeypatch, tok, chunks, *, sandbox=None, eos=False, max_new_tokens=400, max_rounds=2):
    ids = script_ids(tok, chunks, eos=eos)
    queue = list(ids)

    def fake_sample(logits, temperature, rng):
        if not queue:
            raise RuntimeError("script exhausted")
        return queue.pop(0), -0.5

    monkeypatch.setattr(rollout, "_sample", fake_sample)
    lm = FakeLM(len(tok))
    sb = sandbox or RecordingSandbox()
    res = rollout.run_rollout(lm, tok, rollout.format_prompt([1, 2, 3], 6), sb,
                              max_rounds=max_rounds, max_new_tokens=max_new_tokens,
                              temperature=1.0, device="cpu")
    return res, lm, sb, queue


CODE_ROUND = ["<think>\nplan\n</think>\n", "<python>\nprint(1+2)\n</python>"]
ANSWER = ["ok\n<answer>\n1+2+3\n</answer>"]   # no leading whitespace: isolates defects from the re-execution bug (test 3)


# ---------------------------------------------------------------------------
# 1. The model must have been fed exactly the ids we train on.
# ---------------------------------------------------------------------------
def test_fed_context_equals_training_ids(monkeypatch, tok):
    res, lm, sb, _ = run_scripted(monkeypatch, tok, CODE_ROUND + ANSWER)
    assert res.code_rounds == 1 and res.has_answer
    # the last sampled token (closing </answer>) is never fed: training sees all_ids,
    # generation conditioned on all_ids[:-1]
    expected = res.all_ids[:-1]
    if lm.last_fed != expected:
        # locate first divergence for the report
        k = next(i for i, (a, b) in enumerate(zip(lm.last_fed + [None] * len(expected), expected)) if a != b)
        missing = tok.convert_ids_to_tokens([expected[k]])
        pytest.fail(f"generation context != training ids: first divergence at position {k}; "
                    f"training has token {missing} that the model was never fed "
                    f"(fed {len(lm.last_fed)} vs expected {len(expected)} ids)")


def test_no_context_divergence_without_tools(monkeypatch, tok):
    """Control: with no code round the contexts must match exactly."""
    res, lm, _, _ = run_scripted(monkeypatch, tok, ["<think>\nplan\n</think>\n"] + ANSWER)
    assert res.code_rounds == 0
    assert lm.last_fed == res.all_ids[:-1]


# ---------------------------------------------------------------------------
# 2. Masks: mask==1 tokens decode to exactly what the model generated, mask==0
#    completion tokens decode to exactly the injected text (real sandbox output).
# ---------------------------------------------------------------------------
def test_mask_partition_exact(monkeypatch, tok):
    from countdown_rl.sandbox import Sandbox

    with Sandbox(n_workers=1) as real:
        res, lm, sb, _ = run_scripted(monkeypatch, tok, CODE_ROUND + ANSWER, sandbox=RecordingSandbox(real))
    gen = [t for t, m in zip(res.completion_ids, res.loss_mask) if m == 1]
    env = [t for t, m in zip(res.completion_ids, res.loss_mask) if m == 0]
    assert gen == script_ids(tok, CODE_ROUND + ANSWER), "trainable tokens != model-generated tokens"
    assert tok.decode(env) == "\n<output>\n3\n</output>\n", repr(tok.decode(env))
    # env tokens are contiguous and immediately follow the </python> stop
    first0 = res.loss_mask.index(0)
    assert all(m == 0 for m in res.loss_mask[first0:first0 + len(env)])
    assert all(lp == 0.0 for lp, m in zip(res.old_logprobs, res.loss_mask) if m == 0)
    # first model token after the injection is trainable
    assert res.loss_mask[first0 + len(env)] == 1


# ---------------------------------------------------------------------------
# 3. A whitespace token after an executed </python> must not re-execute it.
# ---------------------------------------------------------------------------
def test_whitespace_after_injection_does_not_reexecute(monkeypatch, tok):
    res, lm, sb, _ = run_scripted(monkeypatch, tok, CODE_ROUND + ["\n"] + ANSWER)  # a lone newline token
    assert len(sb.codes) == 1, f"same code executed {len(sb.codes)}x: {sb.codes}"
    assert res.code_rounds == 1


# ---------------------------------------------------------------------------
# 4. Round cap: a third </python> is never executed.
# ---------------------------------------------------------------------------
def test_third_python_not_executed(monkeypatch, tok):
    chunks = CODE_ROUND + ["hm\n<python>\nprint(2)\n</python>", "hm\n<python>\nprint(3)\n</python>"] + ANSWER
    res, lm, sb, _ = run_scripted(monkeypatch, tok, chunks)
    assert [c.strip() for c in sb.codes] == ["print(1+2)", "print(2)"]
    assert res.code_rounds == 2
    assert res.truncated and res.terminal_reason == "max_rounds"


# ---------------------------------------------------------------------------
# 5. Hallucinated <output> (documented contract, rollout.py docstring): the
#    rollout ends, is flagged, the hallucinated tokens are KEPT as trainable
#    (so the penalty reaches them), and any earlier real injection is intact.
# ---------------------------------------------------------------------------
def test_hallucination_after_code_round_keeps_alignment(monkeypatch, tok):
    chunks = CODE_ROUND + ["next\n<output>\n99\n</output>"] + ANSWER
    res, lm, sb, _ = run_scripted(monkeypatch, tok, chunks)
    assert res.hallucinated_output and res.terminal_reason == "hallucinated_output"
    env_text = tok.decode([t for t, m in zip(res.completion_ids, res.loss_mask) if m == 0])
    assert env_text == "\n<output>\n3\n</output>\n", f"real injected output corrupted: {env_text!r}"
    gen_text = tok.decode([t for t, m in zip(res.completion_ids, res.loss_mask) if m == 1])
    assert gen_text.rstrip().endswith("<output>") and "<answer>" not in gen_text
    assert lm.last_fed == res.all_ids[:-1]
    assert res.outputs == ["3"] and not res.has_answer


def test_hallucination_first_round_doc_behaviour(monkeypatch, tok):
    chunks = ["<think>\nx\n</think>\n<python>\nprint(1+2)\n<output>\n3\n</output>"] + ANSWER
    res, lm, sb, q = run_scripted(monkeypatch, tok, chunks)
    assert res.hallucinated_output and res.truncated
    assert sb.codes == [] and res.outputs == []          # nothing executed, nothing injected
    assert all(m == 1 for m in res.loss_mask)            # hallucinated tokens are trainable
    assert res.generated_text.rstrip().endswith("<output>")


def test_round_cap_notice_mode(monkeypatch, tok):
    chunks = CODE_ROUND + ["hm\n<python>\nprint(2)\n</python>", "hm\n<python>\nprint(3)\n</python>"] + ANSWER
    ids = script_ids(tok, chunks)
    queue = list(ids)
    monkeypatch.setattr(rollout, "_sample", lambda *a: (queue.pop(0), -0.5))
    sb = RecordingSandbox()
    res = rollout.run_rollout(FakeLM(len(tok)), tok, rollout.format_prompt([1, 2, 3], 6), sb,
                              max_rounds=2, max_new_tokens=400, device="cpu", on_round_cap="notice")
    assert len(sb.codes) == 2 and res.code_rounds == 2
    assert rollout.ROUND_CAP_NOTICE in res.completion_text
    assert res.has_answer and res.answer_text == "1+2+3"


# ---------------------------------------------------------------------------
# 6. Greedy decoding of a model that wants to write <output> must terminate.
# ---------------------------------------------------------------------------
def test_greedy_hallucination_terminates(tok):
    seq = tok.encode("<output>", add_special_tokens=False)
    prompt_len = len(tok.encode(rollout.format_prompt([1, 2, 3], 6), add_special_tokens=True))

    def policy(fed):
        # greedy model that always continues spelling "<output>"
        gen = fed[prompt_len:]
        for j in range(len(seq) - 1, 0, -1):
            if len(gen) >= j and gen[-j:] == seq[:j]:
                return seq[j]
        return seq[0]

    lm = FakeLM(len(tok), policy=policy, max_calls=5000)
    max_new = 64
    try:
        res = rollout.run_rollout(lm, tok, rollout.format_prompt([1, 2, 3], 6), RecordingSandbox(),
                                  max_new_tokens=max_new, temperature=0.0, device="cpu")
    except RuntimeError as e:
        pytest.fail(f"non-terminating at temperature 0 ({lm.rebuilds} rollbacks): {e}")
    assert len(res.completion_ids) <= max_new


# ---------------------------------------------------------------------------
# 7. EOS / max length / answer stop with merged boundary tokens
# ---------------------------------------------------------------------------
def test_eos_terminates_without_answer(monkeypatch, tok):
    res, *_ = run_scripted(monkeypatch, tok, ["<think>\nx\n</think>\n"], eos=True)
    assert res.terminal_reason == "eos" and res.truncated and not res.has_answer
    assert res.loss_mask[-1] == 1


def test_max_length_mid_tag(monkeypatch, tok):
    res, *_ = run_scripted(monkeypatch, tok, ["<think>\nx\n</think>\n<pyth"], max_new_tokens=6)
    assert res.terminal_reason == "max_length" and len(res.completion_ids) == 6


def test_answer_stop_with_merged_newline(monkeypatch, tok):
    res, *_ = run_scripted(monkeypatch, tok, ["<answer>\n1+2+3\n</answer>\n"])
    assert res.has_answer and res.answer_text == "1+2+3"


# ---------------------------------------------------------------------------
# 8. completion_text contract + eval's self-correction metric
# ---------------------------------------------------------------------------
def test_completion_text_contains_injected_output(monkeypatch, tok):
    """RolloutResult.completion_text is documented as 'gen + injected'."""
    res, *_ = run_scripted(monkeypatch, tok, CODE_ROUND + ANSWER)
    assert "<output>" in res.completion_text, "completion_text omits the injected <output>"


def test_eval_detects_failing_execution(monkeypatch, tok):
    from countdown_rl.evaluation import evaluate as ev
    from countdown_rl.sandbox import Sandbox

    chunks = ["<think>\nx\n</think>\n", "<python>\nprint(1+2 == 6, 1+2)\n</python>"] + ANSWER
    with Sandbox(n_workers=1) as real:
        res, *_ = run_scripted(monkeypatch, tok, chunks, sandbox=RecordingSandbox(real))
    assert "False 3" in res.transcript  # the model really saw a failing execution
    assert ev.has_failing_execution(res), "eval.has_failing_execution misses a real 'False' output"

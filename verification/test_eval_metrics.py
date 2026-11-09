"""Hand-computed eval metrics on a tiny synthetic set vs eval.compute_metrics.

Run: python -m pytest -q -s verification/test_eval_metrics.py
"""

from __future__ import annotations

import pytest

from countdown_rl.evaluation import evaluate as ev
from countdown_rl.rl.rewards import RewardResult
from countdown_rl.rl.rollout import RolloutResult


def row(tier, correct, text, n_gen):
    rr = RolloutResult(prompt_ids=[0], completion_ids=[1] * n_gen, loss_mask=[1] * n_gen,
                       old_logprobs=[0.0] * n_gen, transcript=text, completion_text=text)
    rw = RewardResult(total=float(correct), correct=float(correct), exec=0.0, format=0.0,
                      length_penalty=0.0, hallucinated=False)
    return {"nums": [1], "target": 1, "tier": tier, "rollout": rr, "reward": rw}


GOOD = "<think>\nx\n</think>\n<python>\nprint(1)\n</python>\n<answer>\n1\n</answer>"
BAD = "<answer>\n1\n</answer>\n<answer>\n2\n</answer>"


def test_hand_computed_metrics():
    rows = [row("easy", 1, GOOD, 10), row("easy", 0, GOOD, 20),
            row("hard", 1, BAD, 30), row("hard", 0, BAD, 40), row("hard", 0, GOOD, 100)]
    m = ev.compute_metrics(rows)
    print("\n", m)
    # by hand
    assert m["n"] == 5
    assert m["pass@1"] == pytest.approx(2 / 5)
    assert m["format_compliance"] == pytest.approx(3 / 5)
    assert m["length_mean"] == pytest.approx((10 + 20 + 30 + 40 + 100) / 5)
    assert m["length_max"] == 100
    assert m["per_tier_pass@1"] == {"easy": pytest.approx(0.5), "hard": pytest.approx(1 / 3)}


def test_self_correction_on_transcripts_with_failing_output():
    """A rollout whose real injected output is a failing check ('False ...') must be
    counted. Rollout contract: completion_text = generated + injected text, and
    RolloutResult.outputs holds the real sandbox stdout."""
    gen_only = "<think>\nx\n</think>\n<python>\nprint(2 == 1, 2)\n</python>\nwrong, retry\n<answer>\n1\n</answer>"
    full = gen_only.replace("</python>\n", "</python>\n<output>\nFalse 2\n</output>\n", 1)
    rr = RolloutResult(prompt_ids=[0], completion_ids=[1], loss_mask=[1], old_logprobs=[0.0],
                       transcript="PROMPT" + full, completion_text=full, outputs=["False 2"],
                       generated_text=gen_only)
    rw = RewardResult(total=1.0, correct=1.0, exec=0.0, format=0.0, length_penalty=0.0, hallucinated=False)
    m = ev.compute_metrics([{"nums": [1], "target": 1, "tier": "easy", "rollout": rr, "reward": rw}])
    print("\n", m)
    assert m["n_failing"] == 1, "failing execution in the transcript is invisible to the metric"


def test_backtrack_cue_false_positive():
    """'correct' is a cue, so plain non-backtracking text counts as backtracking."""
    assert not ev.has_backtracking_text(RolloutResult(
        prompt_ids=[], completion_ids=[], loss_mask=[], old_logprobs=[], transcript="",
        completion_text="<think>\nLet me compute the correct value directly.\n</think>")), \
        "non-backtracking text classified as backtracking"

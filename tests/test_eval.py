import eval as E
from rewards import RewardResult
from rollout import RolloutResult


def _rr(completion_text, answer="", has_answer=False, n_tokens=10):
    # real injected outputs are recorded by the rollout engine; mirror them here
    outputs = E.output_contents(completion_text)
    return RolloutResult(prompt_ids=[], completion_ids=[0] * n_tokens, loss_mask=[1] * n_tokens,
                         old_logprobs=[0.0] * n_tokens, transcript=completion_text,
                         completion_text=completion_text, codes=[], code_oks=[], outputs=outputs,
                         has_answer=has_answer, answer_text=answer)


def _row(tier, correct, completion_text, answer="", has_answer=False, n_tokens=10):
    return {"nums": [1, 2, 3], "target": 6, "tier": tier, "rollout": _rr(completion_text, answer, has_answer, n_tokens),
            "reward": RewardResult(total=correct, correct=correct, exec=0, format=0, length_penalty=0, hallucinated=False)}


def test_has_failing_execution():
    assert E.has_failing_execution(_rr("<output>\nFalse 5\n</output>"))
    assert not E.has_failing_execution(_rr("<output>\nTrue 6\n</output>"))
    assert not E.has_failing_execution(_rr("no output here"))


def test_has_backtracking_text():
    # cues only count AFTER a failing output
    assert E.has_backtracking_text(_rr("<output>\nFalse 5\n</output>\nthat is wrong, retry"))
    assert not E.has_backtracking_text(_rr("that is wrong <output>\nTrue 6\n</output>"))
    assert not E.has_backtracking_text(_rr("<output>\nFalse 5\n</output>\nlet me compute the correct value"))


def test_stratified_subset_spans_tiers():
    probs = [{"tier": t, "i": i} for t in ("easy", "medium", "hard") for i in range(5)]
    sub = E.stratified_subset(probs, 6)
    assert [p["tier"] for p in sub] == ["easy", "medium", "hard"] * 2


def test_output_contents():
    assert E.output_contents("<output>\nFalse 5\n</output>x<output>\nTrue 6\n</output>") == ["False 5", "True 6"]


def test_compute_metrics_hand_computed():
    rows = [
        _row("easy", 1.0, "<think>x</think><python>y</python>\n<output>\nFalse 5\n</output>\n<answer>1+2+3</answer>", "1+2+3", True, 10),
        _row("easy", 1.0, "<think>x</think><answer>2*3</answer>", "2*3", True, 5),
        _row("hard", 0.0, "<answer>1+2</answer>", "1+2", True, 3),
        _row("hard", 0.0, "garbage", "", False, 2),
    ]
    m = E.compute_metrics(rows)
    assert m["n"] == 4
    assert abs(m["pass@1"] - 0.5) < 1e-9
    assert abs(m["per_tier_pass@1"]["easy"] - 1.0) < 1e-9
    assert abs(m["per_tier_pass@1"]["hard"] - 0.0) < 1e-9
    assert m["n_failing"] == 1
    assert abs(m["self_correction_correct"] - 1.0) < 1e-9  # failing rollout answered correctly
    assert abs(m["self_correction_backtrack"] - 0.0) < 1e-9  # no backtracking cue in it
    assert m["length_mean"] == 5.0 and m["length_p50"] == 5 and m["length_max"] == 10
    # 2 of 4: row 3 has no <think> (required since the audit-R2 format fix), row 4 is garbage
    assert abs(m["format_compliance"] - 0.5) < 1e-9

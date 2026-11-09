"""Independent reward-hack suite for rewards.compute_reward.

Rollouts are constructed by hand (no model, no rollout engine) so reward logic
is tested in isolation. Baselines:
  HONEST_CORRECT   think -> relevant python check -> correct answer
  HONEST_INCORRECT think -> relevant python check -> wrong but well-formed answer

Run: uv run pytest -q -s tests/independent/test_reward_hacks.py
"""

from __future__ import annotations

import pytest

from countdown_rl.rl.rewards import RewardConfig, compute_reward
from countdown_rl.rl.rollout import RolloutResult

NUMS, TARGET = [25, 3, 1, 7], 76      # 1 is an input number (very common: pool has 1..10 twice)
NUMS_NO1, TARGET_NO1 = [25, 3, 4, 7], 79


def rr(text, *, codes=(), oks=(), answer=None, halluc=False, n_gen=80, outputs=None):
    if outputs is None:  # what the sandbox would print: non-empty iff the code prints
        outputs = [("out" if ("print(" in c and ok) else "") for c, ok in zip(codes, oks)]
    return RolloutResult(
        prompt_ids=[0] * 10, completion_ids=[1] * n_gen, loss_mask=[1] * n_gen,
        old_logprobs=[0.0] * n_gen, transcript=text, completion_text=text,
        codes=list(codes), code_oks=list(oks), code_rounds=len(codes), outputs=list(outputs),
        hallucinated_output=halluc, has_answer=answer is not None, answer_text=answer or "")


GOOD_CODE = "res = 25*3+1\nprint(res == 76, res)"


def honest(answer, nums=NUMS):
    t = f"<think>\nTry 25*3+1.\n</think>\n<python>\n{GOOD_CODE}\n</python>\nok\n<answer>\n{answer}\n</answer>"
    return rr(t, codes=[GOOD_CODE], oks=[True], answer=answer)


def score(r, nums=NUMS, target=TARGET):
    return compute_reward(r, nums, target, RewardConfig())


HC = score(honest("25*3+1")).total
HI = score(honest("25*3+7")).total


def _wf(code, answer):  # well-formed shell around arbitrary code
    t = f"<think>\nx\n</think>\n<python>\n{code}\n</python>\n<answer>\n{answer}\n</answer>"
    return rr(t, codes=[code], oks=[True], answer=answer)


HACKS = {
    # name: (rollout, nums, target, must_be_below)
    "print(1) [1 in nums]": (_wf("print(1)", "25*3+7"), NUMS, TARGET, "HI"),
    "print(1) [1 not in nums]": (_wf("print(1)", "25*3+7"), NUMS_NO1, TARGET_NO1, "HI"),
    "print(target) literal": (_wf("print(76)", "25*3+7"), NUMS, TARGET, "HI"),
    "print(True)": (_wf("print(True)", "25*3+7"), NUMS, TARGET, "HI"),
    "dummy ref + print(target)": (_wf("x = 25\nprint(76)", "25*3+7"), NUMS, TARGET, "HI"),
    "references nums, no check/print": (_wf("x = 25 + 3", "25*3+7"), NUMS, TARGET, "HI"),
    "hallucinated <output>": (rr("<think>\nx\n</think>\n<answer>\n25*3+7\n</answer>", answer="25*3+7", halluc=True), NUMS, TARGET, "HI"),
    "answer reuses a number": (honest("25*3+25+1-25"), NUMS, TARGET, "HC"),
    "answer = target literal": (honest("76"), NUMS, TARGET, "HC"),
    "answer with '='": (honest("25*3+1=76"), NUMS, TARGET, "HC"),
    "answer unicode ×": (honest("25×3+1"), NUMS, TARGET, "HC"),
    "skip think+python, correct": (rr("<answer>\n25*3+1\n</answer>", answer="25*3+1"), NUMS, TARGET, "HC"),
    "skip think+python, wrong": (rr("<answer>\n25*3+7\n</answer>", answer="25*3+7"), NUMS, TARGET, "HI"),
    "duplicate answers": (rr("<think>\nx\n</think>\n<answer>\n25*3+1\n</answer>\n<answer>\n25*3+1\n</answer>", answer="25*3+1"), NUMS, TARGET, "HC"),
    "broken tag order, correct": (rr("<answer>\n25*3+1\n</answer>\n<think>\nx\n</think>", answer="25*3+1"), NUMS, TARGET, "HC"),
    "answer inside <think>": (rr("<think>\n<answer>\n25*3+1\n</answer>", answer="25*3+1"), NUMS, TARGET, "HC"),
    "code crashes (div0)": (rr(f"<think>\nx\n</think>\n<python>\nprint(25/0)\n</python>\n<answer>\n25*3+7\n</answer>", codes=["print(25/0)"], oks=[False], answer="25*3+7"), NUMS, TARGET, "HI"),
    "no answer (truncated)": (rr(f"<think>\nx\n</think>\n<python>\n{GOOD_CODE}\n</python>", codes=[GOOD_CODE], oks=[True]), NUMS, TARGET, "HI"),
    "padding 2000 tokens": (rr(f"<think>\n{'pad ' * 500}\n</think>\n<python>\n{GOOD_CODE}\n</python>\n<answer>\n25*3+7\n</answer>", codes=[GOOD_CODE], oks=[True], answer="25*3+7", n_gen=2000), NUMS, TARGET, "HI"),
    "empty answer": (rr("<think>\nx\n</think>\n<answer>\n\n</answer>", answer=""), NUMS, TARGET, "HI"),
}


def test_baselines():
    print(f"\nHONEST_CORRECT={HC:.3f}  HONEST_INCORRECT_WELLFORMED={HI:.3f}")
    assert HC > HI > 0


@pytest.mark.parametrize("name", list(HACKS))
def test_hack_scores_below_baseline(name):
    r, nums, target, below = HACKS[name]
    res = score(r, nums, target)
    ref = {"HC": HC, "HI": HI}[below]
    print(f"\n{name:34s} total={res.total:+.3f} (c={res.correct} e={res.exec} f={res.format} "
          f"len={res.length_penalty:.3f})  must be < {below}={ref:.3f}")
    assert res.total < ref, f"{name}: {res.total} not < {below} {ref}"


def test_reward_bounded_and_length_penalty_monotonic():
    cfg = RewardConfig()
    prev = None
    for n in (10, 255, 256, 257, 300, 512, 2000):
        res = score(rr("<answer>\n25*3+1\n</answer>", answer="25*3+1", n_gen=n))
        assert res.length_penalty == pytest.approx(cfg.length_lambda * max(0, n - cfg.length_threshold))
        if prev is not None:
            assert res.length_penalty >= prev
        prev = res.length_penalty
        assert res.total <= cfg.correct + cfg.exec + cfg.format


def test_exec_requires_clean_run_and_reference():
    base = dict(answer="25*3+7")
    assert score(_wf(GOOD_CODE, "25*3+7")).exec > 0
    assert score(rr("<python>\nprint(9)\n</python>", codes=["print(9)"], oks=[True], **base),
                 NUMS_NO1, TARGET_NO1).exec == 0
    assert score(rr("<python>\nprint(25)\n</python>", codes=[GOOD_CODE], oks=[False], **base)).exec == 0

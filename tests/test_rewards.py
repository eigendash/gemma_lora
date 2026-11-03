from rewards import RewardConfig, check_format, compute_reward, references_numbers
from rollout import RolloutResult

NUMS = [100, 75, 3, 5]
TARGET = 875
HONEST_ANSWER = "(100 + 75) * 5"  # == 875, uses 100, 75, 5
HONEST_CODE = "res = (100 + 75) * 5\nprint(res)"
HONEST_COMPLETION = (
    "<think>\n(100+75)*5=875\n</think>\n"
    "<python>\nres = (100 + 75) * 5\nprint(res)\n</python>\n"
    "<output>\n875\n</output>\n"
    "<answer>\n(100 + 75) * 5\n</answer>"
)


def make_rollout(**overrides) -> RolloutResult:
    base = dict(
        prompt_ids=[],
        completion_ids=[0] * 50,
        loss_mask=[1] * 50,
        old_logprobs=[-1.0] * 50,
        transcript=HONEST_COMPLETION,
        completion_text=HONEST_COMPLETION,
        codes=[HONEST_CODE],
        code_oks=[True],
        code_rounds=1,
        hallucinated_output=False,
        has_answer=True,
        answer_text=HONEST_ANSWER,
        truncated=False,
        terminal_reason="answer",
    )
    base.update(overrides)
    return RolloutResult(**base)


def reward(rollout) -> float:
    return compute_reward(rollout, NUMS, TARGET).total


def test_honest_baseline_components():
    r = compute_reward(make_rollout(), NUMS, TARGET)
    assert r.correct == 1.0
    assert r.exec == 0.1
    assert r.format == 0.1
    assert r.length_penalty == 0.0
    assert abs(r.total - 1.2) < 1e-9


def test_references_numbers():
    assert references_numbers("res = (100 + 75) * 5", NUMS)
    assert not references_numbers("print(1)", NUMS)
    assert not references_numbers("print(875)", NUMS)  # target, not an input
    assert not references_numbers("print(999)", NUMS)


def test_check_format():
    assert check_format(HONEST_COMPLETION)
    assert not check_format("no tags here")
    assert not check_format("<answer>x</answer><answer>y</answer>")  # two answers
    # answer before python
    assert not check_format("<answer>x</answer><python>y</python>")


def test_no_answer_zero_correct():
    r = make_rollout(has_answer=False, answer_text="")
    res = compute_reward(r, NUMS, TARGET)
    assert res.correct == 0.0


def test_print_1_hack_scores_below_honest():
    honest = reward(make_rollout())
    hack = reward(make_rollout(codes=["print(1)"], code_oks=[True]))
    assert hack < honest


def test_print_target_hack_scores_below_honest():
    honest = reward(make_rollout())
    hack = reward(make_rollout(codes=["print(875)"], code_oks=[True]))
    assert hack < honest


def test_hallucinated_output_scores_below_honest():
    honest = reward(make_rollout())
    hack = reward(make_rollout(hallucinated_output=True))
    assert hack < honest


def test_reused_numbers_scores_below_honest():
    honest = reward(make_rollout())
    # answer reuses 5 and is wrong (880 != 875): correctness must fail
    hack = reward(make_rollout(answer_text="(100 + 75) * 5 + 5"))
    assert hack < honest


def test_answer_wrong_place_scores_below_honest():
    honest = reward(make_rollout())
    bad_completion = (
        "<answer>\n(100 + 75) * 5\n</answer>\n"
        "<think>x</think>\n<python>\nres = (100 + 75) * 5\nprint(res)\n</python>"
    )
    hack = reward(make_rollout(completion_text=bad_completion))
    assert hack < honest


def test_length_penalty_beyond_threshold():
    short = make_rollout()  # 50 generated tokens
    long = make_rollout(completion_ids=[0] * 400, loss_mask=[1] * 400, old_logprobs=[-1.0] * 400)
    r_short = compute_reward(short, NUMS, TARGET)
    r_long = compute_reward(long, NUMS, TARGET)
    assert r_long.length_penalty > 0
    assert r_long.total < r_short.total


def test_config_weights_apply():
    cfg = RewardConfig(correct=2.0, exec=0.2, format=0.0)
    r = compute_reward(make_rollout(), NUMS, TARGET, cfg)
    assert r.correct == 2.0 and r.exec == 0.2 and r.format == 0.0

from countdown_rl.rl.rewards import RewardConfig, check_format, compute_reward, references_numbers
from countdown_rl.rl.rollout import RolloutResult

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
        outputs=["875"],
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


def test_exec_requires_arithmetic_over_inputs_and_output():
    from countdown_rl.rl.rewards import computes_with_inputs

    assert computes_with_inputs("res = (100 + 75) * 5\nprint(res)", NUMS)
    assert not computes_with_inputs("print(1)", NUMS)
    assert not computes_with_inputs("x = 100\nprint(875)", NUMS)          # dummy reference
    assert not computes_with_inputs("print(100 + 100)", NUMS)              # 100 used twice, given once
    silent = make_rollout(codes=["x = 100 + 75"], code_oks=[True], outputs=[""])
    assert compute_reward(silent, NUMS, TARGET).exec == 0.0


def test_format_requires_think_and_wellformed_answer():
    junk = make_rollout(answer_text="False 875",
                        completion_text=HONEST_COMPLETION.replace("(100 + 75) * 5\n</answer>", "False 875\n</answer>"))
    assert compute_reward(junk, NUMS, TARGET).format == 0.0
    no_think = make_rollout(completion_text="<answer>\n(100 + 75) * 5\n</answer>")
    assert compute_reward(no_think, NUMS, TARGET).format == 0.0
    wrong_but_wellformed = make_rollout(answer_text="100 + 75",
                                        completion_text=HONEST_COMPLETION.replace("(100 + 75) * 5\n</answer>", "100 + 75\n</answer>"))
    assert compute_reward(wrong_but_wellformed, NUMS, TARGET).format == 0.1


# --- contradicted answers: "check -> False -> submit it anyway" earns no shaping ---
WRONG = "(100 + 75) * 3"  # 525
WRONG_COMPLETION = HONEST_COMPLETION.replace("(100 + 75) * 5\n</answer>", WRONG + "\n</answer>")


def test_contradicted_by_printed_value_gets_no_shaping():
    r = make_rollout(codes=["res = (100 + 75) * 3\nprint(res == 875, res)"], outputs=["False 525"],
                     answer_text=WRONG, completion_text=WRONG_COMPLETION)
    x = compute_reward(r, NUMS, TARGET)
    assert x.contradicted and x.exec == 0.0 and x.format == 0.0 and x.total <= 0.0


def test_contradicted_by_expression_in_code_gets_no_shaping():
    r = make_rollout(codes=["print((100 + 75) * 3 == 875)"], outputs=["False"],
                     answer_text=WRONG, completion_text=WRONG_COMPLETION)
    assert compute_reward(r, NUMS, TARGET).contradicted


def test_wrong_but_unchecked_or_differently_checked_keeps_shaping():
    # checked a different candidate (False), then answered a new, also wrong, one: not contradicted
    r = make_rollout(codes=["res = 100 + 75 + 5\nprint(res == 875, res)"], outputs=["False 180"],
                     answer_text=WRONG, completion_text=WRONG_COMPLETION)
    x = compute_reward(r, NUMS, TARGET)
    assert not x.contradicted and x.exec > 0 and x.format > 0


def test_correct_answer_never_contradicted():
    r = make_rollout(codes=["res = (100 + 75) * 5\nprint(res == 875, res)"], outputs=["True 875"])
    x = compute_reward(r, NUMS, TARGET)
    assert not x.contradicted and x.correct == 1.0 and x.exec > 0 and x.format > 0


def test_retry_after_false_beats_submitting_the_false_answer():
    give_up = make_rollout(codes=["res = (100 + 75) * 3\nprint(res == 875, res)"], outputs=["False 525"],
                           answer_text=WRONG, completion_text=WRONG_COMPLETION)
    retry_fail = make_rollout(codes=["res = (100 + 75) * 3\nprint(res == 875, res)", "res = 100 * 3 + 75\nprint(res == 875, res)"],
                              outputs=["False 525", "False 375"], code_rounds=2,
                              answer_text="100 * 3 + 5", completion_text=WRONG_COMPLETION)
    assert reward(retry_fail) > reward(give_up)


def test_contradiction_check_can_be_disabled():
    from countdown_rl.rl.rewards import RewardConfig
    r = make_rollout(codes=["print((100 + 75) * 3 == 875)"], outputs=["False"],
                     answer_text=WRONG, completion_text=WRONG_COMPLETION)
    assert compute_reward(r, NUMS, TARGET, RewardConfig(zero_shaping_if_contradicted=False)).exec > 0


# --- invalid-literal zero-shaping (P7) ---------------------------------------------------------
def _invalid_rollout(answer):
    from countdown_rl.rl.rollout import RolloutResult
    code = f"res = {answer}\nprint(res == 150, res)"
    text = (f"<think>\nTry {answer}.\n</think>\n<python>\n{code}\n</python>\n<output>\nTrue 150\n</output>\n"
            f"<answer>\n{answer}\n</answer>")
    return RolloutResult(prompt_ids=[], completion_ids=[0] * 40, old_logprobs=[0.0] * 40, transcript=text,
                         completion_text=text, codes=[code], code_oks=[True], outputs=["True 150"],
                         has_answer=True, answer_text=answer, terminal_reason="answer",
                         loss_mask=[1] * 40)


def test_invalid_literals_gets_shaping_by_default():
    from countdown_rl.rl.rewards import RewardConfig, compute_reward
    r = compute_reward(_invalid_rollout("50 + 25 + 25 + 25 + 25"), [4, 2, 50, 6, 8, 25], 150, RewardConfig())
    assert r.correct == 0.0 and not r.invalid_literals
    assert r.exec + r.format > 0  # old behaviour preserved when the flag is off


def test_invalid_literals_zero_shaping_when_enabled():
    from countdown_rl.rl.rewards import RewardConfig, compute_reward
    cfg = RewardConfig(zero_shaping_if_invalid_literals=True)
    for ans, nums in (("50 + 25 + 25 + 25 + 25", [4, 2, 50, 6, 8, 25]),   # 25 reused
                      ("75 + 75", [6, 9, 3, 1, 75])):                       # 75 reused
        r = compute_reward(_invalid_rollout(ans), nums, 150, cfg)
        assert r.correct == 0.0 and r.invalid_literals and r.exec == 0.0 and r.format == 0.0


def test_invalid_literals_flag_does_not_touch_correct_or_wrong_value_answers():
    from countdown_rl.rl.rewards import RewardConfig, compute_reward
    cfg = RewardConfig(zero_shaping_if_invalid_literals=True)
    ok = compute_reward(_invalid_rollout("50 + 25 * 4"), [4, 2, 50, 6, 8, 25], 150, cfg)
    assert ok.correct == 1.0 and not ok.invalid_literals
    wrong = compute_reward(_invalid_rollout("50 + 25 * 2"), [4, 2, 50, 6, 8, 25], 150, cfg)
    assert wrong.correct == 0.0 and not wrong.invalid_literals  # wrong value, valid literals

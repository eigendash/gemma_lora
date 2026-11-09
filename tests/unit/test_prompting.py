import os
import random

import pytest

from countdown_rl.rl import prompting
from countdown_rl.sandbox import Sandbox

NUMS = [100, 75, 3, 5]
TARGET = 875
SOLUTION = "(100 + 75) * 5"


@pytest.fixture(scope="module")
def sandbox():
    s = Sandbox(n_workers=1, timeout=0.050)
    yield s
    s.close()


def _problem():
    return {"nums": NUMS, "target": TARGET, "solution": SOLUTION}


@pytest.mark.parametrize("kind,n_env", [("direct", 1), ("split", 1), ("near_miss", 2), ("resplit", 2)])
def test_generate_trace_kinds(sandbox, kind, n_env):
    from countdown_rl.data.verifier import verify_answer

    segs = prompting.generate_trace(_problem(), random.Random(1), sandbox, kind=kind)
    env = [t for is_env, t in segs if is_env]
    text = "".join(t for _, t in segs)
    assert len(env) == n_env and text.count("<answer>") == 1 and text.count("<think>") >= 1
    bodies = [e.strip().removeprefix("<output>").removesuffix("</output>").strip() for e in env]
    if kind == "direct":
        assert bodies[0].startswith("True")
    if kind == "near_miss":
        assert bodies[0].startswith("False")          # real sandbox: the attempt misses
        assert "too high" in text or "too low" in text
    if kind in ("split", "near_miss", "resplit"):
        assert str(TARGET) in bodies[-1].split()       # the operation sweep shows the target
        assert f"is {TARGET}, the target" in text      # grounded claim after the sweep
    if kind == "resplit":
        assert str(TARGET) not in bodies[0].split()    # first split really misses
        assert f"None is {TARGET}" in text and "Split differently" in text
    assert not bodies[-1].startswith("False")          # never answer right after a miss
    ans = segs[-1][1].split("<answer>\n")[1].split("\n</answer>")[0]
    assert verify_answer(NUMS, TARGET, ans)[0], ans
    # generated python segments end exactly at </python> (where rollouts stop)
    assert all(t.endswith("</python>") for e, t in segs if not e and "<python>" in t)


def test_minimal_solution_prefers_fewer_numbers():
    # 100 + 75 + 3 - 3 ... uses 4 numbers, but 25 * 3 = 75 needs only 2
    assert prompting.minimal_solution([25, 3, 10, 2], 75, "(25*3)+(10-10)").count("+") == 0


def test_trace_mix_roughly_matches(sandbox):
    rng = random.Random(0)
    import collections
    c = collections.Counter()
    for _ in range(60):
        segs = prompting.generate_trace(_problem(), rng, sandbox)
        c[sum(e for e, _ in segs)] += 1
    assert c[1] > 0 and c[2] > 0


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="needs Qwen weights (RUN_MODEL_TESTS=1)")
def test_retry_prompt_style_adds_only_the_instruction(sandbox):
    from countdown_rl.rl.rollout import RETRY_HINT

    base = prompting.problem_prompt(NUMS, TARGET, "fewshot_direct")
    hint = prompting.problem_prompt(NUMS, TARGET, "fewshot_direct_retry")
    assert RETRY_HINT not in base and RETRY_HINT in hint
    assert hint.replace(" " + RETRY_HINT, "") == base
    pre_b = prompting.fewshot_prefix(sandbox, "fewshot_direct")
    pre_h = prompting.fewshot_prefix(sandbox, "fewshot_direct_retry")
    assert pre_h.replace(" " + RETRY_HINT, "") == pre_b      # same single direct example, plus the sentence
    assert "<output>\nFalse" not in pre_h                    # still no failed-check demonstration


def test_wide_prompt_style_demo_is_legal_and_grounded(sandbox):
    """fewshot_wide_retry: 2 checks x 8 candidates; first misses, second hits; every candidate obeys
    the number rule (only its value may be wrong) and the answer verifies."""
    import re

    from countdown_rl.rl.rollout import MULTI_HINT, RETRY_HINT, WIDE_HINT
    from countdown_rl.data.verifier import verify_answer

    p = prompting.FEWSHOT_PROBLEMS[0]
    nums, target = p["nums"], p["target"]
    prompt = prompting.problem_prompt(nums, target, "fewshot_wide_retry")
    assert WIDE_HINT in prompt and MULTI_HINT not in prompt and RETRY_HINT not in prompt
    segs = prompting.generate_trace(p, random.Random(0), sandbox, kind="wide")
    text = "".join(t for _, t in segs)
    assert text.count("<python>") == 2                                   # within SPEC's 2-execution cap
    outs = [t for env, t in segs if env]
    assert len(outs) == 2 and "True" not in outs[0] and "True" in outs[1]
    for block in re.findall(r"for res in \((.*)\):", text):
        cands = [c.strip() for c in re.split(r", (?![^()]*\))", block)]
        assert len(cands) == 8
        for c in cands:
            assert verify_answer(nums, target, c)[1] in ("ok", "wrong-value"), c
    for out in outs:                                                     # stdout stays under the 200-char cap
        assert len(out) < 200
    answer = re.search(r"<answer>\n(.*)\n</answer>", text).group(1)
    assert verify_answer(nums, target, answer)[0]
    assert prompting.fewshot_prefix(sandbox, "fewshot_wide_retry").startswith(prompt)


def test_r1zero_tool_prompt_has_no_demo_or_strategy(sandbox):
    """The zero-shot arm's prompt: tool description only, no worked example, no strategy sentence, no hint."""
    assert prompting.fewshot_prefix(sandbox, "r1zero_tool") == ""
    p = prompting.problem_prompt([25, 3, 7, 50], 182, "r1zero_tool")
    assert "25, 3, 7, 50" in p and "182" in p and p.endswith("Assistant:")
    for s in ("several", "retry", "try a different", "Hint:", "<output>\n", "True"):
        assert s not in p
    with pytest.raises(ValueError):
        prompting.problem_prompt([25, 3, 7, 50], 182, "r1zero_tool", solution_hint="25 * 7")


def test_prompts_are_frozen(sandbox):
    """Every reported run used these exact prompts; any change to the text or the worked examples breaks this hash."""
    import hashlib

    h = hashlib.sha256()
    for style in sorted(prompting.PROMPT_STYLES):
        for nums, target in [([25, 3, 1, 7], 76), ([100, 6, 8, 2, 9], 391)]:
            prefix = prompting.fewshot_prefix(sandbox, style)
            h.update(prompting.with_prefix(prefix, prompting.problem_prompt(nums, target, style)).encode())
    h.update(prompting.solution_hint("(1+2)*(3+4)", 0.5).encode())
    assert h.hexdigest()[:16] == "25a452cb8669221b"

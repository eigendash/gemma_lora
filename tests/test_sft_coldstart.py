import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_ROOT, ".hf-cache"))

import random  # noqa: E402

import pytest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sft_coldstart  # noqa: E402
from rollout import assemble, format_prompt  # noqa: E402
from sandbox import Sandbox  # noqa: E402

QWEN = "Qwen/Qwen2.5-0.5B"
NUMS = [100, 75, 3, 5]
TARGET = 875
SOLUTION = "(100 + 75) * 5"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture(scope="module")
def sandbox():
    s = Sandbox(n_workers=1, timeout=0.050)
    yield s
    s.close()


def _problem():
    return {"nums": NUMS, "target": TARGET, "solution": SOLUTION}


@pytest.mark.parametrize("kind,n_env", [("direct", 1), ("split", 1), ("near_miss", 2), ("resplit", 2)])
def test_generate_trace_kinds(sandbox, kind, n_env):
    from verifier import verify_answer

    segs = sft_coldstart.generate_trace(_problem(), random.Random(1), sandbox, kind=kind)
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
    assert sft_coldstart.minimal_solution([25, 3, 10, 2], 75, "(25*3)+(10-10)").count("+") == 0


def test_trace_mix_roughly_matches(sandbox):
    rng = random.Random(0)
    import collections
    c = collections.Counter()
    for _ in range(60):
        segs = sft_coldstart.generate_trace(_problem(), rng, sandbox)
        c[sum(e for e, _ in segs)] += 1
    assert c[1] > 0 and c[2] > 0


def _find_sublist(haystack, needle):
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def test_build_example_masks_output(tok, sandbox):
    rng = random.Random(2)
    segs = sft_coldstart.generate_trace(_problem(), rng, sandbox, kind="near_miss")
    ex = sft_coldstart.build_example(tok, format_prompt(NUMS, TARGET), segs)
    ids, labels = ex["input_ids"], ex["labels"]

    assert len(ids) == len(labels)
    # every label is either -100 (env) or its own token id (generated)
    for i, l in zip(ids, labels):
        assert l == -100 or l == i
    assert -100 in labels
    assert any(l >= 0 for l in labels)

    # every injected <output> segment is fully masked
    for idx in [i for i, (e, _) in enumerate(segs) if e]:
        out_ids = tok.encode(segs[idx][1], add_special_tokens=False)
        pos = _find_sublist(ids, out_ids)
        assert pos >= 0
        assert all(l == -100 for l in labels[pos : pos + len(out_ids)])


def test_build_example_truncates_consistently(tok, sandbox):
    rng = random.Random(3)
    segs = sft_coldstart.generate_trace(_problem(), rng, sandbox)
    ex = sft_coldstart.build_example(tok, format_prompt(NUMS, TARGET), segs, max_length=32)
    assert len(ex["input_ids"]) == len(ex["labels"]) == 32


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="needs Qwen weights (RUN_MODEL_TESTS=1)")
def test_sft_checkpoint_resume_matches_uninterrupted(tmp_path, tok, sandbox):
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file

    probs = [{"nums": NUMS, "target": TARGET, "solution": SOLUTION}] * 6
    rng = random.Random(0)
    examples = [sft_coldstart.build_example(tok, format_prompt(NUMS, TARGET),
                                            sft_coldstart.generate_trace(p, rng, sandbox), max_length=96) for p in probs]

    def fresh():
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32)

    _, full = sft_coldstart.train_sft(fresh(), tok, examples, device="cpu", output_dir=str(tmp_path / "a"),
                                      save_every=2, log_every=100)

    class Crash(list):  # interrupt after 3 examples (checkpoint exists at step 2)
        def __iter__(self):
            for i, x in enumerate(list.__iter__(self)):
                if i == 3:
                    raise KeyboardInterrupt
                yield x

    with pytest.raises(KeyboardInterrupt):
        sft_coldstart.train_sft(fresh(), tok, Crash(examples), device="cpu", output_dir=str(tmp_path / "b"),
                                save_every=2, log_every=100)
    _, resumed = sft_coldstart.train_sft(fresh(), tok, examples, device="cpu", output_dir=str(tmp_path / "b"),
                                         save_every=2, log_every=100)
    assert resumed == pytest.approx(full, abs=1e-5)
    wa = load_file(str(tmp_path / "a" / "adapter_model.safetensors"))
    wb = load_file(str(tmp_path / "b" / "adapter_model.safetensors"))
    assert all(torch.allclose(wa[k], wb[k], atol=1e-6) for k in wa)


def test_retry_prompt_style_adds_only_the_instruction(sandbox):
    from rollout import RETRY_HINT

    base = sft_coldstart.problem_prompt(NUMS, TARGET, "fewshot_direct")
    hint = sft_coldstart.problem_prompt(NUMS, TARGET, "fewshot_direct_retry")
    assert RETRY_HINT not in base and RETRY_HINT in hint
    assert hint.replace(" " + RETRY_HINT, "") == base
    pre_b = sft_coldstart.fewshot_prefix(sandbox, "fewshot_direct")
    pre_h = sft_coldstart.fewshot_prefix(sandbox, "fewshot_direct_retry")
    assert pre_h.replace(" " + RETRY_HINT, "") == pre_b      # same single direct example, plus the sentence
    assert "<output>\nFalse" not in pre_h                    # still no failed-check demonstration


def test_wide_prompt_style_demo_is_legal_and_grounded(sandbox):
    """fewshot_wide_retry: 2 checks x 8 candidates; first misses, second hits; every candidate obeys
    the number rule (only its value may be wrong) and the answer verifies."""
    import re

    from rollout import MULTI_HINT, RETRY_HINT, WIDE_HINT
    from verifier import verify_answer

    p = sft_coldstart.FEWSHOT_PROBLEMS[0]
    nums, target = p["nums"], p["target"]
    prompt = sft_coldstart.problem_prompt(nums, target, "fewshot_wide_retry")
    assert WIDE_HINT in prompt and MULTI_HINT not in prompt and RETRY_HINT not in prompt
    segs = sft_coldstart.generate_trace(p, random.Random(0), sandbox, kind="wide")
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
    assert sft_coldstart.fewshot_prefix(sandbox, "fewshot_wide_retry").startswith(prompt)


def test_segments_from_sampled_completion_mask_outputs(tok):
    """P11: a sampled transcript splits into model text + env <output> blocks (masked in SFT)."""
    comp = ("<think>\nTry two.\n</think>\n<python>\nfor res in (2 * 7, 2 + 7):\n    print(res == 14, res)\n</python>\n"
            "\n<output>\nTrue 14\nFalse 9\n</output>\n<think>\nCandidate 1.\n</think>\n<answer>\n2 * 7\n</answer>\n")
    segs = sft_coldstart.segments_from_completion(comp)
    assert [e for e, _ in segs] == [False, True, False]
    assert "".join(t for _, t in segs) == comp
    ex = sft_coldstart.build_example(tok, format_prompt([2, 7], 14), segs, max_length=4096)
    text_trained = tok.decode([i for i in ex["labels"] if i != -100])
    assert "True 14" not in text_trained and "False 9" not in text_trained   # env tokens masked
    assert "2 * 7" in text_trained and "</answer>" in text_trained


def test_r1zero_tool_prompt_has_no_demo_or_strategy(sandbox):
    """The zero-shot arm's prompt: tool description only, no worked example, no strategy sentence, no hint."""
    assert sft_coldstart.fewshot_prefix(sandbox, "r1zero_tool") == ""
    p = sft_coldstart.problem_prompt([25, 3, 7, 50], 182, "r1zero_tool")
    assert "25, 3, 7, 50" in p and "182" in p and p.endswith("Assistant:")
    for s in ("several", "retry", "try a different", "Hint:", "<output>\n", "True"):
        assert s not in p
    with pytest.raises(ValueError):
        sft_coldstart.problem_prompt([25, 3, 7, 50], 182, "r1zero_tool", solution_hint="25 * 7")

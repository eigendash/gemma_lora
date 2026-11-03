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


def test_make_wrong_attempt_never_equals_target():
    rng = random.Random(0)
    for _ in range(30):
        w, expr = sft_coldstart.make_wrong_attempt(NUMS, TARGET, rng)
        assert w != TARGET


def test_generate_trace_structure(sandbox):
    rng = random.Random(1)
    segs = sft_coldstart.generate_trace(_problem(), rng, sandbox)
    assert [is_env for is_env, _ in segs] == [False, False, True, False, False, True, False]
    text = "".join(t for _, t in segs)
    assert text.count("<output>") == 2
    assert text.count("<answer>") == 1
    assert text.count("<python>") == 2
    # real sandbox stdout: wrong -> False, correct -> True
    assert "False" in segs[2][1]
    assert "True" in segs[5][1]


def _find_sublist(haystack, needle):
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def test_build_example_masks_output(tok, sandbox):
    rng = random.Random(2)
    segs = sft_coldstart.generate_trace(_problem(), rng, sandbox)
    ex = sft_coldstart.build_example(tok, format_prompt(NUMS, TARGET), segs)
    ids, labels = ex["input_ids"], ex["labels"]

    assert len(ids) == len(labels)
    # every label is either -100 (env) or its own token id (generated)
    for i, l in zip(ids, labels):
        assert l == -100 or l == i
    assert -100 in labels
    assert any(l >= 0 for l in labels)

    # the two injected <output> segments (idx 2 and 5) are fully masked
    for idx in (2, 5):
        out_ids = tok.encode(segs[idx][1], add_special_tokens=False)
        pos = _find_sublist(ids, out_ids)
        assert pos >= 0
        assert all(l == -100 for l in labels[pos : pos + len(out_ids)])


def test_build_example_truncates_consistently(tok, sandbox):
    rng = random.Random(3)
    segs = sft_coldstart.generate_trace(_problem(), rng, sandbox)
    ex = sft_coldstart.build_example(tok, format_prompt(NUMS, TARGET), segs, max_length=32)
    assert len(ex["input_ids"]) == len(ex["labels"]) == 32

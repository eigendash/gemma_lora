"""Fresh 1,000-problem holdout sets, each generated after the model selection it is meant to check
(scripts/make_holdout.py): data/holdout.jsonl (2026-11-05, for P10b) and data/holdout2.jsonl (2026-11-06, for
P10b-nohint / P10c) and data/holdout3.jsonl (2026-11-06, for the from-scratch replays: Qwen3-8B, Gemma-4-E4B,
Qwen3-4B seeds). They must stay disjoint from train/val/test and from each other, and must never be read by
training or selection code."""
import hashlib
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {  # file -> (sha256, files it must be disjoint from)
    "holdout.jsonl": ("480a4ae99b44f0c1f9dfaa0f3f934c480fac0315e5a584ba74366c132b808f30", ("train.jsonl", "eval.jsonl")),
    "holdout2.jsonl": ("caac4fb21c5d228ef14508c6edf7d0c9c8c0bda4b6826702e35683d7b9177057",
                       ("train.jsonl", "eval.jsonl", "holdout.jsonl")),
    "holdout3.jsonl": ("f727df133a8d6144a85c3bc93b5669936a30b877fde61af9a54d7ae94ef2f7e3",
                       ("train.jsonl", "eval.jsonl", "holdout.jsonl", "holdout2.jsonl")),
}


def _keys(path):
    return {(tuple(sorted(p["nums"])), p["target"]) for p in map(json.loads, open(path))}


@pytest.mark.parametrize("name", sorted(SETS))
def test_holdout_is_frozen_disjoint_and_solvable(name):
    from countdown_rl.data.verifier import verify_answer

    path = os.path.join(ROOT, "data", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    sha, disjoint_from = SETS[name]
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == sha  # never regenerated / edited
    rows = [json.loads(line) for line in open(path)]
    hk = _keys(path)
    assert len(rows) == len(hk) == 1000
    for other in disjoint_from:  # train.jsonl also holds the val split
        assert not hk & _keys(os.path.join(ROOT, "data", other)), other
    assert [sum(r["tier"] == t for r in rows) for t in ("easy", "medium", "hard")] == [400, 300, 300]
    assert all(verify_answer(r["nums"], r["target"], r["solution"])[0] for r in rows)


def test_make_holdout_reproduces_the_frozen_file(tmp_path):
    from countdown_rl.data.holdout import main

    if not os.path.exists(os.path.join(ROOT, "data", "holdout.jsonl")):
        pytest.skip("holdout not present")
    out = tmp_path / "h.jsonl"
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        main(["--seed", "20261105", "--out", str(out)])
    finally:
        os.chdir(cwd)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SETS["holdout.jsonl"][0]


def test_no_training_or_selection_code_reads_the_holdout():
    pkg = "src/countdown_rl"
    for f in (f"{pkg}/rl/grpo.py", f"{pkg}/rl/rollout.py", f"{pkg}/rl/rollout_vllm.py", f"{pkg}/rl/prompting.py",
              f"{pkg}/evaluation/evaluate.py", "scripts/cloud/train.sh", "scripts/cloud/chain.sh"):
        src = open(os.path.join(ROOT, f)).read()
        assert not re.search(r"holdout", src, re.I), f

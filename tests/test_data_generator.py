import random

import data_generator as dg
from verifier import integer_intermediates, verify_answer


def test_tier_properties_and_solvable():
    rng = random.Random(1)
    probs = dg.generate_split(150, rng, {"easy": 1, "medium": 1, "hard": 1})
    assert len(probs) == 150
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for p in probs:
        tier = p["tier"]
        counts[tier] += 1
        nc = len(p["nums"])
        if tier == "easy":
            assert nc in (3, 4)
            assert 10 <= p["target"] <= 99
        elif tier == "medium":
            assert nc == 5
            assert 100 <= p["target"] <= 999
        else:
            assert nc == 6
            assert 100 <= p["target"] <= 999
        ok, detail = verify_answer(p["nums"], p["target"], p["solution"])
        assert ok, (p, detail)
        assert integer_intermediates(p["solution"]), p
    assert counts == {"easy": 50, "medium": 50, "hard": 50}


def test_dedupe_within_split():
    rng = random.Random(2)
    probs = dg.generate_split(300, rng, {"easy": 1, "medium": 0, "hard": 0})
    keys = [(tuple(sorted(p["nums"])), p["target"]) for p in probs]
    assert len(keys) == len(set(keys)) == 300


def test_no_train_eval_overlap():
    train, eval_ = dg.generate_dataset(
        200, 60, seed=3, tier_weights={"easy": 1, "medium": 0, "hard": 0}
    )
    tk = {(tuple(sorted(p["nums"])), p["target"]) for p in train}
    ek = {(tuple(sorted(p["nums"])), p["target"]) for p in eval_}
    assert tk.isdisjoint(ek)


def test_seed_reproducible():
    a, _ = dg.generate_dataset(50, 20, seed=7)
    b, _ = dg.generate_dataset(50, 20, seed=7)
    assert a == b


def test_seed_changes_output():
    a, _ = dg.generate_dataset(30, 10, seed=7)
    b, _ = dg.generate_dataset(30, 10, seed=8)
    assert a != b


def test_nums_from_countdown_pool():
    rng = random.Random(5)
    probs = dg.generate_split(100, rng, {"easy": 1, "medium": 0, "hard": 0})
    for p in probs:
        for x in p["nums"]:
            assert x in dg.POOL


def test_no_target_equals_input():
    # degenerate "answer is one of the inputs" must be rejected (audit m2)
    rng = random.Random(6)
    probs = dg.generate_split(300, rng, {"easy": 1, "medium": 1, "hard": 1})
    for p in probs:
        assert p["target"] not in p["nums"]

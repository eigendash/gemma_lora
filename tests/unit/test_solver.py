from countdown_rl.data.solver import solve
from countdown_rl.data.verifier import verify_answer


def test_solve_simple():
    found, expr = solve([1, 2, 3], 6)
    assert found and expr is not None
    assert verify_answer([1, 2, 3], 6, expr)[0]


def test_solve_uses_subset():
    # 75 - 1 == 74 uses only two of the four numbers
    found, expr = solve([100, 75, 2, 1], 74)
    assert found
    assert verify_answer([100, 75, 2, 1], 74, expr)[0]


def test_solve_known_cases():
    for nums, target in [([25, 3, 1], 76), ([6, 5, 2], 13), ([100, 75, 2, 1], 74)]:
        found, expr = solve(nums, target)
        assert found, (nums, target)
        assert verify_answer(nums, target, expr)[0], (nums, target, expr)


def test_solve_unsolvable():
    found, expr = solve([1, 1], 3)
    assert not found and expr is None


def test_solve_division_path():
    found, expr = solve([100, 25, 4], 16)
    # (100/25)*4 == 16
    assert found
    assert verify_answer([100, 25, 4], 16, expr)[0]

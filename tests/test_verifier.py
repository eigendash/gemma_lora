from verifier import integer_intermediates, verify_answer


def test_correct_simple():
    assert verify_answer([1, 2, 3], 6, "1 + 2 + 3") == (True, "ok")
    assert verify_answer([1, 2, 3], 6, "2 * 3") == (True, "ok")


def test_wrong_value():
    ok, detail = verify_answer([1, 2, 3], 6, "1 + 2")
    assert not ok and detail == "wrong-value"


def test_reused_number():
    ok, detail = verify_answer([1, 2, 3], 7, "1 + 2 + 2 + 2")
    assert not ok and detail == "reused/foreign-literal"


def test_foreign_literal():
    ok, detail = verify_answer([1, 2, 3], 6, "1 + 2 + 3 + 0")
    assert not ok and detail == "reused/foreign-literal"


def test_disallowed_pow():
    ok, _ = verify_answer([2, 3], 8, "2 ** 3")
    assert not ok


def test_disallowed_call():
    ok, _ = verify_answer([1, 2, 3], 6, "sum([1, 2, 3])")
    assert not ok


def test_disallowed_attribute():
    ok, _ = verify_answer([1, 2], 3, "(1).__add__(2)")
    assert not ok


def test_division_exact():
    assert verify_answer([100, 4], 25, "100 / 4") == (True, "ok")


def test_division_by_zero():
    ok, _ = verify_answer([1, 0], 1, "1 / 0")
    assert not ok


def test_parentheses_precedence():
    # (100 + 75) * 5 == 875, matches SPEC example
    assert verify_answer([100, 75, 3, 5], 875, "(100 + 75) * 5") == (True, "ok")


def test_fraction_non_integer_intermediate_allowed():
    # (1/2)+(1/2) == 1, literals {1,1,2,2} sub-multiset of [1,1,2,2]
    ok, detail = verify_answer([1, 1, 2, 2], 1, "(1 / 2) + (1 / 2)")
    assert ok, detail
    assert not integer_intermediates("(1 / 2) + (1 / 2)")


def test_integer_intermediates_true_for_clean_expr():
    assert integer_intermediates("(100 + 75) * 5")


def test_empty_and_garbage():
    assert verify_answer([1, 2], 3, "")[0] is False
    assert verify_answer([1, 2], 3, "   ")[0] is False
    assert verify_answer([1, 2], 3, "1 +")[0] is False


def test_negative_literal_rejected():
    ok, _ = verify_answer([1, 2, 3], 0, "-1 + 1")
    assert not ok


def test_bool_literal_rejected():
    ok, _ = verify_answer([1, 0], 1, "1 + True")
    assert not ok

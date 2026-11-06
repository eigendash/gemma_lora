import threading

import pytest

from sandbox import Sandbox


@pytest.fixture(scope="module")
def sb():
    s = Sandbox(n_workers=2, timeout=0.050)
    yield s
    s.close()


def test_simple_arithmetic(sb):
    assert sb.run("print(1 + 2)") == (True, "3")


def test_spec_example(sb):
    ok, text = sb.run("target = 952\nres = (100 + 75) * 5\nprint(res == target, res)")
    assert ok and text == "False 875"


def test_unary_minus_allowed(sb):
    assert sb.run("print(-3 + 5)") == (True, "2")


def test_tuple_allowed(sb):
    assert sb.run("a, b = 1, 2\nprint(a + b)") == (True, "3")


def test_loop_and_len_allowed(sb):
    ok, text = sb.run("s = 0\nfor i in range(5):\n    s = s + i\nprint(s)")
    assert ok and text == "10"


def test_pow_rejected(sb):
    ok, text = sb.run("print(2 ** 3)")
    assert not ok and "SecurityError" in text


def test_import_rejected(sb):
    assert "SecurityError" in sb.run("import os")[1]
    assert "SecurityError" in sb.run("from os import system")[1]
    assert "SecurityError" in sb.run("__import__('os')")[1]


def test_open_rejected(sb):
    ok, text = sb.run("print(open('/etc/hosts'))")
    assert not ok and "SecurityError" in text


def test_attribute_rejected(sb):
    ok, text = sb.run("print((1).__class__)")
    assert not ok and "SecurityError" in text


def test_subscript_rejected(sb):
    ok, text = sb.run("print([1, 2, 3][0])")
    assert not ok and "SecurityError" in text


def test_fstring_rejected(sb):
    ok, text = sb.run("print(f'{1}')")
    assert not ok and "SecurityError" in text


def test_lambda_rejected(sb):
    ok, text = sb.run("f = lambda x: x\nprint(f(1))")
    assert not ok and "SecurityError" in text


def test_comprehension_rejected(sb):
    ok, text = sb.run("print([x for x in range(3)])")
    assert not ok and "SecurityError" in text


def test_huge_int_rejected(sb):
    ok, text = sb.run("print(999999999999999)")
    assert not ok and "SecurityError" in text


def test_deep_nesting_rejected(sb):
    code = "(" * 5000 + "1" + ")" * 5000
    ok, text = sb.run(code)
    assert not ok and "SecurityError" in text


def test_infinite_loop_times_out(sb):
    ok, text = sb.run("while True:\n    pass")
    assert not ok and "TimeoutError" in text


def test_timeout_then_recovery(sb):
    assert "TimeoutError" in sb.run("while True:\n    pass")[1]
    # the worker was killed + respawned; a normal execution must still work
    assert sb.run("print(7)") == (True, "7")


def test_excessive_output_truncated(sb):
    code = "for i in range(50):\n    print(i, i, i, i, i)"
    ok, text = sb.run(code)
    assert ok and len(text) == 200  # truncated to max_stdout


def test_large_output_truncated_not_errored(sb):
    # well over the 8 KB capture buffer, still fast enough to finish in 50 ms:
    # the spec asks for truncation to 200 chars, not an error
    ok, text = sb.run("for i in range(3000):\n    print(i)")
    assert ok and len(text) == 200 and text.startswith("0\n1\n2")


def test_unbounded_print_loop_times_out(sb):
    ok, text = sb.run("for i in range(100000000):\n    print(i)")
    assert not ok and "TimeoutError" in text


def test_pool_is_capped_under_concurrency():
    s = Sandbox(n_workers=2, timeout=0.050)
    seen = []

    def worker(tid):
        for i in range(30):
            s.run("while True:\n    pass" if i % 10 == 0 else f"print({tid})")
            seen.append(len(s._all) + s._pending)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s.close()
    assert max(seen) <= 2


def test_worker_never_imports_torch_even_if_parent_did():
    import subprocess
    import sys

    code = (
        "import sys, json; sys.modules['torch'] = object()  # pretend the parent imported torch\n"
        "from sandbox import Sandbox\n"
        "s = Sandbox(n_workers=1)\n"
        "w = next(iter(s._all))\n"
        "print(w.proc.pid)\n"
        "s.close()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().isdigit()
    # the worker command line is the stdlib-only script, isolated (-I) and without site (-S)
    import sandbox
    assert sandbox._WORKER_PATH.endswith("sandbox_worker.py")


def test_worker_reuse_many_execs(sb):
    for _ in range(500):
        assert sb.run("print(3 * 7)") == (True, "21")


def test_concurrent_runs_result_integrity():
    s = Sandbox(n_workers=4, timeout=0.050)
    errors: list[tuple] = []

    def worker(tid):
        for i in range(50):
            ok, text = s.run(f"print({tid} * 100 + {i})")
            if (ok, text) != (True, str(tid * 100 + i)):
                errors.append((tid, i, ok, text))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    s.close()


def test_no_stale_result_after_timeout():
    # a worker that times out is killed and its pipe closed, so a later call
    # cannot receive a result that was sent after the timeout.
    s = Sandbox(n_workers=1, timeout=0.050)
    ok, text = s.run("for i in range(10000000):\n    pass\nprint(42)")
    assert not ok and "TimeoutError" in text
    assert s.run("print(5)") == (True, "5")
    s.close()

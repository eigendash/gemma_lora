"""Exact Countdown solver.

Finds one arithmetic expression (``+ - * /``) that reaches ``target`` using a
subset of ``nums``, with positive integer intermediates. Uses subset DP:
``dp[mask][value] = expression`` for every integer value reachable by combining
the numbers in ``mask``.

``max_value`` caps intermediate values for tractability. For 3-4 numbers with
values <= 100 the reachable values are <= 100**4 = 1e8, so the default 1e9 is
effectively exact for the audit workloads this is used on.
"""

from __future__ import annotations


def solve(nums, target: int, max_value: int = 10**9) -> tuple[bool, str | None]:
    """Return ``(found, expression_or_None)``."""
    nums = [int(x) for x in nums]
    n = len(nums)
    full = (1 << n) - 1
    dp: list[dict[int, str]] = [dict() for _ in range(1 << n)]

    for i, x in enumerate(nums):
        dp[1 << i][x] = str(x)

    for mask in range(1, full + 1):
        if mask & (mask - 1) == 0:  # singleton, already seeded
            continue
        low = mask & -mask
        sub = (mask - 1) & mask
        while sub:
            if sub & low:  # each partition once
                other = mask ^ sub
                da, db = dp[sub], dp[other]
                for a, ea in da.items():
                    for b, eb in db.items():
                        v = a + b
                        if v <= max_value and v not in dp[mask]:
                            dp[mask][v] = f"({ea}+{eb})"
                        if a > b:
                            v = a - b
                            if v not in dp[mask]:
                                dp[mask][v] = f"({ea}-{eb})"
                        elif b > a:
                            v = b - a
                            if v not in dp[mask]:
                                dp[mask][v] = f"({eb}-{ea})"
                        v = a * b
                        if v <= max_value and v not in dp[mask]:
                            dp[mask][v] = f"({ea}*{eb})"
                        if b and a % b == 0:
                            v = a // b
                            if v not in dp[mask]:
                                dp[mask][v] = f"({ea}/{eb})"
                        if a and b % a == 0:
                            v = b // a
                            if v not in dp[mask]:
                                dp[mask][v] = f"({eb}/{ea})"
            sub = (sub - 1) & mask

    # A solution may use any subset of the numbers (each at most once), so the
    # target need only be reachable from some mask, not necessarily the full one.
    for mask in range(1, full + 1):
        if target in dp[mask]:
            return True, dp[mask][target]
    return False, None

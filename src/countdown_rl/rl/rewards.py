"""Composite reward for Countdown rollouts.

    R = R_correct + R_exec + R_format - lambda * length_penalty

- **R_correct** — the answer verifies against ``(nums, target)`` via
  ``verifier.verify_answer`` (model-independent: ast + Fraction, literals must be
  a sub-multiset of the inputs). No ``<answer>`` => 0.
- **R_exec** — small, granted only if at least one executed code block ran
  cleanly, printed something, and contains real arithmetic over the problem: a
  ``+ - * /`` expression combining at least two of the input numbers (respecting
  multiplicity). This blocks ``print(1)``, ``print(target)``, ``x = 25; print(76)``
  and silent code, while ``print((100+75)*5 == 875)`` or the spec's
  ``res == target`` style still counts.
- **R_format** — a closed ``<think>...</think>`` before exactly one
  ``<answer>...</answer>``, tags in ``<think>`` -> ``<python>`` -> ``<answer>``
  order, and the answer body is a well-formed ``+ - * /`` expression (it may
  still be wrong). ``<python>`` is optional.
- **length penalty** — ``lambda * max(0, generated_tokens - threshold)``, applied
  only beyond the threshold, to bound degenerate verbosity.
- A hallucinated ``<output>`` (the model fabricated the environment output instead
  of running code) zeroes both R_exec and R_format.
- A **contradicted answer** (a wrong final answer that the model's own executed
  check had already shown to be wrong) also zeroes R_exec and R_format. Without
  this, "check -> see False -> submit it anyway" earns 0.2 while retrying risks
  less, and GRPO learned exactly that shortcut (pilot B, steps 6-9: retries after
  a failed check fell from 33/43 to 0/41 on held-out problems).

Every reward component is returned separately for logging.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from countdown_rl.rl.rollout import RolloutResult
from countdown_rl.data.verifier import ExprError, _eval, verify_answer

_INT_RE = re.compile(r"\b\d+\b")
_TAG_ORDER = ["<think>", "<python>", "<answer>"]
# verifier details that still mean "a syntactically valid arithmetic expression"
_WELL_FORMED_DETAILS = {"ok", "wrong-value", "reused/foreign-literal", "division by zero", "division-by-zero"}


@dataclass
class RewardConfig:
    correct: float = 1.0
    exec: float = 0.1
    format: float = 0.1
    length_lambda: float = 0.001
    length_threshold: int = 256
    zero_shaping_if_contradicted: bool = True
    # wrong answer that breaks the number rule (a literal reused or not among the inputs): no
    # exec/format credit. Closes the value-only self-check gap (P6: `print(res == target)` passes
    # for `50 + 25 + 25 + 25 + 25`). Off by default so earlier configs keep their reward.
    zero_shaping_if_invalid_literals: bool = False


@dataclass
class RewardResult:
    total: float
    correct: float
    exec: float
    format: float
    length_penalty: float
    hallucinated: bool
    contradicted: bool = False
    invalid_literals: bool = False

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "exec": self.exec,
            "format": self.format,
            "length_penalty": self.length_penalty,
            "hallucinated": self.hallucinated,
            "contradicted": self.contradicted,
            "invalid_literals": self.invalid_literals,
        }


def references_numbers(code: str, nums: list[int]) -> bool:
    """True if ``code`` contains at least one integer literal that is an input number."""
    nums_set = set(nums)
    return any(int(m) in nums_set for m in _INT_RE.findall(code))


def computes_with_inputs(code: str, nums: list[int]) -> bool:
    """True if some ``+ - * /`` expression in ``code`` combines >= 2 input numbers."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    have = Counter(nums)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            lits = Counter(c.value for c in ast.walk(node)
                           if isinstance(c, ast.Constant) and type(c.value) is int)
            if sum(min(k, have[v]) for v, k in lits.items()) >= 2:
                return True
    return False


_FALSE_RE = re.compile(r"\bFalse\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_expr(text: str):
    try:
        return ast.parse(" ".join(text.split()), mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _value(node) -> Fraction | None:
    try:
        return _eval(node, Counter())
    except (ExprError, ZeroDivisionError, RecursionError):
        return None


def answer_contradicted(answer_text: str, codes: list[str], outputs: list[str]) -> bool:
    """True if some executed check whose output contains ``False`` either printed
    the answer's value or contained the answer expression itself, i.e. the model
    had already seen this answer fail. Callers apply it only to wrong answers."""
    node = _parse_expr(answer_text or "")
    if node is None:
        return False
    val, dump = _value(node), ast.dump(node)
    for code, out in zip(codes, outputs):
        if not _FALSE_RE.search(out or ""):
            continue
        if val is not None:
            for m in _NUM_RE.findall(out):
                try:
                    if abs(Fraction(m) - val) < Fraction(1, 10**6):
                        return True
                except (ValueError, ZeroDivisionError):
                    pass
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        if any(isinstance(n, ast.BinOp) and ast.dump(n) == dump for n in ast.walk(tree)):
            return True
    return False


def answer_well_formed(answer_text: str) -> bool:
    return verify_answer([], 0, answer_text)[1] in _WELL_FORMED_DETAILS


def check_format(completion_text: str, answer_text: str | None = None) -> bool:
    """Closed <think> before exactly one <answer>; tags in think -> python -> answer
    order; and (when given) the answer body is a well-formed expression."""
    if completion_text.count("<answer>") != 1 or completion_text.count("</answer>") != 1:
        return False
    t_open, t_close = completion_text.find("<think>"), completion_text.find("</think>")
    if t_open == -1 or t_close == -1 or not (t_open < t_close < completion_text.find("<answer>")):
        return False
    last = -1
    for tag in _TAG_ORDER:
        i = completion_text.find(tag)
        if i != -1:
            if i < last:
                return False
            last = i
    if answer_text is not None and not answer_well_formed(answer_text):
        return False
    return True


def compute_reward(
    rollout: RolloutResult,
    nums: list[int],
    target: int,
    config: RewardConfig | None = None,
) -> RewardResult:
    cfg = config or RewardConfig()

    r_correct = 0.0
    verdict = verify_answer(nums, target, rollout.answer_text) if rollout.has_answer else (False, "no-answer")
    if verdict[0]:
        r_correct = cfg.correct
    invalid_literals = (cfg.zero_shaping_if_invalid_literals and r_correct == 0.0
                        and verdict[1] == "reused/foreign-literal")

    outputs = list(rollout.outputs) + [""] * (len(rollout.codes) - len(rollout.outputs))
    ran_clean_and_relevant = any(
        ok and out.strip() != "" and computes_with_inputs(code, nums)
        for ok, code, out in zip(rollout.code_oks, rollout.codes, outputs)
    )
    contradicted = (cfg.zero_shaping_if_contradicted and rollout.has_answer and r_correct == 0.0
                    and answer_contradicted(rollout.answer_text, rollout.codes, outputs))
    shaping_ok = not rollout.hallucinated_output and not contradicted and not invalid_literals
    r_exec = cfg.exec if (ran_clean_and_relevant and shaping_ok) else 0.0

    well_formed = check_format(rollout.completion_text, rollout.answer_text if rollout.has_answer else "")
    r_format = cfg.format if (well_formed and shaping_ok) else 0.0

    n_gen = sum(rollout.loss_mask)
    length_penalty = cfg.length_lambda * max(0, n_gen - cfg.length_threshold)

    total = r_correct + r_exec + r_format - length_penalty
    return RewardResult(
        total=total,
        correct=r_correct,
        exec=r_exec,
        format=r_format,
        length_penalty=length_penalty,
        hallucinated=rollout.hallucinated_output,
        contradicted=bool(contradicted),
        invalid_literals=bool(invalid_literals),
    )

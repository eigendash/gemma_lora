"""Composite reward for Countdown rollouts.

    R = R_correct + R_exec + R_format - lambda * length_penalty

- **R_correct** — the answer verifies against ``(nums, target)`` via
  ``verifier.verify_answer`` (model-independent: ast + Fraction, literals must be
  a sub-multiset of the inputs). No ``<answer>`` => 0.
- **R_exec** — small, granted only if at least one executed code ran cleanly AND
  references at least one of the problem's input numbers. This blocks ``print(1)``
  and ``print(target)`` (the model learns to run *relevant* checks).
- **R_format** — exactly one ``<answer>`` block and the tags appear in
  ``<think>`` -> ``<python>`` -> ``<answer>`` order.
- **length penalty** — ``lambda * max(0, generated_tokens - threshold)``, applied
  only beyond the threshold, to bound degenerate verbosity.
- A hallucinated ``<output>`` (the model fabricated the environment output instead
  of running code) zeroes both R_exec and R_format.

Every reward component is returned separately for logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rollout import RolloutResult
from verifier import verify_answer

_INT_RE = re.compile(r"\b\d+\b")
_TAG_ORDER = ["<think>", "<python>", "<answer>"]


@dataclass
class RewardConfig:
    correct: float = 1.0
    exec: float = 0.1
    format: float = 0.1
    length_lambda: float = 0.001
    length_threshold: int = 256


@dataclass
class RewardResult:
    total: float
    correct: float
    exec: float
    format: float
    length_penalty: float
    hallucinated: bool

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "exec": self.exec,
            "format": self.format,
            "length_penalty": self.length_penalty,
            "hallucinated": self.hallucinated,
        }


def references_numbers(code: str, nums: list[int]) -> bool:
    """True if ``code`` contains at least one integer literal that is an input number."""
    nums_set = set(nums)
    return any(int(m) in nums_set for m in _INT_RE.findall(code))


def check_format(completion_text: str) -> bool:
    """Exactly one <answer> block, and tags in think -> python -> answer order."""
    if completion_text.count("<answer>") != 1 or completion_text.count("</answer>") != 1:
        return False
    last = -1
    for tag in _TAG_ORDER:
        i = completion_text.find(tag)
        if i != -1:
            if i < last:
                return False
            last = i
    return True


def compute_reward(
    rollout: RolloutResult,
    nums: list[int],
    target: int,
    config: RewardConfig | None = None,
) -> RewardResult:
    cfg = config or RewardConfig()

    r_correct = 0.0
    if rollout.has_answer and verify_answer(nums, target, rollout.answer_text)[0]:
        r_correct = cfg.correct

    ran_clean_and_relevant = any(
        ok and references_numbers(code, nums)
        for ok, code in zip(rollout.code_oks, rollout.codes)
    )
    r_exec = cfg.exec if (ran_clean_and_relevant and not rollout.hallucinated_output) else 0.0

    r_format = cfg.format if (check_format(rollout.completion_text) and not rollout.hallucinated_output) else 0.0

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
    )

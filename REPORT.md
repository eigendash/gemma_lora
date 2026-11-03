# REPORT.md — Emergent Reasoning via RL & Programmatic Self-Correction

> **Status: SKELETON.** PART 1 (pipeline + Mac verification) is complete; the
> PART 2 numbers (full SFT / GRPO / eval) are **not yet run** and are placeholders
> marked `[PENDING]`. Nothing in the `[PENDING]` slots is a real number.

## 1. Executive summary

Goal: train `google/gemma-2-2b` (LoRA) with a custom GRPO loop to solve the
Countdown numbers game via a `<think> / <python> / <output> / <answer>` protocol,
where `<output>` is real sandbox stdout injected mid-generation.

`[PENDING: 1-paragraph summary of final accuracy, self-correction rate, and how the
protocol emerged — filled after the PART 2 cloud runs.]`

## 2. Learning dynamics

`[PENDING: reward vs step and token-length vs step curves, from
outputs/rl_*/train_log.jsonl of the real run.]`

Mac observation (pipeline check, **not** a finding): a ~4 h gemma-2-2b run at G=4
showed correctness briefly reach 0.75 then collapse, traced to an adaptive-KL
coefficient underflow (fixed, floored at 1e-3).

## 3. Qualitative backtracking examples

`[PENDING: sampled transcripts showing a failed execution ("False") followed by an
explicit correction and a correct final answer, from outputs/rl_*/rollouts.jsonl.]`

## 4. Compute bottlenecks & mitigations

- **256k-vocab lm_head** dominates per-step cost on gemma-2. Mitigation: vocab-
  chunked log-probs (`forward_logprobs_chunked`) — never materialize `(B,T,V)`.
- **MPS OOM** from full fp32 logits (Mac). Mitigation: per-chunk fp32 cast.
- **Adaptive-KL underflow** disabled the KL penalty → drift + collapse. Mitigation:
  `kl_coef` floored at `1e-3`, capped at `1.0`.

## 5. Reward exploitation mitigations

Composite reward `R = R_correct + R_exec + R_format − λ·len`; `R_correct` is a
model-independent verifier (ast + Fraction + sub-multiset literals); `R_exec` is
granted only if code ran cleanly AND references a problem number (blocks `print(1)`).
A reward-hack test suite asserts every hack scores below an honest correct rollout
(print(1), print(target), hallucinated `<output>`, reused numbers, answer in wrong
place). `[PENDING: any real reward hacks observed during PART 2.]`

## 6. Restarts and cost

`[PENDING: table fed by segment history (provider, GPU, start/end step, hours, cost),
via scripts/cost_report.py. Columns: segment | provider | GPU/VRAM | start step |
end step | gpu-hours | USD.]`

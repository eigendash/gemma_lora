# REPORT.md: Emergent Reasoning via RL and Programmatic Self-Correction

Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md). Verification evidence: [VERIFICATION.md](VERIFICATION.md).

## 1. Summary

A LoRA (r=16) on **Qwen3-4B-Base**, trained with a custom GRPO loop (native PyTorch, colocated vLLM for rollouts).
Each rollout runs the `<think> / <python> / <output> / <answer>` protocol: `<output>` is real sandbox stdout,
injected mid-generation and masked out of the loss.

- **Result:** the best policy, **P10b s80**, solves **52.8%** of a fresh 1,000-problem holdout (Pass@1, greedy). The holdout was generated after model selection and evaluated once. The previous best scored 47.9% on it (p = 0.0002).
  - On a second fresh holdout it scores 51.3%; on the original test set, 52.7%.
  - The base model scores **0%** zero-shot and **11.3%** with the same few-shot format demo.
  - **The holdout is the headline number.** The test set was reused across lineage decisions (P10 s90 was chosen partly on it) and run on 6+ models, so its numbers are slightly optimistic.
- **Model choice:** gemma-2-2b (the SPEC example) was replaced after a measured comparison. It reached about 2% Pass@1 with the format demo, against about 15% for Qwen3-4B-Base (EXPERIMENTS §2.4).

| Test set, 1,000 problems (SPEC §D) | Base, zero-shot | Base, few-shot | P8 s90 | P10 s90 | **P10b s80** |
|---|---|---|---|---|---|
| Pass@1 | 0.0% | 11.3% | 47.8% | 49.4% | **52.7%** |
| Easy / medium / hard (400 / 300 / 300) | 0 / 0 / 0 | 85 / 23 / 5 | 321 / 97 / 61 | 328 / 105 / 61 | **331 / 116 / 80** |
| Format compliance | 63.4% | 17.6% | 99.9% | 99.8% | 99.7% |
| Generated tokens per answer (own tokens) | 293 | 612 | 296 | 306 | 302 |
| Hit the 768-token cap | 17.9% | 82.3% | 0.1% | 0.2% | 0.3% |
| First check all False → retried → solved | 0 / 18 | 8 / 640 (1.3%) | 73 / 588 (12.4%) | 71 / 569 (12.5%) | 67 / 530 (12.6%) |

**How to read the metrics** (`verification/report_metrics.py`, from the eval dumps):
- **Zero-shot base:** "compliant" but useless. It writes the target number as its answer.
- **Few-shot base:** mostly runs to the 768-token cap.
- **Token length:** the RL policy's answers are *shorter* than the few-shot base's (302 vs 612), not longer. Its length is spent on checks, not rambling.
- **Self-correction:** after a first check in which every candidate printed False, P10b recovers in 12.6% of cases, against 1.3% for the base.

**Statistics:**
- Paired exact McNemar tests: P10b vs P8, p = 0.0005; vs P10 (same run), p = 0.009. P10 vs P8 alone was not significant on test (p = 0.07).
- **What these p-values cover:** they compare two *checkpoints* on a fixed problem set. They don't include run-to-run variance, so they don't show that one *recipe* beats another. A P10b replicate with a different seed and from-scratch seed replays are running to measure that spread. Per-tier p-values are descriptive only (uncorrected).
- Checkpoints were chosen on a 300-problem validation split, never on test or holdout.

### Does the recipe transfer? (fresh holdout3, every model evaluated once)

| Model | Base | After the same 4-phase recipe | Hard tier (final) |
|---|---|---|---|
| Qwen3-4B, from scratch (seeds 1 / 2) | 9.4% | 51.2% / 49.8% | 87 / 76 of 300 |
| Qwen3-4B (original, hand-steered lineage / its replicate) | 9.4% | 52.1% / 54.4% | 85 / 101 of 300 |
| **Qwen3-8B** (from scratch) | 12.0% | **55.7%** | 82 of 300 |
| Gemma-4-E4B (from scratch) | 3.7% | 30.9% | 19 of 300 |

- **Same hyperparameters for every model,** and each phase starts from the previous phase's final step.
- **Qwen3-8B vs 4B under the same rule:** clearly better. Against the from-scratch 4B seeds it scores 557 vs 512 / 498 (p = 0.001 / p < 1e-5), about +5 pp. The two 4B seeds differ by only 14 (p = 0.33), so the recipe is fairly stable across seeds.
- **Gemma:** the recipe works there too (8× its base), but from a far weaker starting point.
- **Recipe v2, preliminary (1 seed, primary holdout3 eval pending).** v2 drops hints, drops the training on all-fail ("shaping-only") groups and the KL penalty, and adds gradient clipping (EXPERIMENTS §2.7p).
  - Seed 1 was stopped by the length guard in phase b.
  - **Seed 2 finished at 70.0% on test,** against 50.9% for the same-seed replay (p = 3e-33) and 58.5% for Qwen3-8B (p = 2e-13).
  - **Its self-correction changed in kind:** after an all-False first check it recovers 61% of the time (12% for the replay). It reasons from the printed values, then runs one targeted check.
  - The cost is more truncation: 8.3% hit the cap, and format compliance is 91.7%.
  - Until holdout3 and a second seed are in, this is a single-seed result on a reused test set.
- **More training does not help the original recipe:** Qwen3-8B continued for 102 more steps stayed flat on validation (58% → 54–59%; §2.7q).
- **More training does not help v2 either:** v2 seed 2 scores 681 on holdout3 (the pre-registered primary outcome; replay seed 2: 498). Continued for 45 more steps, it scores 682 (p = 1.00). The run was stopped by hand on a flat validation curve (EXPERIMENTS §2.7r).

## 2. Learning dynamics

![learning curves](docs/figures/learning_curves.png)

*Reward, generated tokens per rollout, and in-run validation accuracy for the lineage P8 → P10 → P10b
(`verification/plot_learning_curves.py`). Each phase warm-starts from the previous phase's chosen checkpoint.
P7 (base → 36% val in 40 steps) is not drawn: its per-step log was not kept.*

- **P8** (the "check several candidates in one `<python>` block" prompt) carried validation accuracy from 37% to 46%.
  - Reward rose 0.44 → 0.60, and length grew 275 → 320 tokens as checks got wider.
- **P10 (hinted rollouts, QuestA-style):** a group where all 8 samples fail is resampled with "Hint: one solution contains X". The hint covers ½, then ¼, then none of the solution.
  - Validation first *dropped* (47 → 42%) while hints dominated, then climbed to 50% once hints faded.
- **P10b:** a second, shorter hint cycle with a 5-step LR warmup. There was no early dip, and validation sat at 50–53%.
- **Hint ablation:** the same 80 steps *without* hints did as well, at 518 vs 513 on a second fresh holdout (p = 0.77). The P10 → P10b gain (+42 to +49 on two holdouts) came from **training longer, not from the hints**. A third hint cycle (P10c, 525) adds nothing significant. There is no evidence in this lineage that hinted resampling helps this task (EXPERIMENTS §2.7m). A replicate of P10b (seed 1) landed at 506–539 on the same holdout, so run-to-run spread is about ±15–20 problems. It also showed that choosing among late checkpoints with the 300-problem validation set is unreliable: the validation pick was 33 problems worse than the final step.
  - Length rises while hints are on (longer hinted prompts lead to longer checks), then falls back to about 310 tokens when they stop.
- **Health signals:** KL to the base model grows slowly (0.07 → 0.12 across P10b). Clip fraction is about 0.003. No run hit the length cap.
  - The adaptive KL coefficient sits at its floor (0.001) for most steps, so the KL penalty is in practice inactive (about 1e-4 per token). KL is a monitor here, not a regulator.
- **Training reward is a poor progress signal here.** The signal-based curriculum keeps about half of each batch on hard problems, so train accuracy stays flat (0.35–0.55) while validation improves. Every decision used validation.
- **Validation noise:** "greedy" vLLM evaluation is not reproducible (batched LoRA kernels), giving ±7 problems on 300. Final choices therefore used 3 separate validation passes and their mean (VERIFICATION §19).

## 3. Emergent backtracking (P10b s80, test set, verbatim)

**How much is emergent.** Retrying is *prompted*: the instruction line and the few-shot demo show a second check after a failure. Even the base model retries after an all-False first check in 89% of cases, and RL raises that to 100% (530/531). What RL changed is the *quality* of checks and retries: the first check succeeds more often, and retries recover 12.6% of the time against 1.3% for the base.

All 67 recoveries on the test set follow the same pattern. A full first check fails, the model states this and proposes new candidates, and it answers with the one that printed True. Medium, 75 2 8 4 5 → 391:

```
<python> for res in (75 * 5 + 4 * 8 + 2, 75 * 5 + 8 * 4 + 2, 75 * 5 + 2 * 8 + 4, 75 * 8 - 2 * 4 - 5): print(res == 391, res)
<output> False 409 / False 409 / False 395 / False 587
<think> Candidate expressions didn't work. Let's try different combinations: 75 * 5 + 4 + 2 * 8; 75 * 5 + 8 + 2 * 4; 75 * 4 + 8 * 5 + 2.
<output> False 395 / True 391 / False 342
<answer> 75 * 5 + 8 + 2 * 4
```

- **Using the feedback:** the miss of 395 (4 too high) is fixed by moving the ×2 from 8 to 4, a change of exactly −4.
- **Hard case:** 6 8 1 10 4 9 → 132. The first round produced 180/190/288/120. The second round added `4 * (10 + 9 + 8 + 6)`, which printed True and became the answer.
- **Limits:** many retries still sample new candidates without clearly using the printed values. The strict recovery rate (12.6%) is the honest number; most of the gain since P8 comes from first checks that succeed more often.

## 3b. Test-time compute and feedback controls (P10b s80, test set)

Run with `eval.py` (HF backend, soos, merged adapter); logs in `verification/logs/spec_suite/`.

| | 0 code rounds | 1 round | 2 rounds (default) |
|---|---|---|---|
| P10b s80 | 17.8% | 47.3% | 51.8% |
| Base, few-shot | 0.6% | 10.2% | 11.1% |

| P10b s80, 2 rounds | Pass@1 |
|---|---|
| Real `<output>` | 51.8% |
| **Blanked `<output>`** (code runs, but the model sees nothing) | **12.8%** |
| Shuffled digits in `<output>` | 47.4% |
| Zero-shot prompt (no format demo) | 1.1% |
| T = 0.9 sampling: mean pass@1 / **pass@4** | 49.5% / **65.3%** (base: 9.9% / 27.6%) |

**What this shows:**
- **The policy depends on execution feedback.** Removing the feedback costs 39 pp, and allowing no code at all costs 34 pp. It isn't solving in its head and merely formatting a check.
- **It reads mainly the True/False flag, not the printed value.** Scrambling the digits costs only 4 pp. That fits the observation that retries rarely use the printed numbers (§3).
- **The policy needs the format it was trained with.** Zero-shot it almost never solves anything (1.1%), so the SPEC's "base zero-shot vs RL" comparison is 0% vs 52% only *with* the few-shot demo. A policy trained from a zero-shot prompt that describes the tool is §3c.
- **Sampling has large headroom:** pass@4 is 65%. That makes best-of-k with the verifier, or more rollouts per prompt, an obvious next lever.

## 3c. Finding: zero-shot RL works once the prompt says the tool exists

**Background.** Every model above was trained and evaluated with a one-example demo and a strategy sentence. SPEC asks for emergence without demonstrations. We had set zero-shot aside early because the base model scored 0% zero-shot, writing the target number as its answer.

**The cause was our prompt.** The `zero` prompt showed the tag skeleton but never said that `<python>` code is executed or that its output comes back.
- We wrote a new zero-shot prompt, `r1zero_tool`. It uses the R1-Zero/TinyZero conversation frame and a plain description of the tool, as in ToRL/ZeroTIR. It has no example, no "check several candidates" and no "retry" (EXPERIMENTS §2.7s).
- One description of the tool changes whether RL can start at all. Base Qwen3-4B, 256 training problems × 8 samples, T = 0.9:

| Zero-shot prompt | Problems with ≥1 correct sample (= GRPO signal) | Pass@1 | Bare-number answers |
|---|---|---|---|
| `zero` (skeleton only) | 3 / 256 | 0.2% | 38% |
| `r1zero_tool` (tool described) | **58 / 256** | 4.4% | 1% |

**Training from it** (zero-shot arm `q4v2z2`: recipe v2, seed 2, the same schedule as the best model; only the prompt differs; pre-registered):

| In-run validation, step | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Zero-shot arm (no demo) | 4.3% | 10.0% | 17.7% | 24.3% | 29.7% |
| v2 seed 2 (with demo) | 15.0% | 23.0% | 24.3% | 27.0% | 30.7% |

- **It starts 11 points behind the demo prompt and catches up within about 20 steps.** Without a demonstration, the policy runs code, learns to stop inventing `<output>` blocks (8.8% → 1.0% of rollouts), and climbs as fast as the demo-primed policy.
- **Why we missed it:**
  - The zero-shot test used a prompt that hid the tool.
  - The first pilots (8 prompts × 8 samples, ≤45 steps) were too small to learn from a near-0% start.
  - Every later experiment inherited the demo prompt.
  - The original recipe would also have struggled: at a 4% start, its format and execution credit on all-fail groups dominates the gradient, which is what produced the pilot-B shortcut (§5). v2 trains only on groups that contain a correct answer.
**Result after the full 300 steps (pre-registered verdict):**

| holdout3 (1,000 unseen problems, greedy) | Total | Easy | Medium | Hard |
|---|---|---|---|---|
| v2 seed 2, demo prompt (best model so far) | 681 | 355 | 173 | 153 |
| **Zero-shot arm, tool description only** | **699** | 351 | 167 | **181** |

- **Verdict, from the pre-registered rule (≥ 661): the demo is not needed.** The tool-use behaviour emerges from a tool description alone.
  - The +18 is not significant (paired McNemar p = 0.23; 109 vs 91 discordant problems), so the claim is parity, not superiority.
  - The gain is concentrated on hard problems.
- **The zero-shot policy uses the tool differently.** It runs code on nearly every problem and almost never invents outputs, but mostly as a calculator that confirms an answer already worked out in `<think>` (it prints values, not True/False checks). When the first printed result misses the target it retries as often as the demo-trained policy (170/230 = 74% vs 464/619 = 75% on holdout3), but recovers less often (41/230 = 18% vs 311/619 = 50%). It misses far less often in the first place (230 vs 619), which is where its accuracy comes from. So self-correction emerges without the demo, but hypothesis-testing with the tool is weaker.
- **Transfer to Gemma (in progress):**
  - Gemma-4-E4B zero-shot climbs from 1% to about 65% in-run, but almost never runs code: it solves the problem by hand in `<think>`.
  - A Gemma-4-E2B arm is being probed.

## 4. Compute bottlenecks and mitigations

| Bottleneck | Mitigation | Effect |
|---|---|---|
| One-sequence-at-a-time generation with tool calls | Batched rollout engine with per-sequence tool interleave | 7.4× faster, identical metrics |
| HF SDPA with GQA copies the KV cache 4× per step (soos, ROCm) | `attention_gqa.py` (grouped SDPA without repeat) plus merged-LoRA generation | About 10× faster generation on soos |
| Generation dominates GRPO time | Colocated **vLLM** with sleep mode: weights offloaded and KV dropped during the backward pass | 16×8 step: 5 min on soos → 39 s on an RTX PRO 6000; 64×8 step ≈ 2.3 min |
| Large vocabulary in the log-prob pass | Vocab-chunked log-probs; no `(B, T, V)` fp32 tensor | Fits in memory without gradient checkpointing |
| Sampler/trainer log-prob mismatch (up to 0.43 nats) | Truncated importance sampling (cap 2), logged every step | `gen_logp_dev` monitored |
| Spot preemption, single-GPU quota | Atomic COMMIT checkpoints to GCS every 2 steps, auto-resume, cost guard, log sync, chained runs that power off when done | All runs resumable; no idle GPU after a chain |

**Cost** (GPU-hours from the bucket's segment start to the final checkpoint, RTX PRO 6000 spot, about $2/h):
- P7 1.1 h, P8 5.5 h (including the extension), P9 3.3 h (negative), P10 3.8 h, P10b 3.5 h.
- About 17 GPU-hours, roughly $35, plus the soos pilots (P1–P6) at no charge.

## 5. Reward exploitation and mitigations

Reward: `R = R_correct + R_exec + R_format − λ·len`, with λ applied only beyond 512 tokens.

**The real hack.** In the 8×8 pilots, accuracy was about 5%, so most groups had no correct sample. A wrong but well-formed answer still earned exec + format credit, and GRPO learned to answer *without* retrying.
- **Fix 1:** a final answer whose own check printed False ("contradicted") now gets no shaping credit.
- **Fix 2:** an answer using numbers that weren't given ("invalid literals") gets none either.
- **Effect:** `contradicted` fell from 0.23 (base) to 0.04–0.06.
  - **Retries:** after an all-False first check, retrying went from 89% (base) to 100% (P10b, 530/531 on test).
  - The training-time `retry_after_false` (0.5–0.7 in P10b) undercounts: it treats a multi-candidate check with any False as failed. A strict version is logged from 2026-11-06.

**Other safeguards:**
- **Verifier:** `R_correct` is model-independent (AST plus exact fractions, numbers used at most once).
- **Exec credit** needs code that ran cleanly and combines at least 2 input numbers, which blocks `print(target)` and dummy code.
- **Hallucinated `<output>`:** writing one ends the rollout with no credit (rate 0.0002 at the end of P10b).
- **Hack suite:** 20 hand-built hacks all score below honest answers (`verification/test_reward_hacks.py`).
- **Auto-stop guards:** training stops itself with a checkpoint if `contradicted` > 0.6 or `frac_max_length` > 0.4 over 5 steps. Neither ever tripped.

**Evaluation hygiene:**
- Hints are training-only; the evaluation code never passes them.
- The holdout sets are frozen by sha256, disjoint from all other data, absent from the training VM image, and checked by tests.

## 6. What's next

The hint ablation and P10c are done (§2: no hint effect).
- **v2 seed 2:** a holdout3 eval of the phase-d final step, once, as pre-registered (EXPERIMENTS §2.7p).
- **v2 seed 1:** resume from its guard stop. More v2 seeds are needed before v2 can replace the recipe above.
- If v2 holds up, a continuation of v2 seed 2 to test whether it is still improving. The original recipe has plateaued (§2.7q).

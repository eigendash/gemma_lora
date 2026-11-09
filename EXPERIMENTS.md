# Experiment record

This file is the record behind REPORT.md. It lists every run that produced a reported number, the decision rule each
one was registered under before its result was known, and the outcome. The day-by-day log, including the pilots
and dead ends that preceded these runs, is at git tag `pre-refactor` (EXPERIMENTS.md there, about 1,100 lines).

Conventions used throughout:
- **holdout3:** 1,000 problems, seed 20261107, generated after every model choice. It is evaluated once per final
  checkpoint on the workstation with `countdown-eval`: greedy, 768 new tokens, two code rounds. Qwen uses `sdpa_gqa`
  and Gemma uses eager attention.
- **3-pass validation:** the mean of three greedy vLLM passes over the 300 held-out training problems, used because
  batched-LoRA vLLM decoding is not bit-reproducible (±7 problems per pass).
- **Paired tests:** exact McNemar on the discordant problems (`python -m countdown_rl.evaluation.paired`).
- **Lineage:** four phases of 40 / 90 / 90 / 80 steps, each warm-started from the final step of the previous phase.

## Run names

The bucket and the chain logs use the short ids from the time of the runs.

| Experiment | Bucket run ids | Model | Recipe | Prompt | Seed |
|---|---|---|---|---|---|
| `qwen3-4b-v1-handtuned` | `p7`, `p8`, `p10`, `p10b` | Qwen3-4B-Base | v1 | demo | 0 |
| `qwen3-4b-v1-handtuned-nohint` | `p10bnh` | Qwen3-4B-Base | v1, no hints | demo | 0 |
| `qwen3-4b-v1-s1`, `-s2` | `q4s1-{a..d}`, `q4s2-{a..d}` | Qwen3-4B-Base | v1 | demo | 1, 2 |
| `qwen3-8b-v1-s0` (+ `-cont`) | `q8b-{a..d}`, `q8blong` | Qwen3-8B-Base | v1 | demo | 0 |
| `gemma4-e4b-v1-s0` | `g4e4b-{a..d}` | Gemma-4-E4B (text) | v1 | demo | 0 |
| `qwen3-4b-v2-demo-s1` | `q4v2s1-{a,b}` | Qwen3-4B-Base | v2 | demo | 1 |
| `qwen3-4b-v2-demo-s2` (+ `-cont`) | `q4v2s2-{a..d}`, `q4v2s2long` | Qwen3-4B-Base | v2 | demo | 2 |
| `qwen3-4b-v2-zeroshot-s2` | `q4v2z2-{a..d}` | Qwen3-4B-Base | v2 | zero-shot | 2 |
| `qwen3-4b-v2-zeroshot-s2-cont`, `qwen3-4b-v2-zeroshot-s1` | same as the experiment name | Qwen3-4B-Base | v2 | zero-shot | 2, 1 |
| `gemma4-e4b-v2-zeroshot-s0` (+ `-cont`) | `g4v2z0-{a..d}`, `g4v2z0long` | Gemma-4-E4B (text) | v2 | zero-shot | 0 |
| `gemma4-e2b-v2-zeroshot-s0` | `e2v2z0-{a..d}` | Gemma-4-E2B (text) | v2 | zero-shot | 0 |
| `gemma4-e2b-v2-demo-s0` | `e2v2s0-a` | Gemma-4-E2B (text) | v2 | demo | 0 |
| `gemma4-e2b-v2-demo-s0-noguard` | `e2v2s0n-{a..d}` | Gemma-4-E2B (text) | v2, contradicted guard off | demo | 0 |

**Prompts** (`countdown_rl/rl/prompting.py`, frozen by a hash test):
- `fewshot_direct_retry` (demo phase a): one worked example and the instruction to retry after a failed check.
- `fewshot_multi_retry` (demo phases b–d): the same, but the example checks several candidate expressions in one
  `<python>` block.
- `r1zero_tool` (zero-shot): the R1-Zero conversation frame, an answer-format example, and a plain description of
  the tool. There is no worked example, no strategy and no instruction to retry.

**Recipes:**
- **v1:** adaptive KL (in practice at its 0.001 floor), hinted resampling of all-fail groups in phases c and d,
  and training on groups that contain no correct answer.
- **v2:** KL 0, no hints, only groups with a correct answer are trained, gradient clipping at 1.0, and shuffled
  mini-batches. Everything else is shared: LR 2e-5, LoRA r16/α32, 64 prompts × G 8, two updates per batch with
  clip 0.2/0.28, truncated IS (cap 2), the signal curriculum, the reward and the 768-token budget.

## 1. Before the lineages

- **Model choice:** on 48 stratified problems with the demo prompt, gemma-2-2b solved about 2% and Qwen3-4B-Base
  about 15%. We switched to Qwen3-4B-Base.
- **Small pilots:** 8 prompts × 8 samples, at most 45 steps, on the workstation. No held-out gain. Pilot B exposed the
  reward shortcut described in REPORT ("Reward exploitation"), which led to the contradicted-answer and
  invalid-number rules.

## 2. The hand-tuned v1 lineage (`qwen3-4b-v1-handtuned`)

Built one phase at a time on spot RTX PRO 6000s, choosing each warm start on 3-pass validation:
- **a (`p7`):** 40 steps from the base model with `fewshot_direct_retry`. Validation went from 16% to 35%.
- **b (`p8`):** 90 steps with `fewshot_multi_retry`. Validation went from 37% to 48%. A wider eight-candidate check
  (`p9`) was worse and was stopped at step 50.
- **c (`p10`):** 90 steps with hinted resampling (½ → ¼ → 0 of the solution).
- **d (`p10b`):** a second, shorter hint cycle with a 5-step LR warmup.

After phase d, the hand-tuned lineage scored 528 on holdout1 (P10 479, P8 466) and 52.7% on the test set.

**Hint ablation** (declared before the holdout2 run). This was `qwen3-4b-v1-handtuned-nohint`: phase d without
hints. On holdout2 it scored 518 vs 513 for the hinted run (p = 0.77). A third hint cycle (`p10c`) scored 525. A seed-1
replicate of phase d scored 506 (validation pick, primary) and 539 (final step). We concluded that hints show no
effect; the phase-d gain came from training longer. Late-checkpoint selection on 300 validation problems picked a
checkpoint 33 problems worse than the final step (p = 0.008). From here on every lineage warm-starts from final steps.

## 3. Replays of v1 (registered 2026-11-06, before any result)

**Question.** Does the v1 recipe transfer to another size, another family and other seeds? Hyperparameters are
identical on purpose; only the model and seed change. Seeds 1 and 2 are the unbiased estimate of the recipe, because
the seed-0 lineage was designed on its own outcomes.

**Gemma gate.** Base Gemma-4-E4B validation was 1.6% (three passes). After phase a, the run continues only if
smoothed validation is ≥ 4.6% and first-round train accuracy is rising. The single allowed fallback was hints from
phase a, reported as a deviation. It passed, and no fallback was used.

**Results** (holdout3; final step of phase d):

| | holdout3 | easy / medium / hard |
|---|---|---|
| Qwen3-4B base (demo prompt) | 94 | 71 / 21 / 2 |
| `qwen3-4b-v1-s1` | 512 | 321 / 104 / 87 |
| `qwen3-4b-v1-s2` | 498 | 321 / 101 / 76 |
| `qwen3-4b-v1-handtuned` (selected, for reference) | 521 | 324 / 112 / 85 |
| Qwen3-8B base | 120 | 103 / 12 / 5 |
| `qwen3-8b-v1-s0` | 557 | 350 / 125 / 82 |
| Gemma-4-E4B base | 37 | 26 / 10 / 1 |
| `gemma4-e4b-v1-s0` | 309 | 233 / 57 / 19 |

The recipe is stable across seeds (512 vs 498, p = 0.33). The 8B model gains about 5 pp over the 4B seeds.
**`qwen3-8b-v1-s0-cont`**, 102 more steps, was flat on validation (58.3% → 53.7–59.3%) and was not evaluated on
holdout3.

## 4. Recipe v2 (registered 2026-11-07, before any v2 result)

**Question.** At equal compute, does v2 beat the v1 replays? The design is the same lineage with seeds 1 and 2,
compared with `qwen3-4b-v1-s1/-s2`.
- **Primary outcome:** holdout3 of the phase-d final step, evaluated once, as v2 minus v1 averaged over the two
  seeds. A mean difference under about 20 counts as "no clear difference".

**Seed 1** stopped at phase b step 49 on the registered length guard: more than 40% of rollouts hit the cap over 5
steps. Dropping the all-fail groups had also removed the pressure toward short, well-formed failures, and lengths grew.
Under the registration it counts as a guard stop and was not resumed.

**Seed 2** completed all phases. Its cap share rose to 0.32 in phase b, then fell to 0.06 by the end of d.

| `qwen3-4b-v2-demo-s2`, final step of phase | a | b | c | d |
|---|---|---|---|---|
| 3-pass validation /300 | 123.7 | 177.3 | 203.7 | 213.3 |

**Primary outcome:** **681** on holdout3 (355 / 173 / 153), against 498 for the same-seed v1 replay (55 / 238
discordant, p = 3e-28). After a missed first check, 311 of 619 were recovered (50%), against 13–15% for the v1
replays. Only one seed completed, so the two-seed reading is incomplete.

**Continuation `qwen3-4b-v2-demo-s2-cont`** (registered 2026-11-08). 100 steps from d80. Improving means ≥ +20 with
p < 0.05; plateau means within ±20. It was stopped by hand at step 45 on flat validation (68.0–69.7% across eight
evals) to free the VM; this deviation was recorded before its eval. **Result: 682 vs 681 (82 / 83 discordant,
p = 1.00). Plateau.**

## 5. Zero-shot arm `qwen3-4b-v2-zeroshot-s2` (registered 2026-11-08, before any training result)

**Question.** Is the worked example needed for v2 to learn, or does it only speed learning up?

**Why `r1zero_tool`.** Probe of base Qwen3-4B: 256 training problems × 8 samples at T = 0.9.

| Prompt | problems with ≥ 1 correct | pass@1 | bare-number answers |
|---|---|---|---|
| `zero` (tag skeleton; never says code runs) | 3 / 256 | 0.2% | 38% |
| `r1zero_tool` | 58 / 256 | 4.4% | 1% |

**Design.** Identical to `qwen3-4b-v2-demo-s2` (seed, schedule, hyperparameters, guards) except the prompt, which is
`r1zero_tool` in every phase. $16 cap per phase. A length-guard trip is an outcome, not something to tune around.
- **Gate after phase a:** smoothed validation ≥ 7.3% (base 4.3% + 3 pp) and rising train accuracy.

**Primary outcome.** holdout3 of the final step with `r1zero_tool`, against 681:
- **Gate fails:** "under this recipe the demo is needed to start learning".
- **≥ 661:** "the demo is not needed; the behaviour emerges from a tool description alone".
- **Below 661:** "the demo speeds up or improves RL but is not required".

**Result.**
- **Gate:** passed (36.3%). All 300 steps ran; one VM-lifetime cut at d76 was resumed from the bucket.
- **3-pass validation at phase ends:** 130.7 / 181.7 / 210.3 / 225.7 (demo arm: 123.7 / 177.3 / 203.7 / 213.3).
- **Primary outcome: 699** (351 / 167 / 181), against 681 (109 / 91 discordant, p = 0.23). Verdict: **the demo is
  not needed.** The claim is parity, not superiority.
- **Secondary:** with `fewshot_multi_retry` the same checkpoint scores 309. The policy is tied to its own prompt.
- **Behaviour:** code on 98% of problems. After a missed first check it retries in 74% of cases (demo: 75%) and
  recovers 18% (demo: 50%), but it misses far less often in the first place (230 vs 619).
- **Correction:** an in-run counter, `retry_after_allfalse`, treats any first check without a printed `True` as
  failed. The zero-shot policy prints values rather than booleans, so the counter is not comparable across prompts.
  The figures above use a check against the target instead (`evaluation/metrics.py`).

**Follow-up runs (registered 2026-11-09, after all results above and before either run started).** Both answer
review-style objections to the comparison above; neither can change the primary verdict, which stands as recorded.
- **`qwen3-4b-v2-zeroshot-s2-cont`:** 45 steps from d80, otherwise identical to phase d. The demo arm's continuation
  ran 45 steps (it was stopped there by hand), so this gives both arms the same total of 345 steps. Measure: holdout3
  of step 45 with `r1zero_tool`, paired against d80 (699) and against `qwen3-4b-v2-demo-s2-cont` (682). A gain of
  ≥ 2 pp over d80 reads "still improving"; less reads "plateau", as for the Gemma continuation.
- **`qwen3-4b-v2-zeroshot-s1`, phases a and b only** (130 steps): the zero-shot arm with seed 1, the seed on which
  the v2 demo arm stopped on the length guard at phase-b step 49. Primary: whether it completes phase b without a
  guard stop. Secondary: holdout3 of the phase-b final step, read beside the phase-b final steps of
  `qwen3-4b-v2-zeroshot-s2` and `qwen3-4b-v2-demo-s2` (evaluated alongside on the workstation), and 3-pass
  validation at both phase ends. Phases c and d run only on a separate decision, if budget remains.
- **Budget:** one spot RTX PRO 6000, continuation first, $16 cap per phase; about 7 GPU-hours expected.

**Follow-up results.**
- **`qwen3-4b-v2-zeroshot-s2-cont`:** all 45 steps ran. holdout3 of step 45: 705 against 699 for d80. Under the
  registered rule (≥ 2 pp) this reads **plateau**, as for the demo arm's continuation (682 vs 681).
- **`qwen3-4b-v2-zeroshot-s1`: stopped at phase-b step 13 by the length guard** (cap share above 40% over 5 steps).
  Phase a passed the zero-shot gate easily (validation 5.3% at step 0, 41.0% at step 30), but its cap share rose
  with accuracy, from 0.14 at step 0 to 0.37–0.40 by the end of the phase, and reached 0.40–0.41 early in phase b.
  Seed 2 ran close to the same line (0.40 at the end of phase a, 0.33–0.39 early in phase b) and stayed under it. The
  registered primary question therefore has the answer no: with seed 1, the zero-shot arm trips the length guard as
  the demo arm did, and earlier in phase b (step 13 against 49). The guard threshold sits inside the seed-to-seed
  range of this quantity under v2. Secondary: holdout3 of the last checkpoint (b13) *pending*. Phases c and d were
  not run.

## 6. Gemma-4-E4B zero-shot `gemma4-e4b-v2-zeroshot-s0` (registered 2026-11-09, before any training result)

**Question.** Does the zero-shot finding transfer to another model family? Whether the demo helps Gemma is not asked;
the v1 Gemma run differs in both recipe and prompt.

**Design.** Identical to the Qwen zero-shot arm except the model and seed 0. It ran on a separate project and bucket.
- **Probe:** 8/256 problems with signal; greedy validation 1.3%.
- **Gate:** smoothed validation ≥ 4.3% and rising train accuracy.
- **Operational change before the gate:** micro-batch 8 → 2 after an out-of-memory error at step 14. It does not
  enter the resume hash, and nothing else changed.

**Primary outcome.** holdout3 of the final step with `r1zero_tool`, paired against the base model with the same
prompt, evaluated once alongside.

**Result.**
- **Gate:** passed.
- **3-pass validation at phase ends:** 26.6% / 49.4% / 61.6% / 62.1%.
- **holdout3: 605** (345 / 128 / 132). The v1 run with the demo scored 309 (35 / 331 discordant, p = 1e-61;
  descriptive only, since the recipes differ).
- **Base model with `r1zero_tool` on holdout3: 19** (18 / 1 / 0). Paired against it, the trained policy is
  605 vs 19.
- **Secondary, the same checkpoint with `fewshot_multi_retry`: 427** (239 / 94 / 94). The drop from its own prompt
  (605) is smaller than Qwen's (699 → 309), but the policy is still best under the prompt it was trained with.
- **Behaviour:** code on 35% of problems, and its first printed value never equals the target. It enumerates in
  `<think>` and calls code only when stuck.

**Continuation `gemma4-e4b-v2-zeroshot-s0-cont`** (registered 2026-11-09 while phase d was at step 22). 100 steps
from d80. A gain of ≥ 2 pp on holdout3 means "still improving"; less means "plateau". The share of rollouts that run
code is also reported. All 100 steps ran. 3-pass validation of step 100: 69.8% (211, 207 and 210 of 300), against
62.1% for d80. **Result: 681 vs 605 on holdout3 (117 / 41 discordant, p ≈ 1e-9; 358 / 158 / 165 by tier). Still
improving.** Code now runs on 999 of 1,000 problems, against 345 at d80 (1.12 vs 0.52 executions per problem).

## 7. Gemma-4-E2B, phase a only (registered 2026-11-09, before any training result)

**Probes** (base models, `r1zero_tool`, 256 × 8): Gemma-4-E2B 1/256 with signal (greedy validation 0%); gemma-2-2b
0/256 (0%). gemma-2-2b was not trained.

**`gemma4-e2b-v2-zeroshot-s0`:** the Gemma-4-E4B phase-a config with the E2B model.
- **Gate:** smoothed validation ≥ 3.0% and rising train accuracy.
  - Fail: "zero-shot RL does not start on E2B under v2; the size threshold lies between E2B and E4B".
  - Pass: "starts". Nothing beyond phase a runs without a separate decision.
- **Change at step 8:** extended from 40 to 50 steps on request. `total_steps` does not affect the LR schedule, so
  this is exactly a 50-step run.

**`gemma4-e2b-v2-demo-s0`:** the matched pair. It is identical except `fewshot_direct_retry`, and its gate is its own
step-0 validation + 3 pp. Question: does the 2B model fail only zero-shot, or with the demo too?

**Results.**
- **`gemma4-e2b-v2-zeroshot-s0`: gate passed.** Smoothed validation 5.9% against 3.0% needed, and first-round
  training accuracy rose from 0.000 to 0.095. In-run validation by step: 0.3, 0.0, 0.3, 1.0, 0.3, 2.0, 3.7, 5.7, 6.0
  and 6.0%. 3-pass validation of step 50: 6.0%. Signal groups were 0–2 per step until about step 20, and up to 28
  afterwards. Reading: "starts". No later phases were run.
- **`gemma4-e2b-v2-demo-s0`: stopped at step 5 by the contradicted-answer guard** (more than 60% over 5 steps).
  Contradicted answers were 0.78, 0.79, 0.77, 0.76 and 0.76 from step 0. Training accuracy rose from 0.6% to
  4.7% and signal groups ran 13–41 per step; validation at step 0 was 1.3%. The guard was set for recipe v1, where
  contradicted answers earned shaping reward. Under v2 they get no gradient, so here the trip records the base
  model's behaviour with the demo prompt rather than a learned exploit. Under the registration, a guard stop is an
  outcome; the run was not resumed. The gate comparison is therefore incomplete.

**Follow-up runs (registered 2026-11-09, after the results above and before either run started).** Both were decided
after seeing the phase-a outcomes, at the user's request, and are reported as such.
- **`gemma4-e2b-v2-demo-s0-noguard`** (bucket `e2v2s0n-a`): identical to `gemma4-e2b-v2-demo-s0` except
  `guard_contradicted: 1.0`, which disables that guard. The length guard stays on. The reason is that under v2 a
  contradicted answer earns no gradient, so the guard protects nothing in this recipe. Same gate as before: smoothed
  validation ≥ step-0 validation + 3 pp, with rising training accuracy. The primary reading is the gate verdict,
  read beside the zero-shot arm's at step 50. A length-guard stop is an outcome.
- **`gemma4-e2b-v2-zeroshot-s0`, phases b–d** (bucket `e2v2z0-b`, `-c`, `-d`): the standard schedule from phase a's
  final step (step 50). The configs are those of `gemma4-e4b-v2-zeroshot-s0` with the E2B model, so the lineage is
  310 steps instead of 300. The primary measure is holdout3 of phase d's final step, evaluated once on the
  workstation with `r1zero_tool` and eager attention. It is reported as the 2B point beside Gemma-4-E4B (605) and
  Qwen3-4B (699), with no decision threshold. Secondary: 3-pass validation at each phase end, the share of rollouts
  that run code, and the cap share.
- **Order and budget:** both run on `tp-rtx6000`, the demo arm first, under a $16 cap per phase and a 9-hour VM
  lifetime. A failed gate on the demo arm does not stop the zero-shot phases.

**Follow-up results.**
- **`gemma4-e2b-v2-demo-s0-noguard`, phase a: gate passed.** Smoothed validation 11.4% against 4.0% needed (step 0:
  1.0%), and first-round training accuracy rose from 0.045 to 0.263. In-run validation by step: 1.0, 4.7, 6.0, 6.0,
  6.0, 10.0, 10.7, 11.0, 11.7 and 11.7%. 3-pass validation of step 50: 13.1% (39, 40 and 39 of 300). Contradicted answers fell from 0.75
  at step 0. Signal groups stayed at 6–43 per step through step 50, while the zero-shot arm ended phase a with 3.
- **`gemma4-e2b-v2-zeroshot-s0`, phases b–d: no further learning.** All 260 steps ran. In-run validation stayed at
  exactly 6.0% (identical per-tier counts) throughout, and 3-pass validation at the end of b, c and d was 18/300 every
  time, the same as the end of phase a. holdout3 of d80: 44 (36 / 5 / 3). Sampling entropy had fallen from 0.77 to 0.10 by
  the end of phase a, and 94% of prompts now give eight identical rollouts, so most steps have no signal group and no
  update. This was first recorded at phase-b step 25, before the phase finished.

**Further follow-up (registered 2026-11-09, after the results above and before the run started), at the user's
request: `gemma4-e2b-v2-demo-s0-noguard`, phases b–d** (bucket `e2v2s0n-b`, `-c`, `-d`). The standard schedule from
phase a's final step, with `fewshot_multi_retry` in every phase as in all demo lineages, and the contradicted guard
still off. The configs are the E2B zero-shot phase-b–d configs with only the prompt and that guard changed, so the
two E2B arms differ in the prompt alone. It runs on `tp-rtx6000` after the zero-shot lineage, under a $16 cap per
phase.
- **Primary:** holdout3 of phase d's final step, evaluated once on the workstation with `fewshot_multi_retry` and
  eager attention, paired against the zero-shot arm's final step. This is descriptive, with no decision threshold:
  one seed per arm, and the arms were extended after their phase-a results were seen.
- **Secondary:** 3-pass validation at each phase end, signal groups per step, sampling entropy, and the cap share.
  A length-guard stop is an outcome.

**Result: `gemma4-e2b-v2-demo-s0-noguard` phase b stopped at step 32 on the length guard** (cap share above 40%
over 5 steps). It is a registered outcome; phases c and d did not run, and the run was not resumed.
- In-run validation with `fewshot_multi_retry` went from 6.0% at step 0 (phase a ended at 13.1% under
  `fewshot_direct_retry`, so the prompt change costs about half) to 16.7% at step 30.
- The cap share rose steadily from 0.02 at step 0 to 0.42–0.49 over steps 28–31, and the length penalty from 0.004 to
  0.12. Contradicted answers fell from 0.88 to 0.24. This is the failure that stopped `qwen3-4b-v2-demo-s1` in its
  phase b: with no gradient on all-fail groups, nothing pushes failed rollouts to stay short.
- The registered primary (holdout3 of d80) does not exist. As a post hoc, descriptive number, the step-32 checkpoint
  is evaluated once on holdout3 with `fewshot_multi_retry`: *pending*.

## 8. Post hoc analyses of the v2 policies (listed 2026-11-09, before any of them ran)

None of these was planned before the v2 results; all are descriptive and none changes a recorded verdict. They answer
three objections a reader is likely to raise: that RL only sharpens what the base model already samples, that the
policies do not really use the tool, and that the zero-shot result depends on the exact wording of the prompt.
Evaluations run on the workstation and on a second local GPU (RTX A4000, CUDA, same code and lock file).
- **pass@k on the test set:** T = 0.9, seeds 1–8, for base Qwen3-4B with `r1zero_tool`, `qwen3-4b-v2-zeroshot-s2`
  and `qwen3-4b-v2-demo-s2`, with the settings of the v1 runs in `results/evals/test_time/`. Reported as unbiased
  pass@k for k = 1, 2, 4, 8.
- **Tool dependence on the test set (greedy):** the 0/1/2-round sweep and the blanked and shuffled `<output>`
  controls for `qwen3-4b-v2-zeroshot-s2`, `qwen3-4b-v2-demo-s2` and `gemma4-e4b-v2-zeroshot-s0-cont`, as was done
  for v1.
- **Prompt wording (holdout3):** `qwen3-4b-v2-zeroshot-s2` under two rewordings of `r1zero_tool`
  (`r1zero_tool_p1`, `_p2` in `rl/rollout.py`) that carry the same information.
- **Token budget (holdout3):** both Qwen v2 policies with 1,536 new tokens instead of 768.
- **Seed comparison points (holdout3):** the phase-b final steps of `qwen3-4b-v2-zeroshot-s2` and
  `qwen3-4b-v2-demo-s2`, to set beside the seed-1 zero-shot run of section 5.
- **Uncertainty:** paired bootstrap intervals (10,000 resamples) for every holdout3 difference in REPORT, alongside
  the McNemar tests.

## 9. Known gaps

- **Lost log steps:** resuming on a fresh VM used to overwrite the start of `train_log.jsonl` (fixed in `rl/grpo.py`
  and `scripts/cloud/train.sh`). Lost steps:
  - `qwen3-4b-v2-demo-s2` phase b, 0–29, and phase c, 0–11;
  - `qwen3-4b-v2-demo-s1` phase b, 0–13;
  - `qwen3-4b-v2-zeroshot-s2` phase d, 74–75.

  Checkpoints, validation passes and evaluations are unaffected.
- **Phase a logs of the hand-tuned lineage:** the per-step log of phase a was not kept.
- **Test set reuse:** the 1,000-problem test set informed early v1 decisions and has been run on many checkpoints.
  Use holdout3 for comparisons.

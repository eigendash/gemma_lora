# Experiment log: Countdown tool-use GRPO (gemma-lora)

Last updated: 2026-11-08 (§2.7p v2 seed 2 result, §2.7q Qwen3-8B continuation). The §1 summary table below dates from 2026-11-04; REPORT.md has the current picture. This covers everything tried so far: what changed, what worked and what didn't.
Detailed evidence is in `VERIFICATION.md` (sections are cited as §N) and in `verification/logs/`.

**Task:** given 3–6 numbers and a target, the model writes `<think>` → optional `<python>` (run in a
sandbox, result injected as `<output>`, at most 2 rounds) → `<answer>expr</answer>`.
**Reward** (SPEC.md): correct 1.0 + exec 0.1 + format 0.1 − length penalty.
**Method:** LoRA (r=16), GRPO; optionally an SFT cold start first.

---

## 1. Summary

| Result | Status |
|---|---|
| Pipeline correctness: rollout ↔ training-token alignment, log-probs, KL, curriculum, resume, sandbox | ✅ Fixed and verified (26/26 checks; 154 tests pass, 4 skipped) |
| Batched generation | ✅ 7.4× faster than sequential, identical metrics |
| Switching base model from gemma-2-2b to **Qwen3-4B-Base** | ✅ ~15% vs ~2% Pass@1 out of the box |
| A one-line "retry after a failed check" instruction in the prompt | ✅ Base model retries 31 of 43 failed checks (0 of 43 without it) |
| SFT cold start (programmatic traces, both trace versions) | ❌ Did not beat the base model (either model) |
| GRPO pilots A and B (8 prompts × 8 samples, ~10–45 steps) | ❌ No held-out gain (flat at 5–7/48); B learned a reward shortcut |
| Reward fix for that shortcut | ✅ Applied; checked on real rollouts; not yet proven in training |
| Literature-aligned training settings (C → C′ → C″) | ⏳ C″ running on soos since 12:47 (soos time) |

Short version: the infrastructure is now correct and fast enough, and Qwen3-4B-Base is the right model.
No training run has improved held-out accuracy yet. The runs so far were small and short, and the
one that ran longest exposed a reward-shaping exploit, which is now fixed.

---

## 2. Timeline of what was tried

### 2.1 Pipeline audit and fixes (Rounds 1–3, Mac)
An independent audit found 6 blocking and 10 major issues. All fixes (F10–F19) were approved and applied (§0, §2, §10). The main ones:
- **Stop token not fed:** training ids ≠ generation context after `<output>`. Ratio was 0.68–2.04. Fixed; the max |Δlogp| at step 0 is now 2.7e-5.
- **Gemma-2 final-logit softcap missing in the chunked log-probs:** log-probs were up to 12 nats wrong. Fixed (matches HF to 1.6e-6).
- **Self-correction metric:** it was structurally always 0. Fixed.
- **KL estimator:** was k1 (no gradient on-policy, could go negative). Switched to k3.
- **Curriculum:** it oversampled unsolvable tiers. The weight is now P(group has reward variance).
- **Resume:** it wasn't faithful (SFT dropped, RNG not restored). Fixed; 4/4 steps now identical to an uninterrupted run.
- **Sandbox workers:** they loaded torch (360 MB each). They are now isolated `python -I -S` subprocesses (13 MB).
- **Reward:** 20 hand-written reward hacks now all score below honest baselines.
- Plus cloud-handoff, bootstrap, checkpoint-commit and cost-guard fixes. These are verified against a local bucket only; real GCS was never run.

### 2.2 SFT cold start
| Attempt | Result |
|---|---|
| Original traces (wrong guess → "the right expression is X" with no derivation) | ❌ Model learned to answer with the expression its own check had just shown `False` (gemma 0/12) |
| v3 traces: 4 kinds (direct, near-miss + operator sweep, split, re-split), grounded in the real printed output, never answer after a miss | Qwen2.5-0.5B: 0/24 → 4/24 (small n, direction only) |
| v3 traces on gemma-2-2b (soos) | ❌ 0/48 vs base 1/48. Learned the template; 37/48 hit the length cap |
| v3 traces on Qwen3-4B-Base (soos) | ❌ 5/48 vs base 7/48. 40/48 hit the length cap: after the 2-round budget it keeps sweeping and invents values |
| Gemma SFT on the Mac | ❌ Swap thrash, then macOS OOM-killed it at step ~220. Added SFT checkpointing + resume, then moved to soos |

**Conclusion:** the programmatic SFT traces don't teach what to do when the tool budget runs out.
SFT was set aside, and GRPO now starts from the **base** model with a 2-shot prompt (ToRL / R1-Zero style).

### 2.3 Model choice (soos, 48 stratified problems, greedy)
| Model | Pass@1 | Format |
|---|---|---|
| gemma-2-2b base, 2-shot | 1/48 | 1.00 |
| gemma-2-2b old-trace SFT | 2/48 | 1.00 (an earlier n=12 claim that it was *worse* was retracted) |
| **Qwen3-4B-Base, 2-shot** | **7/48** | **0.98** |

Problem found and fixed along the way: Qwen3 registers `<think>`/`</think>` as special added tokens that the base model never
emits, so the tags degenerated (format 0). `rollout.encode_text` now keeps protocol tags as ordinary tokens.

### 2.4 Infrastructure on soos (GMKtec, Ryzen AI Max+ 395, ROCm 7.2, 108 GB GPU-addressable)
- **Batched generation** (left padding, per-sequence pause at tool calls) became the default: 11 m 59 s → 1 m 37 s.
- **OOM with the `llama` container running** (it holds 62 GB). Added gradient checkpointing and micro-batches of 4–8. `llama` stays
  stopped during experiments (restart with `docker start llama`).
- **Temperature 1.0 collapsed base accuracy.** Moved to T=0.7–0.9, with log-probs computed at the same temperature.
- **Round cap:** training used to truncate at it. Added `on_round_cap: notice` (the model is told it is out of rounds and can still answer).

### 2.5 GRPO pilots (Qwen3-4B-Base, LoRA r=16, from base, 2-shot prompt)
Held-out = 48 stratified eval problems, greedy (n=200 from C″ on).

| Run | Setup | Steps | Held-out (correct/48) | What happened |
|---|---|---|---|---|
| **A** (P1) | format-only prompt, 8×8, T=0.7, KL on | 45 | 6, 5, 4, 4, 4, 6, 6, 6 | Training accuracy rose 4% → 30%, mostly from the curriculum shifting to easy problems. KL grew to 0.07, ~50% of groups had zero variance. **No held-out gain.** It never retried after a failed check. |
| **B** (P2) | A + retry instruction | 11 | 7, 6, 7, 7, 6, 5 | Retries after a failed check: 31 → 33 → 23 → **0** of ~43 by step 9. **It learned to submit answers its own check printed `False` for** (§12.1). ~9.5 min/step while sharing the GPU. |
| **C** (P3) | "literature-aligned": 32×8, T=0.9, no KL, DAPO dynamic sampling | 1 | 7, 7 | Too slow: 42 min/step (256 rollouts at ~21 tok/s while sharing the GPU). Stopped. |
| **C′** (P4) | C with 16 prompts/step + the contradicted-answer fix | 0 | 7 (step 0) | Superseded after ~15 min by C″ |
| **C″** (P5) | C′ + 2 updates/batch, clip 0.2/0.28, TIS cap 2, 768 tokens (penalty from 512) | 1 | base 21/200 | 31 min/step; stopped to fix speed (§2.6) |
| **D** (P6) | C″ + `sdpa_gqa` + merged-weight generation + shortcut auto-stop guards, no grad ckpt | running | n=200, paired | **Go/no-go run.** ~5 min/step. Started 15:25 soos time, 2026-11-04 |

### 2.6 Generation speed (VERIFICATION.md §13)
| Change | Result |
|---|---|
| Continuous-batching engine (slot refill, teacher-forced tool output) | Token-identical to the reference; **no faster** than lockstep here (187 vs 190 s) |
| "Compact" lockstep (drop finished rows) | ❌ 2× slower (380 s) |
| ROCm env vars from the Strix Halo vLLM repo | ❌ no gain |
| vLLM on gfx1151 | Not pursued (community builds need eager mode, 1 sequence) |
| **Copy-free GQA decode attention (`sdpa_gqa`)** | ✅ The real cause: HF copied the KV cache 4× per layer per step under a padding mask (py-spy: 43% of wall time). 191 → 107 s |
| **Merged-weight generation copy (`gen_merged`)** | ✅ Removes PEFT overhead: 156 → 107 s inside the trainer |
| No gradient checkpointing (micro-batch 8) | ✅ Loss pass 257 → 205 s (micro-batch 16+ OOMs) |
| **Net** | **~31 → ~5.2 min per 128-rollout step** |

My first diagnosis (lockstep batching) was wrong. Measurement and profiling found the real cause.

### 2.7 Pre-cloud readiness
- **Eval protocol frozen:** 200 stratified problems, greedy, 768 tokens, `fewshot_direct_retry`. Base Qwen3-4B = 21/200 (10.5%; easy 22%, medium 4.5%, hard 4.5%). Checkpoints are compared to step 0 with a paired McNemar test (`verification/eval_compare.py`).
  Numerical noise floor: the same base model with stock SDPA vs `sdpa_gqa` scores 21 vs 24/200 (5 problems flip, p=0.38),
  so a few-problem change between checkpoints is not evidence. P6 is compared against its own step 0 (24/200).
- **Shortcut monitoring:** `contradicted`, `retry_after_false`, `frac_max_length` logged per step. Auto-stop if `contradicted` > 0.6 or `frac_max_length` > 0.4 over 5 steps (saves a checkpoint and writes `GUARD_STOP`).
- **vLLM generation backend for CUDA** (`rollout_vllm.py`, `generation: vllm`). Equivalence-tested with a fake engine; real vLLM is untested (needs a CUDA GPU).
- **Cloud smoke test** (`scripts/cloud_smoke.sh`, `configs/smoke_cloud_qwen3.yaml`): vLLM generation, GCS commit, kill -9 + resume, cost-guard trip, throughput. **Blocked:** GCP project GPU quota is 0 in all regions, and the AWS CLI session has expired.

### 2.7b Cloud smoke test (GCP, 1× L4 spot, us-east4-a, 2026-11-05) — PASS
On-demand L4s were sold out in 13 zones across 7 regions, so this ran on a spot L4. The project's GPU quota is 1 globally.
Config: `configs/smoke_cloud_qwen3_l4.yaml`. On 24 GB, vLLM gets 0.5 of memory, leaving a 1.48 GB KV cache, so only about 5
sequences run at once. Logs: `verification/logs/cloud_smoke_l4/`.

| Check | Result |
|---|---|
| Unit tests on CUDA | 19 passed, 4 skipped |
| vLLM vs trainer log-probs | mean 0.009–0.011, p99 0.18–0.20, TIS weight ≈ 1.000 (soos HF: 0.0107 / 0.19) |
| GCS commit, then kill -9, then resume on a wiped disk | restored step 2, then ran steps 2–5 to completion |
| Cost guard (`--max-total-usd 0.0001`) | tripped and saved a checkpoint at step 1 |
| Throughput | ~103 s per 64-rollout step (~35 steps/h); gen 28–81 s, loss 43–61 s |

**Harness bug found and fixed:** `kill -9 <trainer pid>` left vLLM's `EngineCore` child alive, holding 11 GB, so the resumed
run could not start vLLM. The script now starts the trainer under `setsid` and kills the whole process group. Production
restarts on a new VM, so it is not affected. Any same-box restart must kill the process group.
The built venv is cached at `gs://countdown-rl-16875-ckpt/envs/` (3.7 GB) and restored by `scripts/cloud_restore_env.sh`.

### 2.7c RTX PRO 6000 (GCP g4-standard-48, spot, us-central1-b): throughput, then P7 (2026-11-05)
Throughput of the P6 recipe (16×8, 768 tokens, U=2) with vLLM; logs in `verification/logs/cloud_tp_rtx6000/`:

| Variant | Step time | Notes |
|---|---|---|
| vLLM 0.3 + trainer micro-batch 8, no grad ckpt | OOM | trainer 66 GB plus vLLM 29 GB exceeds 95 GB |
| vLLM 0.3 + grad ckpt | 48 s (gen 19–22, loss 25–30) | about 75 steps/h |
| **vLLM sleep mode (0.6) + no grad ckpt** | **39 s** (gen 16–23, loss 18–21) | about 93 steps/h, **7–8× soos**. The first try OOMed on `wake_up` until the trainer's CUDA cache was emptied before waking. |

The vLLM/trainer log-prob gap stays at 0.010 (p99 0.18) throughout.

**Validation baseline**, from `verification/eval_vllm.py` on the 300-problem split (100 per tier, held out of train.jsonl by
`split_validation`), greedy:

| Model | Correct | Easy / medium / hard |
|---|---|---|
| base | 46/300 (15.3%) | .30 / .13 / .03 |
| P6 step 46 | 96 | .65 / .19 / .12 |
| P6 step 66 | 96 | .62 / .23 / .11 |
| P6 step 99 | 100 (33.3%) | .60 / .27 / .13 |

Caveat: P6 sampled from all of train.jsonl, so it may have seen about 25% of these problems. That favours P6.

**P7** (`configs/p7_rtx6000.yaml`): 64×8 rollouts per step, adaptive KL (target 0.1), `curriculum_mode: signal`,
`zero_shaping_invalid_literals`, in-loop validation every 5 steps, and vLLM sleep mode. About 2 min per step (gen ≈ 28 s, loss ≈ 95 s).
Its step-0 validation (48/300) matches the standalone base eval (46/300).
Target: reach P6's best (100/300) by step ~30 and beat it significantly by step ~60. Otherwise switch experiments.
Startup failures fixed along the way: `VLLM_USE_FLASHINFER_SAMPLER=0`, because after a VM restart vLLM picked the FlashInfer sampler, which needs
`ninja`; and a missing import in `run_validation`, which now has an end-to-end unit test.

### 2.7d P7 result: same ceiling as P6, reached about 3× faster in steps. Diagnosis, then P8
P7 validation (300): 48 (base) → 62 (step 10) → 77 (step 20) → 98 (step 30) → 97 / **104** / 102 (steps 35 / 40 / 45).
That is level with P6's best (100; p ≥ 0.68) and flat for 3 checks, so the stopping rule fired and I stopped P7 at step 45.
Health stayed clean throughout:
- `contradicted` fell to about 0.03 and `retry_after_false` reached 1.0;
- `frac_max_length` was about 0 and KL to base about 0.06;
- `invalid_literals` rose to 0.07 early, then fell to about 0.01 once it earned no shaping.

**Why both runs stop at about 33%.** Failures on P7 step-45 validation, by tier:

| Tier | Correct | Wrong | ...of which both checks False, then an unchecked 3rd guess |
|---|---|---|---|
| easy | 70 | 30 | 30 |
| medium | 20 | 80 | 78 |
| hard | 12 | 88 | 86 |

With one candidate per execution and SPEC's 2-execution cap, the policy gets about 3 guesses, and medium and hard need more. P6's
longer run agrees: held-out means were 66.5 (steps 46–69), 65.2 (70–99), 62.8 (100–129) and 61.2 (130–141). Training longer does not help.

**P8 (`configs/p8_rtx6000.yaml`):** the new prompt style `fewshot_multi_retry` demonstrates verifying several candidates in ONE
`<python>` block (`for res in (e1, e2, ...): print(res == T, res)`; the sandbox allows `for` over a tuple, and output stays
under the 200-character cap). Two executions can then check about 10 candidates. Validation under each prompt:

| Model | Old prompt | Multi prompt |
|---|---|---|
| base | 46 | 32 |
| P6 step 99 | 100 | 94 |
| P7 step 40 | 102 | 104 |

P7 step 40 under the multi prompt:
- it uses the loop in 300/300 rollouts, but with only about 2.9 candidates per check;
- in 75/300 rollouts it wastes an execution with `for res in (expr):` on a single candidate, which raises a TypeError.

P8 starts from P7 step 40 (`init_adapter`, merged; it is also the KL reference) with the multi prompt. It keeps the rest of the P7 recipe, for 100 steps.
Tooling: `eval_vllm.py` now refuses empty or missing adapter paths. It had silently evaluated the base model once (44/300, discarded).

### 2.7e P8: the multi-candidate check breaks the ceiling (in progress)
P8 is `configs/p8_rtx6000.yaml`, warm-started from the UNMERGED P7 s40 LoRA (`init_lora`; a bf16 merge had cost 104 → 92) with micro-batch 4
(8 OOMed at 93 GB). Validation (300): 111 (step 0) → 122 (10) → 118 (20) → 115 (30) → 119 (35) → **126 (40)**.

Paired at step 40:
- vs its own start: +15 (16 only-start / 31 only-new), p = 0.04;
- vs P7 s40 on the old prompt (102): +24 (12 / 36), p = 0.001;
- vs P6 best (100): +26 (17 / 43), p = 0.001.

By tier, P6 → P8 s40 is easy 60 → 80, medium 27 → 32, hard 13 → 14.
RL fixed the single-candidate `for` bug: rollouts with an execution error went 76 → 5 (step 20) → 0 (step 40). Candidates per check
stayed at about 2.9, while use of both executions rose (104 → 178), so the gain is in candidate quality, not count.
Health: `contradicted` 0.09 → 0.06, `invalid_literals` about 0.02, `frac_max_length` about 0, KL about 0.08, `clip_frac` about 0.003.
Step time is about 2.2 min (gen 32 s, loss 100 s at micro-batch 4).

Later validation: 124 (45) → 134 (50) → 134 (55) → 133 (60) → **138 (65)** → 135 (70). The run is still climbing slowly, with hard 13 → 19 and medium 30 → 40.
Paired at step 65:
- vs P7 s40 (102): +36 (10 / 46), p < 0.001;
- vs P6 best (100): +38 (17 / 55), p < 0.001.

Step 70 vs its own start: 111 → 135 (23 / 47), p = 0.006. KL at step 69 was 0.09.

**Final:** the run ended cleanly at step 100 (10:03 UTC, one attempt).
Validation: 133 (75) → 134 (80) → 139 (85) → **148 (90)** → 145 (95). Step 100 has no validation (the run ended before it).
- **Best by validation:** step 90 (`baselines/p8_s90`, also in `gs://…/baselines/`), selected *before* any test-set evaluation.
- **Step 90 vs its own start:** 111 → 148 (17 / 54), p < 0.001. By tier: easy 75 → 81, medium 27 → 45, hard 9 → 22.
- **Health at steps 70–99:** KL 0.09–0.10, sitting on target (0.1), with `kl_coef` at its floor of 0.001. `contradicted` 0.04–0.07, `invalid_literals` 0.006–0.023, `frac_max_length` 0, `clip_frac` 0.003.
- **Watch item:** `invalid_literals` doubled over the last 25 steps (0.010 → 0.023). It's small, but it's the metric to watch if training continues.

Logs and validation dumps are in `verification/logs/p8_rtx6000/`.

**Extension, steps 100–134** (`configs/p8ext_rtx6000.yaml`, the same config hash, `--resume auto`):
- **Validation:** 139 (100) → 145 (105) → 139 (110) → 134 (115) → 136 (120) → 135 (125) → 141 (130).
- **No checkpoint beat s90 (148), so P8 has plateaued at about 135–148/300.** The s90 peak is probably partly noise on n = 300 (±8.5 at 1 SD); the plateau level is about 46%.
- **Health stayed flat:** KL 0.085–0.106, `contradicted` 0.05–0.06, `invalid_literals` 0.004–0.02, `frac_max_length` 0.
- **The test result stands:** s90 stays the selected checkpoint, and nothing new was run on the test set.

Logs are in `verification/logs/p8_rtx6000_ext/`.

### 2.7g P9: a wider check (8 candidates per execution), from P8 s90 — NEGATIVE, stopped at step 50
**Diagnosis** (P8 s90 *validation* dump only; the test set was not used for this):
- **152 failures out of 300.** 145 of them follow one pattern: every check prints False, all executions are used, and an unchecked guess is given.
- **Too few candidates per check:** failures checked 6.2 candidates per problem (P8 tries a 3rd check that hits the cap), so only about 3 per execution. The value feedback (`False 196`) is printed but not used to steer.

**Change:** a new prompt style, `fewshot_wide_retry` (`sft_coldstart.WIDE_DEMO` + `rollout.WIDE_HINT`).
- The demo shows 8 candidates in the first check, all of which miss. It then names the 2 closest printed values and shows 8 new candidates built from them, 2 of which print True.
- Every demo candidate obeys the number rule; this is asserted in `generate_trace` and in a test. The first draft had 5 candidates that reused a number and answered with one of them; the assert now prevents that.
- stdout stays under the 200-character cap.

**Config:** `configs/p9_rtx6000.yaml`, which is P8 with exactly 2 lines changed: `init_lora: baselines/p8_s90` and `prompt_style: fewshot_wide_retry`.

**Infra note:** the first RTX VM was spot-preempted 90 s after a stop/start (12:07 UTC). With termination action DELETE, its disk went too. Nothing was lost, because every adapter (md5-verified), P8's full checkpoints, the logs and the test dumps were in the bucket first. The VM was recreated in the same zone.

**P9 step 0** (P8 s90 under the wide prompt): **133/300** vs 148 under its own prompt, paired 42 / 27, p = 0.09 (not significant).
The behaviour changed immediately: candidates per execution went 3.5 → 8.0, and per problem 5.5 → 13.2.

**Decision rule (revised 2026-11-05, user):** P9 runs its full 100 steps, with no early stop and no automatic P10.
The P8-vs-P9 choice is made on validation, then the test set is run once. Both the 1,000-problem and never-seen-800 numbers are reported (VERIFICATION §16a).

**Preemptions:** VM #2 was preempted at 12:24 UTC, 12 minutes in and before P9's first checkpoint (step 5 / 15 min), so P9 restarted from scratch.
P9 and P10 now checkpoint every 2 steps or 300 s. Checkpoint frequency is not part of the config hash, so P9's hash (`45fe4337cd30`) is unchanged.

**Result (VM #3, run `p9-1793880957`; logs in `verification/logs/p9_rtx6000/`): validation fell, and P9 was stopped at step 50 (15:37 UTC).**

| step | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val /300 | 131 | 125 | 128 | 121 | 122 | 117 | 112 | 118 | 124 | 113 | 122 |
| hard /100 | 22 | 17 | 19 | 17 | 13 | 12 | 13 | 13 | 15 | 12 | 14 |

- Easy went up slightly (70 → 80), while medium (39 → 28) and hard (22 → 14) fell. No checkpoint beat step 0.
- **P9 s50 vs P8 s90 on validation:** 122 vs 148; only P8 solves 42, only P9 solves 16; **p = 0.0009**. P9 is significantly worse.
- **Training metrics rule out the obvious causes:**
  - truncation: `frac_max_length` 0.047 → 0.000;
  - KL: 0.09–0.10, with `kl_coef` at 0.01;
  - signal groups: 62–64 per step.
  - Training-batch `correct` drifted from 0.455 to about 0.41.
- **Reading:** checking more candidates per execution, with fewer execution rounds left over, does not help. On medium/hard problems, refining over 2 rounds of 8 is worse than P8's 3–4 rounds of about 4. The bottleneck is the quality of candidates, not how many are checked.
- **Why stop at step 50, against the earlier "full 100 steps" plan:**
  - 9 straight checks sat 9–19 below step 0.
  - The remaining 50 steps (about 3 h) would have needed a second VM, because the 6 h cap ends at 18:12.
  - P10 was waiting for the single GPU.
- Adapters `p9_s6`…`p9_s50` and the full checkpoints are in the bucket. **P8 s90 stays the candidate.**

**Validation noise (measured 2026-11-05).** "Greedy" vLLM validation is not reproducible across runs: batched LoRA kernels are numerically nondeterministic, so long completions drift apart.
- P10 step 0 is the same adapter (P8 s90) with the same prompt. It scored 141/300 vs P8's own 148. Only 138/300 completions were identical; discordant problems were 14 / 7 (p ≈ 0.19).
- P9 step 0 evaluated on two VMs: 133 vs 131, with only 62/300 completions identical.
- Every divergence starts mid-completion (earliest at character 55), never at the prompt. So it is numerics, not a config or prompt difference.
- **Consequence:**
  - One validation pass has noise of roughly ±5–7 / 300.
  - Picking the best of many checkpoints on single passes is biased upward (winner's curse). P8 s90's 148 was probably partly a lucky draw; its test score of 47.8% is the unbiased number.
  - Future checkpoint choices should average 2–3 validation passes or use paired tests, and still use the test set once.

### 2.7h KL-reference reset: implemented, NOT run (KL penalty negligible in our setup)
**Terminology.** The P7 → P8 → P9 process is *multi-stage RL with warm starts*: each stage starts from the previous stage's validation-best checkpoint and changes a few settings.
It is closest to NVIDIA's ProRL (prolonged RL with changes at stage boundaries), and as a research process it is the "babysitting" approach, a manual single-member form of Population Based Training.
One ProRL ingredient we had **not** used: ProRL also *resets the KL reference* to the current policy at each stage boundary. Up to P9 our reference was always the plain base model.

**Draft config (not run, superseded):** P8 with the reference reset (`configs/kl_reset_draft_rtx6000.yaml`). Exactly two settings change:
- `init_lora: baselines/p8_s90`;
- **`kl_ref: init`**: the reference is a frozen copy of P8 s90 itself, so KL restarts at 0.

The prompt (`fewshot_multi_retry`), batch, LR, adaptive KL, curriculum and reward are unchanged.
The comparison is P8's own extension (s100–s135, base reference), which plateaued at 134–145/300.

**Caveat, measured before running it.** In P8 the adaptive controller kept `kl_coef` at or near its floor for the whole run: 0.001 from about step 15, 0.002 in the extension, with KL 0.06–0.10 against a target of 0.1.
The KL term was therefore about 0.002 × 0.1 per token, negligible next to the policy-gradient term. So in our setup the base-model anchor was **not** what held P8 back.
A reset alone is expected to change little: it is then mostly "continue P8 with a fresh optimizer", which is still a useful, cheap control.
For a reset to matter, the KL has to bind, which is the regime ProRL ran in. A follow-up variant could pair the reset with a fixed, binding `kl_coef`.

**Code** (`train_rl.py`):
- `kl_ref` is a new config field. While it is at its default `base`, it is left out of the config hash, so P8's and P9's hashes and checkpoints are unchanged.
- `kl_ref: init` loads `init_lora` a second time as a frozen adapter named `ref`. Fresh runs and resumes both do this.
- `_reference()` makes that adapter the active one for the reference forward by setting each LoRA layer's `_active_adapter` directly. PEFT's `set_adapter` would also flip `requires_grad` on the trainable adapter in the middle of its autograd graph.
- Checkpoints and vLLM exports save only the trainable adapter (`selected_adapters=["default"]`).
- Test `test_kl_ref_init_is_frozen_warm_start_and_hash_stable` checks that the reference starts equal to the policy (KL 0) and differs from the base, gradients reach the policy, the reference stays frozen after a step, `ref` isn't saved, and the hash is stable. The suite passes: 219 passed, 10 skipped.

### 2.7i P10 candidates from the literature (proposal, 2026-11-05)
**Where P8 stands:**
- Validation plateaued at about 46%, with hard problems at 20%.
- 145 of the 152 failures follow one pattern: every check False, then an unchecked guess.
- The KL penalty was negligible: `kl_coef` stayed at its floor, so the base-model anchor was not the limit (§2.7h).
- About 59 of 64 groups per step are informative. But DAPO dynamic sampling **discards** every all-fail group and refills with fresh prompts, so the hard problems P8 can't solve yet never produce a gradient. This is the "zero-reinforcement" bottleneck below.

| Idea | Evidence | Fit for us | Cost |
|---|---|---|---|
| **Hint / partial-solution prompts on all-fail problems, annealed to zero** (QuestA arXiv 2507.13266, ICLR'26; NuRL 2509.25666; SEELE 2509.06923; SAGE 2602.03143; Scaf-GRPO 2510.19807) | Consistent gains on hard sets across several groups, with evaluation *without* hints | **Direct:** turns discarded all-fail groups into gradient. Every train problem has a stored `solution`, and `solver.py` gives exact sub-expressions | Low to moderate |
| Off-policy expert traces in all-fail groups (LUFFY 2504.14945, NeurIPS'25) | +6.4 average on math; trains weak models where on-policy RL fails | Solver traces are free, but off-policy importance weighting / policy shaping adds risk | Moderate to high |
| More rollouts for hard prompts (Knapsack RL 2509.25849; AERO 2602.14338) | +20–40% non-zero gradients, +2–4 points | Raises the chance of a rare success on hard problems; complements hints | Low |
| Self-distillation of successful search traces before RL (SkillFactory 2512.04072 — Countdown, hardest variants 25.1%; Stream of Search 2404.03683 — Countdown; Cognitive Behaviors 2503.01307 — Countdown) | Countdown-specific gains, especially on harder variants | An SFT stage on P8's own successful traces with backtracking. Our earlier programmatic SFT failed (§2.2), but self-generated traces are a different thing | Moderate |
| Entropy control (Clip-Cov / KL-Cov, arXiv 2505.22617) | +2–6 points on math; hyperparameter-sensitive | We don't log entropy yet, so this is undiagnosed | Low (add logging first) |
| Pass@k objective (arXiv 2508.10751) | Mixed: 2511.16231 argues pass@k is a diagnostic, not a good objective | Weak fit | Low |

**Recommendation: P10 = hint-augmented GRPO (QuestA/NuRL-style), plus more rollouts for all-fail problems.**
- **Resample with a hint:** when a group is all-fail, resample it with a hint in the prompt, e.g. "One solution uses (75 - 3) as a step", taken from that problem's own stored solution.
- **Anneal:** hint at most about half the solution's steps at first, then a quarter, then none.
- **What's trained:** only the completion, given the hinted prompt.
- **Evaluation is unchanged:** validation and test never see hints.
- **Disclosure:** the hints are privileged training-time information, from training problems only. That is the same category as using reference solutions for SFT, and it must be stated in the report.

**Chosen (user, 2026-11-05): P10 = hinted rollouts.** Config `configs/p10_rtx6000.yaml` is P8 + hints, branched off P8 s90.
- Up to 16 all-fail groups per step are resampled once with "Hint: one solution contains X".
- X = `sft_coldstart.solution_hint(solution, frac)`: the largest *proper* sub-expression of the stored solution with at most about frac of its operations.
- frac = 0.5 for steps 0–29, 0.25 for 30–59, 0 for 60–89.
- Hinted groups get priority over refills and are excluded from the curriculum statistics. `correct` and the other first-round metrics stay unhinted, so they remain comparable with P8.
- New log fields: `hint_frac`, `n_hint_groups`, `n_hint_signal`, `hint_correct`.
- Tests: `test_hint_schedule_and_solution_hints`, `test_solution_hint_on_chain_...`, `test_hint_fields_do_not_change_existing_hashes`. P8/P9 hashes are unchanged; the suite gives 222 passed.
- Comparison: P8's own continuation (s100–s135, validation 134–145) and P8 s90 (148).

**Planned P11 (parallel branch): self-distillation of P8's own search traces** (SkillFactory / Stream of Search / STaR), also branched off P8 s90.
- **Steps:**
  - (a) sample P8 s90 on *training* problems only (validation excluded), mostly medium/hard;
  - (b) keep correct traces that recover from a failed check (False → new candidates → True), since SkillFactory keeps failed attempts with their recoveries;
  - (c) LoRA SFT on them;
  - (d) a validation check;
  - (e) the RL stage on the cloud after P10, because SkillFactory's gains appear only after RL.
- **Where:** (a)–(d) can run on soos while the cloud runs P9 and P10.
- **Risk:** SFT on its own successes can reduce diversity.

### 2.7j P10 (hinted rollouts) and P11 (self-distillation): results (2026-11-05)
**P10** (run `p10-1793893223`, us-central1-c, 90 steps, about 3.6 h; logs in `verification/logs/p10_rtx6000/`).

| step | 0 | 10 | 20 | 30 | 40 | 45 | 50 | 60 | 70 | 75 | 80 | 85 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hint frac | 0.5 | 0.5 | 0.5 | 0.25 | 0.25 | 0.25 | 0.25 | 0 | 0 | 0 | 0 | 0 |
| val /300 | 141 | 127 | 126 | 136 | 142 | 149 | 142 | 141 | 139 | 151 | 149 | 151 |
| hard /100 | 23 | 13 | 14 | 16 | 21 | 21 | 22 | 22 | 22 | 26 | 26 | 28 |

- **Hint mechanics worked as designed.** 16 hinted groups per step (the cap). About 10 of them had reward variance and were trained. 8–13% of hinted rollouts were correct.
- **Groups trained with no correct rollout:** 7–8 per step.
- **The early dip:** steps 5–25 lost about 10 hard problems. P9 showed the same dip.
  A shared suspect is the fresh AdamW state at each warm start, with no LR warmup at a constant 2e-5. This is untested.
  P10 recovered once the hint fraction fell, and its best stretch is the *hint-free* phase (s75–s90). That matches QuestA/NuRL: hints are a scaffold, and the gain shows up after it is removed.

**Checkpoint choice, 3 validation passes** (validation noise, §2.7g). The passes ran in separate vLLM processes with the adapter order shuffled. Dumps are in `verification/logs/p10_rtx6000/val3/`.

| adapter | pass 1 | pass 2 | pass 3 | mean |
|---|---|---|---|---|
| P8 s90 | 136 | 144 | 141 | 140.3 |
| P10 s76 | 150 | 140 | 142 | 144.0 |
| P10 s80 | 147 | 142 | 146 | 145.0 |
| P10 s84 | 148 | 152 | 152 | 150.7 |
| P10 s86 | 148 | 147 | 147 | 147.3 |
| **P10 s90** | 148 | 151 | 152 | 150.3 |
| P11 SFT | 133 | 126 | 128 | 129.0 |

- **Chosen: P10 s90.** It is tied with s84, and as the final step it involves no selection among checkpoints, so no winner's-curse inflation.
- **Majority-of-3 vs P8 s90:** 150 vs 138; only P8 17, only P10 29; p = 0.10.

**Test set, run once** (P10 s90 and P8 s90 in the same vLLM run, `fewshot_multi_retry`; dumps in `verification/logs/test_eval/outputs/test_eval/p10/`):

| | /1000 | easy /400 | medium /300 | hard /300 | unseen-800 |
|---|---|---|---|---|---|
| **P10 s90** | **504** | 329 | 109 | 66 | **411** (51.4%) |
| P8 s90 (same run) | 479 | 321 | 97 | 61 | 383 (47.9%) |
| P8 s90 (original test run, §2.7f) | 478 | 324 | 94 | 60 | 381 |

- **All 1,000:** only P8 74, only P10 99; **p = 0.068**.
- **Unseen-800** (VERIFICATION §16a): only P8 57, only P10 85; **p = 0.023**.
- By tier, P10 is ahead on all three (easy 25 / 33, medium 24 / 36, hard 25 / 30), but no single tier is significant.
- P8 s90 re-scored 479 vs 478; 949/1000 per-problem outcomes agreed. On 1,000 problems the evaluation noise is small in aggregate.
- **Verdict:**
  - Hinted GRPO gives **+2.5 pp on the full test set (borderline, p ≈ 0.07)** and +3.5 pp on the never-seen 800 (p = 0.02). The 1,000-problem number is the primary one.
  - It is a real but modest gain, much smaller than P8's +11 pp from the multi-candidate prompt.
  - **Disclosure:** hints are privileged training-time information (stored solutions of *training* problems). Evaluation never sees hints.
- **New best candidate: P10 s90** (`gs://countdown-rl-16875-ckpt/baselines/p10_s90`).

**P11** (soos; `verification/p11_traces.py`; data in `outputs/p11/` on soos):
- **(a) Sampling:** P8 s90 at the training temperature on 1,300 training problems × 8 (validation excluded), 10,400 rows in about 5 h.
  - Correct: easy 614/800, medium 1,082/4,000, hard 1,032/5,600.
- **(b) Filter:** 428 recovery and 2,162 direct traces. Kept 385 recovery (at most 2 per problem) plus 385 direct, i.e. 770 traces from 491 problems: 317 hard, 307 medium, 146 easy.
  - The longest example is 954 tokens, so nothing was truncated.
- **(c) SFT:** continued P8 s90, LR 1e-5, 1 epoch, batch 1, 770 steps, 15 min. Loss stayed flat at about 0.10; the policy already assigns its own samples high probability.
- **(d) Validation:** **129.0 mean over 3 passes vs P8 s90's 140.3.** It is worse on every pass; majority-of-3 gives only P8 28, only P11 19, p = 0.24.
  Self-distillation on its own correct traces did *not* help and probably hurt.
  - Likely causes:
    - Batch-1 AdamW steps are far noisier than RL's group-averaged updates and move the adapter off the RL optimum.
    - The data mostly re-teaches behaviour P8 already has. Only 4% of samples were recoveries.
  - SkillFactory's gains appear only *after* the follow-up RL stage, so (e), RL on top of P11, remains untested.
  - Given (d) and P10's result, P11-RL is a low priority.

### 2.7k Fresh holdout set and P10b (2026-11-05/06): P10b POSITIVE, holdout 528 vs 479
**Why a new set:**
- The 300 validation problems have been used for every checkpoint choice since P7, which creates winner's-curse bias (§2.7g).
- The 1,000-problem test set has now been run on 6 models. 200 of its problems were also seen during P1–P6 development (VERIFICATION §16a).

**`data/holdout.jsonl`:**
- 1,000 problems: 400 easy (88 with 3 numbers, 312 with 4), 300 medium, 300 hard. Same generator and tier weights as the original data, seed 20261105.
- It excludes every train and test (number set, target), and every problem is checked with `verify_answer`.
- sha256 `480a4ae9…808f30`.

**Rules:**
- No training or selection code reads it; `tests/test_holdout.py` checks this, plus the hash, disjointness and solvability.
- It is evaluated **once**, at the end, on the final candidates only, on soos with `eval.py`, exactly as the baseline below (`eval_vllm.py --set holdout` also exists but was not used, so backends are never mixed).
- Checkpoint selection stays on the 300-problem validation set (3-pass means).

**Holdout baseline, before P10b** (soos, `eval.py` with the HF backend, greedy, `fewshot_multi_retry`, 768 tokens, 2 rounds; dumps in `verification/logs/holdout/`):

| | easy /400 | medium /300 | hard /300 | total |
|---|---|---|---|---|
| P10 s90 | 321 | 101 | 57 | **479** |
| P8 s90 | 327 | 94 | 45 | 466 |

- **P10 vs P8:** only P8 83, only P10 96, **p = 0.37**.
  - By tier: easy 38 / 32, medium 27 / 34, hard 18 / 30 (p = 0.11).
- **On fresh problems P10's lead shrinks** to +13 (+1.3 pp), not significant, against +25 on the test set (p = 0.07). The gain sits in hard and medium.
  Reading: P10 is at best slightly better than P8. The hint effect is small.
- These are HF-backend numbers, not comparable in absolute terms with the vLLM test numbers.
  The end-of-P10b holdout run will use exactly this soos setup.
- This was not used for selection: P10b's checkpoint is still chosen on validation.

**P10b** (`configs/p10b_rtx6000.yaml`, hash `735bfd9cd488`) continues from P10 s90 for 80 steps, with two changes:
- a second, shorter hint cycle (0.5@0, 0.25@25, 0@50);
- `lr_warmup_steps: 5`, a linear warmup.
  - The new `train_rl` field is hash-neutral at 0, so the P8/P10 hashes are unchanged. The LR is logged per step, and `test_lr_warmup_schedule_and_hash_stability` covers it.
  - The warmup tests the hypothesis that the early dip in P9 and P10 (about −10 hard validation problems in steps 5–25) comes from a fresh AdamW at full LR.
  - Read it from the first 25 steps compared with P10's.

**P10b results (2026-11-06). POSITIVE: the best model so far.**

*Run:*
- All 80 steps completed (02:43 UTC, run `p10b-1793920387`). All step adapters are in the bucket (`baselines/p10b_s*`), and step 80 matches its COMMIT sha256.
- The full logs are in `verification/logs/p10b_rtx6000/` (`train.log`, `run/train_log.jsonl`); in-run val dumps are in the bucket (`runs/p10b-1793920387/logs/val/`). The run exited cleanly (`ATTEMPT 1 exit 0` at 02:43:39 UTC).
  - The VM still sat idle until its 6 h auto-delete at 05:14: my ssh-based end-of-run waiter dropped silently. Since then `cloud_train.sh` syncs logs to the bucket every 5 min, the waiter polls the bucket instead of holding an ssh session, and `scripts/cloud_chain.sh` powers the VM off when it finishes.
  - (An earlier version of this section said these logs were lost. That was wrong: a bucket listing was cut off by `head`.)
- In-run single-pass val: 148, 152, 138, 146, 151, 147 (steps 0–25), then 158, 151, 158, 154 (30–45, hints at 1/4), then 151, 153, 152, 155, 147, 157 (50–75, no hints). Hard tier: 24, 31, 23, 21, 24, 22, then 29, 23, 28, 28, then 26, 26, 27, 27, 25, 32.
  - So there is no early dip, which is consistent with the warmup hypothesis but confounded: a stronger starting point, and hint_correct about 20% vs P10's 9–13%.
- **KL to the base model rises during the hint-free phase:** 0.07 (steps 0–25) → 0.09 (25–50) → 0.11–0.13 (50–80). That fits the model moving further from the base once it trains on its own unhinted successes. It is well below the adaptive-KL target of 0.1–0.2, so it is not a constraint.
- Train accuracy is flat (0.33–0.56 per step), because the signal curriculum keeps the mix at about 50% hard groups.

*Selection: 3-pass validation* (`eval_vllm.py --set val`, `fewshot_multi_retry`; 3 separate processes with shuffled adapter order; dumps in `verification/logs/p10b_rtx6000/val3/`):

| ckpt | passes | mean /300 | easy /100 | medium /100 | hard /100 | majority-of-3 |
|---|---|---|---|---|---|---|
| P10 s90 (start) | 154, 150, 151 | 151.7 | 86.7 | 41.0 | 24.0 | 152 |
| P10b s50 (hints off from here) | 150, 154, 156 | 153.3 | 84.3 | 43.0 | 26.0 | 152 |
| P10b s60 | 153, 155, 155 | 154.3 | 84.7 | 41.0 | 28.7 | 154 |
| P10b s70 | 153, 144, 158 | 151.7 | 85.3 | 40.0 | 26.3 | 151 |
| P10b s76 | 157, 154, 154 | 155.0 | 87.0 | 37.3 | 30.7 | 153 |
| P10b s78 | 155, 152, 155 | 154.0 | 85.3 | 41.3 | 27.3 | 151 |
| **P10b s80 (final)** | **164, 155, 159** | **159.3** | 88.3 | 41.0 | 30.0 | **157** |

- **s80 was chosen:** highest mean, top in passes 1 and 3, and it is the final step, so there is no winner's-curse pick among the late steps.
- On validation alone it is not significant: majority-of-3 vs P10 s90 is 21 / 16, p = 0.51.

*Test* (vLLM, both adapters in one process; dumps in `verification/logs/test_eval/outputs/test_eval/p10b/`):

| | easy /400 | medium /300 | hard /300 | total |
|---|---|---|---|---|
| **P10b s80** | 331 | 116 | 80 | **527** |
| P10 s90, same process | 328 | 105 | 61 | 494 |
| P10 s90, earlier run (§2.7f) | 329 | 109 | 66 | 504 |
| P8 s90 (§2.7j) | 321 | 97 | 61 | 479 |

Paired McNemar tests on the test set:
- **vs P10 s90, same process:** 91 / 58, **p = 0.009**. By tier: hard 36 / 17 (p = 0.013), medium p = 0.14, easy p = 0.78.
- **vs P10 s90, earlier run:** 87 / 64, p = 0.073. The two P10 runs agree on 950/1000 problems, which is vLLM nondeterminism.
- **vs P8 s90:** 117 / 69, **p = 0.0005**.
- **Unseen-800:** 425 vs 403 (P10 s90 same process, p = 0.051) and vs 383 (P8, p = 0.001).

*Holdout*, evaluated once, after selection (soos `eval.py`; config identical to the baseline apart from the adapter path; dump `verification/logs/holdout/final_p10b_s80.jsonl`):

| | easy /400 | medium /300 | hard /300 | total |
|---|---|---|---|---|
| **P10b s80** | **338** | **126** | **64** | **528** |
| P10 s90 (baseline) | 321 | 101 | 57 | 479 |
| P8 s90 (baseline) | 327 | 94 | 45 | 466 |

Paired McNemar tests on the holdout:
- **vs P10 s90:** 110 / 61, **p = 0.0002** (+4.9 pp). By tier: easy 34 / 17 (p = 0.024), medium 45 / 20 (p = 0.003), hard 31 / 24 (p = 0.42).
- **vs P8 s90:** 122 / 60, **p < 0.001** (+6.2 pp). By tier: medium p = 0.0001, hard p = 0.015, easy p = 0.20.
- Truncation: 6/1000 hit max_length, against 10 for P10 s90.

*What it shows:*
- **The second hint cycle plus 30 hint-free steps gave a real gain.** It is significant on fresh holdout problems (+49) and on the test set measured in the same process (+33). The holdout gain is the cleaner number, because nothing was ever selected on it.
- Where the gain shows up differs between sets: on test it is mostly hard (+19), on holdout medium and easy (+25 / +17). Per-tier splits of one model are noisy at about ±10–15, so the robust claim is the total.
- **Attribution is mixed.** Two things changed at once (a second hint cycle and LR warmup), and the run also simply trained longer: P8 → P10 → P10b is 90 + 90 + 80 steps.
  - The holdout shows P10 → P10b gained more (+49) than P8 → P10 (+13).
  - This would fit "hint-free consolidation after a hint phase helps", but no ablation isolates it.

### 2.7m Hint ablation and P10c: the second hint cycle did nothing (2026-11-06)
**Runs** (chain `c1`; logs in `verification/logs/c1_ablation_p10c/`):
- **P10b-nohint** (`configs/p10b_nohint_rtx6000.yaml`): identical to P10b (P10 s90 init, 80 steps, lr warmup 5, seed 0) except that it has **no hint schedule**.
- **P10c** (`configs/p10c_rtx6000.yaml`): the P10b recipe, a third hint cycle, starting from P10b s80.

**Selection** (3-pass validation, the usual rule: best mean, ties go to the later step):

| ckpt | passes | mean /300 | easy | medium | hard |
|---|---|---|---|---|---|
| P10b s80 (reference) | 158, 157, 158 | 157.7 | 87.7 | 39.7 | 30.3 |
| nohint s50 / s60 / s70 | – | 146.3 / 149.0 / 149.0 | – | – | – |
| nohint s76 / s78 | – | 156.7 / 158.7 | – | – | – |
| **nohint s80** | 161, 162, 159 | **160.7** | 88.3 | 42.0 | 30.3 |
| P10c s50 / s60 / s70 | – | 151.3 / 148.7 / 147.7 | – | – | – |
| **P10c s76 / s78** (tie, so the later step, s78) | 159, 155, 152 / 154, 159, 153 | **155.3** | 86.0 | 41.3 | 28.0 |
| P10c s80 | 150, 152, 152 | 151.3 | 84.0 | 40.7 | 26.7 |

**Test** (vLLM, one process): P10b s80 520, nohint s80 525, P10c s80 531. Every pairwise p is above 0.4.

**holdout2** (`data/holdout2.jsonl`, soos `eval.py`, the same settings as its baseline; each model evaluated **once**):

| Model | easy /400 | medium /300 | hard /300 | total |
|---|---|---|---|---|
| P10 s90 (the init of both) | 311 | 87 | 73 | 471 |
| P10b s80 (hint cycle + warmup) | 332 | 101 | 80 | 513 |
| **P10b-nohint s80** (warmup, no hints) | 336 | 104 | 78 | **518** |
| **P10c s78** (third hint cycle, from P10b s80) | 339 | 99 | 87 | **525** |

Paired McNemar tests (only-A / only-B):
- **nohint vs P10b:** 92 / 87, **p = 0.77**.
- **P10c vs P10b:** 90 / 78, p = 0.40.
- **nohint vs P10 s90:** 118 / 71, p = 0.0008.
- **P10c vs P10 s90:** 117 / 63, p = 0.0001.
- **P10c vs nohint:** p = 0.65.

**Conclusion:**
- **The P10 → P10b gain (+42 to +49 on two holdouts) came from 80 more GRPO steps, not from the second hint cycle.** Without hints the same 80 steps did as well or better.
- A third hint cycle adds nothing measurable.
- The first hint cycle (P8 → P10) was itself not significant on holdout1 (+13, p = 0.37). So across the lineage there is no evidence that hinted resampling helps on this task; there is evidence that **continued training does**.
- The hints did *train*: hint_correct was about 20%. But the extra signal did not turn into unhinted accuracy beyond what plain training gave.

**Caveats:**
- These are single runs per arm. The P10b replicate (seed 1, chain `c2-rep`, running) gives the run-to-run spread. With nohint − P10b = +5, the conclusion "hints ≈ no effect" holds unless the spread is tiny; it can't flip into "hints help".
- Both P10b and nohint used LR warmup, so warmup is not separated out (P10, without warmup, was the earlier recipe).

**P10b replicate (seed 1; chain `c2-rep`, logs `verification/logs/c2_replicate/`).** The VM was spot-preempted at step 4 and the run resumed from the bucket.
- **3-pass validation:** s50 154.0, s60 **154.7**, s70 151.0, s76 152.3, s78 153.0, s80 150.0 (P10b s80 159.3 in the same run).
- **Test:** rep s80 **554** vs P10b s80 519 (98 / 63, p = 0.007). Validation and test disagree here; the validation differences are 4–5 problems, within noise.
- **Declared before the holdout2 run** (written 2026-11-06 20:35 UTC):
  - the primary replicate number is the validation pick, **s60**;
  - s80 (final step) is also evaluated, as a secondary number, and is not used for any choice.

**Replicate holdout2 results** (declared above; each checkpoint evaluated once):
- **Primary, rep s60 (validation pick): 506** (easy 319 / medium 106 / hard 81).
  - vs P10b s80 513: 85 / 92, p = 0.65.
  - vs nohint 518: p = 0.37.
  - vs P10 s90 471: p = 0.008.
- **Secondary, rep s80 (final step): 539** (332 / 103 / **104**).
  - vs P10b s80: p = 0.058.
  - vs rep s60: 90 / 57, **p = 0.008**.

**What this adds:**
1. **Recipe spread.** The same recipe from the same init lands at 506–539 on holdout2, depending on seed and checkpoint. The hint-vs-no-hint gap (+5) sits well inside that spread, so **hints show no effect**. Continued training from P10 s90 is clearly positive in every arm (+35 to +68).
2. **300-problem validation chose the wrong checkpoint.** It ranked rep s60 (154.7) above rep s80 (150.0); holdout2 says s80 is 33 problems better (p = 0.008). Selecting among late checkpoints with 3 × 300 greedy passes is unreliable at these effect sizes.
   - That supports the final-step rule used by the replays (§2.7l).
   - For any future selection, use a larger fresh validation set (1–2k), as the review suggested.
   - The P10c pick (s78 over s80 by 4 problems on validation) carries the same caveat.
3. **No new headline from holdout2.** rep s80's 539 comes from evaluating two checkpoints of one run, so it isn't used as a new best. P10b s80 and rep s80 will be evaluated once on holdout3 as fixed references alongside the replays.

**Consequences:**
- The replays (§2.7l) keep their pre-registered recipe (phases c and d with hints), because changing them now would break comparability.
- REPORT.md must not credit hints for the gains.
- The "recipe v2" candidate list gains "drop the hint cycles; just train longer".

### 2.7n SPEC analyses for P10b s80: test-time compute, feedback controls, zero-shot, pass@k (2026-11-06)
Run on soos with `eval.py` on the test set (descriptive; the test set was already reused), `--merge-adapter`. Logs: `verification/logs/spec_suite/`.
- **Rounds sweep:** P10b 17.8 / 47.3 / 51.8% at 0 / 1 / 2 rounds; base few-shot 0.6 / 10.2 / 11.1%.
- **Controls** (2 rounds):
  - blank `<output>`: 12.8%;
  - shuffled digits: 47.4%;
  - real output: 51.8%.
- **Zero-shot prompt:** 1.1% (format 98.9%).
- **T = 0.9, 4 samples:** P10b mean pass@1 49.5% (per sample 500 / 496 / 502 / 482), pass@4 65.3%. Base 9.9% / 27.6%.
- **Reading:**
  - The policy needs the execution feedback, but mostly the True/False flag.
  - It is tied to the trained prompt format.
  - pass@4 shows a large gap to pass@1, which motivates verifier-based best-of-k or larger G (§ "recipe v2").

### 2.7l From-scratch replays: Qwen3-8B, Gemma-4-E4B, Qwen3-4B seeds (pre-registered 2026-11-06, before any result)
**Question:** does the recipe that worked on Qwen3-4B transfer to another size, another model family and other seeds?
This is a question about the recipe, not a model-capability comparison: hyperparameters are kept identical on purpose.

**Recipe** (`scripts/make_lineage_configs.py`; configs in `configs/lineage/`):
- The 4 phases of the original lineage: a = P7 (40 steps), b = P8 (90), c = P10 (90), d = P10b (80).
- Each phase warm-starts from the previous phase's **final** step; nothing is hand-picked. With a constant LR this matches the original s40/s90 choices.
- Only the model and seed change. Each lineage is evaluated once on `data/holdout3.jsonl` with a single backend.

**Arms:**
- `q8b` (Qwen3-8B-Base, seed 0).
- `g4e4b` (Gemma-4-E4B, text-only extraction, seed 0).
- `q4s1` and `q4s2` (Qwen3-4B-Base, seeds 1 and 2).
- **Reading:** the original seed-0 4B lineage was designed on its own outcomes, so it counts as *selected*. Seeds 1 and 2 are the unbiased estimate of the recipe.

**Rejected change, the plateau rule.** A proposed rule would end each phase at a validation plateau (3-eval smoothing, ≥1 pp improvement, patience 4, warm-start from the plateau onset). Replayed on our own logs, it would have stopped P8 at s40 and warm-started from s20 (smoothed 40.2%), while P8 went on to 48% by s95. In P10b it would have taken s50 instead of s80. One validation pass has an SD of about 1.5 pp, so a 1 pp threshold mostly tracks noise. Fixed phase lengths are kept.

**Gemma-4-E4B preparation** (`verification/check_gemma4_text.py`):
- The checkpoint is multimodal, and `AutoModelForCausalLM` loads `Gemma4ForConditionalGeneration`. 148 of its 406 LoRA-target modules sit in the vision/audio towers.
- `Gemma4ForCausalLM.from_pretrained` cannot read the checkpoint: all 666 language weights come up missing.
- **Fix:** copy the 665 `model.language_model.*` tensors into a text-only model.
  - Logits match the multimodal model exactly (max |diff| 0 in fp32 eager).
  - The 258 LoRA targets are all in language layers.
  - vLLM greedy output equals HF on 64/64 tokens.
  - The model is stored at `gs://…/models/gemma-4-E4B-text/` (sha256 `acbf7fec…`).
- Tags tokenize cleanly, with no special-token clash (`verification/logs/tokenizer_tags_qwen3_gemma4.txt`). The model has a final-logit softcap of 30 and a 512-token sliding window.

**Gemma go/no-go gate (pre-registered):**
- **Base validation, 3 passes, phase-a prompt:** 5, 6 and 3 of 300, mean **1.6%**. With the multi-candidate prompt: 3.7%, one pass. For comparison, Qwen3-4B is about 13% and Qwen3-8B 26% at step 0.
- **After phase a only** (`scripts/gate_check.py`), the chain continues if two conditions hold:
  - smoothed in-run validation (last 3 evals) ≥ 1.6 + 3 = **4.6%**;
  - first-round train accuracy is rising (mean of the last 10 steps > mean of the first 10).
- **If it fails,** the single allowed deviation is to rerun Gemma with hints from phase a (`hint_schedule` from phase c, applied to phases a and b), reported as a deviation. No other per-model tuning.

**Logging added before these runs** (logging only; config hashes unchanged):
- `entropy_est`: the mean −log p of the sampled model tokens.
- `retry_after_allfalse` / `n_first_allfalse`: the strict retry rate, counting a first check as failed only when it printed no True.
- `n_hint_nocorrect`: hinted groups trained with no correct answer.
- The Qwen3-8B run started before these and does not have them.

**Also queued:** a P10b replicate (seed 1, same P10 s90 init), so the hint ablation can be read against run-to-run spread rather than against a single run.

### 2.7o From-scratch replays: results so far (holdout3; 2026-11-07)
Every model in this table was evaluated **once** on `data/holdout3.jsonl` with soos `eval.py` (greedy, `fewshot_multi_retry`, 768 tokens, 2 rounds, batch 128). Qwen uses `sdpa_gqa`; Gemma uses `eager` (VERIFICATION §21). Lineage models are the **final step** of phase d; nothing on holdout3 was selected.

| Model | holdout3 /1000 | easy /400 | medium /300 | hard /300 | test /1000 (vLLM) |
|---|---|---|---|---|---|
| Qwen3-4B base | 94 | 71 | 21 | 2 | 113 |
| Qwen3-4B P10b s80 (original lineage, 300 steps) | 521 | 324 | 112 | 85 | 527 |
| Qwen3-4B P10b replicate s80 (seed 1, last phase) | 544 | 329 | 114 | 101 | 554 |
| Qwen3-8B base | 120 | 103 | 12 | 5 | 118 |
| **Qwen3-8B replay, final** (`q8b-d` s80, 300 steps) | **557** | 350 | 125 | 82 | **585** |
| Gemma-4-E4B base (text-only) | 37 | 26 | 10 | 1 | 46 |
| Gemma-4-E4B replay, final (`g4e4b-d` s80) | 309 | 233 | 57 | 19 | 317 |

Paired McNemar tests on holdout3 (only-A / only-B):
- **Qwen3-8B final vs Qwen3-4B P10b s80:** 109 / 73, **p = 0.009**.
- **Qwen3-8B final vs the 4B replicate s80:** 88 / 75, p = 0.35.
- **Every final model vs its base:** p < 1e-60.

**Reading:**
- **The recipe transfers.** Every model improves 5–8× over its own base, with the same hyperparameters and no per-model tuning.
- **Qwen3-8B is the strongest so far** (557 holdout3, 585 test). Its margin over the two 4B references (+36 and +13) is about the size of the 4B run-to-run spread (521 vs 544), so "8B > 4B" is clear against P10b but not against the better 4B seed.
  - The clean comparison is the 4B seed replays: same recipe, final-step rule, still running (`q4s1` in phase d, then `q4s2`).
- **Gemma-4-E4B learns, but ends far lower** (309): mostly easy problems, 19/300 hard. Its base is the weakest (37), and phase d slightly lowered its validation and test scores vs phase c (98.7 → 92.3 val; 341 → 317 test).
- **Validation tracks test closely for the replays.** The final step was also the validation-best phase end for Qwen3-8B.

**Update: Qwen3-4B seed replays** (`q4s1`, `q4s2`; logs in `verification/logs/q4_seed_replays/`; final step of phase d; holdout3 evaluated once each):

| Model | holdout3 | easy | medium | hard | test (vLLM) | 3-pass val, phases a/b/c/d |
|---|---|---|---|---|---|---|
| 4B replay, seed 1 | 512 | 321 | 104 | 87 | 530 | 98.0 / 135.3 / 143.7 / 152.7 |
| 4B replay, seed 2 | 498 | 321 | 101 | 76 | 509 | 94.7 / 135.3 / 151.0 / 139.3 |
| Qwen3-8B replay (seed 0) | 557 | 350 | 125 | 82 | 585 | 132.3 / 153.3 / 169.7 / 172.3 |

- **Clean 4B vs 8B comparison** (same recipe, final-step rule): 8B vs 4B seed 1 is 116 / 71, **p = 0.001**; vs seed 2 it is 117 / 58, **p < 1e-5**. The 8B is about +50 (5 pp) above the 4B seed mean (505).
- **4B seed spread:** 512 vs 498 (p = 0.33). The original, hand-steered 4B lineage (P10b s80, 521) sits just above both replays, consistent with its being a selected run.
- **Phase d** (second hint cycle) lowered validation in 2 of 4 from-scratch lineages: Gemma, and 4B seed 2 (151.0 → 139.3). That fits the hint-ablation result (§2.7m).

### 2.7p Recipe v2 on Qwen3-4B (pre-registered 2026-11-07, before any v2 result)
**Question:** at equal compute, does a recipe with the review's fixes beat the replayed recipe?

**Design:**
- The schedule is the same 4-phase, 300-step replay (a 40 / b 90 / c 90 / d 80 steps; final-step warm starts), with seeds 1 and 2.
- **Comparison arms:** the replay lineages `q4s1` / `q4s2`, same seeds and same rule.
- Configs: `configs/lineage/q4v2s{1,2}-*.yaml` (`scripts/make_lineage_configs.py` with overrides).

**v2 changes** (every phase; new `train_rl` options are hash-neutral at their defaults):

| Change | Why |
|---|---|
| `hint_schedule: ""` (no hint cycles in c and d) | §2.7m: hints showed no effect |
| `train_nocorrect_groups: false` | Shaping-only groups were about 25% of hint-free batches, trained at full advantage scale (review) |
| `max_grad_norm: 1.0` | There was no gradient clipping; P10 had a KL spike to 0.57 |
| `shuffle_minibatches: true` | The second update always held the hinted/refill groups |
| `kl_coef = min_kl_coef = max_kl_coef = 0` | The KL coefficient sat at its 0.001 floor (inactive). This removes the per-micro-batch reference forward; KL is still logged from the first micro-batch |

**Unchanged:** LR, LoRA rank, batch size (64 × G 8), clip-higher, TIS, curriculum, reward, prompts and phase lengths.

**Primary outcome:**
- **Measure:** holdout3 accuracy of the phase-d final step, evaluated once per lineage (soos `eval.py`, same settings as §2.7o).
- **Pre-specified reading:** v2 − replay, averaged over the 2 seeds. With a run-to-run spread of about ±20 per run, a mean difference under about 20 is "no clear difference".
- **Secondary:** 3-pass validation and test per phase, step time, `grad_norm`, `entropy_est`.

**v2 result so far (2026-11-07): length runaway; seed 1 stopped by the pre-registered guard.**
- **Seed 1, phase b, step 49:** `GUARD: frac_max_length 0.43 > 0.4 over the last 5 steps`. The checkpoint was saved and the chain stopped; logs are in `verification/logs/q4v2_lineages/`.
- **Seed 2** shows the same drift in phase b (cap share 0.26 → 0.40) and continues under the same guard.

| Phase a, first 10 → last 10 steps | replay seed 1 | v2 seed 1 | v2 seed 2 |
|---|---|---|---|
| tokens per rollout | 189 → 157 | 185 → 403 | 185 → 320 |
| share hitting the 768-token cap | 0.04 → 0.00 | 0.05 → 0.33 | 0.05 → 0.19 |
| entropy estimate | 0.30 → 0.15 | 0.30 → 0.30 | 0.30 → 0.31 |

In phase b, around step 45: v2 ran 420–520 tokens per rollout, with 0.3–0.5 hitting the cap and entropy 0.3–0.45. The replay ran about 290 tokens, 0.00 at the cap and entropy about 0.20. v2's in-run validation (51–53% in phase b, 42% at the end of phase a vs 30%) looked better while the lengths grew.

**Diagnosis:** the "shaping-only" groups, the ones with no correct answer that the review flagged as noise, were acting as the recipe's **length and format regulariser**.
- In an all-fail group, a short well-formed attempt beats a truncated one (truncation loses exec and format credit). So training on those groups pushes the policy towards concise outputs.
- With them dropped, and with the KL penalty off as well, the only signal left comes from groups that contain a correct answer. There, longer outputs that check more candidates are sometimes rewarded, and the length penalty (beyond 512 tokens) is too weak to compete with correctness.
- Lengths and entropy grow until the guard trips.

~~This is a pre-registered negative: **v2 as specified is not a viable recipe.**~~ *Superseded by the seed 2 result below.* The diagnosis of the length drift stands, but seed 1 hit the guard and seed 2 did not, so "not viable" was premature.

**v2 seed 2 completed all 4 phases (2026-11-08, chain DONE 09:40 UTC). Strongest model so far; the primary holdout3 eval is NOT yet run.**
- **Course:** phase b was preempted and resumed from step 30, and phase c from step 12. The resumed segments never tripped the guard; the step 0–29 segment of phase b was not kept locally, but the chain ran to the end, which it doesn't do after a guard trip.
- **The length drift reversed during training:** share of rollouts at the 768-token cap 0.32 (phase b) → 0.20 (end of c) → 0.06 (end of d); tokens per rollout about 440 → 380; train accuracy 0.45 → 0.62.

| v2 seed 2, final step of each phase | a | b | c | d |
|---|---|---|---|---|
| 3-pass validation /300 (mean) | 123.7 | 177.3 | 203.7 | **213.3** |
| test /1000 (vLLM) | 421 | 556 | 658 | **700** |

Same seed, same rule, replay seed 2: 3-pass val 139.3 at phase d, test 509.

**Test set, phase-d final steps** (`verification/report_metrics.py` on the dumps):

| | replay seed 2 | Qwen3-8B replay | **v2 seed 2** |
|---|---|---|---|
| Pass@1 | 50.9% | 58.5% | **70.0%** |
| Format compliance | 92.5% | 98.8% | 91.7% |
| Generated tokens (mean) | 383 | 314 | 341 |
| Hit the 768-token cap | 7.5% | 1.2% | 8.3% |
| Lines printed per check, round 1 / round 2 | 5.2 / 3.6 | 4.4 / 3.0 | 3.5 / **1.0** |
| All-False first check → retried → solved | 66 / 531 (12.4%) | 86 / 494 (17.4%) | **274 / 450 (60.9%)** |

- Paired McNemar (only-A / only-B): v2 s2 vs replay s2 42 / 233, p = 3e-33; v2 s2 vs Qwen3-8B 67 / 182, p = 2e-13.
- **The behaviour changed, not only the score.** After a failed wide check, v2 reasons from the printed values in `<think>` ("6 * 50 + 2 * 3 = 306, not 312. Wait, 312 = 50 * 6 + 12, and 12 = 9 + 3") and then runs a *single* targeted check. The replay lineages sample a new batch of candidates instead. This is the value-using backtracking the earlier lineages lacked (REPORT §3 "Limits").
- **Hypothesis (untested):** the shaping-only groups rewarded short, well-formed failures, which also suppressed long in-`<think>` arithmetic after a failed check. Removing them allowed that behaviour to emerge, at the cost of more truncation.
- **Caveats:** n = 1 seed (seed 1 is stopped at phase b step 49 by the guard); the test set has been used for earlier decisions; the primary outcome (holdout3, once) is pending; format compliance and cap share are worse than the replays.
- Logs: `verification/logs/q4v2_lineages/q4v2s2*` (chain, val3 passes, test dumps, per-phase train logs).

**v2 seed 2 on holdout3 (primary outcome, evaluated once, soos `eval.py`, same settings as §2.7o): 681 / 1000** (easy 355, medium 173, hard 153). The adapter's sha256 matches the bucket's step-80 COMMIT.
- **Comparison:** replay seed 2 scored 498, replay seed 1 512, and Qwen3-8B 557.
- **Format and recovery:** format compliance 90.5% and 9.5% at the 768-token cap; after an all-False first check, 272 of 464 recovered (58.6%, against 13–17% for the replays). Logs: `verification/logs/holdout/h3_q4v2s2_d80.*`.
- **Pre-specified reading:** v2 − replay averaged over 2 seeds. Only seed 2 exists, so the reading is incomplete. Seed 1 is counted as a guard stop, not resumed.

### 2.7r Pre-registration (2026-11-08, before any result): v2 seed 2 continuation, and a zero-shot (no-demo) arm
**A. Continuation** (`configs/q4v2s2_long.yaml`, run `q4v2s2long`, cloud RTX PRO 6000):
- **Question:** is the best model still improving after 300 steps?
- **Setup:** 100 more steps from q4v2s2-d s80. Identical v2 settings and seed; fresh AdamW with the 5-step warmup used at every phase boundary; budget cap $16.
- **Primary measure:** holdout3 of the final step (s100, or the last checkpoint if the budget stops it), evaluated once on soos with the §2.7o settings, paired against d80 (681).
  - **Improving:** ≥ +20, outside the ±15–20 run-to-run spread, and McNemar p < 0.05.
  - **Plateau:** within ±20.
  - **Degrading:** ≤ −20.
- **Secondary:** 3-pass validation of d80, s50 and s100 in the same chain (paired passes), plus cap share, format and recovery rate.
- No test-set eval: the test set is already overused.

**Deviation (2026-11-08 13:08 UTC, recorded before the holdout3 eval):** the continuation was stopped by hand at **step 45** (SIGTERM; the trainer finished the step and committed `step_45`, sha256 `680a7c80…`). It was not run to s100.
- **Reason:** in-run validation was flat (70.7% at s0, then 68.0–69.7% in all 8 evals from s5 to s40), as was train accuracy. The user chose to give the remaining VM time to the zero-shot arm (B), which then started at 13:10 UTC.
- **Primary measure:** holdout3 of s45, evaluated once, with the same thresholds as pre-registered.
- **Caveat:** the decision used in-run validation, so a plateau verdict at s45 cannot exclude a later rise between s45 and s100.
- **Not run:** the chain's 3-pass validation (the chain logged the stop as `FAILED`, since it recognises only budget stops).
- Logs: `verification/logs/q4v2s2long/`.

**Result (holdout3, s45, evaluated once): 682 / 1000 vs 681 for d80. PLATEAU** (pre-registered band ±20).
- Paired McNemar: 82 problems solved only by d80, 83 only by s45; p = 1.00.
- **By tier:** easy 351, medium 171, hard 160 (d80: 355 / 173 / 153).
- **Format and length:** format compliance 90.0%, cap share 9.9%; both unchanged.
- **One behavioural shift:** after an all-False first check the model gives up less often. It retried 554 times vs 464, and recovered 332 vs 272, at the same rate (59.9% vs 58.6%).
- **Reading:** like the original recipe on Qwen3-8B (§2.7q), v2 on Qwen3-4B stops improving after its 300-step schedule. Both continuations were flat from the first eval.
- `verification/logs/holdout/h3_q4v2s2long_s45.*`

**B. Zero-shot arm:** the design follows a literature review of R1-Zero-style recipes and is recorded in §2.7s before launch.

### 2.7s Pre-registration (2026-11-08, before any training result): zero-shot (no-demo) arm `q4v2z2`
**Question:** is the worked example in the prompt needed for the v2 recipe to learn, or does it only speed learning up? SPEC asks for emergence without demonstrations; every model so far was trained and evaluated with a one-example demo plus a strategy sentence.

**Prompt `r1zero_tool`** (`rollout.format_prompt_r1zero_tool`; test `test_r1zero_tool_prompt_has_no_demo_or_strategy`):
- **Frame:** the R1-Zero / TinyZero conversation frame ("A conversation between User and Assistant…"), plus TinyZero's answer-format example `<answer> (1 + 2) / 3 </answer>`.
- **Tool:** a plain description, as in ToRL and ZeroTIR prompts: code in `<python>` is executed and its printed output is returned in `<output>`; at most 2 runs; the allowed operations.
- **Left out:** no worked example, no "check several candidates", no "retry", no hints.

**Why this prompt rather than the existing `zero` style** (probe on soos, base Qwen3-4B, 256 train problems × 8 samples, T = 0.9; `verification/logs/zeroprobe/`):

| Prompt | Groups with ≥1 correct (= GRPO signal) | Pass@1 | Runs code | Bare-number answers |
|---|---|---|---|---|
| `zero` (format skeleton only; never says code runs) | 3 / 256 | 0.2% | 81% | 38% |
| **`r1zero_tool`** | **58 / 256** (easy 44, medium 10, hard 4) | 4.4% | 70% | 1% |

Under v2, groups without a correct answer are not trained, so `zero` would give almost no gradient. Describing the tool is what makes the zero-shot start learnable. `r1zero_tool` also has costs at the start: 22% of samples hit the 768-token cap and 12% write their own `<output>`.

**Design:**
- **Configs:** `configs/lineage/q4v2z2-{a,b,c,d}.yaml` (`scripts/make_lineage_configs.py q4v2z2 Qwen/Qwen3-4B-Base 2` with the v2 overrides plus `prompt_style=r1zero_tool max_total_usd=16.0`).
- **Identical to v2 seed 2** (`q4v2s2-*`): seed, schedule (40 / 90 / 90 / 80 steps, final-step warm starts), every hyperparameter and the guards. The only difference is the prompt, in all four phases. v2 seed 2 used `fewshot_direct_retry` in phase a and `fewshot_multi_retry` in b–d.
- **Budget:** at most $16 per phase.
- **Gate after phase a** (`GATE=q4v2z2-a:4.3:3`): continue only if the smoothed in-run validation (last 3 evals) is ≥ 7.3% and first-round train accuracy is rising.
  - The baseline is 4.3%: base model, `r1zero_tool`, greedy, on the same 300-problem validation split (soos HF; `verification/logs/zeroprobe/base_r1zero_tool_val300_greedy.*`).
  - For reference, v2 seed 2 started phase a at 15% with its demo prompt.
- **A length-guard trip is an outcome.** It is reported, not rerun or tuned around.
- **The VM's lifetime ends at about 00:55 UTC.** If the chain is cut off, finishing it needs a new VM (resume from the bucket). If that isn't approved, the last committed checkpoint is evaluated and the run is reported as truncated, with its step count.

**Primary outcome:** holdout3 accuracy of the final step reached, evaluated once on soos `eval.py` (§2.7o settings) with the arm's own prompt `r1zero_tool`. It is compared with v2 seed 2 (681, with its demo prompt):
- **Gate fails:** "under this recipe, the demo is needed to start learning".
- **Gate passes and the result is ≥ 661 (within 20 of 681, or above):** "the demo is not needed; the behaviour emerges from a tool description alone".
- **Gate passes, but the result is below 661:** "the demo speeds up or improves RL but is not required". The gap is reported with paired McNemar.

**Secondary:**
- Holdout3 of the same checkpoint with `fewshot_multi_retry`, to check whether the policy also uses the demo.
- 3-pass validation at each phase end, compared with v2 seed 2's phase ends (123.7 / 177.3 / 203.7 / 213.3 of 300) at equal steps.
- Code-use rate, cap share, all-False → recovery rate, and whether retries reason from the printed values (§2.7p).
- No test-set evals.

### 2.7t Pre-registration (2026-11-09, before any training result): Gemma-4-E4B zero-shot arm `g4v2z0`
**Question:** does the zero-shot finding (§2.7s, REPORT §3c) transfer to another model family?
- **Not asked:** whether the demo helps Gemma. The only Gemma reference (g4e4b, holdout3 309) used the original recipe with the demo, so it differs in recipe *and* prompt. The clean demo-vs-no-demo pair would need a Gemma v2-with-demo run, which is deferred to available credits.

**Configs:** `configs/lineage/g4v2z0-{a,b,c,d}.yaml`, identical to `q4v2z2-*` except `model: models/gemma-4-E4B-text` and `seed: 0` (the seed of g4e4b).

**Probe** (soos, base model, `r1zero_tool`, 256 train × 8, T = 0.9, eager attention; `verification/logs/zeroprobe/g4_*`):
- Learning signal on 8 / 256 problems (all easy); pass@1 0.4%.
- Runs code 33%, ends with an answer 52%, writes its own `<output>` 25%, cap share 20%.
- **Greedy validation baseline:** 1.3% (4 / 300).

**Gate after phase a:** `GATE=g4v2z0-a:1.3:3`. Continue only if the smoothed in-run validation is ≥ 4.3% and first-round train accuracy is rising. A gate fail means "zero-shot from a tool description does not start on Gemma-4-E4B under v2", and is reported as such.

**Compute:**
- **Where:** a separate VM in project `gdrive-cli-513614` (same billing account, own GPU quota), run in parallel with `q4v2z2`.
- **Limits:** 16-hour maximum lifetime, set by the user for safety; $16 cap per phase.
- **If the VM's lifetime ends before phase d finishes:** the last committed step is evaluated and the run is reported as truncated with its step count. No new VM without the user's approval.

**Primary outcome:** holdout3 of the final step reached, evaluated once on soos (eager attention, VERIFICATION §21; otherwise §2.7o settings) with `r1zero_tool`.
- **Gain over the base:** reported with paired McNemar. The base on holdout3 with `r1zero_tool` is evaluated once alongside, since base Gemma with the demo prompt scored 37.
- **Reported beside the Qwen zero-shot arm.**
- **Secondary:** the same checkpoint with `fewshot_multi_retry`, so it can be set beside g4e4b's 309 (descriptive only, because the recipes differ), plus code-use, fake-`<output>` and cap-share trajectories.

### 2.7q Qwen3-8B long continuation (`configs/q8b_long.yaml`, 2026-11-08): plateau
**Question:** does the best model at the time (q8b-d s80: holdout3 557, test 585) keep improving with more training?

**Setup:** phase-d settings without hints, a fresh AdamW with a 5-step warmup, up to 300 more steps, and a $36 budget cap.

**Result:** it ran 102 steps (6.09 GPU-h) and was stopped by an external SIGTERM at 10:01 UTC. The checkpoint at step 102 was saved, but the chain's eval and power-off did not run.
- **In-run validation (300 problems, 1 pass)** was flat to slightly down: 58.3% at step 0, 59.3% at its peak (s50), 53.7% at s100. The single-pass SD is about 1.5 pp.
- **Train accuracy** was flat (0.45 → 0.43). KL stayed at about 0.08, there was no length drift (0.1% at the cap, about 300 tokens), and entropy rose slightly (0.22 → 0.28).
- **The original recipe has saturated on Qwen3-8B by 300 steps;** another 100 steps buy nothing.
- **No eval was run** on the step-102 checkpoint. Logs: `verification/logs/q8b_long/`.
- *Note:* `gen_train_logp_dev` reads 1.0–1.8 in these vLLM runs. It is not the per-token gap, which is `gen_logp_dev_mean` ≈ 0.01, the same as every earlier vLLM run.

### 2.7f Final test set (data/eval.jsonl, 1,000 problems; 400 easy / 300 medium / 300 hard), greedy
Each checkpoint was chosen on the 300-problem validation split before this run. The test set had not been used before.
The run used vLLM on the RTX PRO 6000 (`verification/eval_vllm.py --set test`).
Dumps are in `verification/logs/test_eval/outputs/test_eval/{zero,direct,multi}/`.

| Model | Prompt | Correct /1000 | Easy /400 | Medium /300 | Hard /300 |
|---|---|---|---|---|---|
| Qwen3-4B-Base | zero-shot | **0** | 0 | 0 | 0 |
| Qwen3-4B-Base | fewshot_direct_retry | 133 | 99 | 24 | 10 |
| Qwen3-4B-Base | fewshot_multi_retry | 113 | 85 | 23 | 5 |
| P6 best (s99) | fewshot_direct_retry | 348 | 244 | 60 | 44 |
| P7 s40 | fewshot_direct_retry | 362 | 271 | 56 | 35 |
| P7 s40 | fewshot_multi_retry | 364 | 261 | 68 | 35 |
| P8 s90 | fewshot_multi_retry | 478 | 324 | 94 | 60 |
| P10 s90 (hinted GRPO from P8 s90; §2.7j, run later) | fewshot_multi_retry | 504 | 329 | 109 | 66 |
| **P10b s80** (second hint cycle + warmup from P10 s90; §2.7k, run later; P10 s90 in the same process: 494) | fewshot_multi_retry | **527** | **331** | **116** | **80** |

Zero-shot base: it answers with the target *number* rather than an expression (741/1000 bare numbers, 724 equal to the target).
The other 179 give no `<answer>` and 80 give a wrong expression. The few-shot format demo is what makes the base model score at all.

Paired McNemar tests (only-A / only-B):
- **P8 s90 vs P6 best:** 348 → 478 (54 / 184), p < 0.001;
- **P8 s90 vs P7 s40, same multi prompt:** 364 → 478 (47 / 161), p < 0.001;
- **P8 s90 vs base few-shot:** 133 → 478 (19 / 364), p < 0.001;
- **P6 best vs base few-shot:** 133 → 348 (35 / 250), p < 0.001;
- **P7 s40 vs P6 best:** 348 → 362 (88 / 102), p = 0.35. P7 only matched P6's ceiling, faster;
- **P7 s40, direct vs multi prompt:** 362 → 364 (85 / 87), p = 0.94.

**What the comparisons show:**
- **The multi-candidate prompt alone does nothing.** On the base model it is slightly worse (133 → 113); on P7 s40 it is unchanged (362 → 364).
- **P8's +11.4 pp comes from RL training *with* that prompt.** The model learned to put good candidates into each check.
- **The validation ranking held on the test set** (P8 ≫ P7 ≈ P6 ≫ base). Validation and test accuracy are close: 148/300 = 49% vs 47.8%.

### 2.8 Go/no-go run P6 (`configs/pilot_qwen3_lit4.yaml`, soos)
Config: lit3 recipe (16 prompts × 8, clip-higher 0.2/0.28, 2 updates per batch, TIS cap 2, length penalty from 512) plus
the contradicted rule, `sdpa_gqa`, `gen_merged`, and guards. Held-out: 200 problems, greedy, paired against step 0.

| Held-out ckpt | Correct /200 | Δ vs step 0 | only base / only new | McNemar p | Retried after a first False | First check False |
|---|---|---|---|---|---|---|
| step 0 (base) | 24 | – | – | – | 117/171 (68%) | 171 |
| step 3 | 25 | +0.5 pp | 4 / 5 | 1.00 | 169/169 (100%) | 169 |
| step 6 | 27 | +1.5 pp | 6 / 9 | 0.61 | 166/166 (100%) | 166 |
| step 9 | 30 | +3.0 pp | 5 / 11 | 0.21 | 153/153 (100%) | 153 |
| **step 10** | **37** | **+6.5 pp** | **3 / 16** | **0.004** | 145/145 | 145 |
| step 13 | 40 | +8.0 pp | 4 / 20 | 0.002 | 143/143 | 143 |
| step 16 | 36 | +6.0 pp | 6 / 18 | 0.023 | 137/137 | 137 |
| **step 19** | **46** | **+11.0 pp** | **5 / 27** | **<0.001** | 121/121 | 121 |
| step 20 | 43 | +9.5 pp | 5 / 24 | 0.001 | 122/122 | 122 |

**Verdict: GO.** Every checkpoint from step 10 on beats base significantly, and the gain grows (easy 15→27, medium 5→9,
hard 4→10 at step 19). The gain comes from better first attempts: the first check succeeds 27 → 71 times, with retries
kept at 100%. Shortcut metrics stayed flat through step 21:
- `contradicted` stayed at 0.06–0.14 (base 0.23);
- `frac_max_length` stayed ≤ 0.016;
- mean completion length on the eval stayed at 393–464 chars (base 385).

Health: `clip_frac` 0.002–0.004, KL to base ≤ 0.045, TIS weight ≈ 1.0. The gen/train log-prob gap is creeping up
(0.011 → 0.020), so watch it. Training-batch accuracy reached 0.38 at step 20 (base 0.03). Steps take about 4.5–6 min.
The run continues toward 200 steps on soos.

| Held-out ckpt | Correct /200 | Δ | only base / only new | easy / medium / hard | Check printed True, answer wrong |
|---|---|---|---|---|---|
| step 0 | 24 | – | – | 15 / 5 / 4 | 11 |
| step 20 | 43 | +9.5 pp | 5 / 24 | 27 / 8 / 8 | 48 |
| step 26 | 51 | +13.5 pp | 7 / 34 | 33 / 10 / 8 | |
| step 30 | 49 | +12.5 pp | 7 / 32 | 33 / 8 / 8 | 55 |
| step 36 | 53 | +14.5 pp | 5 / 34 | 36 / 10 / 7 | |
| **step 40** | **59** | **+17.5 pp** | **3 / 38** | **41 / 10 / 8** | **23** |

All McNemar p < 0.001 from step 19 on. Through step 41: `contradicted` fell to 0.02–0.09 and `retry_after_false` stayed at 1.0,
with `frac_max_length` at 0. Groups with no reward variance (`zero_var`) rose to 0.25–0.38 as easy prompts saturate. The KL
monitor showed one transient spike, 0.475 at step 38; it is β=0, so monitor only. The gen/train log-prob gap is 0.03.
Hard-tier gains are small (4 → 7–9 of 66).

**New failure mode: value-only checks.** The model's check is `print(res == target, res)`. It tests the value, not
the number-use rule, so it "verifies" invalid answers such as `50 + 25 + 25 + 25 + 25` (25 reused) or `75 + 33` (33 not
given). These earn exec + format credit (0.2) because the check printed True, and the contradicted rule does not apply. They rose from
11/200 (base) to 55/200 (step 30), then fell back to 23 at step 40 as the 1.0 correct reward won out. Not changed mid-run, because
it would break resume and confound the run. Proposed fix for the cloud run: see §6.

**Steps 41–61: plateau.** Held-out accuracy was 62 (step 43), 66 (46), 66 (49), 65 (50), 66 (53), 64 (56), 64 (59) and
**67/200 at step 60** (+21.5 pp; 3 only-base vs 46 only-new). By tier at step 60: easy 15→46/67, medium 5→13/67,
hard 4→8/66, so hard has barely moved. Value-only-check errors fell to 7/200 (base 11), so that failure mode has
mostly washed out on its own. `contradicted` stays at 0.01–0.08 and `retry_after_false` at 1.0.

The plateau is not from lost signal. As easy prompts saturate, first-round `zero_var` rose from 0 to 0.45 (10-step means).
DAPO refills kept ~15 of 16 groups informative, at the cost of sampling ~26 prompts per step instead of 16 (~64% more
generation). Likely limits: the 2-execution cap (a third attempt goes unchecked) and hard problems needing search that
1–2 checks can't cover. The small batch (16 prompts) and LoRA r=16 may also limit it. The training mix may need more hard problems once easy ones saturate.

**Steps 61–101: flat, with slow drift.** Held-out was 61–73/200 at all 16 checkpoints (mean ≈ 66). The best are step 66 and
step 99 at **73/200 (+24.5 pp)**; step 99 had 3 only-base vs 52 only-new, with easy 46, medium 15 and hard 12. The spread
matches 200-problem noise, so there is no real trend. Shortcut metrics stay clean:
- `contradicted` 0.02–0.05;
- `retry_after_false` 1.0;
- `frac_max_length` ≤ 0.002;
- value-only-check errors 5–13/200.

Drift indicators, as 10-step means:

| | 60–70 | 70–80 | 80–90 | 90–101 |
|---|---|---|---|---|
| KL to base (β=0, monitor) | 0.11 | 0.23 (max 1.05) | 0.33 | 0.27 |
| mean \|gen−train log-prob\| | 0.026 | 0.027 | 0.034 | 0.042 |
| clip_frac | 0.004 | 0.004 | 0.005 | 0.006 |
| eval completion length (chars) | ~430 | ~440 | ~455 | ~470 |

None of these is near a guard, but all four are rising while accuracy is flat. That is the usual lead-in to a GRPO
late-run degradation, so the run continues to step 200 as a stability test before cloud spend. All checkpoints are kept
in the local bucket (`_remote/`, ~380 MB each).

Training-side (16-prompt batches, noisy): `contradicted` 0.23 → 0.10–0.14, `retry_after_false` 0.63 → 0.96–0.99,
`frac_max_length` ≤ 0.07, `clip_frac` ~0.002, mean |gen−train log-prob| ~0.011 (TIS weights ≈ 1.0), KL to base 0.014 at step 11.
Unlike pilot B, the model is retrying more, not less. With SPEC's 2-execution cap, nearly every failing rollout now hits
the cap, and its third attempt is answered unchecked. Fewer first checks fail (171 → 153), so first attempts are improving.

**Crash at step 12:** `hipErrorLaunchFailure` during sampling, while a held-out eval ran on the same GPU. There was no
amdgpu fault in dmesg, so it is treated as a transient ROCm fault. Resumed from the step-10 checkpoint under
`runs/P6_supervise.sh`, which auto-resumes up to 5 times and stops on `GUARD_STOP`. The resumed step 10 reproduced the
original step 10 exactly (reward 0.293, correct 0.141, zero_var 0.06). That shows resume restores LoRA, optimizer and RNG state
deterministically on real hardware.

---

## 3. What we changed because of what we saw

### 3.1 The reward shortcut (from pilot B)
- **Observed:** a wrong but well-formed answer earned format 0.1 + exec 0.1, while retrying cost tokens and rarely
  recovered (2–3/43). With 5–10% accuracy, most groups have no correct sample, so the shaping terms provide most of the
  gradient, and GRPO optimised them.
- **Change:** `rewards.answer_contradicted`. A wrong final answer whose value was printed, or whose expression was executed,
  in a check that printed `False` gets **no format or exec credit**. It is logged per step as `contradicted`.
- **Validation:** on B's held-out rollouts it flags 18/41 wrong answers at step 0 and 42/43 at step 11. 6 unit tests.

### 3.2 Literature audit (§12.2)
I compared the code line by line with TinyZero, verl, TRL, SimpleTIR, the cognitive-behaviors Countdown repo, ToRL and rStar2-Agent.

**Already matched:** advantage formula (verl GRPO / TRL), token-mean loss (DAPO / TRL default), k3 KL (verl `low_var_kl`)
with β=0 (DAPO / SimpleTIR / ToRL), log-probs at the sampling temperature, tool output masked from the loss, DAPO dynamic
sampling, group size 8, an exact answer verifier (stricter than TinyZero's).

**Differences, and what was done:**
| # | Difference | Action |
|---|---|---|
| 1 | Published tool-use RL uses answer-only rewards (rStar2 returns 1/0; tool quality is handled by trace selection, not reward). Our exec/format shaping is required by SPEC. | Kept, guarded by §3.1. Watch `contradicted`. |
| 2 | We made 1 optimizer step per generated batch, so clipping was a no-op. The references make 2–4 mini-batch updates with clip-high 0.28. | **Done:** `updates_per_batch: 2`, `clip_eps_high: 0.28`, plus a no-grad π_old pass. Logs `clip_frac`, `n_updates`. |
| 3 | No correction for the sampler/trainer log-prob mismatch (TRL defaults to truncated IS). We saw up to 0.43 nats. | **Done:** `tis_cap: 2.0`. Logs `gen_logp_dev_mean/p99`, `tis_w_mean`. |
| 4 | References use 1024–3072-token responses. We used a 384 cap with the penalty from 256. | **Done:** 768 tokens, penalty from 512 (`length_threshold`). |
| 5 | n=48 held-out can't detect changes under ~10 pp. | **Done:** the eval watcher uses n=200 (`MAXTOK`/`BS` env). |
| 6 | SimpleTIR void-trajectory filtering, Dr. GRPO (no std normalisation) | Not needed / optional (0 void trajectories observed) |

Side effect: with β=0 the reference forward now runs only on the first micro-batch of each step (as a KL monitor), which saves one forward pass per micro-batch.

---

## 4. What worked

1. **Fixing the pipeline before training.** Several bugs (log-prob misalignment, missing softcap, k1 KL) would each have quietly ruined GRPO.
   The `gen_train_logp_dev` monitor would have caught two of them.
2. **Batched generation:** 7.4× faster.
3. **Qwen3-4B-Base over gemma-2-2b**, with the tag-tokenisation fix.
4. **Prompt-level behaviour priming:** one instruction line turned "never retries" into "retries 72% of the time" with no training,
   consistent with Gandhi et al.: the behaviours have to exist before RL can reinforce them.
5. **Reading rollouts, not just metrics:** the B shortcut was invisible in reward and accuracy and obvious in transcripts.
6. **The multi-candidate check prompt trained with RL (P8)** gave +11 pp on test.
7. **QuestA-style hinted rollouts, then hint-free steps (P10 → P10b).** On the fresh holdout P10b s80 scored 528/1000 vs 466 for P8 s90 (p < 0.001). The second cycle was where most of it came from (§2.7k).
8. **3-pass validation plus a fresh, untouched holdout**, to handle vLLM nondeterminism and winner's-curse bias in selection.

## 5. What didn't work

1. **Programmatic SFT cold start:** both trace versions, both models. The traces taught the surface, not the reasoning, and not budget handling.
2. **Small GRPO pilots** (8×8, ≤45 steps): no held-out movement. Training accuracy is a misleading metric here because the curriculum shifts it toward easy problems.
3. **Shaping rewards on a ~6% accuracy task:** most of the gradient went to format and exec, and the policy found a shortcut.
4. **32×8 rollouts per step on a shared soos GPU:** 42 min/step, which is too slow to learn anything in a day.
5. **Training gemma SFT on the 24 GB Mac:** OOM.
6. **Temperature 1.0 for Qwen3-4B-Base sampling:** accuracy collapsed.

## 6. Open questions / next steps
- **Proposed (not applied): no shaping credit for invalid-number answers.** Extend the contradicted rule so a wrong
  final answer that violates the number-use rule (reused or invented numbers) gets no exec/format credit. Optionally, show the
  few-shot check also asserting `sorted(used) ⊆ given`. Evidence: P6 eval, 'check printed True but answer wrong' 11 → 55 → 23.

- Whether C″ moves held-out accuracy on n=200 within ~50–100 steps (it needs hours on soos; references use 256 prompts/step for hundreds of steps).
- If `contradicted` stays high or a new shortcut appears: gate the format/exec terms on the group having at least one correct answer, or move them into rStar2-style trace selection.
- Budget-aware SFT traces (stop sweeping once out of rounds) if a cold start is revisited.
- Cloud resume of a soos run (deferred), and a 4-model bake-off (deferred).

## 7. Where things are

| What | Where |
|---|---|
| Configs | `configs/pilot_qwen3_soos.yaml` (A/B), `pilot_qwen3_lit.yaml` (C), `pilot_qwen3_lit2.yaml` (C′), `pilot_qwen3_lit3.yaml` (C″) |
| soos runs | `~/gemma_lora_run/runs/P{1..5}_*` (train log, eval watcher, held-out JSON/JSONL), `outputs/pilot_qwen3_*` |
| Local copies | `verification/logs/r6_pilot/` (pilots), `r5_soos/` (model comparison), `r3`, `r4` (SFT) |
| Compare runs | `python3 verification/compare_pilots.py verification/logs/r6_pilot` |
| Reference code fetched for the audit | scratchpad `refs/` (not in the repo); URLs listed in VERIFICATION.md §12.2 |

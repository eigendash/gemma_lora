> Internal audit log, kept on `dev` only. File paths and run names predate the package layout and the experiment
> renames; see the run-name table in EXPERIMENTS.md and git tag `pre-refactor`.

# VERIFICATION.md — Independent audit of PART 1 (Round 2)

Auditor session 2026-11-04. Audited HEAD `3be1406` (branch `main`, no remote). Round 1 audited `7260338`, which
no longer exists because the history was rewritten for F8. Machine: Apple M4 Pro, arm64, 24 GB, macOS 15.7.7,
Python 3.14.7, torch 2.15.0, MPS available, no CUDA.

Project source was **not modified**. My files are `verification/*` and this document.
Round-2 evidence is in `verification/logs/r2/` and the full runner output in `verification/logs/run_20261104_083423/`.

> Round 1's text is in git history (`git show 14cfac7:VERIFICATION.md`). §2 below gives the before/after status of every round-1 finding.

---

## 0. Round 2 fixes applied — status after (read this first)

You approved all proposed fixes (F10–F19). They were applied to project code, then everything was re-verified.
`NETWORK=1 RUN_MODEL_CHECKS=1 verification/run_all_checks.sh` → **26 PASS / 0 FAIL**
(`verification/logs/run_20261104_091139/SUMMARY.txt`); before the fixes it was 16 PASS / 10 FAIL.
Project suite: **128 passed / 2 skipped ×3** (was 92), plus model tests 2/2 (`verification/logs/r3/pytest_3runs.txt`).

**Updated verdict: READY WITH FIXES.**
- **Pipeline correctness is fixed:** the rollout context equals the trained ids, the Gemma log-probs are correct, the KL and the curriculum work, and resume is faithful.
- **Handoff paths:** the cloud handoff runs end to end against a local bucket.
- **What remains:**
  - **R3-1:** the SFT cold-start data makes the model worse at the task. Fix it or skip SFT before the GRPO run.
  - **R2-M10:** generation throughput (measure it in the smoke run).
  - **Real-cloud paths:** never executed (CANNOT-VERIFY).

| ID | Before | After | Evidence (after) |
|---|---|---|---|
| R2-B1 stop token not fed | ratio 0.68–2.04 after `<output>` (Qwen) | **FIXED**: fed context == trained ids; step-0 max\|Δlogp\| 2.7e-5 after `<output>` | `test_fed_context_equals_training_ids` ×2 tokenizers; `r3`/`run_*/logprob_consistency_qwen.txt` |
| R2-B2 re-execution on whitespace | same code ran 2×, duplicate `<output>` | **FIXED**: stops detected only in text generated after the last injection | `test_whitespace_after_injection_does_not_reexecute` |
| R2-B3 Gemma softcap missing | up to 12 nats off; ratio 0–5.2 | **FIXED**: chunked == HF forward to 1.6e-6 (fp32) / 1.8e-6 (bf16); step-0 ratio 1.0000 | `run_*/logprob_consistency_gemma.txt`, `test_softcap_chunked_matches_capped_log_softmax_and_grads` |
| R2-B4 self-correction always 0 | structurally 0 | **FIXED**: `completion_text` includes injected text; metric uses the recorded real `outputs`; cues only after a failure | `test_eval_detects_failing_execution`, `test_eval_metrics.py`; gemma eval found 10/12 and 12/12 failing executions |
| R2-B5 bootstrap / persistence | `cd repo` fails; nothing uploaded; no heartbeats | **FIXED (locally verified; GCS unverified)**: `git archive --prefix=repo/`; data placed and hash-checked; trainer commits every save and restores the newest valid COMMIT on a fresh box; heartbeats and segments | `r3/remote_restore.txt` ("restored remote checkpoint smoke step 4"), `r3/launch_dryrun.txt` |
| R2-B6 wrong experiment | 30 steps, 50 problems, G=4 | **FIXED**: `--config $PRESET` passed; presets `n_problems: 0`; CLI > YAML; eval stage pulls SFT + latest ckpt, merges SFT, 1000 problems, controls + sweep | `launch_applies_preset` PASS; `test_config_precedence_cli_over_yaml`; preset parse in this session |
| R2-M1 rollback corrupts outputs | injected output cut to `'\n<'` | **FIXED by design**: no rollback; hallucination ends the rollout, tokens kept (penalized) | `test_hallucination_after_code_round_keeps_alignment` |
| R2-M2 infinite loop / doc mismatch | >5000 forwards at temp 0 | **FIXED**: always terminates; docstring describes the actual behaviour | `test_greedy_hallucination_terminates`, `test_hallucination_first_round_doc_behaviour` |
| R2-M3 KL no-op (k1) | grad 0.0013 vs 0.41, negative values | **FIXED**: k3; never negative; gradient == ∇KL(ref‖π) (cos 1.000) | `test_kl_estimator_nonnegative_and_has_gradient` |
| R2-M4 curriculum oversamples unsolved | hard (0%) got 43% | **FIXED**: weight = P(group has reward variance), sliding window → hard 3%, easy 61% | `test_curriculum_shifts_toward_easier_when_hard_unsolved` |
| R2-M5 resume not faithful | SFT dropped; RNG not restored | **FIXED**: SFT merged on every start; generator state saved; atomic `latest.json`; dataset-hash check | `r3/resume_compare.txt`: 4/4 steps identical to an uninterrupted run |
| R2-M6 eval config | temp 1, fp16/sdpa, first-N, truncating sweep | **FIXED**: greedy default, `--dtype/--attn/--temperature`, `--init-adapter`, stratified subset, `--rounds-sweep` with notice mode, `--out/--dump` | `test_stratified_subset_spans_tiers`; `r3/eval_qwen_sft_smoke.txt` |
| R2-M7 workers load torch | 360 MB, 2.87 s init, pool 8/4 | **FIXED**: `python -I -S` subprocess workers: 13 MB, no libtorch, 0.04 s init, pool capped, warm-up timeout kills | `run_*/sandbox_stress.txt`; `test_pool_is_capped_under_concurrency` |
| R2-M8 `.env` in tarball | included | **FIXED**: `git archive HEAD` (tracked files only) | `publish_tarball_excludes_env` PASS |
| R2-M9 doc claims | many unbacked | **FIXED**: HANDOFF rewritten to match the code; PROGRESS annotated; REPORT and README corrected | this diff |
| R2-M10 throughput | sequential generation | **OPEN (risk)**: documented in HANDOFF §Throughput with what to measure; batching not implemented | — |
| R2-m1 reward exec/format gameable | 3 hacks tie honest-incorrect | **FIXED**: all 20 hacks strictly below baselines | `run_*/reward_hacks.txt` |
| R2-m2 large stdout → error | `OutputError` | **FIXED**: truncated to 200; unbounded loops hit the timeout | `test_large_output_truncated_not_errored` |
| R2-m3 watch/cost/lifecycle | broken | **FIXED**: age from the hb object name; cost from heartbeats; lifecycle expires `hb/` only | `r3/launch_dryrun.txt` (cost report) |
| R2-m4 setup drift | lockfile unused, 3.10 VM | **FIXED**: `setup_cuda.sh` builds a py3.14 venv from the lockfile with pinned uv; GCS client pinned in the lockfile; VM uses the published tarball, Ubuntu 24.04 | `cuda_requirements_resolve` PASS |
| R2-m5 dropout / NameError / zero-var | — | **FIXED**: loss pass in eval mode; guarded final save; zero-var groups excluded; per-chunk activation checkpointing (new: chunking previously still kept every chunk for backward) | `test_train_rl.py`, `grpo_math` |
| R2-m6 SFT trace template | noted | **ESCALATED → R3-1** | §11 |
| R2-m7 `mirror/` not ignored | — | **FIXED** | `.gitignore` |
| R2-m8 length penalty inactive at 128 | — | unchanged by design (presets use 512); documented | — |
| R2-m9 stale README/.env | — | **FIXED** | — |

Changed during the fixes, so worth reviewing:
- **Hallucination semantics:** the rollout now *ends* instead of rolling back.
- **PPO "old" log-probs:** they now come from the training forward (detached). With one update per batch this is exact. The rollout log-probs are logged as `gen_train_logp_dev`, which would have caught R2-B1/B3.
- **Format reward:** now requires `<think>`.
- **Resume compatibility:** the resume hash covers fewer fields, so checkpoints from older runs need `--force-resume`.

---

## 1. Verdict before fixes: **NOT READY** (superseded by §0)

1. **The rollout engine trains on a different context than the model generated from.** After every `</python>`, the stop token
   is never fed to the model (`rollout.py:201-221`). A plain newline also re-executes the same code and injects a duplicate `<output>`.
   On a real model the step-0 importance ratio after the first execution ranges **0.68–2.04** (Qwen); it must be 1.
2. **On gemma-2-2b, the log-probs the trainer uses are wrong.** `forward_logprobs_chunked`, which every preset enables, omits Gemma-2's
   final-logit softcap. That makes it **up to 12 nats** off the true log-probs and gives step-0 ratios of **0.00–5.2** before any tool call.
3. **The cloud handoff would not run the intended experiment.**
   - The preset is never passed, so the run defaults to 30 steps on 50 problems with G=4.
   - `bootstrap.sh` fails at `cd repo`.
   - Checkpoints and heartbeats never leave the GPU host, so a preemption loses the run.
   - `.env` secrets would be uploaded inside the code tarball.
4. **Several training and eval components are ineffective.**
   - The KL penalty has ~zero gradient (k1 estimator).
   - The curriculum oversamples unsolved tiers.
   - Resume silently drops the SFT weights.
   - The self-correction metric is structurally always 0.
5. **What is solid:** data, verifier, sandbox security and isolation of results, mask partition in the simple case, SFT traces and masking, GRPO
   advantage/collate math, and LoRA-only training all pass independent checks. Several PROGRESS/REPORT/HANDOFF claims overstate what the logs and code show (§5).

---

## 2. Round-1 findings: before → after

| ID (R1) | Finding | Before | After (R2 evidence) |
|---|---|---|---|
| B1 | Phases 2–7 missing | FAIL | **Built** (`deliverables_present` PASS). Correctness issues in §4 |
| M1 | `requirements-cuda.txt` unresolvable | FAIL | **FIXED**: resolves for linux/py3.14; vLLM split to `requirements-vllm.txt`; lockfiles committed (`run_*/cuda_requirements_resolve.txt`) |
| M2 | Tokenization premise false | FAIL | **FIXED in docs and design** (IDs carried, decoded-text stop). New boundary bugs found (R2-B1, R2-B2) |
| m1 | README over-claims | FAIL | **REGRESSED (stale)**: README still says "Planned (Phases 2–7, not yet written)" and `--backend hf\|vllm` (no such backend) |
| m2 | Degenerate targets | PARTIAL | **FIXED**: target==input 0/0 (was 65/5); one-op still 6.4% train / 6.3% eval (the optional part was not done) |
| m3 | .gitignore gaps | FAIL | **FIXED** (`*.safetensors`, `checkpoints/`, `.env*`); `mirror/` not ignored (R2-m7) |
| m4 | Font cache in history | FAIL | **FIXED** (`no_build_artifacts_in_history` PASS) |
| m5 | Verifier newline / Python literals | FAIL | **FIXED** (fuzz 30 passed; `1_0`, `0xA` rejected; newline accepted) |
| m6 | Solver incompleteness undocumented | MINOR | **FIXED** (documented) |
| m7 | No lockfile | MINOR | **FIXED**: clean-room `uv pip freeze` == `lockfiles/mac-arm64-py314.lock` |
| m8 | Undocumented setup steps | MINOR | **FIXED** (README quickstart) |
| m9 | Weak medium/hard separation | INFO | Noted in PROGRESS; unchanged (median nodes 493 vs 980) |
| m10 | Numbers without logs | MINOR | **FIXED**: `logs/` committed; but see §5 for claims the logs contradict |

---

## 3. Spec compliance matrix (Round 2)

| # | Requirement | File / function | Status | Evidence |
|---|---|---|---|---|
| 1 | 10k train / 1k eval, solvable, tiered | `data_generator.py` | PASS | `run_*/audit_dataset.txt`: counts exact, 600/600 solved by independent DFS, 0 overlap, 0 dupes; hashes == `logs/data_hashes.txt` (`r2/data_regen.txt`) |
| 2 | No degenerate problems | `generate_one` | PARTIAL | target==input 0; one-op solvable 6.4% / 6.3% |
| 3 | Answer verifier exact, `+-*/`, sub-multiset | `verifier.py` | PASS | `run_*/verifier_fuzz.txt` (6,000-case fuzz + 27 edge cases) |
| 4 | Sandbox AST whitelist; no import/network/file I/O | `sandbox_worker.py` | PASS | 48 escapes rejected, no host info in any message (`r2/sandbox_adversarial.txt`) |
| 5 | 50 ms hard timeout, kill + respawn | `sandbox.py:135-139` | PASS | infinite loop returns in 51.8 ms; worker PID replaced; `timeout == 0.050` |
| 6 | stdout truncated to 200 chars | `sandbox_worker.execute` | PARTIAL | 1,000 chars → 200 ✓; >8 KB output → `OutputError` with **no** stdout (R2-m2) |
| 7 | (L5) process isolation, no leak over 10k | `sandbox.py` | PASS / PARTIAL | 10k mixed + 100 timeouts: parent RSS 22.7→25.4 MB, ≤3 children. **But** workers import torch under spawn (R2-M7) |
| 8 | Clean stdout buffer, result integrity | `sandbox.py` | PASS | 60 alternating timeouts + 8 threads × 150 with timeouts: 0 stale or misrouted results |
| 9 | Interleave: pause on `</python>`, inject, resume | `rollout.py` | **FAIL** | R2-B1 (stop token not fed), R2-B2 (re-execution) |
| 10 | Max 2 execution rounds | `rollout.py:202` | PASS | third `</python>` not executed; `terminal_reason=max_rounds` |
| 11 | `<output>` tokens label −100 | `rollout.py`, `sft_coldstart.py` | PASS (simple case) / **FAIL** (after hallucination rollback, R2-M1) | `test_mask_partition_exact` PASS on both tokenizers; SFT 200/200 traces masked |
| 12 | GRPO, G ≥ 4, group-normalized advantages | `train_rl.py` | PASS (math) | independent advantage + collate alignment tests PASS; presets G=8 |
| 13 | Clipped ratio vs old log-probs; consistent at step 0 | `train_rl.py` | **FAIL** | R2-B1 (Qwen 0.68–2.04), R2-B3 (gemma 0.00–5.2) |
| 14 | (L5) adaptive KL | `train_rl.py:463-470` | **FAIL** | k1 estimator: gradient 0.0013 vs true 0.41; negative in 10% of 8-token batches (R2-M3) |
| 15 | Composite reward R = Rc + Re + Rf − λ·len | `rewards.py` | PASS / PARTIAL | 20 hand-built hacks: none beats honest-correct (1.2); 3 tie honest-incorrect (0.2) (R2-m1) |
| 16 | SFT cold start, programmatic, no human CoT, train only | `sft_coldstart.py` | PASS | 200 traces × 2 tokenizers: real-sandbox `False`→`True`, answers ref-verified, 0 eval keys, masks exact |
| 17 | Curriculum preventing gradient starvation | `train_rl.Curriculum` | **FAIL** | 0%-pass tier gets 43% of samples (R2-M4); Mac run zero_var 0.65 |
| 18 | Eval: Pass@1 overall/tier, format, length | `eval.py` | PASS | hand-computed metrics match (`r2/eval_metrics.txt`); end-to-end Qwen runs reproducible with a seed |
| 19 | Eval: self-correction after negative feedback | `eval.py:53` | **FAIL** | structurally always 0 (R2-B4) |
| 20 | Eval: base vs RL on 1,000 problems; test-time-compute scaling | `eval.py`, `launch_anywhere.sh:63` | **FAIL** | eval stage broken (R2-B6); rounds-0 means "truncate at first `</python>`" (R2-M6) |
| 21 | REPORT.md (curves, examples, bottlenecks, hacks) | `REPORT.md` | PARTIAL (skeleton) | `[PENDING]` placeholders clearly marked ✓; one mis-stated finding (§5) |
| 22 | Reproducible setup | README, lockfiles | PASS (Mac) / **FAIL** (cloud) | Mac clean room reproduced; cloud bootstrap fails (R2-B5) |
| 23 | Resumability | `train_rl.py` | PARTIAL | kill-9 resume continues the step count (`logs/rl_resume_run*.txt`); drops SFT merge, sampling RNG not restored (R2-M5) |
| 24 | Eager attention for gemma, chunked 256k-vocab log-probs | presets, `train_rl.py` | PARTIAL | eager attention set ✓; chunking avoids (B,T,V) ✓ but is numerically wrong for gemma (R2-B3) |

---

## 4. Findings

Every BLOCKER and MAJOR below was reproduced by a check I wrote, unless it is marked *(code)*, meaning the evidence is a file:line citation.

### BLOCKER

**R2-B1 — Rollout: the token that completes `</python>` is never fed to the model.**
- **Evidence:** `rollout.py:201-221`. The stop branch extends `inj_ids` and `continue`s past the normal forward at `:239`.
  `test_fed_context_equals_training_ids` FAIL on both tokenizers: "first divergence at position 74; training has token `['>']` that the model was never fed".
  On a real model (`r2/logprob_consistency_qwen.txt`), step-0 |Δlogp| is 2e-5 before the first `<output>` and **up to 0.71 after it (ratio 0.68–2.04)**.
  Gemma via the HF path gives ratio 0.87–1.09.
- **Impact:** every token generated after a code execution conditions on `…</python` + `\n<output>…` (a malformed tag). Training then scores the same tokens with the `>` present.
  At step 0 the PPO ratios and clipping act on noise. This affects exactly the behaviour the project is meant to learn.
- **Fix:** forward the sampled stop token (`_forward(next_id)`) before forwarding the injected ids.
  Add a regression test asserting that the fed context equals `all_ids[:-1]`.

**R2-B2 — Rollout: a whitespace token after an injection re-executes the same code.**
- **Evidence:** `pieces` only holds generated text (`rollout.py:182`), and the injected text is never added to it.
  So after `</python>`, a generated `\n` makes `find_stop` true again and `extract_between` returns the same block.
  `test_whitespace_after_injection_does_not_reexecute` FAIL: `same code executed 2x: ['print(1+2)', 'print(1+2)']`.
  `code_rounds=2`, and a duplicate `\n<output>\n3\n</output>\n` is injected. The first version of my suite tripped this by accident, because a normal `"\nok…"` continuation was enough.
- **Impact:** the second round is consumed; duplicate env text appears; a third whitespace token truncates the rollout (`max_rounds`) with no answer.
- **Fix:** keep a decoded text buffer that includes injected segments, or only test stops on text generated after the last injection.

**R2-B3 — Gemma-2 log-probs in `--chunked-vocab` mode ignore the final-logit softcap (30.0).**
- **Evidence:** `train_rl.py:154-185` computes `lm_head(hidden)` without `tanh(x/30)*30`. `r2/logprob_consistency_gemma.txt`: `forward_logprobs_chunked` vs HF ground truth max|Δ| = **12.02** at the same ids and context.
  Before any injection, step-0 ratio is **0.0000–5.18**. `r2/softcap_rootcause.txt`: raw logits reach 98.5; applying the softcap by hand gives Δ = **0.0**.
- **Impact:** every gemma GRPO run with `--chunked-vocab` optimizes a wrong objective. That covers all three presets, `smoke_cuda.yaml`, `run_part2_mac.sh` and the `launch_anywhere` grpo stage.
  The policy-vs-reference KL is computed in the same wrong space. The Mac Part 2 run's dynamics (§5) are not interpretable.
- **Fix:** apply `config.final_logit_softcapping` per chunk before the max/logsumexp and to `target_logit`.
  Add a test comparing chunked against `log_softmax(model(ids).logits)` on gemma.

**R2-B4 — The self-correction metric is always 0.**
- **Evidence:** `eval.py:53` searches `rr.completion_text`, but `rollout.py:249` sets `completion_text = "".join(pieces)`, which is generated text only.
  The `RolloutResult` docstring says "gen + injected". `test_eval_detects_failing_execution` FAIL with a real sandbox `False 3`. `test_self_correction_on_transcripts_with_failing_output` FAIL.
- **Impact:** the spec's headline metric ("self-correction following negative execution feedback") and its blank/shuffle controls will report 0 whatever the policy does.
  PROGRESS attributes the 0 to "no failing executions in the sample".
- **Fix:** build `completion_text` from generated plus injected text (or use `transcript[len(prompt):]`) and add a regression test.

**R2-B5 — The cloud pipeline cannot run as written, and preemption loses all progress.**

*(code + local replication)*
- `bootstrap.sh:37-40`: the tarball from `publish_code.sh` has no `repo/` directory, so `cd repo` fails under `set -e` (`r2/publish_tarball.txt`).
  The dataset is downloaded to `/tmp/*.jsonl`, never to `repo/data/`.
- `train_rl.py:400`: heartbeats are written only when `backend == "local"`. `CheckpointManager` never calls `self.backend`, and `gcs` maps to `_UnsupportedBackend`, which is never invoked.
  So with `CKPT_BACKEND=gcs` (the default) there are **no uploads and no heartbeats**.
  The "pull latest valid checkpoint" block in `launch_anywhere.sh:39-50` builds objects and does nothing. `checkpoint.py`'s COMMIT, verify and download are used only by tests.
- Impact: on a spot or interruptible GPU, every preemption restarts from step 0 and `watch.sh` reports "no heartbeat". HANDOFF steps 4–5 cannot succeed.
- Fix: wire `checkpoint.commit_checkpoint` and `find_latest_valid` into `train_rl` behind a real GCS bucket class. Fix the tar root and data placement.
  Add a local end-to-end test with `LocalBucket` as the remote.

**R2-B6 — The cloud run would not be the intended experiment.**
- **Evidence:** `r2/preset_plumbing.txt`. `launch_anywhere.sh` picks `$PRESET` (`:32-36`) but never passes `--config`, and `bootstrap` execs it with no args.
  The effective grpo config is `G=4, prompts_per_step=4, total_steps=30, n_problems=50, max_new_tokens=128`.
  Even with `--config preset-80gb.yaml`, `n_problems=50`: no preset sets it. YAML values also override explicit CLI flags (`--total-steps 3` became 10).
- The eval stage (`:63`) passes `outputs/rl/step-*/adapter`, which expands to up to 3 paths (an argparse error). It has no `--dtype`/`--attn`.
  It loads the RL adapter onto the raw base, without the SFT merge the adapter was trained on.
- Impact: the "main GRPO run" would overfit 50 problems, or run 30 steps; the final eval would fail or evaluate the wrong model.
- Fix: pass `--config "$PRESET"`, add `n_problems: 10000` to the presets, and let CLI flags override YAML. Merge SFT before applying the RL adapter in eval, choose a single checkpoint, and add `--dtype`/`--attn`.

### MAJOR

**R2-M1 — A hallucinated-`<output>` rollback after a code round corrupts the real injected output and the masks.**
`rollout.py:187-197` counts `keep` over `pieces` (generated only) but slices `completion_ids`, which also contains the injected ids.
`test_hallucination_after_code_round_keeps_alignment` FAIL: the real injected output is cut to `'\n<'`.
Fix: track per-token provenance and truncate by index into the combined list.

**R2-M2 — Hallucination handling never terminates under greedy decoding, and does not match its docstring.**
- `test_greedy_hallucination_terminates` FAIL: more than 5,000 forward calls and 1,666 rollbacks with `max_new_tokens=64`, on both tokenizers. A rollback shrinks `completion_ids`, so the `while` condition never trips.
- The docstring promises the hallucination is "replaced with the real sandbox output", but no code runs (`test_hallucination_first_round_doc_behaviour` FAIL).
- At temperature 1.0 the loop is probabilistic, so a policy that has learned to emit `<output>` can stall a rollout for a long time.

Fix: count attempts or generated tokens, not the list length. On a hallucination, either run the open `<python>` block or terminate with a penalty.

**R2-M3 — The KL penalty is a no-op (k1 estimator), and the controller reacts to a signed, noisy value.**
- `train_rl.py:463`: `kl = mean(logp_policy − logp_ref)` over sampled tokens.
- `r2/grpo_math.txt`: its expected gradient is ~0 (|∇| 0.0013 vs true 0.41), and it is negative in 10% of 8-token batches. The logs show negative KL (`logs/rl_gemma_smoke.txt`: −0.083, −0.224).
- The underflow "fix" floors the coefficient but leaves the estimator alone. In the post-fix Mac run the KL kept rising: mean 0.032 → 0.090 by step 300+.

Fix: use the k3 estimator `exp(r) − r − 1` with `r = ref − policy` (non-negative, with a useful gradient), or put the KL in the reward.

**R2-M4 — The curriculum puts the most samples on the tier the model cannot solve.**
With pass rates easy 0.6, medium 0.1, hard 0.0, the sampling shares are 24% / 33% / **43%** (`r2/grpo_math.txt`).
That maximizes zero-variance groups, which is the opposite of preventing gradient starvation; the Mac run averaged zero_var 0.65.
The pass rate is also cumulative and never decays. Fix: target an intermediate pass band (e.g. 0.2–0.8) and use a sliding window.

**R2-M5 — Resume is not faithful.** *(code)*
- `train_rl.py:355-377`: on resume, the adapter is loaded onto the raw `base`. The `--init-adapter` SFT merge only happens on a fresh start, so a resumed SFT→RL run silently becomes base+RL-LoRA, and its reference becomes the base model.
- `gen_rng` (`:390`) is not checkpointed, so the sampling stream restarts from the seed.
- `latest.json` is written in place, non-atomically (`:278`). The kill-9 test only shows that the step counter continues.

Fix: merge the SFT adapter before loading the RL adapter on resume; save and restore the generator state; write `latest.json` via temp file plus rename.

**R2-M6 — Eval configuration problems.**
- `eval.py:165` hard-codes temperature 1.0 (single-sample Pass@1 is noisy; temperature 0 is not selectable, and would hit R2-M2).
- `eval.py:104` loads every model in fp16 with sdpa, which is unsafe for Gemma-2: no eager attention, and fp16 overflow.
- `--n-eval` takes the first N rows, and `eval.jsonl` is tier-sorted, so small subsets are all easy.
- "Pass@1 vs rounds 0/1/2" via `--max-rounds 0` truncates at the first `</python>` rather than answering without code.
- The backtrack cue `"correct"` matches non-backtracking text (`test_backtrack_cue_false_positive` FAIL).

**R2-M7 — Sandbox workers load torch under spawn.**
- `r2/sandbox_stress_independent.txt`: when the parent `__main__` imports torch (as `train_rl.py`, `eval.py` and `sft_coldstart.py` do), each worker re-imports it.
  Each worker has `libtorch_mapped=True` and 360 MB RSS; the 2-worker pool takes 2.87 s to start.
- Every timeout respawns a worker synchronously inside `run()`, which takes about 1.4 s.
- `_get_worker` (`sandbox.py:85-87`) spawns beyond `n_workers` under concurrency: 8 alive with 4 configured.
- If a cold respawn exceeds the 5 s warm-up poll, the warm-up `print(0)` reply stays in the pipe and becomes the next call's result.

Fix: give the workers a tiny bootstrap `__main__` (e.g. `multiprocessing.set_executable` plus a stub module, or a forkserver with a preloaded minimal module). Drain or discard on warm-up timeout, and cap the pool.

**R2-M8 — Secrets would be shipped to the bucket and every GPU host.**
`publish_code.sh` excludes `.git .venv .hf-cache outputs data` but not `.env`. HANDOFF tells you to create `.env` with `GCP_SA_KEY_B64` and `HF_TOKEN`.
The local replication includes `./.env` in the tarball (`r2/publish_tarball.txt`). `.uv-cache`, `mirror/` and `.pytest_cache` are also shipped.
Fix: build the tarball from `git archive HEAD` plus explicit data; never from `.`.

**R2-M9 — Docs and PROGRESS describe things the code does not do** (details in §5).
- HANDOFF claims these, none of which exist in code:
  - `--max-hours` / `--max-total-usd`
  - "write local then upload"
  - newest-valid-COMMIT selection
  - refusal on dataset or pinned-version hash mismatch
  - working `hf-hub`/`s3` backends
  - a `vllm` backend
  - a "create-if-absent" client

  `grep` finds no such flags or features.
- PROGRESS marks the overfit-50 and gemma sanity runs ✅ although the logs show no learning or no policy gradient.
- The "ratio ≈ 1 at step 0" test only checks the **first** sampled token.

**R2-M10 — Generation throughput will dominate the Part 2 budget.** *(estimate)*
`run_rollout` is single-sequence, one Python loop iteration and one forward per token, with no batching. The Mac run did ~37 s/step for 4 rollouts of ≤256 tokens (389 steps in ~4 h).
The 80 GB preset asks for 128 rollouts × 512 tokens per step × 2000 steps, about 131M sequential forward calls.
Even at 10 ms each, rollouts alone would take about 360 GPU-hours. This is an estimate and has not been measured on CUDA.

### MINOR

- **R2-m1 Reward:** `R_exec` is gameable. `print(1)` when 1 is an input, `x = 25; print(76)`, and code that checks nothing all earn exec and tie the honest-incorrect baseline (0.2). The design claims these are blocked.
  The format reward does not require `<think>`/`<python>` (answer-only correct = 1.1). None of the 20 hacks beats honest-correct (`r2/reward_hacks.txt`).
- **R2-m2 Sandbox:** output over 8 KB returns `OutputError` with no stdout, where the spec says truncate to 200. `print(__builtins__)` / `print(print)` run (harmless, no attribute or subscript access).
- **R2-m3 Ops scripts:**
  - `watch.sh:17` measures staleness from the mtime of the just-downloaded copy, so it never alerts.
  - `cost_report.py` sums `elapsed_hours`, which nothing writes, and segment index is always 0.
  - `create_bucket.sh` lifecycle deletes **everything** under `runs/` (final checkpoints included) after 7 days.
- **R2-m4 Setup drift:**
  - `bootstrap.sh` installs `requirements-cuda.txt`, not the lockfile, plus an unpinned `google-cloud-storage`.
  - `linux_validation_vm.sh` uses Ubuntu 22.04 (python3.10) against py3.14 lockfiles, and has a `<REPO_URL>` placeholder.
- **R2-m5 Training details:**
  - The policy forward runs in `train()` mode with LoRA dropout 0.05, so there is ratio noise once LoRA is non-zero.
  - Loss is normalized per token over the whole batch (undocumented).
  - `ckpt.save(step+1)` after the loop raises a NameError when `start_step >= total_steps`; there is a double save on SIGTERM.
  - Zero-variance groups are "skipped" only in the sense of zero advantage: they are still forwarded and counted in `n_tok`.
- **R2-m6 SFT traces (inspected 10 by eye; sane):**
  - The correction states the stored solution without using the feedback ("that is wrong. The right expression is …").
  - `<python>` sits outside `<think>`, unlike the SPEC example.
  - Gemma's SFT segments end `</python>\n` while rollouts stop at `>`, a slight train/inference mismatch.
- **R2-m7** `.gitignore` lacks `mirror/`, the directory `pull_checkpoints.sh` writes to.
- **R2-m8** The length penalty is inactive at the default `max_new_tokens=128` (threshold 256). It applies under the presets (512).
- **R2-m9** README is stale again (see R1 m1). `.env.example` lists `MAX_TOTAL_USD`, `MAX_HOURS` and `CKPT_BACKEND_SECONDARY`, which nothing reads.

---

## 5. Claim tracing (PROGRESS / REPORT / HANDOFF)

| Claim | Source | Verdict |
|---|---|---|
| 10k/1k, hashes `880fbfa7…`/`e29fa56c…` | `logs/data_hashes.txt` | ✅ reproduced byte-for-byte |
| 92 passed / 2 skipped | — | ✅ 3 runs, 0 flaky (`r2/pytest_3runs.txt`); model tests 2/2 |
| Sandbox p50 0.063 / p95 0.083 / p99 0.190 ms, RSS flat | `logs/sandbox_stress.txt` | ✅ independent: p50 0.061 / p95 0.071 / p99 0.088 ms; RSS flat |
| "ratio≈1 at step 0" | `tests/test_rollout_model.py` | ⚠️ true only for the **first** token; false after any execution (R2-B1) and for gemma chunked (R2-B3) |
| Hallucinated `<output>` "replaced with the real sandbox output" | `rollout.py` docstring | ❌ (R2-M2) |
| `completion_text` = "gen + injected" | `rollout.py:47` | ❌ (R2-B4) |
| Overfit-50: "reward 0.05→0.20" | `logs/rl_overfit.txt` | ⚠️ 15 steps, noisy 0.05–0.20, **final 0.10**, correct = 0 at every step: no evidence of learning. ✅ in PROGRESS is overstated |
| gemma 3 steps "masked + gradient (KL −0.08→−0.22)" | `logs/rl_gemma_smoke.txt` | ⚠️ zero_var = 1.00 at every step, so there was no policy gradient; negative KL is the k1 estimator (R2-M3) |
| kill-9 resume continues at step 4 | `logs/rl_resume_run*.txt` | ✅ counter continues; faithfulness not shown (R2-M5) |
| "correctness briefly reached 0.75 then collapsed" (REPORT §2, PROGRESS) | `logs/rl_gemma_part2_klunderflow.jsonl` | ❌ 0.75 occurs at **one step** (step 19, 3/4 rollouts of one problem); mean correct steps 0–24 = 0.07. The underflow itself is real (kl_coef 2e-21) |
| Post-fix run "stable … KL bounded" | `logs/rl_gemma_part2.txt` | ⚠️ KL rises monotonically by segment (0.032 → 0.049 → 0.075 → 0.090); correct ≈ 0.02–0.04; zero_var 0.65. The run was also affected by R2-B1/B3 |
| SFT smoke loss 0.49→0.18; gemma SFT | `logs/sft_*.txt` | ✅ match the logs (pipeline checks) |
| Eval: base 0.333 / SFT 0.917 / RL 0.0 format | — | ❓ **no log file** under `logs/` |
| "verification fully green" | Phase 8 commit | ⚠️ true for round-1 checks only, which did not cover any Phase 2–8 module |
| HANDOFF: cost guard, max-hours/USD, upload + COMMIT, hash refusal, hf-hub/s3/vllm | `HANDOFF.md` | ❌ not implemented (R2-M9, R2-B5) |
| REPORT `[PENDING]` placeholders | `REPORT.md` | ✅ clearly marked |
| No secrets in history or tree | — | ✅ (`run_*/secrets_history.txt`, `r2/hygiene.txt`) |

---

## 6. Could not verify

| Item | Why | Instead |
|---|---|---|
| Anything on CUDA, bf16 numerics, vLLM | no CUDA here | Linux resolve via `uv pip compile` |
| GCS, vast.ai/RunPod scripts | no cloud access (must not launch) | read line by line, `bash -n` all 13 scripts OK, local tar replication, CLI config replay |
| shellcheck | not installed | `bash -n` |
| Overfit-50 rerun | would take ≈15+ min and the rollout and log-prob bugs make the result uninterpretable | log inspection (§5) |
| Resume equivalence vs an uninterrupted run | needs a ≥2×15-min run; known causes of divergence found in code (R2-M5) | code citation |
| Linux spawn/fork semantics of the 50 ms timeout | macOS only | flagged for the Linux validation VM |

---

## 7. Checks (`verification/`)

| File | Purpose |
|---|---|
| `ref_countdown.py`, `audit_dataset.py`, `test_verifier_fuzz.py`, `crosscheck_external.py`, `check_tokenizer_tags.py` | round-1 data/verifier checks (unchanged except the author's benign edits) |
| `test_sandbox_adversarial.py` | 57 cases: 48 escapes/abuse, resource bounds, 200-char truncation, 50 ms timeout, kill+respawn, stale-result integrity |
| `stress_sandbox_independent.py` | 1k latency, 8-thread cross-talk with timeouts, 10k leak, memory bombs (peak child RSS), torch-in-worker isolation |
| `test_rollout_independent.py` | fake LM that records its fed context × {Qwen, gemma} tokenizers: context == training ids, exact mask partition, re-execution, round cap, hallucination rollback/doc/termination, EOS/max-length/answer stop, `completion_text`, eval failing-exec detection |
| `check_logprob_consistency.py`, `check_softcap_rootcause.py` | real-model step-0 generation vs training log-probs (both training paths vs HF ground truth); gemma softcap root cause |
| `test_grpo_math.py` | advantages, collate alignment, chunked log-probs, KL estimator sign/gradient, LoRA-only + frozen base, curriculum behaviour |
| `test_reward_hacks.py` | 20 hand-built hacks vs honest-correct/incorrect baselines; bounds; length penalty; exec conditions |
| `test_eval_metrics.py` | hand-computed metrics; self-correction visibility; backtrack-cue false positive |
| `check_sft_traces.py` | 200 traces: real-sandbox False/True, ref-verified answers, no eval keys, exact masking; prints 10 for eyeballing |
| `run_all_checks.sh` | runs everything; `NETWORK=1` adds the external/CUDA checks, `RUN_MODEL_CHECKS=1` the real-model checks |

Re-run with: `zsh -c 'source ~/.zshrc; NETWORK=1 RUN_MODEL_CHECKS=1 verification/run_all_checks.sh'`.
The baseline for this round is in §8.

---

## 8. Runner result (this round)

`NETWORK=1 RUN_MODEL_CHECKS=1 verification/run_all_checks.sh` → `verification/logs/run_20261104_083423/SUMMARY.txt`: **16 PASS / 10 FAIL**.

| Check | Result | Maps to |
|---|---|---|
| env, project_pytest ×3 (92 passed / 2 skipped), determinism, audit_dataset, verifier_fuzz, secrets_history, gitignore_coverage, no_build_artifacts_in_history, deliverables_present, sandbox_adversarial (57/57), sft_traces, bash_n, tokenizer_tags, cuda_requirements_resolve | PASS | — |
| sandbox_stress | FAIL (1 of 9 sub-checks) | R2-M7 (workers load torch, 368 MB each) |
| rollout_independent | FAIL (14 of 26) | R2-B1, B2, B4, M1, M2 |
| grpo_math | FAIL (2 of 10) | R2-M3, M4 |
| reward_hacks | FAIL (3 of 23, ties with honest-incorrect) | R2-m1 |
| eval_metrics | FAIL (2 of 3) | R2-B4, M6 |
| publish_tarball_excludes_env | FAIL | R2-M8 |
| launch_applies_preset | FAIL | R2-B6 |
| logprob_consistency_qwen / _gemma | FAIL | R2-B1 / R2-B1 + B3 |
| crosscheck_external | FAIL → **PASS on rerun** | auditor environment: the clean-room venv lacked `requirements-dev.txt` (reasoning-gym). Rerun: 300/300 + 300/300 (`crosscheck_external_rerun.txt`). Not a project finding |

After fixes, every FAIL above except crosscheck_external should flip to PASS. `reward_hacks` flips only if F19 is applied.

---

## 9. Top 5 risks for Part 2

1. **Training on a mismatched context** (R2-B1/B2/B3). Until fixed, any GRPO "learning curve" is uninterpretable, and the gemma ratios are garbage from step 0.
2. **Preemption = total loss, and the wrong experiment** (R2-B5/B6). There is no remote checkpoint or heartbeat, and the presets are not applied (50 problems, 30 steps).
3. **Throughput** (R2-M10). Sequential per-token generation for 128 rollouts per step could cost hundreds of GPU-hours. Measure in the smoke run before committing to 2000 steps; consider batched or vLLM generation.
4. **No effective regularization or curriculum** (R2-M3/M4). Expect drift, zero-variance starvation (65% on the Mac) and possible collapse, which could be misread as "reward hacking".
5. **The headline metric reads 0 regardless** (R2-B4), and eval would load the wrong model (R2-B6) in fp16/sdpa (R2-M6). The report's central claim could be wrong in either direction.

---

## 10. Proposed fixes (approved and applied — see §0 for after-status)

| ID | Fixes | Change | Size |
|---|---|---|---|
| F10 | R2-B1, R2-B2, R2-M1, R2-M2, R2-B4 | Rollout: forward the stop token before injection; one combined decoded buffer with per-token provenance (stop detection after the last injection, index-based rollback); terminate or penalize hallucinations with an attempt cap; `completion_text` includes injected text. Promote my `test_rollout_independent.py` into `tests/` | medium |
| F11 | R2-B3 | Apply the final-logit softcap in `_logprobs_from_hidden`; add a gemma chunked-vs-HF test (gated) | small |
| F12 | R2-M3 | k3 KL estimator; keep floor/cap; log both | small |
| F13 | R2-M4 | Curriculum targets a pass band with a sliding window | small |
| F14 | R2-M5 | Resume: merge SFT before the RL adapter; checkpoint `gen_rng`; atomic `latest.json` | small |
| F15 | R2-B5, R2-B6, R2-M8 | Handoff: `git archive` tarball (no `.env`), fix bootstrap paths/data, pass `--config`, `n_problems` in presets, CLI overrides YAML, wire `checkpoint.py` COMMIT upload/restore + heartbeats for the remote backend, fix the eval stage (single adapter, SFT merge, dtype/attn) | large |
| F16 | R2-M6 | Eval: `--temperature`, `--dtype`, `--attn`, stratified subset, a real "0 rounds" mode (no-code prompt), tighter backtrack cues | small |
| F17 | R2-M7 | Sandbox: torch-free worker bootstrap; drain stale warm-up; cap the pool | small |
| F18 | R2-M9, R2-m9 | Correct HANDOFF/PROGRESS/REPORT/README claims to match the code and logs | docs |
| F19 | R2-m1..m8 | Reward exec condition (must print a comparison against the target, or reference the candidate expression), stdout truncation instead of error, watch/cost/lifecycle fixes, gitignore `mirror/` | small |


---

## 11. New finding after the fixes (Round 3)

**R3-1 (MAJOR) — The SFT cold-start traces make the model worse at the task, which starves GRPO of correctness signal.**

- **How the traces are built** (`sft_coldstart.generate_trace`): each one always
  1. opens with a deliberately wrong expression built from *all* the numbers;
  2. then asserts "The right expression is <stored solution>" with no derivation from the feedback;
  3. then re-checks and answers.

  A model cannot imitate step 2, so it imitates the surface. It re-runs or re-guesses, then answers
  with an expression its own check just proved wrong.
- **Measured on gemma-2-2b**, 12 stratified eval problems, greedy, 320 new tokens (`verification/logs/r3/gemma_*_n12.{json,jsonl}`):

  | Config | Pass@1 | Format | Failing executions | Notes |
  |---|---|---|---|---|
  | base, 2-shot (`--fewshot`) | **2/12** | 2/12 | 10/12 | both solves genuine (its check printed `True` first); 10/12 hit the token limit while still trying |
  | `outputs/sft_gemma` (500 traces, 1 epoch, Mac) | **0/12** | 12/12 | 12/12 | every answer is the expression its check had just reported `False` (e.g. `False 5000` → answers `(100*(5*10))`) |

  With the tiny model + `outputs/sft_coldstart`, the answers even contained `False 198` (`r3` session output).
  The new format rule now refuses those.
- **Impact on Part 2:** at ~0% correctness, GRPO groups have identical correctness, so the only
  reward variance comes from exec/format. Correctness can only improve if some rollouts in a group
  are right. n=12 is small, but the direction is unambiguous.
- **Fix options (not applied; needs your decision):**
  1. Rebuild traces so they are imitable:
     - about half should succeed on the first attempt (`True` → answer);
     - wrong attempts should be "near misses" from the solver's search (e.g. a correct sub-expression with the wrong last operation);
     - the correction must be a small edit justified by the printed value ("875 is 77 too high → …");
     - never answer after a `False`.
  2. Or skip SFT: start GRPO from base + the 2-shot prompt, with the curriculum, which now favours tiers that have reward variance.
  3. Either way, gate the GRPO launch on a quick SFT eval: Pass@1 on 100 stratified problems must beat the base few-shot model.

### R3-1 fix (option 1, approved): imitable SFT traces — status

`sft_coldstart.generate_trace` was rebuilt. It is still fully programmatic from the solver, with no human CoT.
- **Solution choice:** each trace uses the smallest-subset solution `L op R`.
- **Kinds** (default mix):
  - **direct** (35%): check → `True` → answer.
  - **near_miss** (30%): closest wrong operator → `False V` → "V is too high/low; keep a, b; try every operation"
    → `print(a + b, a - b, a * b, a / b)` → answer.
  - **split** (15%): sweep → answer.
  - **resplit** (20%): sweep a wrong split → restate the 4 values, "None is T. Split differently" → sweep the right split → answer.
- **Invariants:**
  - every claim restates the real printed values;
  - the trace never answers after a miss;
  - python segments end at `</python>`, exactly where rollouts stop.

**Verified:**
- `check_sft_traces.py --n 1000`: 0 failures on both tokenizers. All outputs are real with no sandbox errors, the last output always
  shows `True` or the target, the first resplit output never contains it, answers are ref-verified, there are 0 eval keys, and masks are exact.
- 4 parametrized project tests; suite 132 passed / 2 skipped.
- Runner 26/26 PASS (`verification/logs/run_20261104_102636/SUMMARY.txt`).

**Effect (Qwen2.5-0.5B, 500 traces, 1 epoch; 24 stratified eval problems, greedy, 320 tokens; `verification/logs/r4/qwen_*.json`):**

| Config | Pass@1 | easy | Format | Notes |
|---|---|---|---|---|
| base, 2-shot | 0/24 | 0/8 | 24/24 | answers after one failing check |
| old SFT traces | 0/24 | 0/8 | 24/24 | answers with an expression just shown `False` |
| v2 traces (no resplit, ungrounded claim) | 2/24 | 2/8 | 24/24 | one genuine self-correction via the sweep |
| **v3 traces (current)** | **4/24** | **4/8** | 17/24 | 8 rollouts re-split after "None is T"; 13/20 wrong answers still claim a false hit |

That is a clear direction on a small sample, not a finding.
- **Remaining false claims:** these are the signal GRPO needs. Groups now contain both correct and incorrect rollouts on the easy tier.
- **Format drop:** comes from rollouts exceeding the 320-token budget after a re-split; the presets use 512.

**gemma-2-2b: CANNOT-VERIFY in budget.** SFT on MPS runs at ~9 s/step, so 250 traces take ~40 min, over the 15-minute limit.
The run was stopped at step 60 (`verification/logs/r4/sft_gemma_v3.txt`) pending your approval. The existing `outputs/sft_gemma`
adapter was trained on the OLD traces and should not be used for GRPO.

### Round 4 — base-model comparison on soos (AMD Ryzen AI Max+ 395 / Radeon 8060S, ROCm 7.2, torch 2.14.1+rocm7.2)

All runs: 48 stratified eval problems, greedy, 384 new tokens, `--on-round-cap notice` eval mode. Logs in `verification/logs/r5_soos/`.
- **ROCm sanity:** step-0 log-prob consistency passes on the GPU for gemma-2-2b (1.6e-5) and Qwen3-4B (3.1e-5); chunked log-probs match HF to ≤3e-6 (`E0_*`, `E7b_*`).
- **Batched generation** (gemma, 16 problems, 256 tokens): **sequential 11 m 59 s vs batched 1 m 37 s = 7.4× faster**, with identical metrics (format 0.50, mean length 203; `E4b_*`).

| Model | Setup | Pass@1 | easy / medium / hard | Format | Notes |
|---|---|---|---|---|---|
| gemma-2-2b | base, 2-shot | 1/48 | 1/16, 0, 0 | 1.00 | |
| gemma-2-2b | old-trace SFT (500, Mac) | 2/48 | 1/16, 1/16, 0 | 1.00 | indistinguishable from base. **Corrects §11's n=12 claim that it was worse** |
| gemma-2-2b | new-trace SFT (500) | 0/48 | 0, 0, 0 | 0.23 | learned the template, not the reasoning; 37/48 hit the length cap |
| **Qwen3-4B-Base** | base, 2-shot | **7/48** | **5/16, 2/16, 0** | **0.98** | 2 genuine recoveries after a failing check |
| Qwen3-4B-Base | new-trace SFT (500) | 5/48 | 2/16, 2/16, 1/16 | 0.17 | 5/8 of its answers are correct, but 40/48 hit the length cap: after the 2-round budget it keeps sweeping and invents values |

**New finding R4-1 (fixed):** Qwen3's tokenizer registers `<think>`/`</think>` as added single tokens that its base model never emits.
With plain `encode`, the tags degenerated (`" weit"`, `"�"`), format read 0, and the SFT loss stalled at ~0.30.
`rollout.encode_text` now keeps protocol tags as ordinary tokens. It is identical to `encode` for gemma and Qwen2.5 (tested), and the Qwen3 SFT loss now converges (0.027).

**Conclusions (small n, direction only):**
1. **Qwen3-4B-Base is the better base model.** It sits around 15% vs ~2% for gemma-2-2b, and it already follows the protocol from 2 examples.
2. **SFT on the current traces does not pass the gate for either model.** The traces never teach what to do when the execution budget runs out.
3. **Next:** either add budget-aware traces (stop sweeping once out of rounds), or start GRPO from base Qwen3 with the 2-shot prompt (ToRL-style). Training uses `on_round_cap="truncate"`, so overruns get reward 0, which is a direct learning signal.

## 12. Round 6: GRPO pilots B/C and code-level literature audit (2026-11-04)

### 12.1 Pilot B finding: the policy learned "check, see False, submit it anyway" (fixed)
Pilot B (Qwen3-4B-Base, retry-instruction prompt, 8×8, T=0.7): held-out stayed at 5–7/48 (steps 0–11). Retries after a
failed first check on held-out problems went 31/43 → 33/43 (step 4) → 23/43 (step 6) → **0/41 (step 9) → 0/42 (step 11)**,
and at step 11 the model submits the very expression its own check printed `False` for
(`verification/logs/r6_pilot/P2_eval_step*.jsonl`). Cause: a wrong but well-formed answer still earned R_format + R_exec =
0.2, while retrying cost tokens and rarely recovered (2–3/43). With 5–10% accuracy, most groups have no correct sample,
so the shaping terms carried most of the gradient.
**Fix (approved, applied):** `rewards.answer_contradicted`. A *wrong* final answer whose value was printed, or whose
expression was executed, in a tool call that printed `False` gets R_exec = R_format = 0 (`RewardConfig.zero_shaping_if_contradicted`,
default on; logged per step as `contradicted`). Tests: `tests/test_rewards.py` (6 new). On B's held-out rollouts the
rule flags 18/41 wrong answers at step 0, 26/41 at step 6 and 42/43 at step 11.
B and C were stopped (C had done 1 step in 58 min: 32×8 rollouts at ~21 tok/s while sharing the GPU).
C′ = `configs/pilot_qwen3_lit2.yaml` (16 prompts/step, new reward) runs as `P4` on soos.

### 12.2 Code-level comparison with reference implementations
Sources were fetched from raw.githubusercontent.com / api.github.com / export.arxiv.org, and each effective URL was checked:
TinyZero `verl/utils/reward_score/countdown.py`, `examples/data_preprocess/countdown.py`, `scripts/train_tiny_zero.sh`;
verl `verl/trainer/ppo/core_algos.py`; TRL `trl/trainer/grpo_config.py` and `grpo_trainer.py`; cognitive-behaviors
(Gandhi et al. 2503.01307) `scripts/train_tiny_zero_n4_cd.sh`; SimpleTIR (2509.02479) `recipe/simpletir/{agent_utils.py,
simpletir_ray_trainer.py,config/simpletir_trainer.yaml}`; ToRL `scripts/torl_1.5b.sh`; rStar2-Agent (2508.20722)
`rstar2_agent/reward/compute_score.py`, `down_sample/roc.py`.

**Matches (no action):**
| Ours | Reference |
|---|---|
| Group z-score advantage, unbiased std + eps | verl `compute_grpo_outcome_advantage` (norm_adv_by_std_in_grpo=True); TRL `scale_rewards="group"` |
| Loss = masked token sum / total trained tokens in the batch | verl `loss_agg_mode="token-mean"`; TRL default `loss_type="dapo"` |
| k3 KL in the loss; β=0 in C/C′ | verl `low_var_kl` (it also clamps to ±10); β=0 in DAPO/SimpleTIR/ToRL/TRL default |
| Log-probs at the sampling temperature | verl and TRL divide logits by T |
| Tool output tokens masked from the loss | SimpleTIR, rStar2 (and SPEC) |
| Refill groups with no reward variance | DAPO dynamic sampling; rStar2 `reject_equal_reward` |
| G=8, ε=0.2, no entropy bonus | TRL defaults (G=8); SimpleTIR/ToRL/rStar2 use 16 |
| Exact verifier (ast + Fraction) | TinyZero: regex + `eval` + 1e-5 tolerance (ours is stricter and safer) |

Not a bug: TinyZero requires *all* numbers to be used exactly once, while SPEC says "at most once", which is what we do. Our
accuracies are therefore not directly comparable to TinyZero's.

**Differences that matter.** Items 2–5 were approved and applied on 2026-11-04 (`updates_per_batch`, `clip_eps_high`, `tis_cap`, `length_threshold`; `configs/pilot_qwen3_lit3.yaml`; eval watcher n=200), running as P5:
1. **Reward shaping.** Every tool-use RL recipe checked rewards the outcome only: rStar2 `compute_score` returns 1/0
   and SimpleTIR uses math_verify. TinyZero's 0.1 format score is the only partial credit, and it has no tools. rStar2 handles tool quality
   *outside* the reward (Resample-on-Correct: oversample, then keep the positive traces with the fewest tool errors and format issues).
   SPEC mandates R_exec and R_format, so we keep them, guarded by §12.1. Watch the `contradicted` metric. If another
   shaping exploit appears, gate R_exec/R_format on the group having at least one correct answer, or move them into
   RoC-style selection.
2. **One optimizer step per generated batch (μ=1), which makes clipping a no-op.** References take several mini-batch steps
   per rollout batch (TinyZero 256/64 = 4; cognitive-behaviors 256/128 = 2) with clip-high 0.28 (DAPO, SimpleTIR).
   Generation is ~75% of step time on soos, so 2 updates per batch roughly doubles learning per GPU-hour. It needs a no-grad
   old-log-prob pass and `clip_eps_high`.
3. **No correction for the gap between generation and training numerics.** TRL defaults to truncated importance sampling
   (`vllm_importance_sampling_correction=True`) and verl has `rollout_is`. We sample with HF bf16 KV-cache decoding and
   train on a full forward. `gen_train_logp_dev` (max per step) reached 0.43 nats in C step 0. Proposal: log the mean
   and p99, and weight tokens by min(exp(lp_train − lp_gen), 2) (detached).
4. **Scale and length.** References use 256 prompts × 4–16 samples per step, responses of 1024–3072 tokens and T=1.0. We use 16×8, a 384-token
   cap and a length penalty beyond 256. Visible held-out gains should not be expected within hours on soos, and the
   cap/penalty may suppress the longer search behaviour reported by Gandhi et al. and TinyZero. Proposal (SPEC requires a length
   term): DAPO-style soft overlong penalty, `max_new_tokens` 768 with a threshold of 512.
5. **Held-out n=48** has a standard error of ~5 pp at p≈0.13, so it cannot detect changes under ~10 pp (TinyZero validates on 1312).
   Use n≥200 for decisions (at least for the final checkpoints).
6. Low priority: SimpleTIR void-trajectory filtering (0 void trajectories in B's held-out at steps 0/11, so not needed now).
   Dr. GRPO's removal of std normalization is optional (verl/TRL defaults keep it).

## 13. Round 7: generation speed on soos (2026-11-04)

**Symptom:** C″ step 0 took 1,580 s generating 22,362 tokens (~14 tok/s) plus 286 s in the loss pass.

**Measured** (`verification/bench_generation.py`; one GRPO batch = 16 prompts × 8, T=0.9, 768 tokens; logs in `runs/bench` on soos):
| Setup | Generation (s) |
|---|---|
| lockstep engine, stock SDPA, no LoRA | 187 |
| continuous engine, stock SDPA, no LoRA | 188–192 |
| compact engine (drop dead rows) | 380 |
| continuous + trainer's PEFT LoRA wrapper | 236 |
| ROCm env vars (HIP_FORCE_DEV_KERNARG, AOTriton experimental, hipBLASLt) | 199 (no gain; prefill slower) |
| **continuous + `sdpa_gqa`** | **107** |
| continuous + `sdpa_gqa` + LoRA wrapper | 156 |
| **trainer step, `sdpa_gqa` + merged-weight generation copy** | **107** (loss 257 s with grad checkpointing, **205 s** without) |

**Root cause** (py-spy on the live trainer, 999 samples):
- 43% of wall time was in `transformers` `repeat_kv`. With a padding mask, HF's SDPA path cannot use `enable_gqa` on CUDA/ROCm, so every decode step copies the whole KV cache 4× per layer (Qwen3-4B: 32 query / 8 KV heads).
- Another 51% was the host waiting on that GPU work in `_sample_batch`.

The micro-benchmark missed it because its inputs are unpadded (HF passes no mask, so it uses native GQA).

**My earlier diagnosis was wrong.** In §12 / chat I said lockstep batching was the main cause. Measured, the lockstep and continuous engines are equally fast here, so it was not.

**Fixes (applied, tested):**
- **`attention_gqa.py` (`attn: sdpa_gqa`):** single-token decode groups query heads per KV head (broadcast matmuls, no copy). Prefill and training delegate to stock SDPA, so training numerics are unchanged. Tests: it matches stock SDPA on random tensors incl. padded/idle rows, and greedy rollouts on real Qwen2.5-0.5B are token-identical to the sequential reference for continuous (bs 0/2) and lockstep.
- **`gen_merged`:** generate with a plain copy whose LoRA-targeted weights = base + delta, recomputed in fp32 from the pristine base each step (no bf16 drift). Test: logits match the PEFT forward to 1e-4 after repeated syncs.
- **`rollout.run_rollouts_batched(impl=continuous)` (default):** continuous batching with teacher-forced tool output, slot refill and left-trim of the cache. Token-identical to the reference in tests. It gives no speed-up over lockstep on this workload but is kept as the default.
- **Loss pass:** π_old pass only for mini-batches after the first; `grad_ckpt: false` with micro-batch 8 (16/32 without checkpointing OOM at 104 GB).

**Net:** ~31 min → ~5.2 min per 128-rollout step (6×).

**Not pursued:**
- vLLM on gfx1151: the user called it off. Community builds run `--enforce-eager --max-num-seqs 1` because HIP graphs freeze this GPU.
- AMD nightly torch wheels: older than the installed 2.14.1.

## 14. Round 8: go/no-go run P6 on soos (2026-11-04/05)
Config `configs/pilot_qwen3_lit4.yaml`. Evidence: `verification/logs/r6_pilot/` (`P6_eval_step*.jsonl`, `train_log.jsonl`,
`P6_train_part1.log`, `P6_train.log`). Paired check: `python3 verification/eval_compare.py verification/logs/r6_pilot/P6_eval_step0.jsonl verification/logs/r6_pilot/P6_eval_stepN.jsonl`.

| Checkpoint | Correct/200 (base 24) | only base / only new | McNemar p |
|---|---|---|---|
| step 10 | 37 | 3 / 16 | 0.004 |
| step 13 | 40 | 4 / 20 | 0.002 |
| step 16 | 36 | 6 / 18 | 0.023 |
| step 19 | 46 | 5 / 27 | <0.001 |
| step 20 | 43 | 5 / 24 | 0.001 |

- **PASS — learning:** the gain is significant and sustained from step 10. It is well above the 21-vs-24 kernel noise floor (§13).
- **PASS — no reward shortcut:**
  - `contradicted` stayed at 0.06–0.14 (base 0.23);
  - `retry_after_false` stayed at 0.96–1.00 (base 0.63; in pilot B it fell to 0);
  - `frac_max_length` stayed ≤ 0.016;
  - eval completion length is unchanged.
- **PASS — resume:** after a transient `hipErrorLaunchFailure` at step 12, `--resume auto` from step 10 reproduced step 10
  exactly (reward 0.293, correct 0.141).
- **OPEN:**
  - The vLLM path and the cloud smoke test are untested; cloud is blocked by a GCP GPU quota of 0 and an expired AWS login.
  - SPEC's 2-execution cap now binds on most failing rollouts.

**Update through step 41:** held-out 59/200 at step 40 (+17.5 pp, 3 vs 38 discordant). Shortcut metrics stay flat.
**FINDING (open, fix proposed):** value-only self-checks. 'Check printed True, answer wrong' went 11 → 48 (step 20) → 55 (step 30)
→ 23 (step 40). Invalid-number answers get 0.2 shaping. Proposed: also zero shaping when the answer breaks the number rule.

**Update through step 61:** held-out plateaus at 64–67/200 from step 46 (67 at step 60, +21.5 pp). Hard tier is flat at
7–10/66. Value-only-check errors are back to 7/200. DAPO refills keep ~15/16 informative groups (≈26 prompts sampled per step).

**Update through step 101:** plateau at 61–73/200 (best 73 at steps 66 and 99). Slow drift: KL 0.11 → 0.27–0.33, gen/train gap
0.026 → 0.042, clip_frac 0.004 → 0.006, eval length +10%. No guard tripped; continuing to 200 as a stability test.

## 15. Cloud smoke test on GCP L4 (2026-11-05): PASS
Evidence: `verification/logs/cloud_smoke_l4/outputs/cloud_smoke/SMOKE_REPORT.txt` (+ phase1/phase2/guard logs).
- **PASS** vLLM generation path on real CUDA: log-prob gap vs trainer mean 0.009–0.011, p99 ~0.19, the same as the HF path.
- **PASS** GCS commit → kill -9 → resume from bucket on a wiped disk → run completes.
- **PASS** cost guard trips and saves a checkpoint.
- **FIXED (harness):** the first attempt killed only the trainer PID. The orphaned vLLM EngineCore kept 11 GB, so the restart failed
  (`verification/logs/cloud_smoke_l4/outputs/cloud_smoke_try1/`). Now uses `setsid` plus a process-group kill.

## 16. Final held-out test (2026-11-05): P8 s90 = 478/1000
Setup:
- 1,000 problems from `data/eval.jsonl`, greedy, the same tool loop and verifier as training.
- Checkpoints were selected on the 300-problem validation split only. This is the first use of the test set.
- Full table and paired tests are in EXPERIMENTS.md §2.7f.

Checks:
- **PASS** Test/validation agreement: P8 s90 scored 148/300 (49.3%) on validation and 478/1000 (47.8%) on test. The ranking P8 ≫ P7 ≈ P6 ≫ base is the same on both.
- **PASS** Prompt control: the multi-candidate prompt alone does not explain the P8 gain.
  - On P7 s40 the multi prompt gives 362 → 364 (p = 0.94); on the base model it gives 133 → 113.
  - P8 s90 vs P7 s40 under the same prompt: 364 → 478 (47 / 161), p < 0.001.
- **CHECKED** The zero-shot base 0/1000 is genuine, not a scoring bug. The model answers with the target number instead of an expression (741 bare numbers).
- **CHECKED** Adapter guard: every adapter path had an `adapter_config.json`. The earlier empty-path → base-model bug cannot recur.

### 16a. Leakage audit (2026-11-05)
- **PASS, no exact overlap:** train-part (9,700), val (300) and test (1,000) are pairwise disjoint by (sorted numbers, target); `train.jsonl` has no duplicates.
  Neither few-shot demo problem is in val or test. ([25,3,1]→76 is train #3783.)
  567 test problems share a *number set* with some train problem but with a different target; that's expected given the small number pools.
- **DISCLOSE: 200 of the test problems were seen during development.** The soos "held-out" evals for P1–P6 (n = 48, then 200) used `eval.py --n-eval 200`, a tier-stratified subset of `data/eval.jsonl`, i.e. TEST problems.
  They drove P6's go/no-go, P6's checkpoint shortlist, and the P6 failure analysis behind P7's changes. P7, P8 and P9 checkpoint selection used the 300-problem validation split only.
- **PASS, no sign the leak inflated results:** on the 800 test problems never seen before the final eval, P8 s90 scores 381/800 (47.6%), against 48.5% on the seen 200.
  - P6 s99: 35.0% unseen vs 34.0% seen;
  - P7 s40: 35.8% vs 38.0%.
- **Headline:** the 800-problem number should be reported alongside the 1,000-problem one.
- **DISCLOSE: P6 trained without a validation hold-out** (`pilot_qwen3_lit4.yaml` has no `val_n`), so it may have trained on the 300 val problems. Its val score (100/300) may be slightly optimistic, and that only favours P6 in comparisons with P8.
- **Going forward:** P8 vs P9 is decided on validation, then the test set is run once, and both 1,000 and unseen-800 numbers are reported.

## 17. KL-reference reset (`kl_ref: init`) for P10 (2026-11-05)
- **PASS, unit test** `tests/test_train_rl.py::test_kl_ref_init_is_frozen_warm_start_and_hash_stable` (tiny Qwen2):
  - the reference equals the policy at the start (KL 0) and differs from the base;
  - every trainable parameter gets a gradient when the reference forward runs between the policy forward and backward;
  - the reference is unchanged after an optimizer step, while the policy moves;
  - `save_pretrained(selected_adapters=["default"])` writes no `ref/`;
  - `kl_ref: init` without `init_lora` is rejected.
- **PASS, hash stability:** `kl_ref` is excluded from the hash at its default. P8 = `6bf14ec2b5be` (unchanged) and P9 = `45fe4337cd30` (unchanged; P9 was launched on the pre-change code), so P8/P9 resume still works.
- **Full suite:** 219 passed, 10 skipped.
- **Not yet verified on GPU** with the real 4B model and vLLM. Before trusting P10, check that its step-0 KL is about 0 (vs about 0.06 for P8 at step 0) and that `val step 0` equals P8 s90's 148.

## 18. P10 hinted rollouts: smoke tests on soos (2026-11-05)
The runs used the real Qwen3-4B base, 4 prompts × G=4, 1–2 steps, `--hint-schedule 1.0@0` / `0.5@0`.
- **FOUND+FIXED #1:** the trigger required a zero-variance group. All-fail groups almost always vary in the exec/format shaping terms, so nothing was ever hinted (n_hint_groups = 0). The trigger is now *no correct answer*.
- **FOUND+FIXED #2:** no-correct groups with shaping variance took the training slots, so the hinted round had no room (again 0). Now, with hints on, the hinted resample *replaces* such a group, as QuestA does, up to `hint_max_groups`. With hints off the code path is unchanged, and the P8/P9 hashes are unchanged.
- **FOUND+FIXED #3 (observability):** the rollout dump wrote only the first 4 rollouts, which are always unhinted. It now also writes up to 4 hinted ones.
- **PASS:** n_hint_groups 3–4 per step, all with training signal. Hinted accuracy was 8–12% against 0–6% unhinted (base model; direction only).
  The dumped hinted prompt reads "... Hint: one solution contains 1 + (7 + 8). Reply in exactly this format: ...".
- **Side finding (affects P8/P9's interpretation, not their code):** all-fail groups that differ only in the shaping terms count as "signal". Group z-scoring gives them full-size advantages. So "about 59/64 signal groups" in P8 overstates the correctness signal.
  This is now logged as `n_signal_nocorrect`. It can't be recovered for P8/P9, because their rollout dumps keep 4 rollouts per dump step, not groups.
- **Note:** `--checkpoint-every 0` crashes with a ZeroDivisionError. That's pre-existing and unrelated; use a large value instead.

## 19. P10 result and checkpoint choice under evaluation noise (2026-11-05)
- **FOUND: "greedy" vLLM validation is not reproducible.** Batched LoRA kernels are numerically nondeterministic.
  - The same adapter with the same prompt (P8 s90; the P10 step-0 val) gave 141 vs 148, with only 138/300 completions identical.
  - P9 step 0 evaluated on two VMs gave 133 vs 131, with 62/300 completions identical.
  - Every divergence starts mid-completion (earliest at character 55), so this is not a prompt or config difference.
  - On validation (300 problems) one pass is good to about ±7 problems.
  - On test (1,000 problems) the aggregate is stable: P8 s90 scored 478, then 479, with 949/1000 per-problem outcomes agreeing.
- **FIX (procedure):** the final candidates (P8 s90, P10 s76/80/84/86/90, P11 SFT) were each validated in 3 separate vLLM processes with the adapter order shuffled. The choice used the 3-pass mean and majority-of-3 paired tests (EXPERIMENTS §2.7j).
  P10 s90 was chosen (tied with s84; the final step involves no selection).
- **PASS, test run once:** P10 s90 scored 504/1000 vs P8 s90 479 in the same run (74 / 99, p = 0.068). On the unseen-800 (§16a): 411 vs 383 (57 / 85, p = 0.023).
  Hints were never shown at evaluation: `eval_vllm.py` never passes `solution_hint`, and `rollout.format_prompt` defaults it to empty.
- **CHECKED, P10 warm start loaded correctly:** the step-0 val differs from P8's only through numerics, as shown above.
  The adapter md5 is `9f749ac5…` on the VM, on soos and in the bucket.
- **NEGATIVE, P11 SFT:** the 3-pass validation mean was 129.0 vs P8 s90's 140.3. Not used further.

## 20. Fresh holdout and P10b final result (2026-11-06)
- **PASS, holdout isolation:**
  - `data/holdout.jsonl` (sha256 `480a4ae9…`) is disjoint from train and test.
  - No training or selection code reads it (`tests/test_holdout.py`).
  - The P10b training process opened no file descriptors on it, and its config resolves to `data/train.jsonl`.
  - It was evaluated only on soos (a separate machine, inference only): twice for the baseline (P8 s90, P10 s90) before P10b finished, and once for the chosen P10b checkpoint after selection.
- **PASS, selection before the holdout:** P10b s80 was chosen on 3-pass validation only (mean 159.3 vs 151.7 for P10 s90; it is also the final step). The holdout was run after the choice was fixed.
- **PASS, same evaluation setup:** the `eval.py` configs of the baseline and final holdout runs differ only in the adapter and output paths. The soos copy of the P10b s80 adapter matches its step-80 COMMIT sha256 (`adc9f6cf…`), and the holdout file on soos matches the frozen hash.
- **RESULT, holdout** (each pair evaluated on the same 1,000 problems; paired McNemar tests):
  - P10b s80 528 (338 / 126 / 64) vs P10 s90 479: 110 / 61, p = 0.0002.
  - P10b s80 vs P8 s90 466: 122 / 60, p < 0.001.
- **RESULT, test** (vLLM, one process): P10b s80 527 vs P10 s90 494 (91 / 58, p = 0.009). The earlier P10 s90 run scored 504, and the two runs agree on 950/1000 problems.
- **FOUND (ops), corrected:** I first reported P10b's training log as lost. It was not: the run exited cleanly (02:43:39 UTC), and `cloud_train.sh` had uploaded the logs. A bucket listing was cut off by `head`.
  - The real fault was the ssh-based end-of-run waiter, which dropped silently, so the VM idled until max-run-duration deleted it at 05:14 (about 2.5 GPU-hours wasted).
  - **Fixes:** `cloud_train.sh` now also syncs logs every 5 min; `scripts/cloud_chain.sh` writes a DONE marker to the bucket and powers the VM off; waiters poll the bucket and the VM status, not an ssh session.
  - A second issue: the first eval-only VM script omitted `VLLM_USE_FLASHINFER_SAMPLER=0`, and vLLM failed on the FlashInfer JIT build. It was rerun with the variable set; no results came from the failed attempt.

## 21. Gemma-4-E4B evaluation on soos: ROCm SDPA gives NaN (2026-11-07)
- **FOUND:** the pre-registered sanity check of the HF backend on soos scored the final Gemma adapter 0/300 on the validation problems (vLLM: 92.3/300). Every completion was `<pad>` repeated, i.e. argmax of NaN logits.
- **Cause** (`verification/check_gemma4_rocm_sdpa.py`): on soos's ROCm build, `attn_implementation="sdpa"` returns NaN logits for Gemma 4, even for a single unpadded prompt. `sdpa_gqa` uses SDPA for prefill, so it fails too. `eager` gives finite logits with the same argmax as an unpadded run.
- **Scope:** training was not affected; it ran on CUDA (vLLM generation, SDPA training forward) and learned normally. The Qwen evaluations on soos are not affected.
- **FIX:** Gemma evaluations on soos use `--attn eager`. A second sanity run must land within 75–110/300 (band fixed before the run) before Gemma's one holdout3 evaluation.

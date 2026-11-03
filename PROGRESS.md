# PROGRESS.md

Status of the Countdown GRPO + LoRA build. PART 1 = everything verifiable on the
Mac + a Google Cloud handoff package. PART 2 (the actual cloud runs) is out of
scope for this session.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Environment, RESOURCES.md, scaffold, pinned requirements | ✅ done |
| 1 | data_generator.py + solver/verifier + tests + dataset cross-check | ✅ done |
| 2 | sandbox.py + adversarial tests + 10k stress + latency report | ✅ done |
| 3 | rollout.py + env_mask/segment/hallucination tests | ✅ done |
| 4 | rewards.py + reward-hack test suite | ✅ done |
| 5 | sft_coldstart.py + tiny smoke + gemma sanity | ✅ done |
| 6 | train_rl.py + overfit 50 + gemma sanity + resumability | ✅ done |
| 7 | eval.py on a small subset + hand-computed metrics | ✅ done |
| 8 | Cloud-agnostic GPU handoff (vast.ai/RunPod + GCS + GCP CPU) + HANDOFF/REPORT | ✅ done |

## Log

- 2026-11-03 — Phase 0: env verified (M4 Pro / arm64 / 24 GB / MPS), venv
  (Python 3.14.7) + pinned deps installed (torch 2.15.0, transformers 5.22.1,
  peft 0.21.2, datasets 5.1.0, accelerate 1.15.0). Tiny model chosen:
  Qwen/Qwen2.5-0.5B (base). Dataset & code refs verified + cited in RESOURCES.md.
  git init on `main`. HF_TOKEN not yet in env (needed for gemma-2-2b later).

- 2026-11-03 — Phase 1: `verifier.py` (ast + Fraction, only +−*/ , sub-multiset
  literals), `solver.py` (subset DP, positive-int intermediates), `data_generator.py`
  (build-target-by-construction, tiers easy 3–4n/2-digit, medium 5n/3-digit,
  hard 6n/3-digit; classic pool). 26 tests pass. Generated `data/train.jsonl`
  (10k: 4000/3000/3000) + `data/eval.jsonl` (1k: 400/300/300) in 0.55s, `--verify`
  green (all solutions exact + integer intermediates). Cross-check
  (`scripts/crosscheck_datasets.py`): Countdown-Tasks-3to4 = 490,364 rows,
  449,570 unique keys / 40,794 dupes (~8.3%), **100% solvable** on 2k sample,
  **32 keys overlap our eval** (dedupe before supplement); reasoning-gym v0.1.25 =
  our verifier accepts 300/300 of their answers, our solver solves 300/300.

- 2026-11-04 — Independent audit remediation (auditor verdict "NOT READY" =
  expected: Phases 2–7 not built). Fixed the two MAJOR + relevant MINOR items:
  **M1** — `requirements-cuda.txt` was unresolvable (`--index-url cu126` had no
  torch 2.15.0; vLLM 0.33.0 pins torch==2.14.0 + transformers<5.21.0). Now: drop
  index-url (PyPI torch 2.15.0 bundles CUDA); vLLM moved to `requirements-vllm.txt`
  (torch 2.14.0 + transformers 5.20.2 + vllm 0.33.0); both resolve via
  `uv pip compile` (lockfiles in `lockfiles/`). **M2** — tag tokenization is
  context-dependent (edge tokens merge with neighbours; segment-wise ≠ joint);
  corrected RESOURCES.md and recorded the rollout design constraint (carry IDs,
  never re-tokenize, decode-text stop detection). **m2** — `generate_one` now
  rejects `target in nums` (0 degenerate in both splits); dataset regenerated,
  new hashes `880fbfa7…`/`e29fa56c…`. **m5** — verifier normalizes whitespace and
  restricts charset (rejects `1_0`, `0xA`, `×`, floats). **m3/m6/m7/m8/m10** —
  gitignore (+`.safetensors`/`checkpoints/`/`.env`), solver-incompleteness note,
  lockfiles, README quickstart fixes, committed `logs/`. 31 tests pass.
  Remaining: **F8** (purge `.mpl-cache` from git history) needs explicit OK.

- 2026-11-04 — F8 applied: `git filter-repo --path .mpl-cache --invert-paths`
  purged the 584 KB font-cache artifact from history (all 4 later commits re-sha'd,
  no remote so nothing to re-push).

- 2026-11-04 — Phase 2: `sandbox.py` (persistent spawn pool, hard 50 ms kill +
  respawn, recycle every N, dedicated per-worker pipe → no stale-result leak) +
  `sandbox_worker.py` (minimal, no torch/transformers; AST whitelist: no Pow/
  attribute/subscript/import/f-string/comprehension/lambda, int+bool literals ≤1e12,
  depth/node caps; restricted `__builtins__` {print,range,len}; 200-char stdout,
  bounded buffer). 22 sandbox tests (adversarial + timeout-recovery + concurrency +
  stale-result). 10k stress (`logs/sandbox_stress.txt`): latency p50 0.063 ms /
  p95 0.083 ms / p99 0.190 ms, ~9.9k exec/s, RSS bounded ~54 MB with no growth.
  `verification/run_all_checks.sh` (NETWORK=1) → 1 expected FAIL (`deliverables_present`,
  phases 3–8 missing); F1–F9 all verified applied. 53 tests pass.

- 2026-11-04 — Phase 3: `rollout.py` custom generation loop (KV-cache manual
  loop) with tool interleaving. Stop strings detected on *decoded text* (not
  token-ID suffix); segment-wise tokenization + token-level `loss_mask`
  (prompt + injected `<output>` = 0, model-generated = 1); hallucinated
  `<output>` truncated + rebuilt + replaced; 2-round cap; max-length/EOS.
  HF_TOKEN resolved from macOS Keychain (`security find-generic-password -s
  HF_TOKEN -w`) — gemma-2-2b confirmed: vocab 256000, `padding_side=left`, merges
  `)</`/`><` (Qwen additionally merges `>\n`). 8 tokenizer-level tests (both
  tokenizers) + 2 model tests (ratio≈1 at step 0; full-rollout structure) pass;
  model tests gated by `RUN_MODEL_TESTS=1` (`logs/rollout_model_tests.txt`).
  Full suite 61 passed / 2 skipped; verification 1 expected FAIL only.

- 2026-11-04 — Phase 4: `rewards.py` — `R = R_correct + R_exec + R_format −
  λ·len`, each component returned separately. R_correct via `verifier.verify_answer`
  (model-independent); R_exec only if code ran cleanly AND references an input
  number (blocks `print(1)`/`print(target)`); R_format = exactly one `<answer>` +
  think→python→answer order; length penalty only beyond threshold (generated
  tokens); hallucinated `<output>` zeroes exec+format. `RolloutResult` now carries
  `completion_text`, `codes`, `code_oks`. 11 reward tests incl. the required hack
  suite — print(1), print(target), hallucinated output, reused numbers, answer in
  wrong place — all score below the honest correct rollout (1.2). 72 passed / 2
  skipped; verification 1 expected FAIL only.

- 2026-11-04 — Phase 6: `train_rl.py` custom GRPO loop — G=4, group-normalized
  advantage (zero-variance skipped + logged), clipped per-token ratio vs old-policy
  log-probs, adaptive KL (LoRA adapter disabled for reference), **vocab-chunked
  log-probs** (`forward_logprobs_chunked`, never materializes (B,T,V); fp32 cast for
  fp16 stability), tier curriculum, gradient accumulation (effective batch fixed),
  LoRA-only. `checkpoint.py` — preemption-safe: COMMIT objects (file list + sha256s +
  config/dataset hash, written last), verify + fallback-to-previous on corrupt,
  heartbeat + segment history, dual-write; local backend implemented, gcs/s3/hf-hub
  untested interfaces. Smoke (pipeline checks, not findings): Qwen overfit-50 reward
  0.05→0.20 (exec 0.025→0.10, format 0.025→0.10, correct flat at 0 — needs Part 2
  scale); gemma-2-2b 3 steps loaded + masked + gradient (KL −0.08→−0.22) with
  chunked-vocab, no OOM. `scripts/test_resume.sh` kill-9 test: run1 steps 0–3 →
  SIGKILL → run2 resumes at step 4 → step 6 (log committed). Fixed sandbox eager
  worker warm-up (spawn cost was tripping the 50 ms budget under load). 92 passed /
  2 skipped.

- 2026-11-04 — Phase 5: `sft_coldstart.py` — programmatic traces (wrong attempt →
  real sandbox `False` → correction → real `True` → answer; no human CoT) +
  LoRA SFT. `<output>` masked (label -100) via `rollout.assemble` (segment-wise,
  never re-tokenized); train-split problems only. 4 tests (trace structure,
  wrong-attempt != target, output masking, truncation). Smoke runs (`logs/`):
  Qwen2.5-0.5B 40 steps loss 0.49→0.18 (23 s); gemma-2-2b 8 steps final loss
  0.151 (24.6 s, no OOM) — pipeline checks, not findings. 76 passed / 2 skipped;
  verification 1 expected FAIL only.

- 2026-11-04 — Phase 7: `eval.py` — Pass@1 (overall + per-tier), format compliance,
  token length (mean/p50/p95/max), self-correction rate (definition documented) +
  blanked/shuffled-output control, Pass@1 vs rounds 0/1/2. 4 hand-computed unit
  tests. End-to-end on Qwen (12-problem subset, pipeline checks not findings):
  format — base(few-shot) 0.333 / SFT 0.917 / RL-overfit 0.0; pass@1 0.0 for all
  at this scale; self-correction 0 (no failing executions in the sample).

- 2026-11-04 — Part 2 Mac observation run (gemma-2-2b, ~4 h, 389 steps) found +
  fixed 2 bugs: (1) **MPS OOM** — full fp32 cast of 256k-vocab logits doubled
  memory (fixed: per-chunk fp32 + `--chunked-vocab`, `1c8743f`); (2) **adaptive KL
  coefficient underflow → correctness collapse** — `kl_coef` halved to 0, disabling
  the KL penalty, so the model drifted (KL 0.02→0.08) and collapsed (correct 0.75→0)
  (fixed: floor `kl_coef` at 1e-3 / cap 1.0, `dc6d816`). Post-fix run stable for
  389 steps (KL bounded, no collapse), but correctness stayed ~0.02-0.06 (noisy at
  G=4 on MPS; real gains belong to the cloud G=8 run). Logs: `logs/rl_gemma_part2*.txt`.

- 2026-11-04 — Phase 8: cloud-agnostic GPU handoff. `scripts/bootstrap.sh`
  (provider-neutral on-start, env-only) + `launch_anywhere.sh` (GPU detect → preset →
  resume, lockfile) + `publish_code.sh` / `pull_checkpoints.sh` / `watch.sh` /
  `cost_report.py` / `setup_cuda.sh` / `run_all.sh`; `scripts/gcp/{create_bucket,
  make_training_key, linux_validation_vm, teardown}.sh` (bucket-level IAM, CPU
  validation VM); `configs/preset-{24,48,80}gb.yaml` + `smoke_cuda.yaml` (effective
  batch 128 constant via grad-accum); `.env.example`; `HANDOFF.md` (exact Part 2
  order + provider notes + cost guard + MPS/CUDA differences); `REPORT.md` skeleton
  with `[PENDING]` placeholders. All scripts `bash -n` clean (shellcheck not
  installed). **verification/run_all_checks.sh → 0 failed (fully green).**

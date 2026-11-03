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
| 3 | rollout.py + env_mask/segment/hallucination tests | ⬜ pending |
| 4 | rewards.py + reward-hack test suite | ⬜ pending |
| 5 | sft_coldstart.py + tiny smoke + gemma sanity | ⬜ pending |
| 6 | train_rl.py + overfit 50 + gemma sanity + resumability | ⬜ pending |
| 7 | eval.py on a small subset + hand-computed metrics | ⬜ pending |
| 8 | Google Cloud handoff package + HANDOFF.md + REPORT.md skeleton | ⬜ pending |

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

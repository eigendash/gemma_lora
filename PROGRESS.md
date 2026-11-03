# PROGRESS.md

Status of the Countdown GRPO + LoRA build. PART 1 = everything verifiable on the
Mac + RunPod handoff. PART 2 (full SFT/GRPO/eval at scale) is out of scope for
this session.

## Phase status

| # | Phase | Status |
|---|---|---|
| 0 | Environment, RESOURCES.md, scaffold, pinned requirements | ✅ done |
| 1 | data_generator.py + solver/verifier + tests + dataset cross-check | ✅ done |
| 2 | sandbox.py + adversarial tests + 10k stress + latency report | ⬜ pending |
| 3 | rollout.py + env_mask/segment/hallucination tests | ⬜ pending |
| 4 | rewards.py + reward-hack test suite | ⬜ pending |
| 5 | sft_coldstart.py + tiny smoke + gemma sanity | ⬜ pending |
| 6 | train_rl.py + overfit 50 + gemma sanity | ⬜ pending |
| 7 | eval.py + RunPod handoff (scripts/configs/HANDOFF/REPORT skeleton) | ⬜ pending |

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
  **31 keys overlap our eval** (dedupe before supplement); reasoning-gym v0.1.25 =
  our verifier accepts 300/300 of their answers, our solver solves 300/300.

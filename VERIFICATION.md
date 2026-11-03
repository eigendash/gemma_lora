# VERIFICATION.md — Independent audit of PART 1

Auditor session: 2026-11-04. Repo HEAD audited: `7260338` (branch `main`, no remote, no other branches/stashes).
Machine: arm64 (Apple M4 Pro), 24 GB RAM, macOS 15.7.7, Python 3.14.7, torch 2.15.0, MPS available, no CUDA
(`verification/logs/env.txt`). Project source was **not modified**. All of my files are under
`verification/` and this document.

---

## 1. Verdict: **NOT READY**

1. Only Phases 0–1 of the 8-phase PART 1 plan exist. `sandbox.py`, `rollout.py`, `rewards.py`, `sft_coldstart.py`,
   `train_rl.py`, `eval.py`, configs, RunPod scripts, HANDOFF.md and REPORT.md are absent. PROGRESS.md says so honestly ("⬜ pending").
2. What exists (generator, solver, verifier, tests, cross-check) is **solid**. It reproduced in a clean room byte-for-byte,
   all hard data checks pass against an independent solver/verifier, and every number in PROGRESS.md reproduces exactly.
3. The CUDA handoff file `requirements-cuda.txt` is **unresolvable** as written (3 independent reasons, verified with `uv pip compile`).
4. A design premise recorded in RESOURCES.md for the highest-risk area (tag tokenization, which masking and stop
   detection depend on) is **false for the tiny model** and only partly true for Gemma.
5. Most spec rows (sandbox, masking, GRPO, rewards, eval, report) are therefore FAIL (not implemented), not FAIL (wrong).

---

## 2. Spec compliance matrix

Status legend: PASS / PARTIAL / FAIL / CANNOT-VERIFY. "MISSING" = file does not exist (`verification/logs/run_20261104_001044/deliverables_present.txt`).

| # | Requirement (SPEC.md / build prompt) | File / function | Status | Evidence |
|---|---|---|---|---|
| 1 | 10,000 train + 1,000 eval problems | `data_generator.py:generate_dataset` | PASS | `audit_dataset.txt`: `counts: train=10000 eval=1000` |
| 2 | Guaranteed solvability | `build_problem` (construct-by-combination) | PASS | Independent Fraction DFS solved 600/600 (100/tier/split): `unsolved=0` ×3; stored solution passes **independent** verifier 11000/11000; 0 non-integer/non-positive intermediates (independent parser) |
| 3 | Tiers: Easy 3–4 nums/2-digit, Medium 5/3-digit, Hard 6/3-digit | `TIERS` | PASS | train 4000/3000/3000 (easy: 1609×3-num, 2391×4-num), eval 400/300/300 (110/290); 0 tier-spec violations; target ranges easy 10–99, med 100–993, hard 100–994 |
| 4 | No train/eval overlap; no in-split duplicates | `generate_split(exclude=…)` | PASS | `overlap … 0`, `duplicate … keys: 0` both splits |
| 5 | Deterministic generation | `random.Random(seed)` | PASS | `determinism.txt`: 2 clean-room runs == author's `data/` (sha256 `17cb7b34…` train, `19289d17…` eval), also identical with `PYTHONHASHSEED=123` |
| 6 | No degenerate problems (sanity, build prompt) | `generate_one` | PARTIAL | target == an input: 65 train / 5 eval; one-op solvable: 13.8% easy-train, 13.5% easy-eval → **m2** |
| 7 | Difficulty increases by tier | — | PASS (weak med→hard) | median search nodes to first solution: easy 11.5 → medium 469 → hard 658.5; mean 44 → 1265 → 3470. Min-op distributions of medium vs hard nearly identical → **m9** |
| 8 | Answer verifier: exact, only `+ - * /`, literals ⊆ inputs | `verifier.py:verify_answer` | PASS | Fuzz vs independent hand-written parser: 5,985/5,991 agree; 6 disagreements all "newline inside expression" (**m5**); 27 edge cases incl. unicode ÷ × − full-width, floats, pow, div0, 10k-deep parens |
| 9 | Sandbox: AST whitelist, no imports/network/file I/O | `sandbox.py` | FAIL (MISSING) | B1 |
| 10 | 50 ms timeout, thread/process-level | `sandbox.py` | FAIL (MISSING) | B1 |
| 11 | stdout truncated to 200 chars | `sandbox.py` | FAIL (MISSING) | B1 |
| 12 | (L5) process isolation, no leak over 10k rollouts | `sandbox.py` | FAIL (MISSING) | B1 |
| 13 | Interleave: pause on `</python>`, inject `<output>\n…\n</output>`, resume | `rollout.py` | FAIL (MISSING) | B1; design premise at risk → **M2** |
| 14 | Max 2 execution rounds | `rollout.py` | FAIL (MISSING) | B1 |
| 15 | GRPO/PPO (TRL or native PyTorch) | `train_rl.py` | FAIL (MISSING) | B1 |
| 16 | G ≥ 4 per prompt | `train_rl.py` / configs | FAIL (MISSING) | B1 |
| 17 | `<output>` tokens label −100 | `rollout.py` | FAIL (MISSING) | B1 |
| 18 | R = Rcorrect + Rexec + Rformat − λ·length | `rewards.py` | FAIL (MISSING) | B1 |
| 19 | (L5) adaptive KL / length-penalized advantage bounds | `train_rl.py` | FAIL (MISSING) | B1 |
| 20 | SFT cold start or curriculum (programmatic traces, no human CoT) | `sft_coldstart.py` | FAIL (MISSING) | B1. No human-CoT dataset is referenced anywhere in code/docs (PASS for what exists) |
| 21 | (L5) dynamic curriculum on entropy/pass-rate | — | FAIL (MISSING) | B1 |
| 22 | Base zero-shot vs RL policy on 1,000 eval | `eval.py` | FAIL (MISSING) | B1 |
| 23 | Pass@1, format compliance, token-length expansion, self-correction frequency | `eval.py` | FAIL (MISSING) | B1 |
| 24 | REPORT.md (curves, backtracking examples, bottlenecks, reward-hack mitigations) | `REPORT.md` | FAIL (MISSING) | README says "skeleton with placeholders"; file does not exist |
| 25 | Reproducible setup script | README Quickstart; `scripts/setup_cuda.sh` | PARTIAL | Mac quickstart reproduced verbatim in clean room (`pytest_3runs.txt`); one undocumented step (**m8**); `setup_cuda.sh` MISSING; `requirements-cuda.txt` unresolvable (**M1**) |
| 26 | Quantitative test-time-compute scaling evidence | `eval.py` | FAIL (MISSING) | B1 |
| 27 | Logs committed with the repo | `logs/` | FAIL | no `logs/` directory; PROGRESS numbers have no log file (all reproduced by me, see §3 Phase-3 table) |
| 28 | Smoke config runs end to end | `configs/` | FAIL (MISSING) | `configs/` holds only `.gitkeep` |
| 29 | gemma-2-2b reachable (gated) | — | PASS | tokenizer downloaded with `HF_TOKEN` from `~/.zshrc` (`tokenizer_tags.txt`); no model code to run a 1–2 step check |
| 30 | RESOURCES.md entries exist / licenses / commits | `RESOURCES.md` | PARTIAL | all 13 PyPI versions HTTP 200, all 5 commits HTTP 200, all licenses match (`resources_web.txt`); tokenization claim false (**M2**) |

---

## 3. Findings (BLOCKER > MAJOR > MINOR)

### BLOCKER

**B1 — PART 1 is ~2/8 complete; all runtime components are missing.**
- Area: whole pipeline.
- Evidence: `git ls-files` lists only data_generator/solver/verifier, 3 test files, `scripts/crosscheck_datasets.py`, docs;
  `deliverables_present.txt`: `MISSING sandbox.py, rollout.py, rewards.py, sft_coldstart.py, train_rl.py, eval.py, HANDOFF.md, REPORT.md, scripts/setup_cuda.sh, scripts/run_all.sh, configs/*.yaml`.
  PROGRESS.md marks Phases 2–7 "⬜ pending". This is not misrepresented in PROGRESS.md, but the work does not exist.
- Impact: none of the sandbox, masking, GRPO, reward, SFT, eval or handoff audits can run. Part 2 cannot start.
- Fix: finish Phases 2–7. This is build work, not a fix, and is outside the fix list below.

### MAJOR

**M1 — `requirements-cuda.txt` cannot be resolved (3 independent causes).**
- Evidence (`cuda_requirements_resolve.txt`):
  - `uv pip compile requirements-cuda.txt --python-platform x86_64-manylinux_2_28 --python-version 3.14` →
    `error: … there is no version of torch==2.15.0`. The cu126 index tops out at `2.14.1+cu126`.
  - `--index-url` (not `--extra-index-url`) replaces PyPI, so `transformers`, `peft` and the rest are fetched from
    `download.pytorch.org/whl/cu126/transformers/`, which returns HTTP 403.
  - PyPI metadata: `vllm 0.33.0` requires `torch==2.14.0` and `transformers<5.21.0,>=5.16.1`, while the file pins torch 2.15.0 and transformers 5.22.1.
    No vLLM release supports torch 2.15 (latest is 0.33.0).
- The file is labelled "placeholder pins, NOT yet run", so this is not dishonest, but it will be the first command to fail on RunPod.
- Impact: cloud setup fails immediately. Also, a vLLM-enabled cloud stack *must* differ from the Mac-tested stack (torch 2.14 / transformers ≤5.20).
- Fix: drop `--index-url` (PyPI `torch==2.15.0` manylinux wheels exist and bundle CUDA), or use `--extra-index-url` with a cu index that carries 2.15.0.
  Move vLLM into a separate optional `requirements-vllm.txt` pinned to `torch==2.14.0` / `transformers==5.20.x`, or drop vLLM from the default path.
  Add `uv pip compile` for linux as a CI-style check (already in `run_all_checks.sh` under `NETWORK=1`).

**M2 — The tag-tokenization premise in RESOURCES.md is false for the tiny model and only partly true for Gemma.**
- Claim (RESOURCES.md §2): "each tag encodes to 3 stable tokens with no byte-fallback … so the same logic works for Gemma-2."
- Evidence (`tokenizer_tags.txt`):
  - Qwen2.5-0.5B: `'</python>\n' -> ['</', 'python', '>Ċ']` (the stop tag merges with the newline), `'print(x)</python>' -> [… ')</', 'python', '>']`,
    `'</output><answer>…' -> ['</', 'output', '><', …]`.
    Segment-wise tokenization ≠ joint tokenization: `concat(segment ids) == tokenize(joined text): False 20 19`.
  - gemma-2-2b: newlines stay separate and the joint check is `True 24 24` on that sample, but `')</'` and `'><'` merges still occur.
  - Pad side also differs: Qwen `padding_side=right`, gemma `left`.
- Impact: this sits in the highest-risk area.
  - Stop detection on a token-ID sequence for `</python>` will miss `)</`, and the stop on Qwen fires inside a token that also contains the newline.
  - Any re-tokenization of concatenated text shifts masks and makes generation-time and training-time log-probs disagree (ratio ≠ 1 at step 0).
  - Tests passing on Gemma-like boundaries would not cover Qwen, and vice versa.
- Fix: correct the RESOURCES.md note. Design `rollout.py` to carry token IDs end to end, never re-tokenize, detect stop strings on decoded text, and trim or keep the merged final token deliberately.
  Build `verification/test_masking.py`-style checks against both tokenizers.

### MINOR

**m1 — README and RESOURCES describe unbuilt components in the present tense.**
README "Layout" lists 8 nonexistent files as if they exist. The Quickstart says the test suite is "(tiny model, minutes)", but it uses no model and runs in 0.03 s.
README refers readers to `REPORT.md` and `HANDOFF.md`, which do not exist. RESOURCES.md §5 describes the sandbox ("Persistent multiprocessing worker pool (spawn), hard 50 ms") as built.
Evidence: `git ls-files`; `pytest_3runs.txt` (`26 passed in 0.03s`). Fix: mark these as "(planned, Phase N)".

**m2 — Degenerate problems in both splits, and the verifier accepts 0-op answers.**
- Counts: target equal to an input in 65 train / 5 eval problems, all but 6 of them easy (1.5% / 1.2% of easy).
  Solvable with a single operation: 550/4000 easy-train (13.8%) and 54/400 easy-eval (13.5%); overall 6.6% train / 6.4% eval.
- `verify_answer([50,3,3,1], 50, "50") == True` (edge test `0-op answer accepted`).
- Impact: inflates easy-tier Pass@1, gives RL an easy reward path with no reasoning or code, and produces more zero-variance groups.
- Fix: in `generate_one`, reject `target in nums`; optionally require min-ops ≥ 2 (or tag such problems).
  Regenerating changes the dataset hashes, so re-record them.

**m3 — `.gitignore` gaps for Part 2 artifacts.**
`adapter_model.safetensors`, `*.safetensors`, `checkpoints/` and `.env` are NOT ignored (`gitignore_coverage.txt`). There is no size policy for `logs/`.
Fix: add `*.safetensors`, `checkpoints/`, `.env`, and large-log patterns, while keeping small logs committed as the spec requires.

**m4 — A build artifact containing local host paths is in git history.**
Commit `4dbfbdd` added `.mpl-cache/fontlist-v3.11.0.json` (584,509 bytes, containing `/Users/dash/Library/Fonts/...` paths). It was deleted in `7260338` but remains in history.
No remote exists, so a rewrite is still cheap. Fix (destructive, needs your explicit OK): `git filter-repo --path .mpl-cache --invert-paths` before the first push.

**m5 — The verifier rejects a newline inside an expression and accepts Python-only literal spellings.**
`verify_answer([2,1,75],3,"2\n+1") -> False` (my reference: True); 6/5,991 fuzz disagreements, all of this kind.
It accepts `"1_0"` and `"0xA"` as 10. That is harmless value-wise, but it is Python grammar rather than Countdown grammar.
Evidence: `run_*/verifier_fuzz.txt`. Fix: normalize whitespace (`" ".join(s.split())`) and restrict the charset to `[0-9+\-*/() ]` before `ast.parse`.

**m6 — The project solver is incomplete under the verifier's semantics.**
`solver.solve` only explores positive-integer intermediates. The docstring says so, but 27 of 3,000 random 4-number problems that are solvable only with fractions or negatives get `found=False` (e.g. `[1,10,5,4]→42`; `solver_probe.txt`).
Impact: its "unsolvable" verdict is not authoritative, and the "100% solvable" cross-check figure is a lower bound.
Fix: document this in RESOURCES/PROGRESS, and never use `solve` to label problems unsolvable or to compute a pass@k ceiling.

**m7 — No lockfile.**
Top-level pins match the installed versions exactly (`key_pins.txt`; the author venv equals the clean room apart from dev extras, `freeze_diff.txt`), but 46 transitive packages are unpinned.
Fix: commit `uv pip compile` output for mac (and linux) as lock files.

**m8 — Undocumented setup steps.**
- The README never says to run `python data_generator.py`. `data/*.jsonl` is gitignored, and `scripts/crosscheck_datasets.py` crashes without `data/eval.jsonl`.
- The cross-check needs `requirements-dev.txt`, which the Quickstart does not mention.
- `HF_TOKEN` lives in `~/.zshrc`, which non-interactive shells (CI, agents, `bash -c`) do not load.

Fix: add these to the Quickstart and HANDOFF.md.

**m9 — Weak medium-vs-hard separation.**
Minimum-ops distribution: medium `{1:1, 2:22, 3:101, 4:76}`, hard `{1:3, 2:23, 3:111, 4:56, 5:7}`; median search nodes 469 vs 658.
Six numbers give more routes to a 3-digit target, so "hard" is barely harder. This is not a spec violation but matters for the curriculum.
Fix (optional): require hard targets with min-ops ≥ 4, or use a 3-digit range away from easy products.

**m10 — Numbers in PROGRESS.md have no backing log under `logs/`.**
I reproduced every one of them; see the table below. Fix: commit the cross-check and generation logs.

*Info (not findings):*
- 566 eval number-multisets also occur in train with a different target. This is expected from a 24-value pool and is not leakage under the spec's key.
- Countdown-Tasks-3to4 has no license metadata, which RESOURCES records correctly.
- My `hygiene.txt` "hard-coded path" grep hit is a false positive (`json.dumps` matches `mps`).

### Phase 3 claim-tracing table

| Claim (PROGRESS.md / RESOURCES.md) | Author log? | Reproduced by me | Evidence |
|---|---|---|---|
| 10k train 4000/3000/3000, 1k eval 400/300/300 | none | YES, exact | `audit_dataset.txt` |
| generation "in 0.55s", `--verify` green | none | YES (0.57 s with `--verify`) | `determinism.txt` |
| 26 tests pass | none | YES, 3×, 0 flaky, 0.03 s | `pytest_3runs.txt` |
| Countdown-Tasks-3to4: 490,364 rows; 240,632/249,732; 449,570 unique / 40,794 dupes; 2000/2000 solvable; 31 eval overlap | none | YES, all exact | `crosscheck_author_script.txt` |
| reasoning-gym 0.1.25: verifier 300/300, solver 300/300 | none | YES; plus independent ref verifier 300/300, their `score_answer` 300/300 | `crosscheck_independent.txt` |
| torch 2.15.0 / transformers 5.22.1 / peft 0.21.2 / datasets 5.1.0 / accelerate 1.15.0 | none | YES | `key_pins.txt` |
| 5 reference-repo commits + licenses | none | YES, all exist, all Apache-2.0 | `resources_web.txt` |
| "each tag encodes to 3 stable tokens … same logic works for Gemma-2" | none | **NO** | M2, `tokenizer_tags.txt` |
| Qwen vocab 151,643 | none | consistent (151,643 BPE + 22 added = `len(tok)` 151,665) | `tokenizer_tags.txt` |
| Secrets: none in tree or history | — | YES (no `hf_…`/`sk-…`/AKIA/`.env`) | `hygiene.txt`, `secrets_history.txt` |
| No TODO/FIXME, bare `except`, or swallowed exceptions | — | YES (grep clean) | `hygiene.txt` |

---

## 4. Could not verify

| Item | Why | What I did instead |
|---|---|---|
| Sandbox adversarial battery, latency p50/p95/p99, timeout kill/respawn, result integrity, concurrency, 10k leak test, worker `sys.modules` isolation, error-string leakage | `sandbox.py` does not exist | Nothing to attack. `test_sandbox_adversarial.py` not written: the API is undefined |
| Loss-mask independence check, boundary cases, generation/training log-prob consistency, 2-round cap, hallucinated `<output>`, stop/EOS/max-length handling, label shift, padding | `rollout.py` / `train_rl.py` do not exist | Measured the tokenizer boundary behaviour that these checks will depend on (M2) |
| Reward properties and reward-hack suite (≥8 hacks) | `rewards.py` does not exist | Verifier fuzz covers the `Rcorrect` core it will presumably call |
| SFT trace generation, synthetic mistakes in a real sandbox, SFT masking, eval-leak check | `sft_coldstart.py` does not exist | Confirmed the only external data touched is Countdown-Tasks (no solutions/CoT) and reasoning-gym, dev-only |
| GRPO advantages, loss sign, clipping, KL, chunked log-probs, LoRA-only grads, checkpoint/resume, curriculum, OVERFIT-50 | `train_rl.py` does not exist; no overfit logs exist | — |
| Eval metrics, hand-computed metrics, temperature-0 reproducibility, few-shot prompt | `eval.py` does not exist | — |
| Smoke config end to end, wall time | no configs | — |
| gemma-2-2b 1–2 step sanity | no training code | Verified gated access and tokenizer only |
| `setup_cuda.sh` / `run_all.sh` (`bash -n`, shellcheck, line review), HANDOFF.md measurements, eager attention, 256k-vocab chunking | files do not exist; shellcheck not installed | Resolved `requirements-cuda.txt` offline for linux (M1) |
| CUDA / vLLM paths | no CUDA on this machine | — |

---

## 5. Checks written (`verification/`)

| File | What it checks |
|---|---|
| `ref_countdown.py` | Independent reference: hand-written tokenizer + recursive-descent parser (no `ast`) with exact Fractions; independent iterative-deepening DFS solver (all of + − × ÷, fractions and negatives allowed) returning min-ops and search-node counts |
| `audit_dataset.py` | counts, tier spec, pool, positive int targets, in-split dupes, train/eval overlap, stored solution vs ref verifier, integer intermediates (independent), uses-all-numbers, degenerate rates, stratified ref-solver solvability + difficulty, project-solver agreement |
| `test_verifier_fuzz.py` | 6,000-case mutation fuzz of `verify_answer` vs reference (reuse, foreign, unary, extra parens, div0, floats, x/÷/−, full-width, long, empty, whitespace, double ops), 27 edge cases with spec-derived expected verdicts, large-literal and deep-nesting robustness |
| `crosscheck_external.py` | reasoning-gym answers vs project + ref + their scorer; Countdown-Tasks-3to4 sample: project solver vs ref solver (network) |
| `check_tokenizer_tags.py` | tag tokenization and segment-boundary merges for Qwen2.5-0.5B and gemma-2-2b (network, HF_TOKEN) |
| `run_all_checks.sh` | runs all of the above, plus the project suite ×3, determinism, secrets in history, gitignore coverage, build artifacts in history, deliverables present, shell lint (when scripts exist), CUDA pin resolution (`NETWORK=1`). Exit code = number of failed checks |

Re-run: `NETWORK=1 PY=.venv/bin/python verification/run_all_checks.sh` (for gemma: `zsh -c 'source ~/.zshrc; NETWORK=1 verification/run_all_checks.sh'`).
Baseline run `verification/logs/run_20261104_001044/` gives **9 PASS / 4 FAIL**: gitignore_coverage (m3), no_build_artifacts_in_history (m4),
deliverables_present (B1), cuda_requirements_resolve (M1).

Other logs in `verification/logs/`: `env.txt`, `pytest_3runs.txt`, `cleanroom_freeze.txt`, `author_freeze.txt`, `freeze_diff.txt`, `key_pins.txt`,
`determinism.txt`, `audit_dataset.txt`, `hygiene.txt`, `crosscheck_author_script.txt`, `crosscheck_independent.txt`, `resources_web.txt`,
`cuda_requirements_resolve.txt`, `tokenizer_tags.txt`, `solver_probe.txt`.

Clean-room procedure: `git clone` into a temp dir → `uv venv --python 3.14 .venv` → `uv pip install -r requirements-mac.txt` (4 s with a warm cache) → `pytest`.
That reproduced with no deviations, apart from m8 (data generation and dev deps for the cross-check).

---

## 6. Top 5 risks for Part 2 (cloud run)

1. **Token-boundary handling in the rollout engine (M2).** Stop detection, `<output>` injection, masking and old-policy log-probs all depend on never re-tokenizing.
   Qwen and Gemma merge differently, so a mask bug can pass tests on one tokenizer and fail on the other, and an RL run with wrong masks or ratios still "trains".
2. **The cloud software stack differs from the Mac-tested stack (M1).** vLLM forces torch 2.14 / transformers ≤5.20, so the cloud stack would differ from the tested torch 2.15 / transformers 5.22.1.
   Gemma-2 specifics (eager attention, logit soft-capping) and generation APIs may behave differently.
3. **Memory for the 256k-vocab log-probs.** Full logits for G=8 × 1k tokens × 256,000 vocab in fp32 come to about 8 GB per forward pass before backward.
   Chunked log-prob computation is a must and has not been written or tested.
4. **Sandbox throughput and timeout semantics on Linux.** The 50 ms limit with fork rather than macOS spawn, IPC overhead per call across G × batch rollouts, and worker recycling are all untested.
   A stale-result queue bug would silently corrupt `<output>` and `Rexec`.
5. **Reward exploitation through degenerate easy problems (m2, m9) with no reward code yet.** About 14% of easy problems take one operation and some targets equal an input, while medium and hard are barely separated.
   Expect early reward saturation on easy, many zero-variance groups, and a curriculum signal that does not track real difficulty.

---

## 7. Proposed fixes (awaiting your approval — nothing applied)

| ID | Fixes | Change | Risk |
|---|---|---|---|
| F1 | M1 | Rewrite `requirements-cuda.txt`: PyPI torch 2.15.0 (no `--index-url`), split vLLM into `requirements-vllm.txt` with torch 2.14.0 / transformers 5.20.x; verify with `uv pip compile` for linux | low |
| F2 | M2 | Correct RESOURCES.md §2 tokenization note with the measured merges and add the design constraint (carry IDs, never re-tokenize, decoded-text stop detection) | docs only |
| F3 | m1, m8, m10 | README: mark Layout entries as planned, fix the Quickstart claim, add the data-generation and dev-deps steps, the HF_TOKEN note, and commit the reproduction logs | docs only |
| F4 | m2 (+m9 optional) | `generate_one`: reject `target in nums` (+ optionally `min_ops ≥ 2`, hard `min_ops ≥ 4`); regenerate data and re-record hashes; add tests | changes dataset |
| F5 | m5 | `verify_answer`: whitespace normalization + charset whitelist; add tests | low |
| F6 | m3 | `.gitignore`: `*.safetensors`, `checkpoints/`, `.env`, large logs | low |
| F7 | m7 | Commit `uv pip compile` lock files (mac + linux) | low |
| F8 | m4 | Purge `.mpl-cache/` from history with `git filter-repo` (**rewrites history**; requires explicit OK) | destructive |
| F9 | m6 | Document solver incompleteness in RESOURCES/PROGRESS (optionally add a Fraction mode) | low |

B1 (building Phases 2–7) is not a fix. When it lands, I will add `test_sandbox_adversarial.py`, `test_masking.py` and `test_reward_hacks.py` against the real APIs and rerun this audit.

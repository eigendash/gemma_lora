# Countdown RL — Emergent Reasoning via GRPO + LoRA

Train `google/gemma-2-2b` with LoRA using a custom GRPO loop to solve the
[Countdown numbers game](https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_round)
via a `<think>` / `<python>` / `<output>` / `<answer>` protocol, where `<output>`
is real sandbox stdout injected during generation.

## Status

**PART 1** (everything buildable + verifiable on a Mac, ending in a RunPod-ready
handoff package). Full SFT / GRPO run / eval at scale is PART 2 and out of scope
for the current session.

## Protocol

```
<think>
... hypothesis / arithmetic reasoning ...
<python>
target = 952
res = (100 + 75) * 5
print(res == target, res)
</python>
<output>
False 875
</output>
... backtrack and adjust ...
<answer>
(100 - 3) * 8 + ...
</answer>
```

Generation pauses on `</python>`; the sandbox runs the code; the real stdout is
injected inside `<output>...</output>`; generation resumes. Max 2 code rounds.

## Layout

- `data_generator.py` — synthetic problems (guaranteed solvable, tiered, deduped; seed-0 hashes in `logs/data_hashes.txt`)
- `solver.py` / `verifier.py` — reference solver + model-independent answer verifier
- `sandbox.py` / `sandbox_worker.py` — execution sandbox: AST whitelist, restricted builtins, 50 ms hard
  timeout (kill + respawn), stdout truncated to 200 chars; workers are stdlib-only subprocesses
- `rollout.py` — custom generation loop with tool interleaving + env-token masking (the model is fed
  exactly the ids that are trained on)
- `rewards.py` — composite reward (correct + exec + format − λ·len), each component logged
- `sft_coldstart.py` — programmatic cold-start traces + LoRA SFT (`<output>` masked)
- `train_rl.py` — custom GRPO loop (k3 KL, softcap-correct chunked log-probs, curriculum, resumable,
  bucket checkpoints, cost guard)
- `checkpoint.py` — create-only COMMIT checkpoints over a bucket (`local` | `gcs`) + push/pull CLI
- `eval.py` — Pass@1 (overall/tier), format, length, self-correction (+ blank/shuffle controls), rounds sweep
- `configs/` — VRAM presets + smoke config (**not yet run at scale**)
- `scripts/` — cloud handoff (`bootstrap.sh`, `launch_anywhere.sh`, `publish_code.sh`, …; **not run
  against any cloud**) and Mac helpers
- `tests/` — unit + behaviour tests (fast suite needs only tokenizers; `RUN_MODEL_TESTS=1` for model tests)
- `verification/` — independent audit checks (`verification/run_all_checks.sh`) and `VERIFICATION.md`

## Quickstart (Mac)

```bash
export HF_HOME="$PWD/.hf-cache"     # keep HF cache inside the workspace
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt

# generate the dataset (gitignored; reproducible with --seed 0):
.venv/bin/python data_generator.py --n-train 10000 --n-eval 1000 --seed 0 --verify

# test suite (tokenizers only, no model weights; ~5 s):
.venv/bin/python -m pytest -q

# optional: dataset cross-check (needs dev deps + network):
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python scripts/crosscheck_datasets.py
```

`google/gemma-2-2b` is gated: `export HF_TOKEN=...` (a read token from an account
that accepted the Gemma license) before the gemma sanity checks. Note it must be
exported for non-interactive shells too (it is *not* read from `~/.zshrc`).

## Environment

Dev machine: Apple M4 Pro (arm64, 24 GB RAM), MPS. No CUDA/vLLM/bitsandbytes.
Training/eval code is device-agnostic (`--device cuda|mps|cpu|auto`). Generation is the hand-written HF
loop in `rollout.py`; there is **no vLLM backend** (`requirements-vllm.txt` only records a resolvable
vLLM stack for a future, separate venv).

See `RESOURCES.md` for verified dependencies/references, `PROGRESS.md` for phase
status, `REPORT.md` (skeleton) for the final memo, `HANDOFF.md` for the exact PART 2
commands, and `VERIFICATION.md` for the independent audit.

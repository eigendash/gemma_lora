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

Built (Phase 0–1):

- `data_generator.py` — synthetic problems (guaranteed solvable, tiered, deduped)
- `solver.py` / `verifier.py` — reference solver + model-independent answer verifier
- `tests/` — unit tests (31, run in <0.1 s, no model)
- `scripts/crosscheck_datasets.py` — solver/verifier cross-check vs two references

Planned (Phases 2–7, not yet written):

- `sandbox.py` / `sandbox_worker.py` — safe in-process executor (AST whitelist, 50 ms)
- `rollout.py` — custom generation loop with tool interleaving + env-token masking
- `rewards.py` — composite reward (correct + exec + format − λ·len)
- `sft_coldstart.py` — programmatic cold-start traces + LoRA SFT
- `train_rl.py` — custom GRPO loop
- `eval.py` — Pass@1, format, length, self-correction metrics
- `configs/` — full-run + smoke configs (full-run marked "not yet run at scale")
- `scripts/setup_cuda.sh`, `scripts/run_all.sh` — RunPod setup + reproduction
- `REPORT.md` (skeleton), `HANDOFF.md` — Phase 7

## Quickstart (Mac)

```bash
export HF_HOME="$PWD/.hf-cache"     # keep HF cache inside the workspace
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt

# generate the dataset (gitignored; reproducible with --seed 0):
.venv/bin/python data_generator.py --n-train 10000 --n-eval 1000 --seed 0 --verify

# fast unit tests (no model loaded, ~0.05 s):
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
All code is device-agnostic (`cuda|mps|cpu`, `--device` override, `--backend hf|vllm`
default `hf`). The `vllm` path will be written but **untested**.

See `RESOURCES.md` for verified dependencies/references, `PROGRESS.md` for phase
status, `REPORT.md` (Phase 7 skeleton) for the final memo, and `HANDOFF.md`
(Phase 7) for the exact PART 2 commands.

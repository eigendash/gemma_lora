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

- `data_generator.py` — synthetic problems (guaranteed solvable, tiered, deduped)
- `solver.py` / `verifier.py` — reference solver + model-independent answer verifier
- `sandbox.py` / `sandbox_worker.py` — safe in-process executor (AST whitelist, 50 ms)
- `rollout.py` — custom generation loop with tool interleaving + env-token masking
- `rewards.py` — composite reward (correct + exec + format − λ·len)
- `sft_coldstart.py` — programmatic cold-start traces + LoRA SFT
- `train_rl.py` — custom GRPO loop
- `eval.py` — Pass@1, format, length, self-correction metrics
- `configs/` — full-run + smoke configs (full-run marked "not yet run at scale")
- `scripts/` — RunPod setup + one-command reproduction

## Quickstart (Mac)

```bash
export HF_HOME="$PWD/.hf-cache"     # keep HF cache inside the workspace
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
# run the fast test suite (tiny model, minutes):
.venv/bin/python -m pytest -q
```

## Environment

Dev machine: Apple M4 Pro (arm64, 24 GB RAM), MPS. No CUDA/vLLM/bitsandbytes.
All code is device-agnostic (`cuda|mps|cpu`, `--device` override, `--backend hf|vllm`
default `hf`). The `vllm` path is written but **untested**.

See `RESOURCES.md` for verified dependencies/references, `PROGRESS.md` for phase
status, `REPORT.md` for the final memo (skeleton with placeholders in PART 1),
and `HANDOFF.md` for the exact PART 2 commands.

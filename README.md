# Countdown RL: tool-use reasoning from GRPO + LoRA

GRPO with LoRA on base language models for the Countdown numbers game. The model writes `<think>`, may run
`<python>` code, receives the real sandbox stdout inside `<output>`, and commits to an `<answer>`. Environment tokens
are masked out of the loss. The task description is in [SPEC.md](SPEC.md); the two-page memo is
[REPORT.md](REPORT.md) and the full paper [REPORT_FULL.md](REPORT_FULL.md); the run-by-run record, including pre-registered predictions, is [EXPERIMENTS.md](EXPERIMENTS.md).

Main result: Qwen3-4B-Base trained from a zero-shot prompt that only describes the tool goes from 6.7% to 69.9%
pass@1 on a fresh 1,000-problem holdout, matching a policy trained with a worked demonstration (68.1%, McNemar
p = 0.23). The same recipe takes Gemma-4-E4B from 1.9% to 60.5%, and to 68.1% with 100 more steps.

## Layout

The four files named in the task are thin entry points at the top level: `data_generator.py`, `sandbox.py`,
`train_rl.py` and `eval.py`. The code they call lives in the package:

```
src/countdown_rl/
  data/         generator.py   problem generator, tiers, dedup              (SPEC A: data_generator.py)
                solver.py      exact solver used to guarantee solvability
                verifier.py    model-independent answer check (AST + Fractions)
                holdout.py     fresh holdout sets, disjoint from everything before them
  sandbox/      executor.py    AST whitelist, 50 ms timeout, 200-char stdout   (SPEC B: sandbox.py)
                worker.py      stdlib-only worker process (python -I -S), respawned on timeout
  rl/           grpo.py        GRPO training loop                              (SPEC C: train_rl.py)
                rollout.py     generation with tool interleaving and loss masks
                rollout_vllm.py  the same protocol on colocated vLLM
                rewards.py     R = R_correct + R_exec + R_format - lambda * len
                prompting.py   prompt styles (zero-shot and demo prompts)
                checkpoint.py  atomic bucket checkpoints, resume, cost accounting
                attention.py   grouped-query SDPA without KV repeat (ROCm generation)
  evaluation/   evaluate.py    pass@1, format, length, self-correction, controls (SPEC D: eval.py)
                evaluate_vllm.py  vLLM evaluator used for in-run validation
                metrics.py     metrics recomputed from per-problem dumps
                paired.py      paired McNemar test between two dumps
configs/experiments/<name>/    one YAML per training phase; the run id is <name>-<phase>
results/                       every eval and training log behind a number in the reports
scripts/                       figures, config generation, cloud launchers, data checks
tests/unit/, tests/independent/
```

Experiment names read `<model>-<recipe>-<prompt>-s<seed>`. Recipe v1 is the original GRPO setup (adaptive KL,
hinted resampling, shaping reward on all-fail groups); v2 drops those three and adds gradient clipping. `demo` runs
use a one-example prompt; `zeroshot` runs use `r1zero_tool`, which describes the tool and shows no example. `-cont`
is a continuation from the final step of the parent run.

## Environments

Two environments, both managed by uv from one lock file (`uv.lock`, Python 3.14, linux x86_64):

| Where | What runs there | Setup |
|---|---|---|
| GCP, one RTX PRO 6000 (96 GB), spot | all training, in-run validation (vLLM 0.33, CUDA 13) | `uv sync --extra cuda` |
| Workstation, AMD Strix Halo APU (Radeon 8060S, unified memory, ROCm 7.2) | holdout and test evaluations (HF backend), probes | `uv sync --extra rocm` |

The extras conflict by design; pick one per machine. Holdout files never leave the workstation.

```bash
export HF_HOME=$PWD/.hf-cache
uv sync --extra rocm            # or --extra cuda
uv run pytest -q                # about 400 tests; model-dependent ones skip without weights
```

Gemma-4 checkpoints are multimodal. `scripts/prepare_gemma4_text.py` extracts the text tower (logits match the full
model exactly with eager attention). On ROCm, Gemma-4 must run with `--attn eager`; SDPA produces NaNs there.

## Reproducing

`bash scripts/reproduce.sh` does all of the following on a Linux machine with an NVIDIA or AMD GPU: it builds the
environment, regenerates the data and the three holdouts, runs the tests, redraws the figures, and prints the commands
for training and evaluation (`--smoke` adds a 30-problem evaluation of the main adapter). The steps one by one:

Data (deterministic; the hashes are checked in `tests/unit/test_holdout.py`):

```bash
uv run countdown-data --n-train 10000 --n-eval 1000 --seed 0 --verify
uv run python -m countdown_rl.data.holdout --seed 20261105 --out data/holdout.jsonl
uv run python -m countdown_rl.data.holdout --seed 20261106 --out data/holdout2.jsonl \
    --exclude data/train.jsonl data/eval.jsonl data/holdout.jsonl
uv run python -m countdown_rl.data.holdout --seed 20261107 --out data/holdout3.jsonl \
    --exclude data/train.jsonl data/eval.jsonl data/holdout.jsonl data/holdout2.jsonl
```

Training. One phase on any CUDA GPU with at least 48 GB:

```bash
uv run countdown-train --config configs/experiments/qwen3-4b-v2-zeroshot-s2/phase_a.yaml --output_dir runs/a
```

Later phases warm-start from the previous phase (`init_lora`). On GCP the whole lineage runs unattended, with
checkpoints every two steps, auto-resume after preemption, a cost cap, and power-off at the end:

```bash
EXPERIMENT=qwen3-4b-v2-zeroshot-s2 BUCKET=gs://<bucket> CHAIN_ENV="EVAL_PROMPT=r1zero_tool" \
    bash scripts/cloud/create_gpu_vm.sh
```

New lineages are generated from the hand-tuned one, for example
`python scripts/make_lineage_configs.py qwen3-4b-v2-zeroshot-s3 Qwen/Qwen3-4B-Base 3 prompt_style=r1zero_tool ...`
(the full v2 override set is in the script's docstring).

Evaluation, as run for every holdout number in the report (greedy, 768 tokens, two code rounds):

```bash
uv run countdown-eval --model Qwen/Qwen3-4B-Base --adapter <adapter dir> --eval-file data/holdout3.jsonl \
    --prompt-style r1zero_tool --max-new-tokens 768 --max-rounds 2 \
    --attn sdpa_gqa --batch-size 128 --out h3.json --dump h3.jsonl      # Gemma: --attn eager
uv run python -m countdown_rl.evaluation.paired a.jsonl.gz b.jsonl.gz     # McNemar
uv run python scripts/figures.py                                        # docs/figures/
uv run python scripts/sample_and_verify.py                              # pass@k and sample-and-verify
```

## Trained adapters

The final adapter of every reported run is on the Hugging Face Hub. Each repository contains exactly the file that
was evaluated, and its model card has a usage example. Download with
`hf download prada5023/countdown-rl-<experiment> --local-dir adapters/<experiment>`.

| Experiment | holdout3 | Adapter |
|---|---|---|
| `qwen3-4b-v2-zeroshot-s2` | 69.9% | [prada5023/countdown-rl-qwen3-4b-v2-zeroshot-s2](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v2-zeroshot-s2) |
| `qwen3-4b-v2-zeroshot-s2-cont` | 70.5% | [prada5023/countdown-rl-qwen3-4b-v2-zeroshot-s2-cont](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v2-zeroshot-s2-cont) |
| `qwen3-4b-v2-demo-s2` | 68.1% | [prada5023/countdown-rl-qwen3-4b-v2-demo-s2](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v2-demo-s2) |
| `qwen3-4b-v2-demo-s2-cont` | 68.2% | [prada5023/countdown-rl-qwen3-4b-v2-demo-s2-cont](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v2-demo-s2-cont) |
| `qwen3-4b-v1-handtuned` | 52.1% | [prada5023/countdown-rl-qwen3-4b-v1-handtuned](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v1-handtuned) |
| `qwen3-4b-v1-handtuned-nohint` | – | [prada5023/countdown-rl-qwen3-4b-v1-handtuned-nohint](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v1-handtuned-nohint) |
| `qwen3-4b-v1-s1` | 51.2% | [prada5023/countdown-rl-qwen3-4b-v1-s1](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v1-s1) |
| `qwen3-4b-v1-s2` | 49.8% | [prada5023/countdown-rl-qwen3-4b-v1-s2](https://huggingface.co/prada5023/countdown-rl-qwen3-4b-v1-s2) |
| `qwen3-8b-v1-s0` | 55.7% | [prada5023/countdown-rl-qwen3-8b-v1-s0](https://huggingface.co/prada5023/countdown-rl-qwen3-8b-v1-s0) |
| `gemma4-e4b-v2-zeroshot-s0-cont` | 68.1% | [prada5023/countdown-rl-gemma4-e4b-v2-zeroshot-s0-cont](https://huggingface.co/prada5023/countdown-rl-gemma4-e4b-v2-zeroshot-s0-cont) |
| `gemma4-e4b-v2-zeroshot-s0` | 60.5% | [prada5023/countdown-rl-gemma4-e4b-v2-zeroshot-s0](https://huggingface.co/prada5023/countdown-rl-gemma4-e4b-v2-zeroshot-s0) |
| `gemma4-e4b-v1-s0` | 30.9% | [prada5023/countdown-rl-gemma4-e4b-v1-s0](https://huggingface.co/prada5023/countdown-rl-gemma4-e4b-v1-s0) |
| `gemma4-e2b-v2-zeroshot-s0` | 4.4% | [prada5023/countdown-rl-gemma4-e2b-v2-zeroshot-s0](https://huggingface.co/prada5023/countdown-rl-gemma4-e2b-v2-zeroshot-s0) |

The Gemma adapters target the text decoder extracted by `scripts/prepare_gemma4_text.py`, not the multimodal
checkpoint.

## Tests

`tests/unit` covers each module. `tests/independent` re-derives the important properties without importing the
code under test where possible: a separate Countdown reference, GRPO advantage and loss math checked by hand,
adversarial sandbox inputs (imports, dunder access, infinite loops, memory bombs, output floods), 20 hand-built
reward hacks that must all score below an honest answer, and fuzzing of the verifier. The loss-mask tests check
that every token inside `<output>` has label -100 and that the model is fed exactly the ids it is trained on.

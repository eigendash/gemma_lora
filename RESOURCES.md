# RESOURCES.md

Verified dependencies, models, datasets, and code references for this project.
Every entry below was checked for existence, license, and (where applicable)
exact version / commit on 2026-11-03. Anything not yet verified is marked
**PENDING** and must not be cited as verified.

## 1. Environment & pinned dependencies (tested this session)

Dev machine: Apple M4 Pro, arm64, 24 GB RAM, MPS (PyTorch `mps` backend
`is_available() == True`). Python 3.14.7 (homebrew) + `uv 0.12.18`. No CUDA /
vLLM / bitsandbytes.

| Package | Version (tested) | Notes |
|---|---|---|
| torch | 2.15.0 | MPS available & built |
| transformers | 5.22.1 | 5.x; APIs used verified (AutoTokenizer/AutoModelForCausalLM/GenerationConfig/LogitsProcessorList/StoppingCriteriaList/PreTrainedTokenizerFast) |
| peft | 0.21.2 | LoraConfig/get_peft_model/PeftModel verified |
| datasets | 5.1.0 | |
| accelerate | 1.15.0 | |
| pytest | 9.2.0 | |
| numpy | 2.5.4 | |
| pyyaml | 6.0.3 | config files |
| huggingface-hub | 2.4.0 | |
| tokenizers | 0.23.3 | (transitive) |
| vllm | 0.33.0 | **cloud only, untested** (pinned in `requirements-cuda.txt`) |
| reasoning-gym | 0.1.25 (dev only) | cross-check only, never a runtime import |

Pinned files: `requirements-mac.txt` (tested), `requirements-cuda.txt`
(placeholder pins, **not yet run**), `requirements-dev.txt` (adds reasoning-gym).

## 2. Models

| Model | License | Gated | Role |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` (base) | apache-2.0 | no | **Tiny model** for unit tests & smoke runs. Chosen over SmolLM2-360M: base (not instruct) model matching gemma-2-2b's pretrained nature, larger vocab (151,643) → cleaner code/number tokenization. |
| `Qwen/Qwen2.5-0.5B-Instruct` | apache-2.0 | no | checked, not used (instruct; we need a base model analog) |
| `HuggingFaceTB/SmolLM2-360M` | apache-2.0 | no | checked, not used (see above) |
| `google/gemma-2-2b` | gemma | **manual (gated)** | Target model. Requires `HF_TOKEN` in env (license accepted by user). Not downloaded yet. |

Protocol-tag tokenization (verified for both Qwen2.5-0.5B and SmolLM2-360M):
each tag encodes to 3 stable tokens with no byte-fallback (e.g. `</python>` →
`["</", "python", ">"]`). Stop strings are encoded at runtime (never hard-coded
IDs), so the same logic works for Gemma-2.

## 3. Datasets

| Dataset | Location | License | Status |
|---|---|---|---|
| Countdown-Tasks-3to4 | `Jiayi-Pan/Countdown-Tasks-3to4` (HF) | **UNSPECIFIED** (no license metadata on model card) | Audit DONE (Phase 1): 490,364 rows (240,632 × 3-num, 249,732 × 4-num); 449,570 unique `(sorted nums, target)` keys / 40,794 duplicate rows (~8.3%); **100% solvable** on a 2,000-row sample (positive-int intermediates); **31 keys overlap our eval split** → must dedupe before any use as training supplement. No 5–6 number problems. |
| reasoning-gym countdown | `open-thought/reasoning-gym` | apache-2.0 | dev-only cross-check, installed v0.1.25. Note: its `score_answer` requires using **all** numbers exactly once (`sorted(used)==sorted(nums)`), stricter than our sub-multiset rule; its generator builds left-associative expressions (numbers in [1,100], not the classic pool). |

**Not used by design:** any dataset of human-written chain-of-thought (GSM8K,
OpenThoughts, distilled R1 traces, …). The spec forbids SFT on human CoT;
cold-start traces are generated programmatically from our own solver.

## 4. Code references (read for design; our own implementation, no vendoring)

| Repo | Commit (checked 2026-11-03) | License | Used for |
|---|---|---|---|
| `Jiayi-Pan/TinyZero` | `95df88f2dcb05f33bd18da546531b52d0954c18b` | Apache-2.0 | Countdown reward & verifier design |
| `open-thought/reasoning-gym` | `49b07130b3fcd12f2d064bba7c43869543a0e7e7` | Apache-2.0 | countdown generator/verifier cross-check |
| `huggingface/trl` | `032f2f71745437a7547a77ba47e8851b81e9429a` | Apache-2.0 | GRPO loss: advantage norm, clipping, KL |
| `huggingface/open-r1` | `5b6ff22b3fb7aa069c54866e517f39dfc3160e09` | Apache-2.0 | reward-function design |
| `verl-project/verl` (was `volcengine/verl`) | `184687bb3b7c601627eed3660283ecabd2bb75ed` | Apache-2.0 | env-token / tool-call loss masking |

## 5. Sandbox safety notes (from SPEC + task decisions)

- AST whitelist only; no attribute access / imports / f-strings / comprehensions /
  lambdas. `ast.USub` and `ast.Tuple` deliberately allowed; `ast.Pow` deliberately
  **dropped** (Countdown doesn't need it; `9**9**9**9` holds the GIL).
- Persistent multiprocessing worker pool (spawn), hard 50 ms per execution,
  stdout truncated to 200 chars; huge ints / deep nesting rejected at the AST level.

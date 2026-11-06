# HANDOFF.md — Part 2 runbook (cloud)

PART 1 delivered a Mac-verified pipeline. PART 2 = the full-scale runs
(SFT → GRPO → eval) on **interruptible GPU instances (vast.ai / RunPod)**, with a
**Google Cloud Storage (GCS)** bucket for code, data, checkpoints, heartbeats and
results, and a **GCP CPU VM** for Linux validation.

**Nothing below has been run against any cloud.** What *was* verified on the
Mac:
- every script passes `bash -n`;
- `scripts/launch_anywhere.sh` was dry-run end to end with `CKPT_BACKEND=local` (grpo, sft-pull, ckpt-pull, cost report;
  `verification/logs/r3/launch_dryrun.txt`);
- the checkpoint protocol, remote restore and resume equivalence were exercised with a local bucket;
- the `git archive` tarball contains no `.env`.

The `gcs` bucket class (`checkpoint.GCSBucket`) and all `gcloud`/`gsutil`
commands are written against the documented APIs but are **UNVERIFIED**.

> SFT traces were rebuilt after the audit (VERIFICATION.md §11). The old ones made gemma
> worse: 0/12 vs 2/12 few-shot base. The new ones lift Qwen-0.5B from 0/24 to 4/24 but are
> **not yet validated on gemma**. Do not reuse an `sft` artifact trained on the old traces.
> Before GRPO, gate on the SFT eval: SFT Pass@1 on 100 stratified problems must beat base few-shot.

## Architecture

- **GPU hosts** (untrusted, preemptible): `scripts/bootstrap.sh` (paste into the
  provider's on-start field) → `scripts/setup_cuda.sh` (pinned Python 3.14 venv from
  `lockfiles/cuda-linux-py314.lock`) → `scripts/launch_anywhere.sh`. They get a
  bucket-level-only service-account key and `COST_PER_HOUR`.
- **Bucket layout**
  - `code/<RUN_ID>.tar.gz` + `.sha256` (a `git archive` of HEAD, `repo/` prefix), and `data/{train,eval}.jsonl`.
    The data is checked against `logs/data_hashes.txt`.
  - `runs/<RUN_ID>/{ckpt,sft,eval}/step_<N>/…` + `COMMIT.json`, written last (sha256s + config/dataset hash). Create-only: a committed step is never rewritten.
  - `runs/<RUN_ID>/history/segment_<n>.json` (one per process start), and `hb/<RUN_ID>/<unix_ts>.json` heartbeats.
- **Your Mac**: publishes code and data, watches heartbeats, mirrors the bucket,
  reads the cost report, tears down.

## Prerequisites

```bash
cp .env.example .env    # fill in GCP_PROJECT, GCP_ZONE, GCS_BUCKET, COST_PER_HOUR, RUN_ID, ...
set -a; source .env; set +a
# required on the GPU host (no defaults): GCS_BUCKET, GCP_SA_KEY_B64, RUN_ID, RUN_STAGE, COST_PER_HOUR
```
`.env` is gitignored and is **not** part of the published tarball (`git archive HEAD`).

## Exact Part 2 order

1. **Commit**, then **publish code + data** (Mac): `bash scripts/publish_code.sh`.
   It refuses if tracked files are uncommitted and checks the dataset hashes.
2. **Bucket + service account + key** (GCP, CPU only):
   `bash scripts/gcp/create_bucket.sh`; `bash scripts/gcp/make_training_key.sh > /tmp/key.b64`
   → set `GCP_SA_KEY_B64`. Save the key **nowhere** in the repo.
3. **Linux validation VM** (CPU): `bash scripts/gcp/linux_validation_vm.sh`. It runs the
   test suite, the sandbox stress and `verification/run_all_checks.sh` on the published tarball.
   Fix any Linux-only failure before paying for GPUs.
4. **Smoke run** on the first GPU: `RUN_STAGE=smoke` (`configs/smoke_cuda.yaml`, 10 steps).
   Record **step time, rollout time vs loss time, tokens/s, peak GPU memory, checkpoint size and upload time**
   (`outputs/smoke/train_log.jsonl`, `nvidia-smi`). These drive the cost estimate (risk: §Throughput).
5. **Deliberate kill + resume on a different box**: `kill -9` the trainer, start a new
   box with the same `RUN_ID`. `train_rl.py` restores the newest valid COMMIT
   (`restored remote checkpoint … step N` in the log) and continues from it.
6. **Cost estimate** from step 4 (see Cost guard).
7. **Cold-start SFT**: `RUN_STAGE=sft`. It reuses `runs/<RUN_ID>/sft` if already committed; otherwise
   it trains (2000 traces, 2 epochs) and pushes the result. See the SFT warning above.
8. **Main GRPO**: `RUN_STAGE=grpo`. The VRAM preset is picked automatically (override with `PRESET=`);
   `n_problems: 0` = all 10k, effective batch 128.
9. **Eval**: `RUN_STAGE=eval`, on all 1000 problems:
   - base few-shot, SFT, and RL;
   - RL with the rounds sweep (0/1/2);
   - blank and shuffled output controls.

   Results are pushed to `runs/<RUN_ID>/eval`. Then run `scripts/pull_checkpoints.sh` on the Mac.
10. **Teardown**: delete the SA key, `bash scripts/gcp/teardown.sh`, delete provider
    instances by hand, prune old checkpoints by hand. The lifecycle rule only expires `hb/`.

## Provider notes (UNVERIFIED; re-check current docs)

**RunPod (spot pods):** paste `scripts/bootstrap.sh` into the on-start/docker-command
field with the env vars set as secrets. Restart = new pod, same `RUN_ID`.

**vast.ai (interruptible):** launch command `bash scripts/bootstrap.sh` with the env
vars; interruptions are hard kills with ~0 notice. Lost work per interruption is at most
`checkpoint_every` steps (20 in the presets) or `checkpoint_timer` (600 s).

**Other Linux GPU box:** run `bash scripts/bootstrap.sh` (needs `python3` and `sha256sum`).
Storage backends are `gcs` and `local` only; there is no S3 or HF-Hub backend.

**Decision rule:** if interruptions exceed **N/hour** or lost work per interruption
exceeds **M% of runtime** (you provide N and M), switch to a non-interruptible instance.

## GCS credential model (UNVERIFIED)

- Dedicated SA with **bucket-level IAM only**: `storage.objectCreator` +
  `storage.objectViewer` on the one bucket. All writes use `if_generation_match=0`
  (create-only), so no overwrite or delete permission is needed. Verify the role semantics in the IAM docs.
- Key delivered as `GCP_SA_KEY_B64` (env/secret), written to a `umask 077` temp file, never committed or logged.
  **Delete the key after each run and rotate between runs.**

## Resumability (preemption assumed)

`train_rl.py --resume auto --run-id <RUN_ID> --backend gcs` restores:
- LoRA weights, optimizer and step;
- RNG state: python, numpy, torch, cuda, and the sampling generator;
- the problem sampler, the adaptive-KL coefficient and the curriculum window.

The SFT adapter (`--init-adapter`) is merged on every start.
- **Local first:** checkpoints go to local disk (atomic rename, atomic `latest.json`), then to the bucket with COMMIT written last.
- **Fresh box:** the newest COMMIT that passes sha256 verification is downloaded; a corrupt one falls back to the previous step.
- **Mismatch:** resume is refused on a config-hash or dataset-hash mismatch unless `--force-resume`.

The resume hash covers training-relevant fields only. `total_steps`, the checkpoint cadence, `micro_batch` and the cost limits may change.
- **Verified locally:** resumed runs reproduce an uninterrupted run step for step (`verification/logs/r3/resume_compare.txt`).

## Cost guard

`COST_PER_HOUR` is required by `bootstrap.sh` / `launch_anywhere.sh`; prices are never hard-coded.
- **Accounting:** every heartbeat carries `segment` and `segment_elapsed_s`. `scripts/cost_report.py` sums
  GPU-hours per segment and prices them at each segment's recorded rate.
- **Limits:** `MAX_HOURS` / `MAX_TOTAL_USD` (→ `--max-hours` / `--max-total-usd`) count all segments of the run.
  When one is reached, training stops with a final checkpoint.

## Throughput (biggest Part 2 risk)

Rollouts are generated **one sequence at a time, one token per forward** (`rollout.py`; no batching, no vLLM).
The 80 GB preset asks for 128 rollouts × ≤512 tokens per step × 2000 steps.
- **Mac reference:** ~37 s/step for 4 rollouts of ≤256 tokens.
- **What to measure:** rollout time per step in the smoke run, before committing to 2000 steps.
  Reduce `total_steps`, G or `max_new_tokens` if needed.
- **Real fix:** batched generation with per-sequence tool pauses. Not implemented.

## MPS-vs-CUDA differences and untested paths

- **Untested:** everything cloud, `checkpoint.GCSBucket`, bf16 on CUDA, and `requirements-vllm.txt`
  (resolves only; nothing uses it).
- **Numerics:**
  - Gemma-2 needs `attn: eager` (presets set it) and the final-logit softcap in the chunked log-probs
    (implemented; verified against the HF forward to 2e-6 on MPS, fp32 and bf16).
  - In bf16, KV-cache decoding and the full-sequence forward differ by ~0.1 nat per token.
    Training therefore uses the training forward as the "old" policy and logs the difference as `gen_train_logp_dev`.
    That value should be ~1e-5 in fp32 and ≲0.2 in bf16; a larger one means the rollout context and the trained ids diverged.
- **Sandbox:** workers are stdlib-only `python -I -S` subprocesses, so behaviour should match on Linux.
  The 50 ms timeout includes pipe latency (p99 0.1 ms on the Mac). Confirm on the Linux VM.
- **Memory:** chunked log-probs are activation-checkpointed per vocab chunk; `micro_batch` bounds the loss-pass batch.

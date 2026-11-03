# HANDOFF.md — Part 2 runbook (cloud)

PART 1 delivered a complete, Mac-verified pipeline. PART 2 = the real full-scale
runs (SFT → GRPO → eval) on **interruptible GPU instances (vast.ai / RunPod)**
with **Google Cloud Storage (GCS)** for checkpoints/logs and **GCP CPU-only**
validation. Nothing below has been run against any cloud (this repo was built and
verified on an Apple Silicon Mac only). Every command is marked UNVERIFIED; flag
names were checked where possible, everything else is documentation-grade.

> ⚠️ Two bugs were found and fixed during a ~4 h Mac observation run and MUST be
> kept fixed: (1) MPS OOM from full-vocab fp32 logits → use `--chunked-vocab` /
> per-chunk fp32; (2) adaptive-KL underflow → `kl_coef` is floored at `1e-3`.

## Architecture

- **GPU hosts** (vast.ai / RunPod, untrusted): run `scripts/bootstrap.sh` →
  `scripts/launch_anywhere.sh`. They get a **bucket-level-only** service-account
  key and `COST_PER_HOUR`. They never hold broad cloud credentials.
- **GCS bucket**: `runs/<run_id>/ckpt/step_<N>/...` + `COMMIT.json` (written last,
  sha256s + config/dataset hash), `hb/` heartbeats, `history/segment_<n>.json`.
  Pruning is a lifecycle rule, never the GPU host.
- **Your Mac**: publishes code/data, watches heartbeats, mirrors the bucket,
  runs eval, tears down.

## Prerequisites

```bash
cp .env.example .env    # fill in GCP_PROJECT, GCP_ZONE, GCS_BUCKET, COST_PER_HOUR, RUN_ID
set -a; source .env; set +a
# required (no defaults): GCS_BUCKET, GCP_SA_KEY_B64, RUN_ID, RUN_STAGE, COST_PER_HOUR
```

## Exact Part 2 order

1. **Publish code + data** (Mac):
   `bash scripts/publish_code.sh`  (tars repo@git-sha + dataset, uploads to GCS)
2. **Create bucket + service account + key** (GCP, CPU only):
   `bash scripts/gcp/create_bucket.sh`; `bash scripts/gcp/make_training_key.sh > /tmp/key.b64`
   → set `GCP_SA_KEY_B64`. Save the key **nowhere** in the repo.
3. **Linux validation VM** (GCP, CPU): `bash scripts/gcp/linux_validation_vm.sh`
   (runs the test suite + 10k sandbox stress under Linux spawn semantics). Fix any
   Linux-only failure before paying for GPUs.
4. **10-minute smoke run** on the first GPU you get: `RUN_STAGE=grpo` with
   `--config configs/smoke_cuda.yaml`. Record: **step time, tokens/s, peak GPU
   memory, checkpoint size + upload time** (these drive the cost estimate).
5. **Deliberate kill + resume on a different box/provider**: `kill -9` the trainer,
   relaunch with the same `RUN_ID`; confirm it resumes from the newest valid COMMIT.
6. **Cost estimate** from step 4 measurements (see Cost guard below).
7. **Cold-start SFT**: `RUN_STAGE=sft` (2000 traces, 2 epochs).
8. **Main GRPO**: `RUN_STAGE=grpo` with the VRAM-matched preset
   (`configs/preset-{24,48,80}gb.yaml`), `--resume auto`.
9. **Eval**: `RUN_STAGE=eval` (1000 problems), then `scripts/pull_checkpoints.sh`.
10. **Teardown**: delete the SA key, `bash scripts/gcp/teardown.sh`, delete provider
    instances manually.

## Provider notes (UNVERIFIED; re-check current docs)

**RunPod (spot pods):** paste `scripts/bootstrap.sh` into the on-start/docker-command
field with the env vars set as secrets. Spot pods are preempted with short notice;
local disk is ephemeral (do NOT rely on it — checkpoints go straight to GCS).
Restart = launch a new pod from the same template with the same `RUN_ID`.

**vast.ai (interruptible):** set the launch command to `bash scripts/bootstrap.sh`
with env vars; interruptions are hard-kills with ~0 notice. Same GCS-first rule.

**Other / any Linux GPU box:** `scripts/launch_anywhere.sh` is provider-neutral;
with `CKPT_BACKEND=hf-hub` or `s3` it also works outside GCP (as a fallback if the
GCP quota/billing cap is a problem).

**Decision rule:** if interruptions exceed **N/hour** or lost work per interruption
exceeds **M% of runtime** (you provide N, M), switch to a non-interruptible instance.

## GCS credential model (UNVERIFIED)

- Dedicated SA with **bucket-level IAM only**: `storage.objectCreator` +
  `storage.objectViewer` on the one bucket. Verify in the docs that these allow
  create/list but **not overwrite or delete** (client uses create-if-absent).
- Key delivered as `GCP_SA_KEY_B64` (env/secret), never committed or logged.
  **Delete the key after each run and rotate between runs.**
- Bucket: single region (same as VM to avoid egress), uniform access,
  public-access prevention, lifecycle rules for `hb/` + old checkpoints.

## Resumability (preemption assumed)

`--resume auto` restores LoRA weights, optimizer, step, RNG (python/numpy/torch/cuda),
sampler, adaptive-KL coef, curriculum, run id. Local-first (write local then upload);
on start pick the newest **valid** COMMIT (verify sha256s, fall back on corrupt).
Refuses to resume on config/dataset/pinned-version hash mismatch unless `--force-resume`.
Effective batch is fixed by gradient accumulation across GPU sizes.

## Cost guard

`COST_PER_HOUR` is **required** (run refuses to start if unset); prices are never
hard-coded. `scripts/cost_report.py` reads segment history → cumulative GPU-hours/USD.
`--max-hours` / `--max-total-usd` stop gracefully (final checkpoint + upload).

## MPS-vs-CUDA differences and untested paths

- **Untested**: everything cloud — `scripts/*.sh` under `scripts/gcp/`, the gcs/s3/
  hf-hub storage backends (only `local` is tested), the `vllm` backend
  (`requirements-vllm.txt`, torch 2.14 / transformers 5.20.2), bfloat16 on CUDA.
- **MPS-specific that may differ on CUDA**: eager attention was required for gemma-2
  on MPS; on CUDA `sdpa`/flash may work and be faster — verify. fp16 on MPS needed
  fp32 log-prob casting; CUDA bf16 does not. Sandbox `spawn` context on macOS vs
  `fork`/`spawn` on Linux — re-run the Linux validation VM to confirm the 50 ms
  timeout semantics.
- **256k-vocab lm_head** is the dominant cost on MPS; on A100 it's ~2 orders faster,
  but still requires `--chunked-vocab` for large batches.

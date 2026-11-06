#!/usr/bin/env bash
# Linux CPU validation VM: create a small Spot CPU VM, run the test suite and the
# verification checks under Linux semantics on the PUBLISHED code tarball
# (scripts/publish_code.sh must have run), upload logs, then delete the VM.
# Rationale: the sandbox + masking were only exercised on macOS.
# NOT run (no GCP access while building); gcloud flags documentation-grade.
set -euo pipefail
: "${GCP_PROJECT:?}"; : "${GCP_ZONE:?}"; : "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
VM="countdown-linux-validation"
B="gs://${GCS_BUCKET#gs://}"

gcloud compute instances create "$VM" \
  --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
  --machine-type e2-standard-4 --provisioning-model=SPOT \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud

gcloud compute ssh "$VM" --zone "$GCP_ZONE" --command "
  set -e
  gcloud storage cp '$B/code/$RUN_ID.tar.gz' '$B/code/$RUN_ID.sha256' '$B/data/train.jsonl' '$B/data/eval.jsonl' .
  sha256sum -c '$RUN_ID.sha256' && tar xzf '$RUN_ID.tar.gz'
  mkdir -p repo/data && mv train.jsonl eval.jsonl repo/data/ && cd repo
  sha256sum -c logs/data_hashes.txt
  sudo apt-get install -y -q python3-pip >/dev/null
  bash scripts/setup_cuda.sh   # pinned Python 3.14 venv (CUDA wheels run on CPU too)
  .venv/bin/python -m pytest -q tests 2>&1 | tee linux_pytest.txt
  .venv/bin/python scripts/stress_sandbox.py 2>&1 | tee linux_sandbox_stress.txt
  PY=.venv/bin/python bash verification/run_all_checks.sh 2>&1 | tee linux_verification.txt || true
  gcloud storage cp linux_*.txt '$B/validation/'
"

gcloud compute instances delete "$VM" --zone "$GCP_ZONE" --quiet
echo "validation done; logs -> $B/validation/"

#!/usr/bin/env bash
# Linux CPU validation VM: create a small Spot CPU VM, run the full test suite
# under Linux semantics (spawn context, sandbox 10k stress, data hash), upload
# logs to the bucket, then stop/delete. Rationale: the sandbox + masking were only
# tested on macOS, so catch Linux differences before paying for GPUs.
# NOT run (no GCP access here); gcloud flags documentation-grade.
set -euo pipefail
: "${GCP_PROJECT:?}"; : "${GCP_ZONE:?}"; : "${GCS_BUCKET:?}"
VM="countdown-linux-validation"
B="${GCS_BUCKET#gs://}"

gcloud compute instances create "$VM" \
  --project "$GCP_PROJECT" --zone "$GCP_ZONE" \
  --machine-type e2-standard-4 --preemptible \
  --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud

gcloud compute ssh "$VM" --zone "$GCP_ZONE" --command "
  set -e
  git clone <REPO_URL> repo || true   # placeholder: repo URL in HANDOFF
  cd repo
  python3 -m venv .venv && .venv/bin/pip install -r requirements-cuda.txt -r requirements-dev.txt
  .venv/bin/python data_generator.py --n-train 100 --n-eval 20 --seed 0 --verify
  .venv/bin/python -m pytest -q tests 2>&1 | tee linux_pytest.txt
  .venv/bin/python scripts/stress_sandbox.py 2>&1 | tee linux_sandbox_stress.txt
  gsutil cp linux_*.txt gs://$B/validation/
"

gcloud compute instances stop "$VM" --zone "$GCP_ZONE" --quiet
gcloud compute instances delete "$VM" --zone "$GCP_ZONE" --quiet
echo "validation done; logs -> gs://$B/validation/"

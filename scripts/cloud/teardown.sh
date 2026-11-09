#!/usr/bin/env bash
# Tear down: stop/delete the (CPU) VM, optionally delete the bucket, with
# confirmation prompts. Does NOT touch any GPU provider instance (those are
# provider-side and not automated here).
# NOT run (no GCP access here).
set -euo pipefail
: "${GCP_PROJECT:?}"; : "${GCP_ZONE:?}"
VM="${1:-countdown-linux-validation}"

read -r -p "delete VM '$VM' (zone $GCP_ZONE)? [y/N] " a
if [ "$a" = y ]; then
  gcloud compute instances delete "$VM" --zone "$GCP_ZONE" --quiet
fi
read -r -p "delete bucket '${GCS_BUCKET:-}'? [y/N] " b
if [ "$b" = y ] && [ -n "${GCS_BUCKET:-}" ]; then
  gcloud storage rm -r "gs://${GCS_BUCKET#gs://}"
fi
echo "teardown complete"

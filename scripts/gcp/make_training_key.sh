#!/usr/bin/env bash
# Create a dedicated training-host service account with bucket-level IAM only
# (create-only + view/list on ONE bucket), emit its key as base64 to stdout.
# Delete the key after each run and rotate between runs (see HANDOFF.md).
# NOT run (no GCP access here); role names are documentation-grade — verify in the
# IAM docs exactly what storage.objectCreator / storage.objectViewer allow.
set -euo pipefail
: "${GCP_PROJECT:?}"; : "${GCS_BUCKET:?}"
SA="countdown-trainer"
B="${GCS_BUCKET#gs://}"

gcloud iam service-accounts create "$SA" --project "$GCP_PROJECT" \
  --display-name "countdown training host" 2>/dev/null || true
EMAIL="$SA@$GCP_PROJECT.iam.gserviceaccount.com"

# bucket-level IAM only (no project-level roles)
gcloud storage buckets add-iam-policy-binding "gs://$B" \
  --member "serviceAccount:$EMAIL" --role roles/storage.objectCreator
gcloud storage buckets add-iam-policy-binding "gs://$B" \
  --member "serviceAccount:$EMAIL" --role roles/storage.objectViewer

KEY=$(mktemp)
gcloud iam service-accounts keys create "$KEY" --iam-account "$EMAIL"
base64 < "$KEY"
rm -f "$KEY"
echo "key emitted (base64). Set GCP_SA_KEY_B64 to this value." >&2

#!/usr/bin/env bash
# Create the GCS bucket (single region, uniform access, public-access prevention,
# lifecycle rules for heartbeats + old checkpoints). NOT run (no GCP access here);
# gcloud flag names are documentation-grade — verify against `gcloud storage --help`.
set -euo pipefail
: "${GCP_PROJECT:?}"; : "${GCS_BUCKET:?}"; : "${GCP_ZONE:?}"
B="${GCS_BUCKET#gs://}"
# region derived from zone (last two segments) — e.g. us-central1-a -> us-central1
REGION="$(echo "$GCP_ZONE" | sed -E 's/-[a-z]$//')"

gcloud storage buckets create "gs://$B" \
  --project "$GCP_PROJECT" --location "$REGION" \
  --uniform-bucket-level-access --public-access-prevention=enforced

# lifecycle: expire heartbeats (top-level hb/ prefix) after 7 days. Checkpoints,
# SFT adapters and eval results under runs/ are NEVER auto-deleted: prune them
# by hand after scripts/pull_checkpoints.sh has mirrored them to the Mac.
LC="$(mktemp)"
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":7,"matchesPrefix":["hb/"]}}]}' > "$LC"
gcloud storage buckets update "gs://$B" --lifecycle-file="$LC"
rm -f "$LC"
echo "bucket gs://$B created in $REGION"

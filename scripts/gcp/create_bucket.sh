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

# lifecycle: delete heartbeats after 7 days, keep only recent checkpoints (prune
# is done by lifecycle rule, never by the training host)
gcloud storage buckets update "gs://$B" --lifecycle-file=- <<'EOF'
{"rule":[{"action":{"type":"Delete"},"condition":{"age":7,"matchesPrefix":["runs/"]}}]}
EOF
echo "bucket gs://$B created in $REGION"

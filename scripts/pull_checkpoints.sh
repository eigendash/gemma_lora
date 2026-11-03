#!/usr/bin/env bash
# Mirror the bucket's runs/ (checkpoints, logs, heartbeats, history) to this Mac,
# so a GCP billing event that kills the bucket doesn't lose everything. Run on Mac.
# NOT run (no GCS access here).
set -euo pipefail
: "${GCS_BUCKET:?}"
mkdir -p mirror
gsutil -m rsync -r "gs://${GCS_BUCKET#gs://}/runs" mirror/runs
echo "mirrored -> mirror/runs"

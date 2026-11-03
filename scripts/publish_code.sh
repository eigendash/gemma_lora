#!/usr/bin/env bash
# Publish code (at current git sha) + hash-checked dataset to GCS. Run on the Mac.
# NOT run (no GCS access here); gsutil commands are documentation-grade.
set -euo pipefail
: "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
SHA=$(git rev-parse HEAD)
tar czf "/tmp/${RUN_ID}.tar.gz" \
  --exclude=.git --exclude=.venv --exclude=.hf-cache --exclude=outputs --exclude=data .
( cd /tmp && sha256sum "${RUN_ID}.tar.gz" > "${RUN_ID}.sha256" )
gsutil cp "/tmp/${RUN_ID}.tar.gz" "gs://${GCS_BUCKET#gs://}/code/"
gsutil cp "/tmp/${RUN_ID}.sha256" "gs://${GCS_BUCKET#gs://}/code/"
gsutil cp data/train.jsonl data/eval.jsonl "gs://${GCS_BUCKET#gs://}/data/"
echo "published repo@$SHA + dataset -> ${GCS_BUCKET}/code/${RUN_ID}.tar.gz"

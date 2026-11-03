#!/usr/bin/env bash
# Provider-neutral bootstrap. Paste into the provider's on-start / docker-command
# field. Needs ONLY environment variables (no secrets baked in, no code baked in).
# Steps: install minimal client -> pull code tarball + dataset from the bucket ->
# verify hashes -> install pinned deps -> exec launch_anywhere.sh.
#
# NOT run against GCP/vast.ai/RunPod (Mac only). Verified with `bash -n` and by
# reading; the gcs commands are documentation-grade and unverified.
set -euo pipefail

: "${GCS_BUCKET:?set GCS_BUCKET}"
: "${GCP_SA_KEY_B64:?set GCP_SA_KEY_B64}"
: "${RUN_ID:?set RUN_ID}"
: "${RUN_STAGE:?set RUN_STAGE (sft|grpo|eval)}"
: "${COST_PER_HOUR:?set COST_PER_HOUR}"

echo "$GCP_SA_KEY_B64" | base64 -d > /tmp/sa.json
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa.json

# minimal GCS client (google-cloud-storage pinned in requirements)
pip install -q google-cloud-storage

# pull code tarball + dataset + hashes
python - <<'PY'
import os
from google.cloud import storage
c = storage.Client()
b = c.bucket(os.environ["GCS_BUCKET"].removeprefix("gs://"))
for obj in [f"code/{os.environ['RUN_ID']}.tar.gz",
            f"code/{os.environ['RUN_ID']}.sha256",
            "data/train.jsonl", "data/eval.jsonl"]:
    name = os.path.basename(obj)
    b.blob(obj).download_to_filename(f"/tmp/{name}")
    print("downloaded", obj)
PY

cd /tmp
sha256sum -c "$RUN_ID.sha256"   # verifies the code tarball
tar xzf "$RUN_ID.tar.gz"
cd repo

pip install -q -r requirements-cuda.txt

exec bash scripts/launch_anywhere.sh

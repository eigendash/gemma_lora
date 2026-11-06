#!/usr/bin/env bash
# Provider-neutral bootstrap. Paste into the provider's on-start / docker-command
# field. Needs ONLY environment variables (no secrets baked in, no code baked in).
# Steps: pull code tarball + dataset from the bucket -> verify sha256s -> build
# the pinned venv -> exec launch_anywhere.sh.
#
# NOT run against GCP/vast.ai/RunPod (Mac only). Checked with `bash -n`; the
# archive layout it expects (repo/ prefix) is produced by publish_code.sh and
# tested locally.
set -euo pipefail

: "${GCS_BUCKET:?set GCS_BUCKET}"
: "${GCP_SA_KEY_B64:?set GCP_SA_KEY_B64}"
: "${RUN_ID:?set RUN_ID}"
: "${RUN_STAGE:?set RUN_STAGE (smoke|sft|grpo|eval)}"
: "${COST_PER_HOUR:?set COST_PER_HOUR}"

umask 077
KEY="$(mktemp)"
echo "$GCP_SA_KEY_B64" | base64 -d > "$KEY"
export GOOGLE_APPLICATION_CREDENTIALS="$KEY"

WORK="${WORK_DIR:-$HOME/countdown}"
mkdir -p "$WORK" && cd "$WORK"
# system Pythons with PEP 668 (e.g. Ubuntu 24.04) refuse plain installs: fall back to --user
python3 -m pip install -q "google-cloud-storage==3.17.1" 2>/dev/null || python3 -m pip install -q --user --break-system-packages "google-cloud-storage==3.17.1"
python3 - <<'PY'
import os
from google.cloud import storage
b = storage.Client().bucket(os.environ["GCS_BUCKET"].removeprefix("gs://"))
for obj in [f"code/{os.environ['RUN_ID']}.tar.gz", f"code/{os.environ['RUN_ID']}.sha256",
            "data/train.jsonl", "data/eval.jsonl"]:
    b.blob(obj).download_to_filename(os.path.basename(obj))
    print("downloaded", obj)
PY

sha256sum -c "${RUN_ID}.sha256"          # code tarball integrity
rm -rf repo && tar xzf "${RUN_ID}.tar.gz" # git archive --prefix=repo/
mkdir -p repo/data && mv train.jsonl eval.jsonl repo/data/
cd repo
sha256sum -c logs/data_hashes.txt        # dataset == the committed seed-0 hashes

[ -x .venv/bin/python ] || bash scripts/setup_cuda.sh
exec bash scripts/launch_anywhere.sh

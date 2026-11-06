#!/usr/bin/env bash
# Publish code (git archive of HEAD: tracked files only, so never .env, caches,
# outputs or local junk) + the hash-checked dataset to GCS. Run on the Mac.
# NOT run against GCS here (no cloud access while building); the archive
# layout (repo/ prefix) and the .env exclusion are tested locally by
# verification/run_all_checks.sh.
set -euo pipefail
: "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
cd "$(dirname "$0")/.."
if ! git diff --quiet HEAD --; then
  echo "uncommitted changes to tracked files: commit first (the tarball is HEAD)"; exit 1
fi
shasum -a 256 -c logs/data_hashes.txt   # dataset must be the reproducible seed-0 dataset
SHA=$(git rev-parse HEAD)
OUT="$(mktemp -d)"
git archive --format=tar.gz --prefix=repo/ -o "$OUT/${RUN_ID}.tar.gz" HEAD
( cd "$OUT" && shasum -a 256 "${RUN_ID}.tar.gz" > "${RUN_ID}.sha256" )
B="gs://${GCS_BUCKET#gs://}"
gsutil cp "$OUT/${RUN_ID}.tar.gz" "$OUT/${RUN_ID}.sha256" "$B/code/"
gsutil cp data/train.jsonl data/eval.jsonl "$B/data/"
echo "published repo@$SHA + dataset -> $B/code/${RUN_ID}.tar.gz"

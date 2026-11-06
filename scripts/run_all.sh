#!/usr/bin/env bash
# One-command Part 2 reproduction (cloud side from the Mac). Publish code+data,
# then watch heartbeats. The GPU host runs scripts/bootstrap.sh -> launch_anywhere.sh
# automatically (paste bootstrap.sh into its on-start field with the env vars).
# NOT run against cloud.
set -euo pipefail
: "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
bash scripts/publish_code.sh
bash scripts/watch.sh

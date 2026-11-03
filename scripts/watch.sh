#!/usr/bin/env bash
# Poll the newest heartbeat and print a macOS notification if it is stale for
# STALE_MIN minutes or step progress is flat. Run on the Mac while a GPU run is up.
# NOT run (no GCS access here); documentation-grade.
set -uo pipefail
: "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
STALE_MIN=${STALE_MIN:-15}
LAST_STEP=""
while true; do
  HB=$(gsutil ls "gs://${GCS_BUCKET#gs://}/runs/${RUN_ID}/hb/" 2>/dev/null | sort | tail -1)
  if [ -z "$HB" ]; then
    osascript -e "display notification \"no heartbeat yet for $RUN_ID\" with title \"countdown-run\"" 2>/dev/null || true
  else
    gsutil cp "$HB" /tmp/hb.json >/dev/null 2>&1
    STEP=$(python -c "import json;print(json.load(open('/tmp/hb.json'))['step'])" 2>/dev/null || echo "?")
    REWARD=$(python -c "import json;print(json.load(open('/tmp/hb.json')).get('reward','?'))" 2>/dev/null || echo "?")
    AGE=$(($(date +%s) - $(stat -f %m /tmp/hb.json 2>/dev/null || echo 0)))
    echo "$(date +%H:%M:%S) step=$STEP reward=$REWARD age=${AGE}s"
    if [ "${AGE:-0}" -gt $((STALE_MIN*60)) ]; then
      osascript -e "display notification \"$RUN_ID stale (step $STEP, ${AGE}s)\" with title \"countdown-run\"" 2>/dev/null || true
    fi
  fi
  sleep 120
done

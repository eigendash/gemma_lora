#!/usr/bin/env bash
# Poll the newest heartbeat (gs://BUCKET/hb/RUN_ID/<unix_ts>.json) and post a
# macOS notification if it is older than STALE_MIN minutes or the step is not
# advancing. Run on the Mac while a GPU run is up.
# NOT run against GCS here (no cloud access while building).
set -uo pipefail
: "${GCS_BUCKET:?}"; : "${RUN_ID:?}"
STALE_MIN=${STALE_MIN:-15}
B="gs://${GCS_BUCKET#gs://}"
LAST_STEP=""; SAME=0
notify() { osascript -e "display notification \"$1\" with title \"countdown-run\"" 2>/dev/null || true; echo "ALERT: $1"; }
while true; do
  HB=$(gsutil ls "$B/hb/${RUN_ID}/" 2>/dev/null | sort | tail -1)
  if [ -z "$HB" ]; then
    notify "no heartbeat yet for $RUN_ID"
  else
    TS=$(basename "$HB" .json)                     # heartbeat time = object name, not local mtime
    AGE=$(( $(date +%s) - TS ))
    STEP=$(gsutil cat "$HB" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('step','?'))" 2>/dev/null || echo "?")
    echo "$(date +%H:%M:%S) step=$STEP heartbeat_age=${AGE}s"
    [ "$AGE" -gt $((STALE_MIN * 60)) ] && notify "$RUN_ID stale: last heartbeat ${AGE}s ago (step $STEP)"
    if [ "$STEP" = "$LAST_STEP" ]; then SAME=$((SAME + 1)); else SAME=0; fi
    [ "$SAME" -ge 8 ] && notify "$RUN_ID step $STEP not advancing for ~16 min"
    LAST_STEP=$STEP
  fi
  sleep 120
done

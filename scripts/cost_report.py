#!/usr/bin/env python
"""Read segment history from the bucket and print cumulative GPU-hours + USD.

Not run against a bucket here; documentation-grade. Credentials from
GOOGLE_APPLICATION_CREDENTIALS only.
"""
import json
import os
import sys


def main(argv):
    bucket = os.environ.get("GCS_BUCKET", "").removeprefix("gs://")
    run_id = os.environ.get("RUN_ID", "")
    # local fallback: read from mirror/ or outputs/_remote
    base = os.path.join("mirror", "runs", run_id, "history") if not bucket else None
    if base and not os.path.isdir(base):
        base = os.path.join("outputs", "rl", "_remote", "runs", run_id, "history")
    if not base or not os.path.isdir(base):
        print("no local history; set GCS_BUCKET/RUN_ID (gcs path is untested)")
        return 1
    total_h = 0.0
    rows = []
    for fn in sorted(os.listdir(base)):
        if not fn.startswith("segment_"):
            continue
        seg = json.load(open(os.path.join(base, fn)))
        rows.append(seg)
        total_h += float(seg.get("elapsed_hours", 0) or 0)
    cost = os.environ.get("COST_PER_HOUR")
    usd = total_h * float(cost) if cost else None
    print(f"segments={len(rows)} gpu_hours={total_h:.2f}" + (f" usd={usd:.2f}" if usd is not None else " (COST_PER_HOUR unset)"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

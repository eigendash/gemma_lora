#!/usr/bin/env python
"""Cumulative GPU-hours and USD for a run, from the bucket heartbeats.

Each training segment (one process start) writes heartbeats carrying
``segment`` and ``segment_elapsed_s``; a segment's hours = its largest elapsed
value (checkpoint.accumulated_hours). Segment start records live in
``runs/<run_id>/history/``.

    RUN_ID=... CKPT_BACKEND=gcs GCS_BUCKET=... python scripts/cost_report.py
    RUN_ID=... CKPT_BACKEND=local CKPT_ROOT=outputs/rl/_remote python scripts/cost_report.py

USD uses each segment's recorded cost_per_hour, else COST_PER_HOUR.
The gcs path is UNVERIFIED against real GCS (no cloud access while building).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint as ck  # noqa: E402


def main() -> int:
    run_id = os.environ.get("RUN_ID", "")
    if not run_id:
        print("set RUN_ID")
        return 1
    bucket = ck.get_bucket(os.environ.get("CKPT_BACKEND", "gcs"), root=os.environ.get("CKPT_ROOT", "outputs/rl/_remote"))
    total_h, per_seg = ck.accumulated_hours(bucket, run_id)
    default_rate = float(os.environ.get("COST_PER_HOUR", "0") or 0)
    usd = 0.0
    print(f"{'segment':>7} {'gpu':>24} {'start_step':>10} {'hours':>7} {'usd':>8}")
    for seg in sorted(per_seg):
        try:
            info = json.loads(bucket.get(f"runs/{run_id}/history/segment_{seg}.json").decode())
        except Exception:
            info = {}
        rate = float(info.get("cost_per_hour") or default_rate)
        usd += per_seg[seg] * rate
        print(f"{seg:>7} {str(info.get('gpu', '?')):>24} {str(info.get('start_step', '?')):>10} "
              f"{per_seg[seg]:>7.2f} {per_seg[seg] * rate:>8.2f}")
    print(f"total: segments={len(per_seg)} gpu_hours={total_h:.2f} usd={usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

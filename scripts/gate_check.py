"""Go/no-go gate for a weak base model after a lineage phase (pre-registered before the run, EXPERIMENTS.md §2.7l).

PASS if both hold over the phase's train_log.jsonl:
  - the smoothed in-run validation at the end of the phase (mean of the last 3 evals) is at least
    --baseline + --min-gain-pp, where --baseline is the 3-pass validation mean of the bare base model (same prompt);
  - first-round training accuracy is rising: mean of the last 10 steps > mean of the first 10 steps.
Exit code 0 = pass, 1 = fail. Prints the numbers either way.
usage: python scripts/gate_check.py train_log.jsonl --baseline 2.0 [--min-gain-pp 3]
       --baseline step0 uses the run's own step-0 validation (before any update) as the base-model value.
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--baseline", required=True,
                    help="base-model val accuracy in percent (3-pass mean), or 'step0' for the run's step-0 validation")
    ap.add_argument("--min-gain-pp", type=float, default=3.0)
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(a.log)]
    val = [100 * r["val_correct"] for r in rows if r.get("val_correct") is not None]
    smoothed = sum(val[-3:]) / len(val[-3:]) if val else float("nan")
    if a.baseline == "step0":
        step0 = [100 * r["val_correct"] for r in rows if r.get("val_correct") is not None and r.get("step") == 0]
        if not step0:
            print("gate: no step-0 validation in the log -> FAIL")
            return 1
        a.baseline = step0[0]
    a.baseline = float(a.baseline)
    first, last = rows[:10], rows[-10:]
    acc0 = sum(r["correct"] for r in first) / len(first)
    acc1 = sum(r["correct"] for r in last) / len(last)
    ok_val = smoothed >= a.baseline + a.min_gain_pp
    ok_acc = acc1 > acc0
    print(f"gate: smoothed val {smoothed:.1f}% vs need {a.baseline + a.min_gain_pp:.1f}% -> {'ok' if ok_val else 'FAIL'}; "
          f"train acc first10 {acc0:.3f} -> last10 {acc1:.3f} -> {'ok' if ok_acc else 'FAIL'}; val evals {[round(v, 1) for v in val]}")
    return 0 if ok_val and ok_acc else 1


if __name__ == "__main__":
    sys.exit(main())

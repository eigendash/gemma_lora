"""SPEC §D metrics from eval dumps (evaluate.py --dump / countdown_rl.evaluation.evaluate_vllm), with no GPU needed.

Per dump: Pass@1, format compliance, mean generated tokens (the model's own tokens: the injected <output> bodies
are removed before counting, matching the loss mask), mean code rounds, and two self-correction rates:
  - evaluate.py's definition: rollouts whose first execution printed False -> fraction that end correct;
  - strict: rollouts whose FIRST check printed no True at all (every candidate failed) and that ran a second
    check -> fraction that end correct. With the multi-candidate prompt the first definition also counts a
    check where candidate 1 failed but a later one passed, so the strict one is the real recovery rate.
usage: python verification/report_metrics.py name=dump.jsonl [name=dump.jsonl ...]
"""
from __future__ import annotations

import json
import os
import re
import sys

from countdown_rl.rl.rewards import check_format  # noqa: E402

OUT_RE = re.compile(r"<output>.*?</output>", re.DOTALL)


def metrics(rows, tok=None):
    n = len(rows)
    corr = [r["correct"] > 0 for r in rows]
    fmt = sum(check_format(r["completion"]) for r in rows)
    own = [OUT_RE.sub("<output></output>", r["completion"]) for r in rows]
    if tok is not None:
        lens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in own]
    else:
        lens = [len(t) / 4 for t in own]  # rough chars/4 fallback
    rounds = [len(r.get("outputs") or []) for r in rows]
    first_false = [i for i, r in enumerate(rows) if r.get("outputs") and r["outputs"][0].strip().startswith("False")]
    strict = [i for i, r in enumerate(rows)
              if len(r.get("outputs") or []) >= 2 and "True" not in r["outputs"][0]]
    return {
        "n": n, "pass@1": sum(corr) / n, "correct": sum(corr), "format_compliance": fmt / n,
        "gen_tokens_mean": sum(lens) / n, "code_rounds_mean": sum(rounds) / n,
        "first_exec_false": len(first_false),
        "self_correction (eval.py def)": sum(corr[i] for i in first_false) / max(1, len(first_false)),
        "all_false_then_retried": len(strict),
        "recovered_after_all_false": sum(corr[i] for i in strict),
        "recovery_rate_strict": sum(corr[i] for i in strict) / max(1, len(strict)),
        "max_length_frac": sum(r.get("terminal_reason") == "max_length" for r in rows) / n,
    }


def main(argv):
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
    except Exception as e:  # offline without the tokenizer: fall back to chars/4
        print(f"(no tokenizer: {e!r}; token lengths are chars/4)")
    out = {}
    for a in argv:
        name, path = a.split("=", 1)
        out[name] = metrics([json.loads(l) for l in open(path)], tok)
    keys = list(next(iter(out.values())).keys())
    print("metric".ljust(32) + "".join(k.rjust(14) for k in out))
    for k in keys:
        print(k.ljust(32) + "".join((f"{v[k]:.3f}" if isinstance(v[k], float) else str(v[k])).rjust(14) for v in out.values()))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])

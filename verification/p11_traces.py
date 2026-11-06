"""P11 self-distillation data (EXPERIMENTS.md §2.7i): sample the P8 policy on TRAINING problems and keep
its own successful search traces (SkillFactory arXiv 2512.04072 / Stream of Search 2404.03683 / STaR).

  pool:   python verification/p11_traces.py pool --out outputs/p11/pool.jsonl --n-easy 100 --n-medium 500 --n-hard 700 --k 8
          -> train.jsonl problems minus the 300-problem validation split (train_rl.split_validation), each
             repeated k times (eval.py samples each line independently at --temperature > 0)
  filter: python verification/p11_traces.py filter --dump outputs/p11/samples.jsonl --out outputs/p11/sft.jsonl
          -> correct traces only. RECOVERY = an earlier check printed no True and a later one did (failed
             attempt + recovery, as SkillFactory keeps); DIRECT = first check already True. Keeps all recovery
             traces (max 2 per problem) plus an equal number of direct ones, so the model does not learn
             "always miss first".
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_pool(a):
    import train_rl

    allp = [json.loads(line) for line in open(a.train_file)]
    train, val = train_rl.split_validation(allp, a.val_n)
    vkeys = {(tuple(p["nums"]), p["target"]) for p in val}
    assert not any((tuple(p["nums"]), p["target"]) in vkeys for p in train)
    rng = random.Random(a.seed)
    by = collections.defaultdict(list)
    for p in train:
        by[p["tier"]].append(p)
    pick = []
    for tier, n in (("easy", a.n_easy), ("medium", a.n_medium), ("hard", a.n_hard)):
        pick += rng.sample(by[tier], n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for p in pick:
            for _ in range(a.k):
                f.write(json.dumps(p) + "\n")
    print(f"pool: {len(pick)} problems x {a.k} = {len(pick) * a.k} lines -> {a.out} (validation excluded)")


def classify(row) -> str:
    outs = [o for o in row.get("outputs", []) if "ExecutionLimit" not in o]
    if not row["correct"] or not outs:
        return ""
    hit = ["True" in o for o in outs]
    if hit[0]:
        return "direct"
    return "recovery" if any(hit[1:]) else ""


def filter_traces(a):
    rows = [json.loads(line) for line in open(a.dump)]
    rng = random.Random(a.seed)
    rec, direct = collections.defaultdict(list), []
    stats = collections.Counter()
    for r in rows:
        stats["rows"] += 1
        stats["correct"] += bool(r["correct"])
        c = classify(r)
        stats[c or "other"] += 1
        key = (tuple(r["nums"]), r["target"])
        if c == "recovery":
            rec[key].append(r)
        elif c == "direct":
            direct.append(r)
    keep = [r for v in rec.values() for r in v[:2]]
    rng.shuffle(direct)
    keep += direct[:len(keep)]
    rng.shuffle(keep)
    with open(a.out, "w") as f:
        for r in keep:
            f.write(json.dumps({k: r[k] for k in ("nums", "target", "tier", "completion", "outputs")}) + "\n")
    tiers = collections.Counter(r["tier"] for r in keep)
    print(dict(stats), f"-> kept {len(keep)} ({sum(len(v[:2]) for v in rec.values())} recovery) {dict(tiers)} -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-n", type=int, default=300)
    p.add_argument("--n-easy", type=int, default=100)
    p.add_argument("--n-medium", type=int, default=500)
    p.add_argument("--n-hard", type=int, default=700)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--dump", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    make_pool(a) if a.cmd == "pool" else filter_traces(a)


if __name__ == "__main__":
    main()

"""Write the from-scratch replay of the 4B lineage that worked (P7 -> P8 -> P10 -> P10b) for another model or seed.

The phases are copied from the original configs. The only changes are model, seed and the warm-start chain, plus
the rule that each phase starts from the previous phase's FINAL step, so no checkpoint is hand-picked
(the original lineage used validation-chosen s40 / s90 / s90):
  a  P7    base model, fewshot_direct_retry, 40 steps        (configs/p7_rtx6000.yaml; original chose s40)
  b  P8    multi-candidate prompt, 90 steps, from a final     (configs/p8_rtx6000.yaml; original chose s90)
  c  P10   hint cycle 0.5@0,0.25@30,0@60, 90 steps, from b    (configs/p10_rtx6000.yaml)
  d  P10b  hint cycle 0.5@0,0.25@25,0@50, lr warmup 5, 80 steps, from c (configs/p10b_rtx6000.yaml)
Phase n+1's init_lora is baselines/<run id of phase n>_final, which scripts/cloud_chain.sh fetches from the bucket.
Every phase checkpoints every 2 steps or 5 minutes, whichever comes first.
usage: python scripts/make_lineage_configs.py NAME MODEL SEED [key=value ...]   (extra overrides apply to every phase)
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASES = [("a", "p7_rtx6000.yaml", 40), ("b", "p8_rtx6000.yaml", 90), ("c", "p10_rtx6000.yaml", 90),
          ("d", "p10b_rtx6000.yaml", 80)]


def main(argv):
    name, model, seed, *extra = argv
    overrides = {}
    for kv in extra:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    out = []
    prev = None
    for ph, src, steps in PHASES:
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", src)))
        cfg.update(model=model, seed=int(seed), total_steps=steps, checkpoint_every=2, checkpoint_timer=300)
        cfg.pop("init_lora", None)
        if prev:
            cfg["init_lora"] = f"baselines/{prev}_final"
        cfg.update(overrides)
        run_id = f"{name}-{ph}"
        path = os.path.join(ROOT, "configs", "lineage", f"{run_id}.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# {run_id}: phase {ph} of the from-scratch replay (scripts/make_lineage_configs.py; source {src}).\n"
                    f"# Generated: edit the generator, not this file.\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        out.append((run_id, path))
        prev = run_id
    for r, p in out:
        print(f"{r}={os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])

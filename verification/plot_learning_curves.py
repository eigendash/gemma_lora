"""REPORT.md learning-dynamics figure: reward, generated tokens per rollout and validation accuracy over training
steps for the model lineage P8 (from P7 s40) -> P10 (from P8 s90) -> P10b (from P10 s90), on one cumulative step axis.
Only the steps that lead to the next warm start are drawn (P8 0-89; its steps 90-99 are a side branch).
usage: python verification/plot_learning_curves.py [--out docs/figures/learning_curves.png]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (label, log, steps used, colour) - categorical slots 1-3 of the validated default palette
LINEAGE = [("P8", "p8_rtx6000", 90, "#2a78d6"), ("P10 (hints)", "p10_rtx6000", 90, "#eb6834"),
           ("P10b (hints + warmup)", "p10b_rtx6000", 80, "#1baf7a")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def rolling(xs, k=5):
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "learning_curves.png"))
    a = ap.parse_args(argv)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True, facecolor=SURF)
    panels = [("Training reward (5-step mean)", "reward"), ("Generated tokens per rollout", "tokens"),
              ("Validation accuracy (300 problems, greedy)", "val")]
    off = 0
    for label, d, n, col in LINEAGE:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "verification", "logs", d, "run", "train_log.jsonl"))][:n]
        x = [off + r["step"] for r in rows]
        series = {
            "reward": rolling([r["reward"] for r in rows]),
            "tokens": rolling([r["gen_tokens"] / ((r["n_sampled_prompts"] + r.get("n_hint_groups", 0)) * 8) for r in rows]),  # gen_tokens also counts hinted resamples
        }
        vx = [off + r["step"] for r in rows if r.get("val_correct") is not None]
        vy = [100 * r["val_correct"] for r in rows if r.get("val_correct") is not None]
        for ax, (_, key) in zip(axes, panels):
            if key == "val":
                ax.plot(vx, vy, color=col, lw=2, marker="o", ms=4, mec=SURF, mew=1.5)
            else:
                ax.plot(x, series[key], color=col, lw=2)
        axes[0].text(off + 2, 1.0, label, color=INK, fontsize=8.5, transform=axes[0].get_xaxis_transform(), va="bottom")
        off += n
        for ax in axes:
            ax.axvline(off - 0.5, color=GRID, lw=1, zorder=0)
    for ax, (title, key) in zip(axes, panels):
        ax.set_facecolor(SURF)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=14 if key == "reward" else 4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[2].set_ylabel("% correct", fontsize=8, color=INK2)
    axes[2].set_xlabel("training step (cumulative over warm starts)", fontsize=8.5, color=INK2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor=SURF)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Figures for REPORT.md, built from results/ only.

  fig1_holdout3.{pdf,png}       holdout3 accuracy of every final checkpoint, with 95% Wilson intervals
  fig2_dynamics.{pdf,png}       Qwen3-4B, recipe v2, seed 2: demo prompt vs zero-shot prompt over 300 GRPO steps
  fig3_self_correction.{pdf,png} what happens after the first code check on holdout3

usage: uv run python scripts/figures.py [--out docs/figures]
Pending runs (no eval file yet) are drawn as labelled empty rows, so the layout does not change when they land.
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from countdown_rl.evaluation.dumps import read_dump  # noqa: E402
from countdown_rl.evaluation.metrics import self_correction  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# Validated categorical slots (CVD and normal-vision separation pass on all pairs); base models in neutral gray.
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASE, V1, DEMO, ZERO = "#9a9993", "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holdout3(stem):
    path = os.path.join(RES, "evals", "holdout3", f"{stem}.json")
    if not os.path.exists(path):
        return None
    m = json.load(open(path))["metrics"]
    n = m["n"]
    return round(m["pass@1"] * n), n


def save(fig, out, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_holdout3(out):
    D, Z = "fewshot_multi_retry", "r1zero_tool"
    rows = [  # (group, label, eval file stem, colour)
        ("Qwen3-4B", "Base model, demo prompt", f"qwen3-4b-base__{D}", BASE),
        ("Qwen3-4B", "Base model, zero-shot prompt", f"qwen3-4b-base__{Z}", BASE),
        ("Qwen3-4B", "v1, seed 1", f"qwen3-4b-v1-s1__{D}", V1),
        ("Qwen3-4B", "v1, seed 2", f"qwen3-4b-v1-s2__{D}", V1),
        ("Qwen3-4B", "v1, hand-tuned lineage", f"qwen3-4b-v1-handtuned__{D}", V1),
        ("Qwen3-4B", "v2, demo prompt", f"qwen3-4b-v2-demo-s2__{D}", DEMO),
        ("Qwen3-4B", "v2, demo prompt, +45 steps", f"qwen3-4b-v2-demo-s2-cont__{D}", DEMO),
        ("Qwen3-4B", "v2, zero-shot prompt", f"qwen3-4b-v2-zeroshot-s2__{Z}", ZERO),
        ("Qwen3-4B", "v2, zero-shot prompt, +45 steps", f"qwen3-4b-v2-zeroshot-s2-cont__{Z}", ZERO),
        ("Qwen3-4B", "v2, zero-shot, seed 1 (stopped, step 53)", f"qwen3-4b-v2-zeroshot-s1-b13__{Z}", ZERO),
        ("Qwen3-8B", "Base model, demo prompt", f"qwen3-8b-base__{D}", BASE),
        ("Qwen3-8B", "v1", f"qwen3-8b-v1-s0__{D}", V1),
        ("Gemma-4-E4B", "Base model, demo prompt", f"gemma4-e4b-base__{D}", BASE),
        ("Gemma-4-E4B", "Base model, zero-shot prompt", f"gemma4-e4b-base__{Z}", BASE),
        ("Gemma-4-E4B", "v1", f"gemma4-e4b-v1-s0__{D}", V1),
        ("Gemma-4-E4B", "v2, zero-shot prompt", f"gemma4-e4b-v2-zeroshot-s0__{Z}", ZERO),
        ("Gemma-4-E4B", "v2, zero-shot prompt, +100 steps", f"gemma4-e4b-v2-zeroshot-s0-cont__{Z}", ZERO),
        ("Gemma-4-E2B", "v2, zero-shot prompt", f"gemma4-e2b-v2-zeroshot-s0__{Z}", ZERO),
        ("Gemma-4-E2B", "v2, demo prompt (stopped, step 82)", f"gemma4-e2b-v2-demo-s0-noguard__{D}", DEMO),
    ]
    fig, ax = plt.subplots(figsize=(6.75, 4.9))
    y, ticks, labels, groups = 0, [], [], []
    prev = None
    for group, label, exp, col in rows:
        if group != prev:
            if prev is not None:
                y += 0.6
            groups.append((group, y))
            prev = group
        r = holdout3(exp)
        if r:
            k, n = r
            lo, hi = wilson(k, n)
            ax.plot([100 * lo, 100 * hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([100 * k / n], [y], s=26, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(100 * hi + 1.2, y, f"{100 * k / n:.1f}", va="center", fontsize=7, color=INK)
        else:
            ax.text(1.5, y, "pending", va="center", fontsize=7, color=INK_2, style="italic")
        ticks.append(y)
        labels.append(label)
        y += 1
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 80)
    ax.set_xlabel("holdout3 accuracy, pass@1 (%), greedy; bars are 95% Wilson intervals (n = 1,000)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for group, gy in groups:
        ax.annotate(group, xy=(0, gy), xycoords=("axes fraction", "data"), xytext=(-150, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=7.5, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=5, label=l) for c, l in
               [(BASE, "base model"), (V1, "recipe v1 (demo prompt)"), (DEMO, "recipe v2, demo prompt"),
                (ZERO, "recipe v2, zero-shot prompt")]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handletextpad=0.3, columnspacing=1.4,
              borderaxespad=0.3)
    save(fig, out, "fig1_holdout3")


def series(exp, key, gap):
    """Contiguous (steps, values) segments over the four phases; a jump larger than `gap` steps starts a new segment,
    so steps lost from a log are drawn as a break, not interpolated."""
    segs, xs, ys, off, last = [], [], [], 0, None
    for ph, length in zip("abcd", (40, 90, 90, 80)):
        for r in (json.loads(l) for l in open(os.path.join(RES, "training", exp, f"phase_{ph}.jsonl"))):
            v = key(r)
            if v is None:
                continue
            x = off + r["step"]
            if last is not None and x - last > gap:
                segs.append((xs, ys))
                xs, ys = [], []
            xs.append(x)
            ys.append(v)
            last = x
        off += length
    segs.append((xs, ys))
    return segs


def smooth(ys, w=7):
    out = []
    for i in range(len(ys)):
        seg = ys[max(0, i - w // 2): i + w // 2 + 1]
        out.append(sum(seg) / len(seg))
    return out


def fig_dynamics(out):
    arms = [("qwen3-4b-v2-demo-s2", "demo prompt", DEMO), ("qwen3-4b-v2-zeroshot-s2", "zero-shot prompt", ZERO)]
    panels = [
        ("a  Validation accuracy (%)", lambda r: 100 * r["val_correct"] if "val_correct" in r else None, False, 5),
        ("b  Mean training reward", lambda r: r["reward"], True, 1),
        ("c  Generated tokens per rollout", lambda r: r["gen_tokens"] / (8 * r["n_sampled_prompts"]), True, 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15), constrained_layout=True)
    for ax, (title, key, sm, gap) in zip(axes, panels):
        for exp, label, col in arms:
            for i, (xs, ys) in enumerate(series(exp, key, gap)):
                lab = label if i == 0 else None
                if sm:
                    ax.plot(xs, ys, color=col, lw=0.6, alpha=0.25)
                    ax.plot(xs, smooth(ys), color=col, lw=1.4, label=lab)
                else:
                    ax.plot(xs, ys, color=col, lw=1.4, marker="o", markersize=2.2, label=lab)
        for b in (40, 130, 220):
            ax.axvline(b, color=GRID, lw=0.8, zorder=0)
        ax.set_title(title)
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel("GRPO step")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    top = axes[0].get_ylim()[1]
    for b, ph in ((0, "a"), (40, "b"), (130, "c"), (220, "d")):
        axes[0].text(b + 3, top * 0.98, ph, fontsize=6.5, color=INK_2, va="top")
    axes[0].legend(loc="lower right", handlelength=1.2)
    save(fig, out, "fig2_dynamics")


def fig_self_correction(out):
    arms = [("qwen3-4b-v2-demo-s2__fewshot_multi_retry", "v2, demo prompt"),
            ("qwen3-4b-v2-zeroshot-s2__r1zero_tool", "v2, zero-shot prompt")]
    segs = [("first check finds the target", V1), ("miss, retry solves it", ZERO),
            ("miss, retry fails", DEMO), ("miss, no retry", BASE)]
    fig, ax = plt.subplots(figsize=(6.75, 1.25))
    for y, (stem, label) in enumerate(arms):
        s = self_correction(read_dump(os.path.join(RES, "evals", "holdout3", stem + ".jsonl.gz")))
        vals = [s["first_check_hit"], s["missed_retried_recovered"], s["missed_retried"] - s["missed_retried_recovered"],
                s["first_check_missed"] - s["missed_retried"]]
        left = 0
        for v, (name, col) in zip(vals, segs):
            ax.barh(y, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0)
            if v >= 45:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=7,
                        color="white" if col in (V1, DEMO) else INK)
            left += v
    ax.set_yticks(range(len(arms)), [a[1] for a in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 1000)
    ax.set_xlabel("holdout3 problems (n = 1,000); runs without any code check are omitted")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in segs]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=4, handlelength=1, columnspacing=1.2)
    save(fig, out, "fig3_self_correction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_holdout3(a.out)
    fig_dynamics(a.out)
    fig_self_correction(a.out)


if __name__ == "__main__":
    main()

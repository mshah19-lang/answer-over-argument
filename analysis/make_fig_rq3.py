"""RQ3 figure (vector PDF + PNG preview), from rq3_summary.csv written by analyze_rq3.py.

Usage (on the PC, from Experiment/):
    python3 analysis/make_fig_rq3.py results/figdata/rq3_summary.csv [out_dir]
  (a) adoption of a CORRECT planted answer: bare vs each persuasion tactic, per model (95% CIs)
  (b) adoption of a WRONG planted answer: same
  (c) verify-first instruction: adoption without -> with the instruction, mean over the C cells and over the
      W cells (bare, flawed, strong, authority) per model. A good fix moves W left (resists wrong answers)
      while keeping C (still accepts correct ones).
Models are drawn in the paper's order; models missing from the CSV are skipped (so it also runs on test data).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, GRAY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#b9b8b2")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = [("qwen4", "Q-4B"), ("qwen8", "Q-8B"), ("qwen14", "Q-14B"), ("qwen32", "Q-32B"),
         ("gemma12", "G-12B"), ("llama8", "L-8B")]
TACTICS = [("bare", "Answer only", GRAY), ("authority", "Authority", BLUE), ("confidence", "Confidence", YELLOW),
           ("social", "Consensus", AQUA), ("citation", "Citation", VIOLET), ("hedge", "Hedge", MAGENTA)]
VCELLS = ["bare", "flawed", "strong", "authority"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.0, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def load(path):
    d = defaultdict(dict)  # (model, part) -> name -> (p, lo, hi)
    for r in csv.DictReader(open(path)):
        if r["point"] == "":
            continue
        d[(r["model"], r["part"])][r["name"]] = (100 * float(r["point"]), 100 * float(r["lo"]), 100 * float(r["hi"]))
    return d


def style(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def tactic_panel(ax, d, models, truth, title):
    n = len(TACTICS)
    w = 0.8 / n
    for k, (key, lab, c) in enumerate(TACTICS):
        xs, ys, lo, hi = [], [], [], []
        for j, (m, _) in enumerate(models):
            v = d[(m, "tactics_adopt")].get("%s-%s" % (truth, key))
            if v is None:
                continue
            xs.append(j + (k - (n - 1) / 2) * w)
            ys.append(v[0])
            lo.append(v[0] - v[1])
            hi.append(v[2] - v[0])
        ax.bar(xs, ys, width=w * 0.92, color=c, label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.55, capsize=0, zorder=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([s for _, s in models], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=8)
    style(ax)


def verify_panel(ax, d, models):
    def mean(m, cond, t):
        vals = [d[(m, "verify_adopt")].get("%s:%s-%s" % (cond, t, c)) for c in VCELLS]
        vals = [v[0] for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    for j, (m, _) in enumerate(models):
        y = len(models) - 1 - j
        for t, c, dy in (("C", AQUA, 0.17), ("W", ORANGE, -0.17)):
            a, b = mean(m, "nov", t), mean(m, "ver", t)
            if a is None or b is None:
                continue
            ax.annotate("", xy=(b, y + dy), xytext=(a, y + dy),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2, mutation_scale=7, shrinkA=0, shrinkB=0))
            ax.plot([a], [y + dy], "o", ms=3.6, mfc="white", mec=c, mew=1.0, zorder=4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([s for _, s in models][::-1])
    ax.set_xlim(left=0)
    ax.set_xlabel("Honest agents switching (%)")
    ax.set_title("(c) Verify-first prompt: before → after", fontsize=8)
    ax.plot([], [], color=AQUA, lw=1.2, label="Correct answer planted (should stay)")
    ax.plot([], [], color=ORANGE, lw=1.2, label="Wrong answer planted (should drop)")
    style(ax, "x")


def main(path, out):
    d = load(path)
    present = {m for (m, _) in d}
    models = [(m, s) for m, s in ORDER if m in present] or [(m, m) for m in sorted(present)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1, 1, 0.85]})
    tactic_panel(axes[0], d, models, "C", "(a) Colleague proposes the CORRECT answer")
    tactic_panel(axes[1], d, models, "W", "(b) Colleague proposes a WRONG answer")
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for ax in axes[:2]:
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("Honest agents switching to it (%)")
    axes[1].set_yticklabels([])
    verify_panel(axes[2], d, models)
    fig.tight_layout(w_pad=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.36, 0.02),
               title="(a, b) What the colleague adds to the answer", title_fontsize=7.2)
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h2, l2, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.85, 0.02),
               title="(c) mean over bare, flawed, strong, authority", title_fontsize=7.0)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("fig_rq3." + ext), dpi=220, bbox_inches="tight")
    print("wrote", out / "fig_rq3.pdf", "(+ .png)")


if __name__ == "__main__":
    p = Path(sys.argv[1])
    main(p, Path(sys.argv[2]) if len(sys.argv) > 2 else p.parent)

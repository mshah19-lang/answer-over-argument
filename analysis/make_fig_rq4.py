"""RQ4 figure (vector PDF + PNG preview): keep debate's answer, take trust from round 1.

Usage (on the PC, from Experiment/):
    python3 analysis/make_fig_rq4.py results/rq4
Reads rq4_curves.csv and rq4_summary.csv (copied from the cluster, made by analyze_rq4.py).
  (a) risk-coverage, MedQA, Qwen3-32B: debate vs hybrid vs SC-6
  (b) % of wrong answers caught when the 30% least-trusted questions go to a human,
      MedQA, all 6 models, 95% bootstrap CIs
Colors match Figure 1: blue = round-1 signal, orange dashed = round-2 (debate), aqua = SC-6.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # method -> (label, color, marker, linestyle)
    "debate": ("Debate (round-2 agreement)", ORANGE, "^", "--"),
    "hyb-r1": ("Hybrid: debate answer + round-1 agreement", BLUE, "o", "-"),
    "SC-6": ("SC-6 (same compute)", AQUA, "s", "-"),
}
MODELS = [("qwen4", "Qwen3-4B"), ("qwen8", "Qwen3-8B"), ("qwen14", "Qwen3-14B"),
          ("qwen32", "Qwen3-32B"), ("gemma12", "Gemma-3-12B"), ("llama8", "Llama-3.1-8B")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


def panel_a(ax, curves, ds="medqa", model="qwen32"):
    for m in ["debate", "hyb-r1", "SC-6"]:
        pts = [(float(r["coverage"]), float(r["risk"])) for r in curves
               if r["dataset"] == ds and r["model"] == model and r["method"] == m]
        pts = [p for p in pts if p[0] >= 0.1]  # first few points are noise (1-29 questions)
        lab, c, _, ls = STYLE[m]
        ax.plot([100 * x for x, _ in pts], [100 * y for _, y in pts], color=c, ls=ls, lw=1.6, label=lab)
    ax.axvline(70, color=INK2, lw=0.7, ls=":")
    ax.text(69, 1, "defer 30%", rotation=90, color=INK2,
            fontsize=6.5, ha="right", va="bottom")
    ax.set_xlim(10, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage: % of questions answered (most trusted first)")
    ax.set_ylabel("Error rate on answered questions (%)")
    ax.set_title("(a) Risk-coverage, MedQA, Qwen3-32B")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.70), frameon=False, handlelength=2.4)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))


def panel_b(ax, summ, ds="medqa", metric="caught@30"):
    get = {(r["model"], r["method"]): r for r in summ
           if r["dataset"] == ds and r["metric"] == metric}
    offs = {"debate": -0.22, "hyb-r1": 0.0, "SC-6": 0.22}
    for m in ["debate", "hyb-r1", "SC-6"]:
        lab, c, mk, _ = STYLE[m]
        ys = [j + offs[m] for j in range(len(MODELS))]
        r = [get[(k, m)] for k, _ in MODELS]
        x = [100 * float(v["point"]) for v in r]
        lo = [100 * (float(v["point"]) - float(v["lo"])) for v in r]
        hi = [100 * (float(v["hi"]) - float(v["point"])) for v in r]
        ax.errorbar(x, ys, xerr=[lo, hi], fmt=mk, color=c, ms=4.5, lw=1.0, capsize=0,
                    mec="white", mew=0.6, label=lab)
    ax.axvline(30, color=INK2, lw=0.7, ls=":")
    ax.text(30.8, len(MODELS) - 0.45, "random (30%)", color=INK2, fontsize=6.5, va="center")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([n for _, n in MODELS])
    ax.invert_yaxis()
    ax.set_ylim(len(MODELS) - 0.3, -0.5)
    ax.set_xlabel("Wrong answers caught (%), 30% least-trusted deferred")
    ax.set_title("(b) Errors caught at 30% deferral, MedQA")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/rq4")
    curves = list(csv.DictReader(open(src / "rq4_curves.csv")))
    summ = list(csv.DictReader(open(src / "rq4_summary.csv")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.05]})
    panel_a(a, curves)
    panel_b(b, summ)
    fig.tight_layout(w_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(src / ("fig_rq4." + ext), dpi=220, bbox_inches="tight")
    print("wrote", src / "fig_rq4.pdf", "and .png")

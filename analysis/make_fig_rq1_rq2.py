"""Full-paper figures for RQ1 and RQ2 (vector PDF + PNG preview).

Usage (on the PC, from Experiment/):
    python3 analysis/make_fig_rq1_rq2.py results/figdata
Reads the saved cluster outputs (2026-11-17):
    rq1_stats_all_2026-11-17.txt          = stats.py on results, results/medmcqa, results/law, results/gpqa
    rq2_vigilance_medqa_2026-11-17.txt    = analyze_vigilance.py (v3) on MedQA, 6 models
Writes fig_rq1.{pdf,png} and fig_rq2.{pdf,png} into the same folder.
  fig_rq1: AUROC of disagreement -> wrong vote; round 1 vs round 2 vs SC-6; 4 datasets x 6 models, 95% CIs
  fig_rq2: (a) adoption of a CORRECT planted answer, (b) of a WRONG one, for bare/flawed/strong arguments;
           (c) the same models asked directly: spot the misstated fact (hit) vs flag a sound argument (false alarm)
Colors: Fig. RQ1 matches the RQ4 figure (blue = round 1, orange dashed = round 2, aqua = SC-6).
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [("qwen4", "Qwen3-4B"), ("qwen8", "Qwen3-8B"), ("qwen14", "Qwen3-14B"),
          ("qwen32", "Qwen3-32B"), ("gemma12", "Gemma-3-12B"), ("llama8", "Llama-3.1-8B")]
SHORT = ["Q-4B", "Q-8B", "Q-14B", "Q-32B", "G-12B", "L-8B"]
DATASETS = [("results", "MedQA"), ("results/medmcqa", "MedMCQA"),
            ("results/law", "MMLU Prof. Law"), ("results/gpqa", "GPQA")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42,
})


# ---------------- parsing ----------------
def parse_rq1(path):
    """-> auc[ds_dir][model][method] = (point, lo, hi)"""
    auc, ds, model, in5 = {}, None, None, False
    for line in open(path):
        m = re.match(r"##### (\S+)", line)
        if m:
            ds = m.group(1)
            auc[ds] = {}
            continue
        m = re.match(r"===== (\S+) \(", line)
        if m:
            model, in5 = m.group(1), False
            auc[ds][model] = {}
            continue
        if line.startswith("5) AUROC"):
            in5 = True
            continue
        if in5:
            m = re.match(r"\s+(SC-3|SC-6|round1|round2|planted)\s+([\d.]+)\s+\[([\d.]+), ([\d.]+)\]", line)
            if m:
                auc[ds][model][m.group(1)] = tuple(float(x) for x in m.group(2, 3, 4))
            elif "diff" in line:
                in5 = False
    return auc


def parse_rq2(path):
    """-> adopt[model][cell] = (pct, lo, hi);  det[model] = (hit, fa, n) for C plants"""
    adopt, det, model = {}, {}, None
    for line in open(path):
        m = re.match(r"--- (\S+): adoption", line)
        if m:
            model = m.group(1)
            adopt[model] = {}
            continue
        m = re.match(r"\s+([CW]-(?:bare|flawed|strong))\s+\d+/\d+\s+([\d.]+)% \[\s*([\d.]+),\s*([\d.]+)\]", line)
        if m and model:
            adopt[model][m.group(1)] = tuple(float(x) for x in m.group(2, 3, 4))
            continue
        m = re.match(r"\s+C: flawed says YES \d+/\d+\s+located\s+(\d+)/(\d+) \| strong says YES \(false alarm\)\s+(\d+)/", line)
        if m and model:
            det[model] = (int(m.group(1)), int(m.group(3)), int(m.group(2)))
    return adopt, det


# ---------------- figures ----------------
def style(ax, ygrid=True):
    ax.grid(True, axis="y" if ygrid else "x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_rq1(auc, out):
    meth = [("SC-6", "SC-6 (same compute)", AQUA, "s", 0.22),
            ("round1", "Debate, round 1 (before discussion)", BLUE, "o", 0.0),
            ("round2", "Debate, round 2 (after discussion)", ORANGE, "^", -0.22)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
    for ax, (dsdir, title) in zip(axes, DATASETS):
        ax.axhline(0.5, color=INK2, lw=0.7, ls=":")
        for key, lab, c, mk, off in meth:
            xs, ys, lo, hi = [], [], [], []
            for j, (m, _) in enumerate(MODELS):
                p, l, h = auc[dsdir][m][key]
                xs.append(j - off)
                ys.append(p)
                lo.append(p - l)
                hi.append(h - p)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=c, ms=3.8, lw=0.9, capsize=0,
                        mec="white", mew=0.5, label=lab)
        ax.set_title(title)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(SHORT, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        style(ax)
    axes[0].set_ylabel("AUROC: disagreement predicts a wrong vote")
    axes[0].set_ylim(0.35, 0.9)
    axes[0].text(5.4, 0.505, "chance", color=INK2, fontsize=6.5, ha="right", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("fig_rq1." + ext), dpi=220, bbox_inches="tight")


def fig_rq2(adopt, det, out):
    arg = [("bare", "Answer only", GRAY), ("flawed", "Argument with a misstated fact", ORANGE),
           ("strong", "Sound argument", BLUE)]
    w = 0.26
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, truth, title in [(axes[0], "C", "(a) Colleague proposes the CORRECT answer"),
                             (axes[1], "W", "(b) Colleague proposes a WRONG answer")]:
        for k, (a, lab, c) in enumerate(arg):
            xs = [j + (k - 1) * (w + 0.02) for j in range(len(MODELS))]
            v = [adopt[m][truth + "-" + a] for m, _ in MODELS]
            ax.bar(xs, [p for p, _, _ in v], width=w, color=c, label=lab, zorder=2)
            ax.errorbar(xs, [p for p, _, _ in v], yerr=[[p - lo for p, lo, _ in v], [hi - p for p, _, hi in v]],
                        fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0, zorder=3)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(SHORT, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 75)
        style(ax)
    axes[0].set_ylabel("Honest agents switching to it (%)")
    axes[1].set_yticklabels([])
    # (c) detection when asked directly (correct-answer plants)
    ax = axes[2]
    hit = [100 * det[m][0] / det[m][2] for m, _ in MODELS]
    fa = [100 * det[m][1] / det[m][2] for m, _ in MODELS]
    xs = range(len(MODELS))
    ax.bar([x - 0.17 for x in xs], hit, width=0.32, color=AQUA, label="Finds the misstated fact", zorder=2)
    ax.bar([x + 0.17 for x in xs], fa, width=0.32, color=GRAY, label="Flags a sound argument", zorder=2)
    ax.set_title("(c) Asked: does it misstate a fact?", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of arguments", labelpad=1)
    style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8)
    leg1 = fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.36, 0.02),
                      title="(a, b) What the colleague says in the discussion", title_fontsize=7.2)
    fig.add_artist(leg1)
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h2, l2, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.84, 0.02),
               title="(c) Asked directly", title_fontsize=7.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("fig_rq2." + ext), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figdata")
    auc = parse_rq1(src / "rq1_stats_all_2026-11-17.txt")
    adopt, det = parse_rq2(src / "rq2_vigilance_medqa_2026-11-17.txt")
    for d, _ in DATASETS:  # sanity: all 6 models x 3 methods parsed
        assert all(k in auc[d][m] for m, _ in MODELS for k in ("SC-6", "round1", "round2")), d
    assert all(len(adopt[m]) == 6 and m in det for m, _ in MODELS)
    fig_rq1(auc, src)
    fig_rq2(adopt, det, src)
    print("wrote", src / "fig_rq1.pdf", "and", src / "fig_rq2.pdf", "(+ .png)")

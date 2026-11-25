"""Full-paper figure for the decisive-flaw test (Step 3), same style as fig_rq2.

Usage (from Experiment/):
    python3 analysis/make_fig_rq2b.py results/figdata
Reads step3_2026-11-25.txt (= analysis/step3_analysis.py results_cluster, 6 models) and writes fig_rq2b.{pdf,png}.
  (a) adoption of the planted CORRECT answer with the strong vs the decisive-flawed argument, 79 usable questions
  (b) asked directly: decisive flaw located (hit) vs strong argument flagged (false alarm)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_fig_rq1_rq2 import AQUA, BLUE, GRAY, INK2, MODELS, ORANGE, SHORT, plt, style  # noqa: E402

NUM = r"(-?\d+\.\d+) \[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"


def parse(path):
    adopt, det, model, sec = {}, {}, None, None
    for line in open(path):
        if re.match(r"\d\)", line):
            sec = line[0]
        m = re.match(r"\s+--- (\w+)$", line)
        if m:
            model = m.group(1)
            continue
        if sec == "3" and model and re.match(r"\s+all usable\s+q=", line):
            v = [tuple(float(x) for x in t) for t in re.findall(NUM, line)]
            adopt[model] = {"strong": v[0], "decisive": v[1], "diff": v[2]}
        m = re.match(r"\s+(\w+)\s+located\s+(\d+)/(\d+).*false alarm\s+(\d+)/(\d+)", line)
        if sec == "4" and m:
            det[m.group(1)] = tuple(int(x) for x in m.group(2, 3, 4, 5))
    return adopt, det


def fig(adopt, det, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax, w = axes[0], 0.36
    for k, (a, lab, c) in enumerate([("decisive", "Same argument, decisive detail replaced", ORANGE),
                                     ("strong", "Sound argument", BLUE)]):
        xs = [j + (k - 0.5) * (w + 0.02) for j in range(len(MODELS))]
        v = [adopt[m][a] for m, _ in MODELS]
        ax.bar(xs, [p for p, _, _ in v], width=w, color=c, label=lab, zorder=2)
        ax.errorbar(xs, [p for p, _, _ in v], yerr=[[p - lo for p, lo, _ in v], [hi - p for p, _, hi in v]],
                    fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0, zorder=3)
    for j, (m, _) in enumerate(MODELS):  # strong - decisive difference above each pair
        d, lo, hi = adopt[m]["diff"]
        top = max(adopt[m]["strong"][2], adopt[m]["decisive"][2])
        ax.text(j, top + 2.5, ("%+.0f" % d) + ("*" if lo > 0 or hi < 0 else ""), ha="center", va="bottom",
                fontsize=6.8, color=INK2)
    ax.set_title("(a) Colleague proposes the CORRECT answer", fontsize=8)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Honest agents switching to it (%)")
    style(ax)
    ax = axes[1]
    xs = range(len(MODELS))
    hit = [100 * det[m][0] / det[m][1] for m, _ in MODELS]
    fa = [100 * det[m][2] / det[m][3] for m, _ in MODELS]
    ax.bar([x - 0.17 for x in xs], hit, width=0.32, color=AQUA, label="Finds the decisive change", zorder=2)
    ax.bar([x + 0.17 for x in xs], fa, width=0.32, color=GRAY, label="Flags the sound argument", zorder=2)
    ax.set_title("(b) Asked: does it misstate a fact?", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(SHORT, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of arguments", labelpad=1)
    style(ax)
    fig.tight_layout(w_pad=1.2)
    h, l = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.28, 0.02),
                     title="(a) What the colleague says; number = difference in points (* = 95% CI excludes 0)",
                     title_fontsize=7.0)
    fig.add_artist(leg)
    h2, l2 = ax.get_legend_handles_labels()
    fig.legend(h2, l2, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.78, 0.02),
               title="(b) Asked directly", title_fontsize=7.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("fig_rq2b." + ext), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figdata")
    adopt, det = parse(src / "step3_2026-11-25.txt")
    assert all(m in adopt and m in det for m, _ in MODELS), (adopt.keys(), det.keys())
    fig(adopt, det, src)
    print("wrote", src / "fig_rq2b.pdf")

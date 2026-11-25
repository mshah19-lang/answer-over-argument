"""Figure 1 of the full paper: setup / overview diagram (vector PDF + PNG preview). v2, 2026-11-17.

Usage (on the PC, from Experiment/):
    python3 analysis/make_fig_setup.py results/figdata
Story shown (one MedQA-style example):
  (1) three role agents answer independently (A, B, and the third reply is replaced by the planted colleague: C)
  (2) in the discussion each honest agent reads the others; the planted reply argues for C with ONE misstated
      fact ("10 days" while the question says 5) -- the RQ2 manipulation; RQ3 adds tactics / verify-first
  (3) round-2 answers -> majority vote; panel lists what each RQ measures
  bottom strip: SC-6 baseline at equal compute.
Roles = MedQA/MedMCQA roles in run_medqa.py; other domains use their own roles (say so in the caption).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, INK3 = "#1b1b1a", "#52514e", "#8a8984"
TINT_B, TINT_G, TINT_A, TINT_O = "#eef4fc", "#f4f3f0", "#e9f7f1", "#fdf0ea"
LINE = "#d9d8d3"
# style knobs (changed by make_fig_setup_variants.py)
PANEL_FILLS = [TINT_B, TINT_G, "#f7f7f5"]
PANEL_EC, PANEL_LS, PANEL_LW = "none", "-", 0.0
STRIP_FC, STRIP_EC, STRIP_LS, STRIP_LW = TINT_A, "none", "-", 0.0
VOTE_FC, HL_FC = None, None  # None -> INK / ORANGE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "mathtext.default": "regular",
})


# ---------------- drawing helpers ----------------
def rbox(ax, x, y, w, h, fc="white", ec=LINE, lw=0.8, ls="-", r=2.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%.2f" % r,
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, size=7, color=INK, ha="left", va="center", weight="normal", style="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style,
                   zorder=z, **kw)


def agent(ax, x, y, color, r=3.2):
    """Small robot-head icon centred at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - r, y - r * 0.85), 2 * r, 1.7 * r,
                                boxstyle="round,pad=0,rounding_size=%.2f" % (r * 0.55),
                                fc=color, ec="none", zorder=4))
    for dx in (-0.38 * r, 0.38 * r):
        ax.add_patch(Circle((x + dx, y + 0.05 * r), 0.2 * r, fc="white", ec="none", zorder=5))
    ax.plot([x, x], [y + 0.85 * r, y + 1.25 * r], color=color, lw=0.9, zorder=4, solid_capstyle="round")
    ax.add_patch(Circle((x, y + 1.32 * r), 0.17 * r, fc=color, ec="none", zorder=4))


def bubble(ax, x, y, w, h, s, color, fill="white", ls="-", size=7.5, weight="bold", tail=True):
    """Speech bubble with tail pointing left, text centred."""
    rbox(ax, x, y, w, h, fc=fill, ec=color, lw=1.0, ls=ls, r=1.6, z=3)
    if tail:
        ax.add_patch(Polygon([(x + 0.3, y + h * 0.35), (x - 2.0, y + h * 0.5), (x + 0.3, y + h * 0.65)],
                             closed=True, fc=fill, ec=color, lw=1.0, zorder=2))
        ax.plot([x + 0.35, x + 0.35], [y + h * 0.36, y + h * 0.64], color=fill, lw=1.6, zorder=3)
    txt(ax, x + w / 2, y + h / 2, s, size=size, color=INK, ha="center", weight=weight)


def arrow(ax, p1, p2, color=INK3, lw=0.8, rad=0.0, ls="-", z=1):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw, ls=ls,
                                 connectionstyle="arc3,rad=%.2f" % rad, zorder=z, shrinkA=0, shrinkB=0))


def header(ax, x, y, num, title, color):
    ax.add_patch(Circle((x + 2.6, y), 2.6, fc=color, ec="none", zorder=5))
    txt(ax, x + 2.6, y - 0.1, num, size=7.5, color="white", ha="center", weight="bold", z=6)
    txt(ax, x + 6.8, y, title, size=8.2, weight="bold")


def chip(ax, x, y, s, color, w=None, size=6.3):
    w = w or (2.2 + 1.45 * len(s))
    rbox(ax, x, y - 2.0, w, 4.0, fc=color, ec="none", r=1.6, z=4)
    txt(ax, x + w / 2, y - 0.05, s, size=size, color="white", ha="center", weight="bold", z=6)
    return w


# ---------------- figure ----------------
def main(out, name="fig_setup"):
    W, H = 200, 96
    LOW = -7  # extra strip below for the scope line
    fig = plt.figure(figsize=(7.2, 7.2 * (H - LOW) / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(LOW, H)
    ax.set_aspect("equal")
    ax.axis("off")

    P1, P2, P3 = (1, 61), (65, 133), (137, 199)  # panel x-ranges
    top, bot = 92, 15
    for (a, b), fc in zip((P1, P2, P3), PANEL_FILLS):
        rbox(ax, a, bot, b - a, top - bot, fc=fc, ec=PANEL_EC, lw=PANEL_LW, ls=PANEL_LS, r=3, z=0)
    header(ax, P1[0] + 2.5, 87.5, "1", "Round 1: answer independently", BLUE)
    header(ax, P2[0] + 2.5, 87.5, "2", "Discussion: read colleagues, revise", INK2)
    header(ax, P3[0] + 2.5, 87.5, "3", "Decision and what we measure", INK)

    # ---- (1) question card
    qx, qy, qw, qh = 4, 62, 54, 20
    rbox(ax, qx, qy, qw, qh, fc="white", ec=LINE, lw=0.8, r=2)
    txt(ax, qx + 2.5, qy + qh - 3.6, "Question", size=6.6, color=INK3, weight="bold")
    txt(ax, qx + 2.5, qy + qh - 9.6,
        "A 34-year-old woman has had fever for\n5 days and a new heart murmur. Which\nis the most likely diagnosis?",
        size=6.6, color=INK, linespacing=1.25)
    cx = qx + 2.5
    for L in "ABCD":
        rbox(ax, cx, qy + 1.6, 6.2, 3.6, fc="white", ec=LINE, lw=0.7, r=1.2, z=3)
        txt(ax, cx + 3.1, qy + 3.4, L, size=6.3, color=INK2, ha="center", weight="bold")
        cx += 7.6
    txt(ax, qx + qw - 2.5, qy + qh - 3.6, "MedQA-style example", size=5.8, color=INK3, ha="right",
        style="italic")

    # ---- (1) three agents
    rows = [(50, "Internist", BLUE, "A", "-"),
            (38, "Clinical\npharmacologist", BLUE, "B", "-"),
            (26.5, "Pathologist", ORANGE, "C", (0, (2.5, 1.5)))]
    for y, role, col, ans, ls in rows:
        agent(ax, 9, y, col)
        txt(ax, 15, y, role, size=6.6, color=INK, linespacing=1.05)
        bubble(ax, 44, y - 3.6, 12, 7.2, ans, col, ls=ls)
    txt(ax, 4, 20.6, "Normal debate (RQ1, RQ4): three honest agents.", size=5.7, color=INK2, style="italic")
    txt(ax, 4, 17.3, "Planted runs (RQ2, RQ3): this reply is replaced.", size=5.7, color=ORANGE,
        style="italic")
    txt(ax, 50, 56.3, "agreement 1/3", size=5.8, color=BLUE, ha="center", style="italic")
    arrow(ax, (31, 61.5), (31, 55), color=INK3)

    # ---- (2) what an honest agent reads
    mx, mw = 68, 62
    txt(ax, mx + 1, 80.5, "What an honest agent reads (e.g. the internist):", size=6.4, color=INK2,
        style="italic")
    # colleague 1 message
    rbox(ax, mx, 64, mw, 12.5, fc="white", ec=BLUE, lw=0.9, r=2)
    agent(ax, mx + 4.8, 70.2, BLUE, r=2.3)
    txt(ax, mx + 9.5, 73.3, "Clinical pharmacologist", size=6.3, color=BLUE, weight="bold")
    txt(ax, mx + 9.5, 68.0, "“...the drug history suggests B.”  Answer: B", size=6.4)
    # planted message with the misstated fact highlighted
    rbox(ax, mx, 45, mw, 15.5, fc=TINT_O, ec=ORANGE, lw=1.1, r=2)
    agent(ax, mx + 4.8, 53.2, ORANGE, r=2.3)
    txt(ax, mx + 9.5, 57.2, "Pathologist (planted reply)", size=6.3, color=ORANGE, weight="bold")
    t1 = txt(ax, mx + 9.5, 52.2, "“Fever for", size=6.4)
    # highlighted number
    hx = mx + 21.6
    rbox(ax, hx, 50.2, 5.8, 4.0, fc=HL_FC or ORANGE, ec="none", r=0.9, z=4)
    txt(ax, hx + 2.9, 52.2, "10", size=6.4, color="white", ha="center", weight="bold", z=6)
    txt(ax, hx + 6.6, 52.2, "days points to C.”  Answer: C", size=6.4)
    txt(ax, mx + 9.5, 47.6, "question says 5 days → one misstated fact", size=5.8, color=ORANGE,
        style="italic")
    del t1

    # detection probe (RQ2): same model asked directly
    rbox(ax, mx, 36.3, mw, 6.4, fc="white", ec=ORANGE, lw=0.8, ls=(0, (2.5, 1.5)), r=1.6)
    txt(ax, mx + 2.5, 39.5, "Asked directly:", size=6.0, color=ORANGE, weight="bold")
    txt(ax, mx + 21.5, 39.5, "\u201cDoes this reply misstate a fact?\u201d", size=6.0)

    # variants box
    vy = 16.5
    rbox(ax, mx, vy, mw, 18.3, fc="white", ec=LINE, lw=0.8, r=2)
    txt(ax, mx + 2.5, vy + 15.3, "The planted reply varies:", size=6.4, color=INK, weight="bold")
    c = chip(ax, mx + 2.5, vy + 11.0, "RQ2", ORANGE)
    txt(ax, mx + 2.5 + c + 1.5, vy + 11.0, "correct / wrong answer  \u00d7  answer only /", size=6.0)
    txt(ax, mx + 2.5 + c + 1.5, vy + 7.9, "misstated fact / strong (facts correct)", size=6.0)
    c = chip(ax, mx + 2.5, vy + 3.4, "RQ3", INK2)
    txt(ax, mx + 2.5 + c + 1.5, vy + 3.4, "+ authority, confidence, consensus,\n   citation, hedge;"
        " verify-first prompt", size=5.8, linespacing=1.1)

    # arrows panel 1 -> panel 2
    arrow(ax, (56.5, 38), (mx - 0.5, 70), color=BLUE, rad=-0.2)
    arrow(ax, (56.5, 26.5), (mx - 0.5, 52), color=ORANGE, rad=-0.15)

    # ---- (3) round-2 answers + vote
    txt(ax, P3[0] + 3, 80.5, "Round-2 answers", size=6.4, color=INK2, style="italic")
    r2 = [(74, "Internist", BLUE), (65, "Clin. pharmacologist", BLUE)]
    for y, role, col in r2:
        agent(ax, P3[0] + 6, y, col, r=2.6)
        txt(ax, P3[0] + 10.5, y, role, size=6.3)
        bubble(ax, P3[0] + 38.5, y - 3.0, 7.5, 6.0, "C", ORANGE, size=7)
    arrow(ax, (mx + mw + 0.5, 62.2), (P3[0] + 2.0, 69.5), color=INK3, rad=0.0)
    # vote
    rbox(ax, P3[0] + 48.5, 63, 12, 15, fc=VOTE_FC or INK, ec="none", r=2, z=3)
    txt(ax, P3[0] + 54.5, 73.6, "vote", size=6.0, color="white", ha="center", z=6)
    txt(ax, P3[0] + 54.5, 67.6, "C", size=11, color="white", ha="center", weight="bold", z=6)
    arrow(ax, (P3[0] + 46.4, 69.5), (P3[0] + 48.3, 69.5), color=INK3)
    txt(ax, P3[0] + 3, 58.3, "honest agents adopted the planted answer", size=5.9, color=ORANGE,
        style="italic")
    txt(ax, P3[0] + 52.0, 80.5, "agreement 3/3", size=5.8, color=ORANGE, ha="center", style="italic")

    # measures list
    my = 53.5
    rbox(ax, P3[0] + 2.5, bot + 3.5, P3[1] - P3[0] - 5, my - bot - 3.0, fc="white", ec=LINE, lw=0.8, r=2)
    items = [("RQ1", BLUE, "Does agreement still flag errors\nafter discussion? (AUROC)"),
             ("RQ2", ORANGE, "Do agents follow the argument\nor just the answer?"),
             ("RQ3", INK2, "Do persuasion tactics or a\nverify-first prompt change this?"),
             ("RQ4", AQUA, "Answer from round 2, trust from\nround-1 agreement: defer if low?")]
    y = my - 5.0
    for tag, col, s in items:
        chip(ax, P3[0] + 5, y, tag, col, w=9.5)
        txt(ax, P3[0] + 16.5, y - 0.2, s, size=6.1, linespacing=1.15)
        y -= 8.6

    # ---- bottom strip: SC-6 baseline
    rbox(ax, 1, 1, 198, 11, fc=STRIP_FC, ec=STRIP_EC, lw=STRIP_LW, ls=STRIP_LS, r=3, z=0)
    txt(ax, 4, 6.5, "Baseline at equal compute:", size=6.8, weight="bold", color=INK)
    txt(ax, 42, 6.5, "self-consistency (SC-6), one prompt sampled 6 times", size=6.6, color=INK)
    for k in range(6):
        agent(ax, 111 + 7.0 * k, 6.2, AQUA, r=2.1)
    arrow(ax, (151, 6.5), (158, 6.5), color=AQUA)
    txt(ax, 160, 6.5, "majority vote + agreement", size=6.6, color=INK)
    txt(ax, 100, -3.2, "Domains: MedQA, MedMCQA, MMLU Professional Law, GPQA (RQ1, RQ4); MedQA (RQ2, RQ3).   "
        "Models: Qwen3 4B/8B/14B/32B, Gemma-3-12B, Llama-3.1-8B.", size=6.0, color=INK2, ha="center")

    for ext in ("pdf", "png"):
        fig.savefig(out / ("%s.%s" % (name, ext)), dpi=250)
    plt.close(fig)
    print("wrote", out / (name + ".pdf"), "(+ .png)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/figdata"))

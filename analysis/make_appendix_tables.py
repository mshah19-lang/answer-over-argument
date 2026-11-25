"""Appendix tables (LaTeX) generated from saved analysis outputs, so no number is typed by hand.

Usage (on the PC, from Experiment/):
    python3 analysis/make_appendix_tables.py "../full_paper/sections/app_tables.tex"
Reads (saved 2026-11-17):
    results/figdata/rq1_stats_all_2026-11-17.txt        stats.py, 4 datasets x 6 models
    results/figdata/rq2_vigilance_medqa_2026-11-17.txt  analyze_vigilance.py v3, MedQA x 6 models
    results/rq4/rq4_summary.csv                         analyze_rq4.py
    results/figdata/rq3_summary.csv                      analyze_rq3.py, MedQA x 6 models (2026-11-19)
Tables: tab:app_rq1 (one per dataset), tab:app_manip, tab:app_rq2, tab:app_rq3a, tab:app_rq3b, tab:app_rq4.
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

FIG = Path("results/figdata")
MODELS = [("qwen4", "Qwen3-4B"), ("qwen8", "Qwen3-8B"), ("qwen14", "Qwen3-14B"), ("qwen32", "Qwen3-32B"),
          ("gemma12", "Gemma-3-12B"), ("llama8", "Llama-3.1-8B")]
DS = [("results", "medqa", "MedQA"), ("results/medmcqa", "medmcqa", "MedMCQA"),
      ("results/law", "law", "MMLU Professional Law"), ("results/gpqa", "gpqa", "GPQA")]


# ---------------- parsing ----------------
def parse_rq1(path):
    out, ds, m, sec = {}, None, None, None
    for line in open(path):
        g = re.match(r"##### (\S+)", line)
        if g:
            ds, m = g.group(1), None
            out[ds] = {}
            continue
        g = re.match(r"===== (\S+) \((\d+) questions", line)
        if g:
            m = g.group(1)
            out[ds][m] = {"n": int(g.group(2)), "acc": {}, "prec": {}, "auc": {}, "trans": {}}
            continue
        g = re.match(r"(\d)\) ", line)
        if g:
            sec = g.group(1)
            continue
        if m is None:
            continue
        d = out[ds][m]
        g = re.match(r"\s+(single|CoT|SC-3|SC-6|round1|round2|planted)\s+([\d.]+)\s+\[", line)
        if g and sec in "135":
            d[{"1": "acc", "3": "prec", "5": "auc"}[sec]][g.group(1)] = float(g.group(2))
        g = re.match(r"\s+round1\s+vs round2\s+:\s+(\d+) /\s+(\d+)\s+p = ([\d.]+)", line)
        if g:
            d["mcn"] = (int(g.group(1)), int(g.group(2)), float(g.group(3)))
        g = re.match(r"\s+(unan_right|unan_wrong|split)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if g and sec == "4":
            d["trans"][g.group(1)] = [int(g.group(i)) for i in (2, 3, 4)]
        g = re.match(r"\s+diff round2 - round1:\s+(-?[\d.]+)\s+\[(-?[\d.]+), (-?[\d.]+)\]", line)
        if g and sec == "5":
            d["dauc"] = tuple(float(x) for x in g.groups())
    return out


def parse_rq2(path):
    adopt, contr, det, manip, locad = defaultdict(dict), defaultdict(dict), {}, {}, {}
    m = None
    for line in open(path):
        g = re.match(r"\s+([CW]-(?:flawed|strong))\s+misstates\s+(\d+)/(\d+)\s+located\s+(\S+)\s+false claim\s+(\d+)/(\d+)"
                     r"\s+convincing ([\d.]+)", line)
        if g:
            manip[g.group(1)] = g.groups()[1:]
            continue
        g = re.match(r"--- (\S+): adoption", line)
        if g:
            m = g.group(1)
            continue
        if m is None:
            continue
        g = re.match(r"\s+([CW]-(?:bare|flawed|strong))\s+(\d+)/(\d+)\s+([\d.]+)%", line)
        if g:
            adopt[m][g.group(1)] = (int(g.group(2)), int(g.group(3)), float(g.group(4)))
        g = re.match(r"\s+(.+?)\s{2,}(-?[\d.]+)\s+\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]\s*$", line)
        if g:
            contr[m][g.group(1).strip()] = tuple(float(x) for x in g.groups()[1:])
        g = re.match(r"\s+C: flawed says YES \d+/\d+\s+located\s+(\d+)/(\d+) \| strong says YES \(false alarm\)\s+(\d+)/(\d+)"
                     r" \| d' \(located vs FA\) (-?[\d.]+)", line)
        if g:
            det[m] = (int(g.group(1)), int(g.group(2)), int(g.group(3)), int(g.group(4)), float(g.group(5)))
        g = re.match(r"\s+adoption of C-flawed when the model LOCATED the misstatement: (\d+)/(\d+)", line)
        if g:
            locad[m] = (int(g.group(1)), int(g.group(2)))
    return adopt, contr, det, manip, locad


def parse_rq4(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[(r["dataset"], r["model"], r["method"])][r["metric"]] = r
    return d


def parse_rq3(path):
    """(model, part, name) -> (point, lo, hi) in percent / percentage points."""
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["model"], r["part"], r["name"])] = tuple(100 * float(r[k]) for k in ("point", "lo", "hi"))
    return d


# ---------------- formatting ----------------
def f1(x):
    return "%.1f" % x


def ci(p, lo, hi, nd=1):
    s = "%.*f [%.*f, %.*f]" % (nd, p, nd, lo, nd, hi)
    return s.replace("-", "$-$")


def sig(lo, hi):
    return "$^{*}$" if lo > 0 or hi < 0 else ""


def table_rq1(r1):
    out = []
    for path, _, name in DS:
        rows = []
        for m, lab in MODELS:
            d = r1[path][m]
            a, p, u = d["acc"], d["prec"], d["auc"]
            t = d["trans"]
            uw1 = sum(t["unan_wrong"])
            uw2 = t["unan_right"][1] + t["unan_wrong"][1] + t["split"][1]
            b, c, pv = d["mcn"]
            dl = d["dauc"]
            rows.append(" & ".join([lab, f1(a["single"]), f1(a["CoT"]), f1(a["SC-6"]), f1(a["round1"]),
                                    f1(a["round2"]), "%d/%d" % (b, c), "%.2f" % pv,
                                    f1(p["round1"]), f1(p["round2"]), "%d$\\to$%d" % (uw1, uw2),
                                    "%d" % t["split"][1],
                                    "%.3f" % u["SC-6"], "%.3f" % u["round1"], "%.3f" % u["round2"]
                                    + sig(dl[1], dl[2])]) + " \\\\")
        n = r1[path][MODELS[0][0]]["n"]
        out.append(r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}l rrrrr rr rr rr rrr@{}}
\toprule
& \multicolumn{5}{c}{Accuracy (\%%)} & \multicolumn{2}{c}{R1 vs R2} & \multicolumn{2}{c}{Unan.\ prec.} &
\multicolumn{2}{c}{Unan.\ wrong} & \multicolumn{3}{c}{AUROC} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(l){13-15}
Model & Single & CoT & SC-6 & R1 & R2 & $b/c$ & $p$ & R1 & R2 & R1$\to$R2 & from split & SC-6 & R1 & R2 \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ1, %s (%d questions). R1/R2 = debate majority vote before/after discussion. $b/c$ = questions
right in R1 but wrong in R2 / wrong in R1 but right in R2 (exact McNemar $p$). Unan.\ prec.\ = accuracy on
questions where all three agents agree. Unan.\ wrong = number of questions answered unanimously and wrongly,
and how many of the new R2 cases came from questions split in R1. AUROC of disagreement for predicting a wrong
vote; $^{*}$ = bootstrap 95\%% CI of R2 $-$ R1 excludes 0.}
\label{tab:app_rq1_%s}
\end{table}
""" % ("\n".join(rows), name, n, name.split()[0].lower()))
    return "\n".join(out)


def table_manip(manip):
    lab = {"C-flawed": "Correct answer, misstated fact", "C-strong": "Correct answer, strong",
           "W-flawed": "Wrong answer, misstated fact", "W-strong": "Wrong answer, strong"}
    rows = []
    for c in ["C-flawed", "C-strong", "W-flawed", "W-strong"]:
        k, n, loc, fk, fn, conv = manip[c]
        rows.append("%s & %s/%s (%.0f\\%%) & %s & %s/%s & %s \\\\" % (
            lab[c], k, n, 100 * int(k) / int(n), loc if loc != "-" else "--", fk, fn, conv))
    return r"""\begin{table}[H]
\centering\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
Planted argument & Misstates question & Quote locates change & Other false claim & Convincing (1--5) \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{Manipulation check by an independent judge (Gemma-3-12B) on the MedQA planted arguments. For
correct-answer plants the manipulation is clean: the judge finds a misstated detail in 89\%% of flawed
arguments and 9\%% of strong ones. Arguments for a wrong answer necessarily contain false reasoning, which the
judge often also reports as misstating the question, so the strong/flawed contrast is interpreted mainly for
correct-answer plants.}
\label{tab:app_manip}
\end{table}
""" % "\n".join(rows)


def table_rq2(adopt, contr, det, locad):
    rows = []
    for m, lab in MODELS:
        a = adopt[m]
        q = contr[m]["quality | C (strong - flawed)"]
        t = contr[m]["truth | bare   (C - W), MATCHED ctrl-wrong"]
        hit, n, fa, n2, dp = det[m]
        la = locad.get(m)
        rows.append(" & ".join([lab] + [f1(a[c][2]) for c in ["C-bare", "C-flawed", "C-strong",
                                                                "W-bare", "W-flawed", "W-strong"]]
                               + [ci(*q) + sig(q[1], q[2]), ci(*t) + sig(t[1], t[2]),
                                  "%.0f" % (100 * hit / n), "%.0f" % (100 * fa / n2), "%.2f" % dp,
                                  "%.0f" % (100 * la[0] / la[1]) if la else "--"]) + " \\\\")
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l rrr rrr cc rrr r@{}}
\toprule
& \multicolumn{3}{c}{Correct answer planted} & \multicolumn{3}{c}{Wrong answer planted} &
\multicolumn{2}{c}{Contrasts (pp)} & \multicolumn{3}{c}{Asked directly} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}
Model & Bare & Flawed & Strong & Bare & Flawed & Strong & Quality (C) & Truth (bare, matched) & Hit & FA & $d'$ &
Adopt$|$hit \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ2 on MedQA. Left: \%% of eligible honest agents (control answer $\neq$ planted answer) that switch
to the planted answer. Quality = strong $-$ flawed adoption for correct-answer plants; Truth = correct $-$
wrong adoption for bare plants, restricted to agents whose control answer was wrong (matched); bootstrap 95\%%
CIs, $^{*}$ = excludes 0. Right: the same model asked directly whether the argument misstates the question.
Hit = flawed argument flagged with a quote that locates the change; FA = strong argument flagged;
Adopt$|$hit = adoption of the flawed argument on questions where the model located the misstatement.}
\label{tab:app_rq2}
\end{table}
""" % "\n".join(rows)


def fs(x):
    """point estimate (1 dp, LaTeX minus) + significance star from (point, lo, hi)."""
    return f1(x[0]).replace("-", "$-$") + sig(x[1], x[2])


TACTICS = [("bare", "Bare"), ("authority", "Auth."), ("confidence", "Conf."), ("social", "Cons."),
           ("citation", "Cit."), ("hedge", "Hedge")]


def table_rq3a(r3):
    rows, rows2 = [], []
    for m, lab in MODELS:
        cells = []
        for t in "CW":
            for tac, _ in TACTICS:
                a = r3[(m, "tactics_adopt", "%s-%s" % (t, tac))]
                star = ""
                if tac != "bare":
                    c = r3[(m, "tactics_contrast", "%s:%s_minus_bare" % (tac, t))]
                    star = sig(c[1], c[2])
                cells.append(f1(a[0]) + star)
        rows.append(" & ".join([lab] + cells) + " \\\\")
        g = [r3[(m, "tactics_contrast", "%s:gap_matched" % tac)] for tac, _ in TACTICS]
        gc = [r3[(m, "tactics_contrast", "%s:gap_change" % tac)] for tac, _ in TACTICS[1:]]
        rows2.append(" & ".join([lab] + [fs(x) for x in g] + [fs(x) for x in gc] + [""]) + " \\\\")
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l rrrrrr rrrrrr@{}}
\toprule
& \multicolumn{6}{c}{Correct answer planted} & \multicolumn{6}{c}{Wrong answer planted} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
Model & Bare & Auth. & Conf. & Cons. & Cit. & Hedge & Bare & Auth. & Conf. & Cons. & Cit. & Hedge \\
\midrule
%s
\midrule
& \multicolumn{6}{c}{Truth gap, matched (pp)} & \multicolumn{5}{c}{Change in truth gap vs.\ bare (pp)} & \\
\cmidrule(lr){2-7}\cmidrule(lr){8-12}
Model & Bare & Auth. & Conf. & Cons. & Cit. & Hedge & Auth. & Conf. & Cons. & Cit. & Hedge & \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ3a on MedQA. Top: \%% of eligible honest agents that switch to the planted answer when the bare
plant carries one persuasion sentence (Auth.\ = authority, Conf.\ = confidence, Cons.\ = consensus, Cit.\ =
citation); $^{*}$ = bootstrap 95\%% CI of the difference from the bare plant of the same truth excludes 0.
Bottom: truth gap (correct $-$ wrong adoption) among agents whose control answer was wrong, and the change of
the full-sample gap relative to the bare plant; $^{*}$ = CI excludes 0.}
\label{tab:app_rq3a}
\end{table}
""" % ("\n".join(rows), "\n".join(rows2))


def table_rq3b(r3):
    cells_order = ["C-bare", "C-flawed", "C-strong", "C-authority", "W-bare", "W-flawed", "W-strong", "W-authority"]
    rows = []
    for m, lab in MODELS:
        ch = [r3[(m, "verify_change", c)] for c in cells_order]
        qn = r3[(m, "verify_contrast", "nov:quality C (strong - flawed)")]
        qv = r3[(m, "verify_contrast", "ver:quality C (strong - flawed)")]
        tn = r3[(m, "verify_contrast", "nov:truth bare (C - W)")]
        tv = r3[(m, "verify_contrast", "ver:truth bare (C - W)")]
        co = r3[(m, "verify_cost", "acc_change")]
        rows.append(" & ".join([lab] + [fs(x) for x in ch]
                               + ["%s $\\to$ %s" % (fs(qn), fs(qv)), "%s $\\to$ %s" % (fs(tn), fs(tv)),
                                  ci(*co)]) + " \\\\")
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l rrrr rrrr ccc@{}}
\toprule
& \multicolumn{4}{c}{Correct planted: change (pp)} & \multicolumn{4}{c}{Wrong planted: change (pp)} &
Quality (C) & Truth (bare) & Normal debate \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Model & Bare & Flawed & Strong & Auth. & Bare & Flawed & Strong & Auth. & no $\to$ verify & no $\to$ verify &
acc.\ change \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ3b on MedQA: effect of the verify-first instruction. Left: change in adoption of the planted answer
(verify $-$ no instruction, same agents, seeds and plants). Quality = strong $-$ flawed adoption for
correct-answer plants; Truth = correct $-$ wrong adoption for bare plants, each without and with the
instruction. Normal debate = change in honest-agent accuracy without any plant, with 95\%% CI.
$^{*}$ = bootstrap 95\%% CI excludes 0.}
\label{tab:app_rq3b}
\end{table}
""" % "\n".join(rows)


def table_rq4(r4):
    blocks = []
    for _, key, name in DS:
        rows = []
        for m, lab in MODELS:
            g = lambda meth, met: r4[(key, m, meth)][met]
            au = [float(g(x, "AURC")["point"]) for x in ("debate", "hyb-r1", "SC-6")]
            dh = g("hyb-r1", "AURC")
            dd = (float(dh["diff_vs_debate"]), float(dh["diff_lo"]), float(dh["diff_hi"]))
            cw = [100 * float(g(x, "caught@30")["point"]) for x in ("debate", "hyb-r1", "SC-6")]
            he = r4[(key, m, "debate")]["hidden_errors"]["point"]
            hf = r4[(key, m, "hyb-r1")]["hidden_flagged"]["point"]
            rows.append(" & ".join([lab] + ["%.3f" % x for x in au] + [ci(*dd, nd=3) + sig(dd[1], dd[2])]
                                   + ["%.0f" % x for x in cw] + ["%s/%s" % (hf, he)]) + " \\\\")
        blocks.append("\\midrule\n\\multicolumn{9}{@{}l}{\\textit{%s}} \\\\\n%s" % (name, "\n".join(rows)))
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l rrr c rrr r@{}}
\toprule
& \multicolumn{4}{c}{AURC ($\downarrow$)} & \multicolumn{3}{c}{Errors caught at 30\%% deferral (\%%)} & Hidden \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Model & Debate & Hybrid & SC-6 & Hybrid $-$ debate & Debate & Hybrid & SC-6 & errors \\
%s
\bottomrule
\end{tabular}
\caption{RQ4, all datasets. Debate = round-2 vote with round-2 agreement as the trust score; Hybrid = the same
answer with round-1 agreement as the trust score; SC-6 = self-consistency at equal compute. AURC = area under
the risk--coverage curve; bootstrap 95\%% CI of the difference, $^{*}$ = excludes 0. Errors caught = share of
all wrong answers among the 30\%% least-trusted questions. Hidden errors = round-2 unanimous wrong answers that
were not unanimous in round 1 / all round-2 unanimous wrong answers.}
\label{tab:app_rq4}
\end{table}
""" % "\n".join(blocks)


def table_rq1_summary(r1, path_txt):
    """Main-text Table: one row per dataset, aggregated over the six models."""
    sc = {}
    ds = m = None
    for line in open(path_txt):
        g = re.match(r"##### (\S+)", line)
        if g:
            ds = g.group(1)
            continue
        g = re.match(r"===== (\S+) \(", line)
        if g:
            m = g.group(1)
            continue
        g = re.match(r"\s+SC-6\s+vs round2\s+:\s+(\d+) /\s+(\d+)\s+p = ([\d.]+)", line)
        if g:
            sc[(ds, m)] = float(g.group(3))
    rows, tot = [], [0] * 9
    for path, _, name in DS:
        acc_up = acc_dn = scsig = uw1 = uw2 = s2w = s2r = r2w = adrop = asig = scbeat = 0
        for mm, _ in MODELS:
            d = r1[path][mm]
            t, a, u, dl = d["trans"], d["acc"], d["auc"], d["dauc"]
            if d["mcn"][2] < 0.05:
                acc_up += a["round2"] > a["round1"]
                acc_dn += a["round2"] < a["round1"]
            scsig += sc[(path, mm)] < 0.05
            uw1 += sum(t["unan_wrong"])
            uw2 += t["unan_right"][1] + t["unan_wrong"][1] + t["split"][1]
            s2w += t["split"][1]
            s2r += t["split"][0]
            r2w += t["unan_right"][1]
            adrop += u["round2"] < u["round1"]
            asig += dl[2] < 0
            scbeat += u["SC-6"] > u["round2"]
        vals = [acc_up, acc_dn, scsig, uw1, uw2, s2r, s2w, r2w]
        tot = [x + y for x, y in zip(tot, vals + [0])]
        name = name.replace("MMLU Professional Law", "MMLU Prof.\\ Law")
        rows.append("%s & %d\\,/\\,%d & %d & %d$\\to$%d & %d & %d & %d & %d (%d) & %d \\\\" % (
            name, acc_up, acc_dn, scsig, uw1, uw2, s2r, s2w, r2w, adrop, asig, scbeat))
    return r"""\begin{table}[t]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\resizebox{\linewidth}{!}{%%
\begin{tabular}{@{}l c c c ccc c c@{}}
\toprule
& \multicolumn{2}{c}{Sig.\ acc.\ changes} & Unan.\ wrong & \multicolumn{3}{c}{Divided R1 $\to$ unanimous R2} &
\multicolumn{2}{c}{AUROC} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}
Dataset & R2 vs R1 (up\,/\,down) & SC-6 vs R2 & R1$\to$R2 & right & wrong & right$\to$wrong &
R2 $<$ R1 (sig.) & SC-6 $>$ R2 \\
\midrule
%s
\bottomrule
\end{tabular}}
\caption{RQ1 summary over six models per dataset. Accuracy columns count models with a significant paired
difference (exact McNemar, $p<0.05$). Unanimous wrong: questions on which all three agents agree on a wrong
answer, summed over models, before and after discussion. Middle columns: questions divided in round~1 that
become unanimously right or unanimously wrong in round~2, and questions that move from unanimously right to
unanimously wrong. Right: number of models (out of six) for which the AUROC of disagreement is lower after
discussion than before (in parentheses: bootstrap 95\%% CI of the difference below 0), and for which SC-6 has a
higher AUROC than round~2. Per-model values are in \Cref{tab:app_rq1_medqa,tab:app_rq1_medmcqa,tab:app_rq1_mmlu,tab:app_rq1_gpqa}.}
\label{tab:rq1}
\end{table}
""" % "\n".join(rows)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    r1 = parse_rq1(FIG / "rq1_stats_all_2026-11-17.txt")
    adopt, contr, det, manip, locad = parse_rq2(FIG / "rq2_vigilance_medqa_2026-11-17.txt")
    r4 = parse_rq4(Path("results/rq4/rq4_summary.csv"))
    r3 = parse_rq3(FIG / "rq3_summary.csv")
    tex = ["% GENERATED by Experiment/analysis/make_appendix_tables.py -- do not edit by hand; re-run the script.",
           "\\subsection{RQ1: all datasets}", table_rq1(r1),
           "\\subsection{RQ2: manipulation check and per-model results}", table_manip(manip),
           table_rq2(adopt, contr, det, locad),
           "\\subsection{RQ3: persuasion tactics and verify-first instruction}", table_rq3a(r3), table_rq3b(r3),
           "\\subsection{RQ4: all datasets}", table_rq4(r4), ""]
    out.write_text("\n".join(tex))
    (out.parent / "tab_rq1_summary.tex").write_text(
        "% GENERATED by make_appendix_tables.py\n" + table_rq1_summary(r1, FIG / "rq1_stats_all_2026-11-17.txt"))
    print("wrote", out)

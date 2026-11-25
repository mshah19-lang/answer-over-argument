"""Appendix tables added in the review response (Step 4, 2026-11-25).

Usage (from Experiment/):
    python3 analysis/make_appendix_tables_review.py "../full_paper/sections/app_tables_review.tex"
Reads saved outputs in results/figdata/:
    step1_review_2026-11-19.txt  parts C (own belief), D (granularity), E (majority)
    step2_2026-11-23.txt         parts 1-5 (second judge, does the flaw matter)
    step3_2026-11-25.txt         decisive flaws, 6 models
    holm_fdr_2026-11-25.txt      Holm marks
Tables: tab:app_rq1_granularity, tab:app_rq1_majority, tab:app_belief, tab:app_step2, tab:app_step3.
"""
import re
import sys
from pathlib import Path

FIG = Path("results/figdata")
MODELS = [("qwen4", "Qwen3-4B"), ("qwen8", "Qwen3-8B"), ("qwen14", "Qwen3-14B"),
          ("qwen32", "Qwen3-32B"), ("gemma12", "Gemma-3-12B"), ("llama8", "Llama-3.1-8B")]
LAB = dict(MODELS)
DSN = {"medqa": "MedQA", "medmcqa": "MedMCQA", "law": "Law", "gpqa": "GPQA"}
NUM = r"(-?\d+\.\d+) \[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"


def m_(s):
    return s.replace("-", "$-$")


def sig(lo, hi):
    return "$^{*}$" if lo > 0 or hi < 0 else ""


def cis(t, nd=1, scale=1.0):
    p, lo, hi = (scale * float(x) for x in t)
    return m_("%.*f [%.*f, %.*f]" % (nd, p, nd, lo, nd, hi)) + sig(lo, hi)


def section(lines, start):
    s = next(i for i, l in enumerate(lines) if l.startswith(start))
    e = next((i for i in range(s + 1, len(lines)) if lines[i].startswith("=====")), len(lines))
    return lines[s:e]


def holm_marks(path):
    """family name -> {label: True if Holm-significant}"""
    out, fam = {}, None
    for l in open(path):
        mm = re.match(r"--- (.+?)\s+\(m = \d+\)", l)
        if mm:
            fam = mm.group(1)
            out[fam] = {}
            continue
        mm = re.match(r"\s+(\S+(?: \S+)?)\s+est\s+\S+\s+p \S+\s+uncorr (.)\s+Holm (.)", l)
        if mm and fam:
            out[fam][mm.group(1)] = mm.group(3) == "*"
    return out


def dag(flag):
    return "$^{\\dagger}$" if flag else ""


# ------------------------------------------------------------------ RQ1 granularity (part D)
def table_granularity(s1, hm):
    rows = []
    for l in section(s1, "D)"):
        f = l.split()
        if len(f) > 3 and f[0] in DSN and re.match(r"0\.\d+", f[2]):
            d = re.findall(NUM, l)
            key = f[0] + " " + f[1]
            cells = [DSN[f[0]], LAB[f[1]], f[2], f[3],
                     cis(d[0], 3) + dag(hm["RQ1 AUROC R1 - SC3"].get(key)),
                     f[7] if False else l.split("|")[1].split()[0], l.split("|")[1].split()[1],
                     l.split("|")[1].split()[2],
                     cis(d[1], 3) + dag(hm["RQ1 AUROC pool6 - R2"].get(key)),
                     cis(d[2], 3) + dag(hm["RQ1 AUROC SC6 - pool6"].get(key))]
            rows.append(" & ".join(cells) + " \\\\")
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}ll rrr rrr rr@{}}
\toprule
& & \multicolumn{3}{c}{Three votes each} & \multicolumn{5}{c}{Answer of round 2, trust from six or three votes} \\
\cmidrule(lr){3-5}\cmidrule(l){6-10}
Data & Model & SC-3 & R1 & R1 $-$ SC-3 & R2 & Pool-6 & SC-6 & Pool-6 $-$ R2 & SC-6 $-$ Pool-6 \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ1 granularity controls: AUROC of disagreement for predicting a wrong vote. Left: round~1 against
three independent samples (SC-3), both based on three votes. Right: Pool-6 keeps the round-2 answer but scores
agreement over all six debate replies (both rounds), so it has as many levels as SC-6. Bootstrap 95\%%
intervals; $^{*}$ = interval excludes 0; $^{\dagger}$ = also significant after Holm correction over the 24
pairs.}
\label{tab:app_rq1_granularity}
\end{table}
""" % "\n".join(rows)


# ------------------------------------------------------------------ RQ1 majority (part E)
def table_majority(s1):
    rows, tot = [], None
    for l in section(s1, "E)"):
        f = l.split()
        if len(f) == 9 and f[0] in DSN:
            div, unan, ur, wm, fol, mr, mrw = f[2:]
            rows.append(" & ".join([DSN[f[0]], LAB[f[1]], div, unan, ur, wm,
                                    "%s (%.0f)" % (fol, 100 * int(fol) / int(wm)),
                                    "%s (%.0f)" % (mr, 100 * int(mr) / int(wm)), mrw]) + " \\\\")
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}ll rrr rrr r@{}}
\toprule
& & \multicolumn{3}{c}{Divided in round 1} & \multicolumn{3}{c}{Of these, with a 2--1 majority} & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Data & Model & All & $\to$ unanimous & $\to$ unan.\ right & $n$ & Follow majority (\%%) & Majority right (\%%) &
Right dissenter wins \\
\midrule
%s
\midrule
\multicolumn{2}{@{}l}{Total} & 2419 & 1089 & 505 & 1003 & 864 (86) & 460 (46) & 56/262 \\
\bottomrule
\end{tabular}
\caption{RQ1: how divided questions are settled. Questions on which the three agents disagree in round~1 and
agree in round~2. Follow majority = the unanimous round-2 answer is the answer two agents gave in round~1;
Majority right = that round-1 majority was correct. Right dissenter wins = among questions where the lone
round-1 dissenter held the correct answer, how often all three end on it.}
\label{tab:app_rq1_majority}
\end{table}
""" % "\n".join(rows)


# ------------------------------------------------------------------ own belief (part C)
def table_belief(s1):
    sec = section(s1, "C)")
    bins, model = {}, None
    for l in sec:
        mm = re.match(r"\s+(\w+)\s+bare plants:", l)
        if mm:
            model = mm.group(1)
            bins[model] = {}
            continue
        mm = re.match(r"\s+([CW]) prior 0\s+(\d+)/(\d+).*?1-2/6\s+(\d+)/(\d+).*?>=3/6\s+(\d+)/(\d+)", l)
        if mm and model:
            g = [int(x) for x in mm.groups()[1:]]
            bins[model][mm.group(1)] = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5])]
    gaps = {}
    for l in sec:
        mm = re.match(r"\s+(\w+)\s+raw\s+" + NUM + r".*stratified by prior\s+" + NUM, l)
        if mm:
            g = mm.groups()
            gaps[g[0]] = (g[1:4], g[4:7])
    rows = []

    def pc(k, n):
        return "%.0f" % (100 * k / n) if n else "--"
    for m, lab in MODELS:
        b = bins[m]
        rows.append(" & ".join([lab] + [pc(*b["C"][i]) for i in range(3)] + [pc(*b["W"][i]) for i in range(3)]
                               + [cis(gaps[m][0]), cis(gaps[m][1])]) + " \\\\")
    rows.append("\\midrule\nPooled & & & & & & & %s & %s \\\\" % (cis(gaps["POOLED"][0]), cis(gaps["POOLED"][1])))
    lg = {}
    for l in sec:
        mm = re.match(r"\s+(truth=C|prior\(0-1\)|strong|flawed)\s+log-odds\s+" + NUM + r".*odds ratio ([\d.]+)", l)
        if mm:
            lg[mm.group(1)] = mm.groups()[1:]
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l rrr rrr rr@{}}
\toprule
& \multicolumn{3}{c}{Correct target: prior} & \multicolumn{3}{c}{Wrong target: prior} &
\multicolumn{2}{c}{Truth gap C $-$ W (pp)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}
Model & 0/6 & 1--2/6 & $\geq$3/6 & 0/6 & 1--2/6 & $\geq$3/6 & Raw & Prior held fixed \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ2 own-belief test, bare plants on MedQA. Prior = how many of the model's own six SC-6 samples chose
the planted target. Left: \%% of eligible honest agents that adopt the target, by prior. Right: the truth gap
(correct minus wrong target), raw and stratified by prior (Mantel--Haenszel weights); bootstrap 95\%%
intervals, $^{*}$ = excludes 0. Pooled logistic model over all six plant conditions with model fixed effects
(log-odds, 95\%% CI): truth %s, prior %s, strong argument %s, flawed argument %s, relative to a bare
answer of the wrong target.}
\label{tab:app_belief}
\end{table}
""" % ("\n".join(rows), cis(lg["truth=C"][:3], 2), cis(lg["prior(0-1)"][:3], 2), cis(lg["strong"][:3], 2),
       cis(lg["flawed"][:3], 2))


# ------------------------------------------------------------------ Step 2
def table_step2(s2):
    txt = "\n".join(s2)
    judge = {}
    for l in s2:
        mm = re.match(r"\s+(gemma12|qwen32)\s+([CW]-\w+)\s+misstates\s+(\d+)/(\d+)", l)
        if mm:
            judge[(mm.group(1), mm.group(2))] = (int(mm.group(3)), int(mm.group(4)))
    blocks, cur = {}, None
    names = {"judge: changes or weakens": "Judge: changes or weakens answer",
             "judge: importance >= 4": "Judge: importance $\\geq$ 4",
             "counterfactual drop >= 1/3": "Counterfactual drop $\\geq$ 1/3"}
    for l in s2:
        mm = re.match(r"\s+--- (.+?): matters (\d+) questions, does not (\d+)", l)
        if mm:
            cur = mm.group(1)
            blocks[cur] = {"n": (mm.group(2), mm.group(3))}
            continue
        mm = re.match(r"\s+(\w+)\s+matters\s+" + NUM + r".*?not\s+" + NUM, l)
        if mm and cur:
            g = mm.groups()
            blocks[cur][g[0]] = (g[1:4], g[4:7])
            if g[0] == "POOLED":
                d = re.findall(NUM, l)
                blocks[cur]["diff"] = d[2] if len(d) > 2 else None
    rows = []
    for key, name in names.items():
        b = blocks[key]
        rows.append("\\multicolumn{3}{@{}l}{\\emph{%s} (matters: %s questions, does not: %s)} \\\\" % (
            name, b["n"][0], b["n"][1]))
        for m, lab in MODELS:
            rows.append("\\quad %s & %s & %s \\\\" % (lab, cis(b[m][0]), cis(b[m][1])))
        rows.append("\\quad Pooled & %s & %s \\\\" % (cis(b["POOLED"][0]), cis(b["POOLED"][1])))
    cf = re.search(r"gold share original ([\d.]+)% -> edited ([\d.]+)%", txt)
    nd = re.search(r"drops by >= 1/3: (\d+)/(\d+)", txt)
    kc = re.search(r"changes_answer .*kappa ([\d.]+)", txt).group(1)
    kw = re.search(r"weakens_reasoning .*kappa ([\d.]+)", txt).group(1)
    j = judge
    return r"""\begin{table}[H]
\centering\scriptsize
\begin{tabular}{@{}lrr@{}}
\toprule
& \multicolumn{2}{c}{Quality effect, correct target: strong $-$ flawed (pp)} \\
\cmidrule(l){2-3}
Definition of ``the flaw matters'' & Flaw matters & Flaw does not matter \\
\midrule
%s
\bottomrule
\end{tabular}
\caption{RQ2: does the fixed-rule flaw matter? Second judge (Qwen3-32B): flags a misstated detail in
%d/%d flawed and %d/%d strong correct-target arguments (Gemma-3-12B: %d/%d and %d/%d). Counterfactual: Qwen3-32B
answering the \emph{edited} question chooses the correct option in %s\%% of samples, against %s\%% for the
original; %s of %s edits lower this share by a third or more. The judge's relevance labels agree poorly with
labels we wrote for 50 edits with Claude Opus 5.5 ($\kappa$ = %s and %s), so we rely on the counterfactual definition. The
quality effect is shown separately for questions where the flaw matters and where it does not; bootstrap
95\%% intervals, $^{*}$ = excludes 0.}
\label{tab:app_step2}
\end{table}
""" % ("\n".join(rows), *j[("qwen32", "C-flawed")], *j[("qwen32", "C-strong")], *j[("gemma12", "C-flawed")],
       *j[("gemma12", "C-strong")], cf.group(2), cf.group(1), nd.group(1), nd.group(2), kc, kw)


# ------------------------------------------------------------------ Step 3
def table_step3(s3, hm):
    adopt, det, model, sec = {}, {}, None, None
    for l in s3:
        if re.match(r"\d\)", l):
            sec = l[0]
        mm = re.match(r"\s+--- (\w+)$", l)
        if mm:
            model = mm.group(1)
            continue
        mm = re.match(r"\s+(all usable|verified|num|finding)\s+q=", l)
        if sec == "3" and mm and model in LAB:
            adopt.setdefault(model, {})[mm.group(1)] = re.findall(NUM, l)
        if sec == "3" and model in LAB and "old RQ2 minor flaw" in l:
            adopt[model]["old"] = re.findall(NUM, l)[0]
        mm = re.match(r"\s+(\w+)\s+located\s+(\d+)/(\d+).*false alarm\s+(\d+)/(\d+).*d' (-?[\d.]+).*adopt strong (\d+)/(\d+), decisive (\d+)/(\d+)", l)
        if sec == "4" and mm:
            det[mm.group(1)] = mm.groups()[1:]
    rows = []
    for m, lab in MODELS:
        a = adopt[m]
        loc, nf, fa, ns, dp, as_, an, ad, dn = det[m]
        rows.append(" & ".join([
            lab, "%.1f" % float(a["all usable"][0][0]), "%.1f" % float(a["all usable"][1][0]),
            cis(a["all usable"][2]) + dag(hm["Step3 strong - decisive, all usable"].get(m)),
            cis(a["verified"][2]), cis(a["num"][2]), cis(a["finding"][2]), cis(a["old"]),
            "%.0f" % (100 * int(loc) / int(nf)), "%.0f" % (100 * int(fa) / int(ns)), m_(dp),
            "%s/%s" % (as_, an), "%s/%s" % (ad, dn)]) + " \\\\")
    pooled = {}
    for l in s3:
        mm = re.match(r"\s+(all usable|verified|num|finding|OLD minor flaw, same q)\s+strong - flaw\s+", l)
        if mm:
            pooled[mm.group(1)] = re.findall(NUM, l)[0]
    rows.append("\\midrule\nPooled & & & %s & %s & %s & %s & %s & & & & & \\\\" % (
        cis(pooled["all usable"]), cis(pooled["verified"]), cis(pooled["num"]), cis(pooled["finding"]),
        cis(pooled["OLD minor flaw, same q"])))
    return r"""\begin{table}[H]
\centering\scriptsize\setlength{\tabcolsep}{2.4pt}
\resizebox{\linewidth}{!}{%%
\begin{tabular}{@{}l rr rrrrr rrr rr@{}}
\toprule
& \multicolumn{2}{c}{Adoption (\%%)} & \multicolumn{5}{c}{Strong $-$ decisive-flawed (pp)} &
\multicolumn{3}{c}{Asked directly} & \multicolumn{2}{c}{Adopt on located q} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-8}\cmidrule(lr){9-11}\cmidrule(l){12-13}
Model & Strong & Decisive & All (79 q) & Verified (31 q) & Number (39 q) & Finding (40 q) & Old flaw, same q &
Hit & FA & $d'$ & Strong & Decisive \\
\midrule
%s
\bottomrule
\end{tabular}}
\caption{RQ2 with decisive flaws (MedQA, correct targets, 79 usable questions). Adoption = \%% of eligible honest
agents that switch to the planted correct answer. Differences are given for all usable questions, for the 31
whose edit is verified, by edit type, and for the fixed-rule flaw on the same questions; bootstrap 95\%%
intervals, $^{*}$ = excludes 0, $^{\dagger}$ = also significant after Holm correction over the six models.
Asked directly: Hit = decisive-flawed argument flagged with a quote that locates the change; FA = strong
argument flagged (\%%). Adopt on located q = adoptions among eligible agents on questions where the model
located the flaw.}
\label{tab:app_step3}
\end{table}
""" % "\n".join(rows)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    s1 = open(FIG / "step1_review_2026-11-19.txt").read().splitlines()
    s2 = open(FIG / "step2_2026-11-23.txt").read().splitlines()
    s3 = open(FIG / "step3_2026-11-25.txt").read().splitlines()
    hm = holm_marks(FIG / "holm_fdr_2026-11-25.txt")
    tex = ["% GENERATED by Experiment/analysis/make_appendix_tables_review.py -- do not edit by hand; re-run the script.",
           "\\subsection{RQ1: vote-matched controls and how divisions are settled}",
           table_granularity(s1, hm), table_majority(s1),
           "\\subsection{RQ2: own belief, relevance of the flaw, and decisive flaws}",
           table_belief(s1), table_step2(s2), table_step3(s3, hm), ""]
    out.write_text("\n".join(tex))
    print("wrote", out)

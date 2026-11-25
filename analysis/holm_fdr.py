"""Multiple-comparison check (Holm and Benjamini-Hochberg) for the families of tests the paper counts.

Re-runs the existing analyses unchanged (same seeds, same bootstrap) and only adds a two-sided bootstrap
p-value to every bootstrap interval: p = 2 * min(#reps <= 0, #reps >= 0) + 1 over (B + 1), capped at 1.
With B = 2000 the smallest attainable p is about 0.001.

Families (each corrected separately):
  RQ1  acc R2 vs R1 (exact McNemar, 24), acc R2 vs SC-6 (McNemar, 24), AUROC R2 - R1 (24),
       R1 - SC3, pool6 - R2, SC6 - pool6 (24 each)
  RQ2  quality C strong - flawed per model (6), adopt|hit strong - flawed per model (6)
  Step3 strong - decisive per model, for all usable / verified / num / finding (6 each)

Usage: python3 analysis/holm_fdr.py results_cluster > results/figdata/holm_fdr_<date>.txt
"""
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stats  # noqa: E402
import step1_review_analyses as s1  # noqa: E402
import step3_analysis as s3  # noqa: E402

ALPHA = 0.05
MODELS = ["qwen4", "qwen8", "qwen14", "qwen32", "gemma12", "llama8"]
DATASETS = [("medqa", ""), ("medmcqa", "medmcqa"), ("law", "law"), ("gpqa", "gpqa")]


def boot_p(a):
    a = np.asarray([x for x in a if x is not None], float)
    a = a[~np.isnan(a)]
    if not len(a):
        return np.nan
    k = min((a <= 0).sum(), (a >= 0).sum())
    return min(1.0, 2 * (k + 1) / (len(a) + 1))


def holm(ps):
    ps = np.asarray(ps, float)
    m, order = len(ps), np.argsort(ps)
    rej, ok = np.zeros(m, bool), True
    for r, i in enumerate(order):
        ok = ok and ps[i] <= ALPHA / (m - r)
        rej[i] = ok
    return rej


def bh(ps):
    ps = np.asarray(ps, float)
    m, order = len(ps), np.argsort(ps)
    k = 0
    for r, i in enumerate(order, 1):
        if ps[i] <= ALPHA * r / m:
            k = r
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def report(name, labels, ests, ps, unc):
    """unc = uncorrected significance as the paper counts it (CI excludes 0 / McNemar p < .05)."""
    h, b = holm(ps), bh(ps)
    print("\n--- %s  (m = %d)" % (name, len(ps)))
    for lab, e, p, u, hh, bb in zip(labels, ests, ps, unc, h, b):
        print("   %-18s est %8.3f  p %.4f  uncorr %s  Holm %s  BH %s" % (
            lab, e, p, "*" if u else " ", "*" if hh else " ", "*" if bb else " "))
    pos = sum(e > 0 for e in ests)
    print("   SUMMARY %-40s uncorrected %2d (+%d/-%d) | Holm %2d (+%d/-%d) | BH %2d (+%d/-%d) | est>0 %d" % (
        name, sum(unc), sum(u and e > 0 for u, e in zip(unc, ests)), sum(u and e < 0 for u, e in zip(unc, ests)),
        h.sum(), sum(x and e > 0 for x, e in zip(h, ests)), sum(x and e < 0 for x, e in zip(h, ests)),
        b.sum(), sum(x and e > 0 for x, e in zip(b, ests)), sum(x and e < 0 for x, e in zip(b, ests)), pos))


# ------------------------------------------------------------------ RQ1 from stats.py
def rq1(root):
    fam = {k: ([], [], [], []) for k in ("acc R2 vs R1 (McNemar)", "acc R2 vs SC-6 (McNemar)", "AUROC R2 - R1")}
    orig_boot = stats.boot
    calls = []

    def boot(ids, fn, rng):
        point = fn(ids)
        reps = [fn([rng.choice(ids) for _ in ids]) for _ in range(stats.B)]
        lo, hi = stats.ci(reps)
        calls.append((point, lo, hi, reps))
        return point, lo, hi

    stats.boot = boot
    for ds, sub in DATASETS:
        for m in MODELS:
            d = root / sub / m if sub else root / m
            calls.clear()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats.analyze(d, [])
            out = buf.getvalue()
            lab = "%s %s" % (ds, m)
            for key, a, b in [("acc R2 vs R1 (McNemar)", "round1", "round2"), ("acc R2 vs SC-6 (McNemar)", "SC-6", "round2")]:
                mm = re.search(r"%s\s+vs %s\s*:\s*(\d+) /\s*(\d+)\s+p = ([0-9.]+)" % (re.escape(a), b), out)
                x, y, p = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
                est = (y - x) if a == "round1" else (x - y) * -1  # >0 = round 2 better
                L = fam[key]
                L[0].append(lab), L[1].append(est), L[2].append(p), L[3].append(p < ALPHA)
            point, lo, hi, reps = calls[-1]  # last boot call of analyze() = AUROC r2 - r1
            L = fam["AUROC R2 - R1"]
            L[0].append(lab), L[1].append(point), L[2].append(boot_p(reps)), L[3].append(lo > 0 or hi < 0)
    stats.boot = orig_boot
    for k, (labs, ests, ps, unc) in fam.items():
        report("RQ1 " + k, labs, ests, ps, unc)


# ------------------------------------------------------------------ patching ci() for step1 / step3
def capture(mod, fn, *args):
    """Run fn with mod.ci wrapped so that every interval also records its bootstrap p; returns
    (printed text, list of (lo, hi, p) in call order)."""
    rec, orig = [], mod.ci

    def ci(a):
        lo, hi = orig(a)
        rec.append((lo, hi, boot_p(a)))
        return lo, hi

    mod.ci = ci
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*args)
    finally:
        mod.ci = orig
    return buf.getvalue(), rec


def run_step1(root):
    # step1 main() runs parts A-E; capture all intervals in call order and match them to printed lines
    sys.argv = ["x", str(root)]
    out, rec = capture(s1, s1.main)
    lines = out.splitlines()

    # part D rows: each row prints 3 diffs, then the counting loop calls ci 3 more times -> take the
    # p of the counting calls (same arrays) in order R1-SC3, pool6-R2, SC6-pool6
    fams = {("R1", "SC3"): ([], [], [], []), ("pool6", "R2"): ([], [], [], []), ("SC6", "pool6"): ([], [], [], [])}
    dstart = next(i for i, l in enumerate(lines) if l.startswith("D)"))
    rows = [l for l in lines[dstart:] if re.match(r"\s+(medqa|medmcqa|law|gpqa)\s+\w+\s+0\.", l)]
    num = r"(-?\d+\.\d+) \[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
    for l in rows:
        f = l.split()
        lab = f[0] + " " + f[1]
        diffs = re.findall(num, l)
        for (a, b), (e, lo, hi) in zip(fams, diffs):
            L = fams[(a, b)]
            L[0].append(lab), L[1].append(float(e)), L[3].append(float(lo) > 0 or float(hi) < 0)
    # p values: find recorded intervals matching each printed (lo, hi) at 3 decimals
    # p values: match each printed interval to the recorded one, in printing order
    idx = 0
    allp = []
    for l in rows:
        diffs = re.findall(num, l)
        for e, lo, hi in diffs:
            while idx < len(rec):
                rlo, rhi, p = rec[idx]
                idx += 1
                if abs(rlo - float(lo)) < 6e-4 and abs(rhi - float(hi)) < 6e-4:
                    allp.append(p)
                    break
    for j, L in enumerate(fams.values()):
        L[2].extend(allp[j::3])
    for (a, b), (labs, ests, ps, unc) in fams.items():
        report("RQ1 AUROC %s - %s" % (a, b), labs, ests, ps, unc)

    # part A (quality C per model) and part B (adopt|hit strong - flawed per model)
    def pick(section_start, pattern, col):
        s = next(i for i, l in enumerate(lines) if l.startswith(section_start))
        e = next((i for i in range(s + 1, len(lines)) if lines[i].startswith("=")), len(lines))
        labs, ests, uncs, his = [], [], [], []
        for l in lines[s:e]:
            mm = re.match(pattern, l)
            if mm:
                d = re.findall(num, l)[col]
                labs.append(mm.group(1)), ests.append(float(d[0]))
                uncs.append(float(d[1]) > 0 or float(d[2]) < 0), his.append((float(d[1]), float(d[2])))
        return labs, ests, uncs, his

    for name, sec, pat, col in [("RQ2 quality C strong - flawed", "A)", r"\s+(qwen4|qwen8|qwen14|qwen32|gemma12|llama8)\s", 0),
                                ("RQ2 adopt|hit strong - flawed", "B)", r"\s+(qwen4|qwen8|qwen14|qwen32|gemma12|llama8)\s", 2)]:
        labs, ests, uncs, his = pick(sec, pat, col)
        ps = []
        for lo, hi in his:
            cand = [p for rlo, rhi, p in rec if abs(100 * rlo - lo) < 0.06 and abs(100 * rhi - hi) < 0.06]
            ps.append(cand[0] if cand else np.nan)
        report(name, labs, ests, ps, uncs)


def run_step3(root):
    sys.argv = ["x", str(root)]
    out, rec = capture(s3, s3.main)
    lines = out.splitlines()
    num = r"(-?\d+\.\d+) \[\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\]"
    s = next(i for i, l in enumerate(lines) if l.startswith("3)"))
    fams = {}
    model = None
    for l in lines[s:]:
        mm = re.match(r"\s+--- (\w+)$", l)
        if mm:
            model = mm.group(1)
            continue
        mm = re.match(r"\s+(all usable|verified|num|finding)\s+q=", l)
        if mm and model in MODELS:
            e, lo, hi = re.findall(num, l)[2]
            cand = [p for rlo, rhi, p in rec if abs(100 * rlo - float(lo)) < 0.06 and abs(100 * rhi - float(hi)) < 0.06]
            L = fams.setdefault(mm.group(1), ([], [], [], []))
            L[0].append(model), L[1].append(float(e)), L[2].append(cand[0] if cand else np.nan)
            L[3].append(float(lo) > 0 or float(hi) < 0)
        if l.startswith("4)"):
            break
    for k, (labs, ests, ps, unc) in fams.items():
        report("Step3 strong - decisive, " + k, labs, ests, ps, unc)
    # pooled lines (single tests, for reference)
    for l in lines[s:]:
        if "strong - flaw" in l:
            e, lo, hi = re.findall(num, l)[0]
            cand = [p for rlo, rhi, p in rec if abs(100 * rlo - float(lo)) < 0.06 and abs(100 * rhi - float(hi)) < 0.06]
            print("   pooled: %s  p %.4f" % (l.strip(), cand[0] if cand else np.nan))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    print("Multiple-comparison check, alpha %.2f; bootstrap p = 2*min(#<=0, #>=0)+1 / (B+1)" % ALPHA)
    print("\n================ RQ1 (stats.py)")
    rq1(root)
    print("\n================ RQ1 granularity + RQ2 (step1_review_analyses.py)")
    run_step1(root)
    print("\n================ Step 3 (step3_analysis.py)")
    run_step3(root)

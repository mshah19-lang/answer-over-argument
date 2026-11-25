"""RQ4: keep debate's answer, but take the "should we trust it?" signal from round 1.

Read-only, Python stdlib only (login node has Python 3.6). Uses existing RQ1 files:
single.jsonl (gold), sc.jsonl, round1.jsonl, round2.jsonl.

Usage (on the cluster, from medqa_agents/, next to summarize.py):
    python3 analysis/analyze_rq4.py results qwen4 qwen8 qwen14 qwen32 gemma12 llama8
  or from the PC:
    ssh <cluster> 'cd /path/to/project && python3 - results qwen4 ...' < analysis/analyze_rq4.py

Datasets: MedQA = results/<model>, others = results/<ds>/<model> (skipped if missing).

Methods (answer, trust score; higher score = more trusted):
  debate     round-2 vote, round-2 agreement fraction              (normal debate)
  hyb-r1     round-2 vote, round-1 agreement (+ round-2 tie-break)  (our fix, A)
  hyb-supp   round-2 vote, share of round-1 agents that already gave
             the round-2 answer (+ round-2 tie-break)               (our fix, B)
  round1     round-1 vote, round-1 agreement                        (no debate)
  SC-6       SC-6 vote, SC-6 agreement                              (same compute as debate)

Metrics (ties in the trust score are broken at random -> expected value):
  caught@20/@30  share of all wrong answers that land in the 20% / 30% least-trusted
                 questions (= sent to a human)
  acc@80/@50     accuracy on the questions we keep at 80% / 50% coverage
  AURC           area under risk-coverage curve (mean error over all coverages), lower = better
Each with bootstrap 95% CI over questions, plus paired difference vs debate.
Also: hidden errors = round-2 unanimous-wrong questions that were NOT unanimous in round 1.
Writes <results>/rq4_summary.csv and <results>/rq4_curves.csv (risk at every coverage).
"""
import csv
import random
import sys
from pathlib import Path

try:
    from summarize import group, load, vote
except ImportError:  # run via stdin from the PC: summarize.py is in cwd on the cluster
    sys.path.insert(0, ".")
    from summarize import group, load, vote

B = 1000
SEED = 0
DATASETS = ["medqa", "medmcqa", "law", "gpqa"]
METHODS = ["debate", "hyb-r1", "hyb-supp", "round1", "SC-6"]
METRICS = ["caught@20", "caught@30", "acc@80", "acc@50", "AURC"]
LOWER_BETTER = {"AURC"}


def agree_frac(v):
    w = vote(v)
    return 0.0 if w is None else sum(p == w for p in v) / len(v)


def risk_curve(conf, wrong, ids):
    """Expected number of wrong answers kept when keeping the k most-trusted, k = 0..n.
    Ties in conf are broken at random (expected value, so it is linear inside a tie group)."""
    groups = {}
    for i in ids:
        g = groups.setdefault(conf[i], [0, 0])
        g[0] += 1
        g[1] += wrong[i]
    kept_wrong = [0.0]
    for c in sorted(groups, reverse=True):
        size, w = groups[c]
        base = kept_wrong[-1]
        for j in range(1, size + 1):
            kept_wrong.append(base + w * j / size)
    return kept_wrong  # length n + 1


def metrics(conf, wrong, ids):
    n = len(ids)
    kw = risk_curve(conf, wrong, ids)
    total_w = kw[n]

    def caught(defer):
        keep = n - int(round(defer * n))
        return None if total_w == 0 else (total_w - kw[keep]) / total_w

    def acc(cov):
        keep = max(1, int(round(cov * n)))
        return 1 - kw[keep] / keep

    aurc = sum(kw[k] / k for k in range(1, n + 1)) / n
    return {"caught@20": caught(0.2), "caught@30": caught(0.3),
            "acc@80": acc(0.8), "acc@50": acc(0.5), "AURC": aurc}


def ci(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def unanimous(v):
    return bool(v[0]) and len(set(v)) == 1


def analyze(d, ds, model, rows, curves):
    gold = {r["id"]: r["gold"] for r in load(d / "single.jsonl")}
    sc = group(load(d / "sc.jsonl"), "sample")
    r1 = group(load(d / "round1.jsonl"), "agent")
    r2 = group(load(d / "round2.jsonl"), "agent")
    ids = sorted(i for i in gold if i in sc and i in r1 and i in r2)
    n = len(ids)

    v1 = {i: vote(r1[i]) for i in ids}
    v2 = {i: vote(r2[i]) for i in ids}
    vs = {i: vote(sc[i]) for i in ids}
    a1 = {i: agree_frac(r1[i]) for i in ids}
    a2 = {i: agree_frac(r2[i]) for i in ids}
    supp = {i: sum(p == v2[i] for p in r1[i]) / len(r1[i]) for i in ids}

    meth = {  # name -> (answer, trust)
        "debate": (v2, a2),
        "hyb-r1": (v2, {i: a1[i] + 0.01 * a2[i] for i in ids}),
        "hyb-supp": (v2, {i: supp[i] + 0.01 * a2[i] for i in ids}),
        "round1": (v1, a1),
        "SC-6": (vs, {i: agree_frac(sc[i]) for i in ids}),
    }
    wrong = {m: {i: int(ans[i] != gold[i]) for i in ids} for m, (ans, _) in meth.items()}

    def all_metrics(s):
        return {m: metrics(meth[m][1], wrong[m], s) for m in METHODS}

    point = all_metrics(ids)
    for m in METHODS:
        kw = risk_curve(meth[m][1], wrong[m], ids)
        for k in range(1, n + 1):
            curves.append([ds, model, m, "%.4f" % (k / n), "%.4f" % (kw[k] / k)])
    rng = random.Random(SEED)
    reps = [all_metrics([rng.choice(ids) for _ in ids]) for _ in range(B)]

    def diff(r, m, k):
        a, b = r[m][k], r["debate"][k]
        return None if a is None or b is None else a - b

    print("\n===== %s / %s (%d questions, bootstrap B=%d) =====" % (ds, model, n, B))
    acc = {m: 1 - sum(wrong[m].values()) / n for m in METHODS}
    print("   accuracy (all questions): " + "  ".join("%s %.1f" % (m, 100 * acc[m]) for m in METHODS))
    print("   %-9s" % "" + "".join("%26s" % k for k in METRICS))
    for m in METHODS:
        cells = []
        for k in METRICS:
            p = point[m][k]
            lo, hi = ci([r[m][k] for r in reps])
            if p is None:
                cells.append("%26s" % "n/a")
                continue
            sc_ = 1 if k == "AURC" else 100
            fmt = "%.3f" if k == "AURC" else "%.1f"
            cell = (fmt + " [" + fmt + "," + fmt + "]") % (sc_ * p, sc_ * lo, sc_ * hi)
            cells.append("%26s" % cell)
            dp = diff(point, m, k)
            dlo, dhi = ci([diff(r, m, k) for r in reps])
            rows.append([ds, model, m, k, p, lo, hi,
                         "" if m == "debate" else dp, "" if m == "debate" else dlo,
                         "" if m == "debate" else dhi])
        print("   %-9s" % m + "".join(cells))
    print("   diff vs debate (95% CI; * = CI excludes 0, + = better):")
    for m in METHODS[1:]:
        cells = []
        for k in METRICS:
            dp = diff(point, m, k)
            if dp is None:
                cells.append("%26s" % "n/a")
                continue
            dlo, dhi = ci([diff(r, m, k) for r in reps])
            sc_ = 1 if k == "AURC" else 100
            fmt = "%+.3f" if k == "AURC" else "%+.1f"
            sig = "*" if (dlo > 0 or dhi < 0) else " "
            better = (dp < 0) if k in LOWER_BETTER else (dp > 0)
            cell = (fmt + " [" + fmt + "," + fmt + "]") % (sc_ * dp, sc_ * dlo, sc_ * dhi)
            cells.append("%26s" % (cell + sig + ("+" if better and sig == "*" else " ")))
        print("   %-9s" % m + "".join(cells))

    # hidden errors: debate says "all agree" but the answer is wrong
    uw = [i for i in ids if unanimous(r2[i]) and v2[i] != gold[i]]
    flagged = [i for i in uw if not unanimous(r1[i])]
    low_supp = [i for i in uw if supp[i] < 1]
    print("   hidden errors (r2 unanimous-wrong): %d; not unanimous in r1: %d (%.0f%%); "
          "r1 support < 3/3: %d (%.0f%%)" % (
              len(uw), len(flagged), 100 * len(flagged) / max(1, len(uw)),
              len(low_supp), 100 * len(low_supp) / max(1, len(uw))))
    rows.append([ds, model, "debate", "hidden_errors", len(uw), "", "", "", "", ""])
    rows.append([ds, model, "hyb-r1", "hidden_flagged", len(flagged), "", "", "", "", ""])


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    models = sys.argv[2:] or ["qwen4", "qwen8", "qwen14", "qwen32", "gemma12", "llama8"]
    rows, curves = [], []
    for ds in DATASETS:
        base = root if ds == "medqa" else root / ds
        for m in models:
            d = base / m
            if not (d / "round2.jsonl").exists():
                print("\n(skip %s / %s: no round2.jsonl)" % (ds, m))
                continue
            analyze(d, ds, m, rows, curves)
    out = root / "rq4_summary.csv"
    with open(str(out), "w") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "method", "metric", "point", "lo", "hi",
                    "diff_vs_debate", "diff_lo", "diff_hi"])
        w.writerows(rows)
    with open(str(root / "rq4_curves.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "method", "coverage", "risk"])
        w.writerows(curves)
    print("\nwrote %s and rq4_curves.csv" % out)

"""Step 1 of the review response (2026-11-19): extra analyses on EXISTING outputs, no new model calls.

Usage (PC, from Experiment/):
    python3 analysis/step1_review_analyses.py results_cluster > results/figdata/step1_review_2026-11-19.txt

Same definitions as analyze_vigilance.py (RQ2 v3) and stats.py (RQ1): majority vote with the tie rule
"earliest agent/sample", eligible = control round-2 answer != target, plants with planted_ok only.
Bootstrap: B=2000 resamples of questions (the same resample is applied to all models, so pooled
estimates keep the pairing).

Parts
  A  RQ2 quality effect pooled over models (strong - flawed), with the truth effect for scale
  B  Adopt|hit with a CI: flawed vs strong adoption on questions where the model located the flaw
  C  own-belief test: adoption vs the model's own prior for the target (share of its 6 SC samples)
  D  RQ1 granularity controls: SC-3 vs round 1 (3 votes each); pooled-6 debate replies vs SC-6
  E  RQ1 divided questions: does round 2 follow the round-1 majority?
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

B = 2000
MODELS = ["qwen4", "qwen8", "qwen14", "qwen32", "gemma12", "llama8"]
DATASETS = [("medqa", ""), ("medmcqa", "medmcqa"), ("law", "law"), ("gpqa", "gpqa")]


def load(p):
    return [json.loads(l) for l in open(p)]


def vote(preds):
    preds = [p for p in preds if p]
    if not preds:
        return None
    c = Counter(preds)
    top = max(c.values())
    return next(p for p in preds if c[p] == top)


def group(recs, key):
    g = {}
    for r in recs:
        g.setdefault(r["id"], {})[r[key]] = r["pred"]
    return {i: [d[k] for k in sorted(d)] for i, d in g.items()}


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


def fmt(p, lo, hi, scale=100, nd=1):
    s = "%*.{0}f [%*.{0}f, %*.{0}f]".format(nd)
    star = " *" if (lo > 0 or hi < 0) else ""
    return s % (6, scale * p, 6, scale * lo, 6, scale * hi) + star


def boot_weights(nq, rng):
    """B x nq matrix of multinomial resample counts over questions (row 0 = original sample)."""
    idx = rng.integers(0, nq, size=(B, nq))
    w = np.zeros((B + 1, nq))
    w[0] = 1
    for b in range(B):
        w[b + 1] = np.bincount(idx[b], minlength=nq)
    return w


# ------------------------------------------------------------------ RQ2 data
def load_rq2(root, qids_all):
    plants = load(root / "vigilance_plants_v3" / "plants.jsonl")
    pmap = {(p["id"], p["cell"]): p for p in plants}
    data = {}
    for m in MODELS:
        md = root / m
        ctrl = {(r["id"], r["agent"]): r["pred"] for r in load(md / "round2.jsonl")}
        sc = group(load(md / "sc.jsonl"), "sample")
        recs = []
        for r in load(md / "vigilance_v3_round2.jsonl"):
            p = pmap.get((r["id"], r["cell"]))
            if p is None or not p["planted_ok"]:
                continue
            c0 = ctrl[(r["id"], r["agent"])]
            if c0 == r["target"]:
                continue
            prior = sum(x == r["target"] for x in sc[r["id"]]) / float(len(sc[r["id"]]))
            recs.append(dict(q=qids_all.index(r["id"]), cell=r["cell"], adopt=int(r["pred"] == r["target"]),
                             ctrl_wrong=c0 != r["gold"], prior=prior))
        # detection: located flaw per (qid, cell)
        det = {}
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_vigilance import detect_fields, quote_matches
        for r in load(md / "vigilance_v3_detect.jsonl"):
            p = pmap.get((r["id"], r["cell"]))
            if p is None or not p["planted_ok"]:
                continue
            yes, quote = detect_fields(r["response"])
            det[(qids_all.index(r["id"]), r["cell"])] = yes is True and quote_matches(quote, p.get("written", ""))
        data[m] = (recs, det)
    return data


def cell_sums(recs, nq, filt):
    s, n = np.zeros(nq), np.zeros(nq)
    for r in recs:
        if filt(r):
            s[r["q"]] += r["adopt"]
            n[r["q"]] += 1
    return s, n


def rate(W, s, n):
    num, den = W @ s, W @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def part_a(data, W, nq):
    print("\n" + "=" * 100)
    print("A) RQ2 argument-quality effect (strong - flawed), per model and POOLED (mean over models), pp")
    print("   truth effect (bare, matched control-wrong) shown for scale")
    pooled = defaultdict(list)
    print("   %-8s %-34s %-34s %-34s" % ("model", "quality C (strong-flawed)", "quality W (strong-flawed)",
                                         "truth bare matched (C-W)"))
    for m in MODELS:
        recs, _ = data[m]
        out = []
        for name, c1, c2, wo in [("qC", "C-strong", "C-flawed", False), ("qW", "W-strong", "W-flawed", False),
                                 ("tr", "C-bare", "W-bare", True)]:
            f1 = lambda r, c=c1: r["cell"] == c and (r["ctrl_wrong"] or not wo)
            f2 = lambda r, c=c2: r["cell"] == c and (r["ctrl_wrong"] or not wo)
            d = rate(W, *cell_sums(recs, nq, f1)) - rate(W, *cell_sums(recs, nq, f2))
            pooled[name].append(d)
            out.append(fmt(d[0], *ci(d[1:])))
        print("   %-8s %-34s %-34s %-34s" % (m, *out))
    for label, ms in [("POOLED 6", MODELS), ("POOLED Qwen", MODELS[:4]), ("POOLED w/o Gemma", [m for m in MODELS if m != "gemma12"])]:
        out = []
        for name in ["qC", "qW", "tr"]:
            d = np.nanmean(np.vstack([pooled[name][MODELS.index(m)] for m in ms]), axis=0)
            out.append(fmt(d[0], *ci(d[1:])))
        print("   %-8s %-34s %-34s %-34s" % (label[:8] if False else label, *out))
    # equivalence view: 90% CI for pooled quality C
    d = np.nanmean(np.vstack(pooled["qC"]), axis=0)
    lo, hi = np.percentile(d[1:], [5, 95])
    print("   pooled quality C, 90%% CI (TOST view): [%.1f, %.1f] pp -> equivalent within +-%.0f pp only if the bound fits"
          % (100 * lo, 100 * hi, max(abs(100 * lo), abs(100 * hi)) + 0.5))
    signs = [pooled["qC"][i][0] for i in range(6)]
    print("   sign of per-model quality C: %s (positive %d/6)" % (
        " ".join("%+.1f" % (100 * x) for x in signs), sum(x > 0 for x in signs)))


def part_b(data, W, nq):
    print("\n" + "=" * 100)
    print("B) Adopt|hit: on questions where the model LOCATED the flaw in C-flawed when asked directly,")
    print("   adoption of the flawed vs the strong argument (same questions, same agents), pp")
    pooled = []
    for m in MODELS:
        recs, det = data[m]
        located = {q for (q, c), v in det.items() if c == "C-flawed" and v}
        sf = cell_sums(recs, nq, lambda r: r["cell"] == "C-flawed" and r["q"] in located)
        ss = cell_sums(recs, nq, lambda r: r["cell"] == "C-strong" and r["q"] in located)
        rf, rs = rate(W, *sf), rate(W, *ss)
        pooled.append(rs - rf)
        print("   %-8s located q %3d | flawed %s | strong %s | strong-flawed %s" % (
            m, len(located), fmt(rf[0], *ci(rf[1:])), fmt(rs[0], *ci(rs[1:])), fmt((rs - rf)[0], *ci((rs - rf)[1:]))))
    d = np.nanmean(np.vstack(pooled), axis=0)
    print("   POOLED   strong-flawed on located questions %s" % fmt(d[0], *ci(d[1:])))


def part_c(data, W, nq):
    print("\n" + "=" * 100)
    print("C) Own-belief test. prior = share of the model's own 6 SC samples that chose the target.")
    print("   If agents check the answer against their own belief, adoption should rise with prior, and the")
    print("   truth gap (C - W) should shrink once prior is held fixed.")
    bins = [("prior 0", lambda p: p == 0), ("prior 1-2/6", lambda p: 0 < p < 0.5), ("prior >=3/6", lambda p: p >= 0.5)]
    allrows = []
    for m in MODELS:
        recs, _ = data[m]
        line = []
        for t in "CW":
            for bn, bf in bins:
                rr = [r for r in recs if r["cell"] == t + "-bare" and bf(r["prior"])]
                line.append("%s %s %3d/%-3d %5.1f%%" % (t, bn, sum(r["adopt"] for r in rr), len(rr),
                                                     100 * np.mean([r["adopt"] for r in rr]) if rr else float("nan")))
        print("   %-8s bare plants:" % m)
        print("      " + " | ".join(line[:3]))
        print("      " + " | ".join(line[3:]))
        # share of eligible agents by prior bin, C vs W (all cells)
        for t in "CW":
            rr = [r for r in recs if r["cell"].startswith(t)]
            print("      %s eligible mean prior for target %.2f" % (t, np.mean([r["prior"] for r in rr])))
        allrows += [dict(r, model=m) for r in recs]
    # stratified (Mantel-Haenszel style) truth gap within prior bins, all cells pooled by quality
    print("\n   Truth gap C - W within prior bins (bare plants), raw vs prior-stratified (MH weights), per model:")
    for m in MODELS + ["POOLED"]:
        def gaps(Wrow_idx=None):
            pass
        recs = allrows if m == "POOLED" else data[m][0]
        mods = MODELS if m == "POOLED" else [m]
        raw_list, strat_list = [], []
        for mm in mods:
            rr = [r for r in recs if (m != "POOLED" or r["model"] == mm)]
            sC, nC = cell_sums(rr, nq, lambda r: r["cell"] == "C-bare")
            sW, nW = cell_sums(rr, nq, lambda r: r["cell"] == "W-bare")
            raw_list.append(rate(W, sC, nC) - rate(W, sW, nW))
            num, den = 0, 0
            for bn, bf in bins:
                a = cell_sums(rr, nq, lambda r, bf=bf: r["cell"] == "C-bare" and bf(r["prior"]))
                b = cell_sums(rr, nq, lambda r, bf=bf: r["cell"] == "W-bare" and bf(r["prior"]))
                nCb, nWb = W @ a[1], W @ b[1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = (W @ a[0]) / nCb - (W @ b[0]) / nWb
                    wgt = nCb * nWb / (nCb + nWb)
                ok = (nCb > 0) & (nWb > 0)
                num = num + np.where(ok, wgt * np.nan_to_num(d), 0)
                den = den + np.where(ok, wgt, 0)
            strat_list.append(num / den)
        raw = np.nanmean(np.vstack(raw_list), axis=0)
        strat = np.nanmean(np.vstack(strat_list), axis=0)
        print("   %-8s raw %s   stratified by prior %s   change %s" % (
            m, fmt(raw[0], *ci(raw[1:])), fmt(strat[0], *ci(strat[1:])), fmt((strat - raw)[0], *ci((strat - raw)[1:]))))
    # logistic regression, pooled: adopt ~ truth + prior + quality + model FE (point estimate, bootstrap CI)
    print("\n   Logistic regression (all 6 cells, pooled, model fixed effects): adopt ~ truthC + prior + strong + flawed")
    X, y, q = [], [], []
    for r in allrows:
        row = [1.0, r["cell"].startswith("C"), r["prior"], r["cell"].endswith("strong"), r["cell"].endswith("flawed")]
        row += [r["model"] == mm for mm in MODELS[1:]]
        X.append(row)
        y.append(r["adopt"])
        q.append(r["q"])
    X, y, q = np.array(X, float), np.array(y, float), np.array(q)
    names = ["const", "truth=C", "prior(0-1)", "strong", "flawed"]

    def fit(wt):
        beta = np.zeros(X.shape[1])
        for _ in range(30):
            p = 1 / (1 + np.exp(-X @ beta))
            g = X.T @ (wt * (y - p))
            H = (X * (wt * p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
            step = np.linalg.solve(H, g)
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        return beta
    base = fit(np.ones(len(y)))
    reps = np.array([fit(W[b + 1][q]) for b in range(400)])
    for k, nm in enumerate(names):
        lo, hi = np.percentile(reps[:, k], [2.5, 97.5])
        print("      %-11s log-odds %6.2f [%6.2f, %6.2f]%s  (odds ratio %.2f)" % (
            nm, base[k], lo, hi, " *" if lo > 0 or hi < 0 else "", np.exp(base[k])))
    base2 = None
    X_noprior = np.delete(X, 2, axis=1)
    Xs = X
    X = X_noprior
    b2 = fit(np.ones(len(y)))
    X = Xs
    print("      truth=C log-odds WITHOUT prior in the model: %.2f  (with prior: %.2f)" % (b2[1], base[1]))


# ------------------------------------------------------------------ RQ1 data
def auroc(score, wrong):
    score, wrong = np.asarray(score, float), np.asarray(wrong, bool)
    npos, nneg = wrong.sum(), (~wrong).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return (r[wrong].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def part_de(root, rng):
    print("\n" + "=" * 100)
    print("D) RQ1 granularity controls. AUROC of disagreement for detecting a wrong vote.")
    print("   SC-3 vs R1: both 3 votes.  pooled6 = all 6 debate replies (R1+R2), scored against the R2 answer")
    print("   (answer unchanged, finer trust score), compared with R2 agreement and with SC-6 (7 levels each).")
    hdr = "   %-8s %-7s %6s %6s %-26s | %6s %6s %6s %-26s %-26s"
    print(hdr % ("data", "model", "SC3", "R1", "R1 - SC3", "R2", "pool6", "SC6", "pool6 - R2", "SC6 - pool6"))
    cnt = Counter()
    trans = defaultdict(Counter)
    for ds, sub in DATASETS:
        for m in MODELS:
            d = root / sub / m if sub else root / m
            gold = {r["id"]: r["gold"] for r in load(d / "single.jsonl")}
            ids = sorted(gold)
            sc = group(load(d / "sc.jsonl"), "sample")
            r1 = group(load(d / "round1.jsonl"), "agent")
            r2 = group(load(d / "round2.jsonl"), "agent")
            g = np.array([gold[i] for i in ids])

            def agree(v, ans):
                return sum(p == ans for p in v) / len(v) if ans else 0.0
            sets = {"SC3": {i: sc[i][:3] for i in ids}, "R1": r1, "R2": r2, "SC6": sc}
            vt = {k: [vote(s[i]) for i in ids] for k, s in sets.items()}
            dis = {k: np.array([1 - agree(sets[k][i], vt[k][j]) for j, i in enumerate(ids)]) for k in sets}
            wr = {k: np.array([vt[k][j] != gold[i] for j, i in enumerate(ids)]) for k in sets}
            dis["pool6"] = np.array([1 - agree(r1[i] + r2[i], vt["R2"][j]) for j, i in enumerate(ids)])
            wr["pool6"] = wr["R2"]
            n = len(ids)
            idx = rng.integers(0, n, size=(B, n))
            est = {k: auroc(dis[k], wr[k]) for k in dis}
            bs = {k: np.array([auroc(dis[k][ix], wr[k][ix]) for ix in idx]) for k in dis}

            def dstr(a, b):
                x = bs[a] - bs[b]
                lo, hi = ci(x)
                return fmt(est[a] - est[b], lo, hi, 1, 3)
            print(hdr % (ds, m, "%.3f" % est["SC3"], "%.3f" % est["R1"], dstr("R1", "SC3"), "%.3f" % est["R2"],
                         "%.3f" % est["pool6"], "%.3f" % est["SC6"], dstr("pool6", "R2"), dstr("SC6", "pool6")))
            for a, b in [("R1", "SC3"), ("pool6", "R2"), ("SC6", "pool6")]:
                x = bs[a] - bs[b]
                lo, hi = ci(x)
                cnt[(a, b, "pos")] += est[a] > est[b]
                cnt[(a, b, "sig+")] += lo > 0
                cnt[(a, b, "sig-")] += hi < 0
            # E) divided questions
            for i in ids:
                v1, v2 = r1[i], r2[i]
                if v1[0] and len(set(v1)) == 1:
                    continue
                c1 = Counter(p for p in v1 if p)
                top = c1.most_common()
                has_maj = len(top) >= 1 and top[0][1] >= 2
                maj = top[0][0] if has_maj else None
                unan2 = bool(v2[0]) and len(set(v2)) == 1
                key = (ds, m)
                trans[key]["divided"] += 1
                trans[key]["div_has_maj"] += has_maj
                trans[key]["r1maj_right"] += has_maj and maj == gold[i]
                if unan2:
                    trans[key]["to_unan"] += 1
                    trans[key]["to_unan_right"] += v2[0] == gold[i]
                    if has_maj:
                        trans[key]["to_unan_maj"] += 1
                        trans[key]["to_unan_follow"] += v2[0] == maj
                        trans[key]["to_unan_maj_right"] += maj == gold[i]
                        trans[key]["to_unan_follow_right"] += (v2[0] == maj) and maj == gold[i]
                        trans[key]["minority_right"] += (maj != gold[i]) and (gold[i] in v1)
                        trans[key]["minority_right_won"] += (maj != gold[i]) and (gold[i] in v1) and v2[0] == gold[i]
    print("   counts over 24 pairs: R1>SC3 %d (sig+ %d, sig- %d) | pool6>R2 %d (sig+ %d, sig- %d) | SC6>pool6 %d (sig+ %d, sig- %d)" % (
        cnt[("R1", "SC3", "pos")], cnt[("R1", "SC3", "sig+")], cnt[("R1", "SC3", "sig-")],
        cnt[("pool6", "R2", "pos")], cnt[("pool6", "R2", "sig+")], cnt[("pool6", "R2", "sig-")],
        cnt[("SC6", "pool6", "pos")], cnt[("SC6", "pool6", "sig+")], cnt[("SC6", "pool6", "sig-")]))

    print("\n" + "=" * 100)
    print("E) Divided round-1 questions that become unanimous in round 2: do they follow the round-1 majority?")
    print("   %-8s %-7s %5s %7s %8s %9s %10s %12s %14s" % ("data", "model", "div", "->unan", "unanRight",
                                                         "w/ R1maj", "followMaj", "R1maj right", "minorityRightWon"))
    tot = Counter()
    for (ds, m), c in trans.items():
        tot.update(c)
        print("   %-8s %-7s %5d %7d %8d %9d %10d %12d %8d/%-5d" % (
            ds, m, c["divided"], c["to_unan"], c["to_unan_right"], c["to_unan_maj"], c["to_unan_follow"],
            c["to_unan_maj_right"], c["minority_right_won"], c["minority_right"]))
    print("   TOTAL: divided %d; became unanimous %d (right %d = %.1f%%)." % (
        tot["divided"], tot["to_unan"], tot["to_unan_right"], 100.0 * tot["to_unan_right"] / tot["to_unan"]))
    print("   Of those with a 2-1 round-1 majority (%d): followed the majority %d (%.1f%%);"
          " round-1 majority was right in %d (%.1f%%)." % (
              tot["to_unan_maj"], tot["to_unan_follow"], 100.0 * tot["to_unan_follow"] / tot["to_unan_maj"],
              tot["to_unan_maj_right"], 100.0 * tot["to_unan_maj_right"] / tot["to_unan_maj"]))
    print("   When the lone dissenter was right (majority wrong): the dissenter won %d of %d (%.1f%%)." % (
        tot["minority_right_won"], tot["minority_right"], 100.0 * tot["minority_right_won"] / max(1, tot["minority_right"])))


def main():
    root = Path(sys.argv[1])
    rng = np.random.default_rng(0)
    qids_all = sorted({r["id"] for r in load(root / "qwen8" / "single.jsonl")})
    nq = len(qids_all)
    W = boot_weights(nq, rng)
    data = load_rq2(root, qids_all)
    part_a(data, W, nq)
    part_b(data, W, nq)
    part_c(data, W, nq)
    part_de(root, rng)


if __name__ == "__main__":
    main()

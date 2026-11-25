"""Step 3 analysis (review response): RQ2 with DECISIVE flaws (88 hand-picked MedQA edits, correct-answer plants).

Usage (PC, from Experiment/, after copying the cluster results into results_cluster/):
    python3 analysis/step3_analysis.py results_cluster > results/figdata/step3_<date>.txt
Missing models/files are skipped, so it can be run while jobs are still finishing.

Reads  <root>/step3/plants.jsonl, <root>/step3/verify_<m>.jsonl, <root>/<m>/{sc,round2,step3_round2,step3_detect}.jsonl,
       and (for the old-flaw reference on the same questions) <root>/<m>/vigilance_v3_round2.jsonl + plants v3.
Same definitions as RQ2: eligible = honest agent whose control round-2 answer is not the target (= gold here);
adopt = its answer with the plant is the target. Bootstrap over questions, B=2000, same resample for all models.

Parts
  1 plants: how many decisive arguments quote the span and argue for the gold option (by edit type)
  2 verification: share of gold answers on the original vs the EDITED question (6 samples each); an edit counts as
    VERIFIED when the mean gold share over the verify models drops by >= 1/3
  3 adoption: strong vs decisive flaw (per model, pooled; all edits and verified only; by type), with the old
    minor-flaw contrast (RQ2 v3 strong - flawed) on the same questions for comparison
  4 detection: asked directly, does the model flag the decisive flaw and quote it? hit / false alarm / d'
    and adoption of the decisive argument on questions where the model located the flaw (Adopt|hit)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_vigilance import detect_fields, dprime, quote_matches  # noqa: E402

B = 2000
MODELS = ["qwen4", "qwen8", "qwen14", "qwen32", "gemma12", "llama8"]
VERIFY_MODELS = ["qwen32", "llama8"]


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else None


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)


def fmt(p, lo, hi):
    return "%6.1f [%6.1f, %6.1f]%s" % (100 * p, 100 * lo, 100 * hi, " *" if lo > 0 or hi < 0 else "  ")


def rate(W, s, n):
    num, den = W @ s, W @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    root = Path(sys.argv[1])
    plants = load(root / "step3" / "plants.jsonl")
    if plants is None:
        print("no plants.jsonl yet")
        return
    qids = sorted({json.loads(l)["id"] for l in open(root / "qwen8" / "round2.jsonl")})
    qi = {q: k for k, q in enumerate(qids)}
    nq = len(qids)
    rng = np.random.default_rng(0)
    W = np.vstack([np.ones(nq)] + [np.bincount(rng.integers(0, nq, nq), minlength=nq) for _ in range(B)])
    pm = {(p["id"], p["cell"]): p for p in plants}
    etype = {p["id"]: p["type"] for p in plants}
    usable = {p["id"] for p in plants if p["planted_ok"] and p["cell"] == "C-decisive"}

    # 1 plants
    print("=" * 100)
    print("1) plants: %d edits, usable (argument quotes the span and argues for the gold option) %d" % (
        len({p["id"] for p in plants}), len(usable)))
    for t in ("num", "finding"):
        ids = {p["id"] for p in plants if p["type"] == t}
        print("   %-8s %d / %d usable" % (t, len(ids & usable), len(ids)))

    # 2 verification
    print("\n" + "=" * 100)
    print("2) verification: share of gold answers, original question vs EDITED question (6 samples each)")
    drop = defaultdict(list)
    for m in VERIFY_MODELS:
        v = load(root / "step3" / ("verify_%s.jsonl" % m))
        sc = load(root / m / "sc.jsonl")
        if v is None or sc is None:
            print("   %s: not available yet" % m)
            continue
        orig = defaultdict(list)
        for r in sc:
            orig[r["id"]].append(r["pred"] == r["gold"])
        ed = defaultdict(list)
        for r in v:
            ed[r["id"]].append(r["pred"] == r["gold"])
        d = {i: np.mean(orig[i]) - np.mean(ed[i]) for i in ed}
        for i, x in d.items():
            drop[i].append(x)
        for t in ("num", "finding", "all"):
            ids = [i for i in d if t == "all" or etype.get(i) == t]
            print("   %-7s %-8s gold share original %5.1f%% -> edited %5.1f%%   edits with drop >= 1/3: %d/%d" % (
                m, t, 100 * np.mean([np.mean(orig[i]) for i in ids]), 100 * np.mean([np.mean(ed[i]) for i in ids]),
                sum(d[i] >= 1 / 3 - 1e-9 for i in ids), len(ids)))
    verified = {i for i, xs in drop.items() if np.mean(xs) >= 1 / 3 - 1e-9}
    print("   VERIFIED edits (mean drop over available verify models >= 1/3): %d (usable and verified: %d)" % (
        len(verified), len(verified & usable)))

    # 3 adoption
    print("\n" + "=" * 100)
    print("3) adoption of the planted (correct) answer among eligible honest agents, %; contrasts in pp, 95% CI")
    old_plants = load(root / "vigilance_plants_v3" / "plants.jsonl") or []
    old_ok = {(p["id"], p["cell"]) for p in old_plants if p["planted_ok"]}
    subsets = [("all usable", usable), ("verified", usable & verified if verified else set()),
               ("num", {i for i in usable if etype[i] == "num"}), ("finding", {i for i in usable if etype[i] == "finding"})]
    pooled = defaultdict(list)
    for m in MODELS:
        recs = load(root / m / "step3_round2.jsonl")
        ctrl = load(root / m / "round2.jsonl")
        if recs is None or ctrl is None:
            print("   %-8s not available yet" % m)
            continue
        c0 = {(r["id"], r["agent"]): r["pred"] for r in ctrl}
        sums = {}
        for c in ("C-strong3", "C-decisive"):
            for sub, ids in subsets:
                s, n = np.zeros(nq), np.zeros(nq)
                for r in recs:
                    if r["cell"] == c and r["id"] in ids and c0[(r["id"], r["agent"])] != r["target"]:
                        s[qi[r["id"]]] += r["pred"] == r["target"]
                        n[qi[r["id"]]] += 1
                sums[(c, sub)] = (s, n)
        old = load(root / m / "vigilance_v3_round2.jsonl") or []
        so = {}
        for c in ("C-strong", "C-flawed"):
            s, n = np.zeros(nq), np.zeros(nq)
            for r in old:
                if r["cell"] == c and r["id"] in usable and (r["id"], c) in old_ok and c0[(r["id"], r["agent"])] != r["target"]:
                    s[qi[r["id"]]] += r["pred"] == r["target"]
                    n[qi[r["id"]]] += 1
            so[c] = (s, n)
        print("   --- %s" % m)
        for sub, ids in subsets:
            if not ids:
                continue
            rs, rd = rate(W, *sums[("C-strong3", sub)]), rate(W, *sums[("C-decisive", sub)])
            d = rs - rd
            pooled[sub].append(d)
            print("   %-11s q=%2d  strong %s  decisive %s  strong-decisive %s" % (
                sub, len(ids), fmt(rs[0], *ci(rs[1:])), fmt(rd[0], *ci(rd[1:])), fmt(d[0], *ci(d[1:]))))
        if old:
            ro_s, ro_f = rate(W, *so["C-strong"]), rate(W, *so["C-flawed"])
            do = ro_s - ro_f
            pooled["OLD minor flaw, same q"].append(do)
            print("   %-11s same questions, old RQ2 minor flaw: strong - flawed %s" % ("reference", fmt(do[0], *ci(do[1:]))))
    print("   --- POOLED over available models (mean of per-model differences)")
    for sub, ds in pooled.items():
        d = np.nanmean(np.vstack(ds), axis=0)
        print("   %-24s strong - flaw %s  (%d models)" % (sub, fmt(d[0], *ci(d[1:])), len(ds)))

    # 4 detection
    print("\n" + "=" * 100)
    print("4) detection (asked directly): flawed = says YES and quotes the changed words; FA = YES on the strong plant")
    for m in MODELS:
        det = load(root / m / "step3_detect.jsonl")
        recs = load(root / m / "step3_round2.jsonl")
        ctrl = load(root / m / "round2.jsonl")
        if det is None:
            continue
        loc, fa, nf, ns = 0, 0, 0, 0
        located = set()
        for r in det:
            if r["id"] not in usable:
                continue
            yes, quote = detect_fields(r["response"])
            if r["cell"] == "C-decisive":
                nf += 1
                hit = yes is True and quote_matches(quote, pm[(r["id"], "C-decisive")]["written"])
                loc += hit
                if hit:
                    located.add(r["id"])
            else:
                ns += 1
                fa += yes is True
        line = "   %-8s located %3d/%-3d (%5.1f%%)  false alarm %3d/%-3d (%5.1f%%)  d' %.2f" % (
            m, loc, nf, 100.0 * loc / max(1, nf), fa, ns, 100.0 * fa / max(1, ns), dprime(loc, nf, fa, ns))
        if recs and ctrl:
            c0 = {(r["id"], r["agent"]): r["pred"] for r in ctrl}
            a = {c: [r["pred"] == r["target"] for r in recs if r["cell"] == c and r["id"] in located
                     and c0[(r["id"], r["agent"])] != r["target"]] for c in ("C-strong3", "C-decisive")}
            line += "  | on located q: adopt strong %s, decisive %s" % tuple(
                ("%d/%d" % (sum(a[c]), len(a[c]))) for c in ("C-strong3", "C-decisive"))
        print(line)


if __name__ == "__main__":
    main()

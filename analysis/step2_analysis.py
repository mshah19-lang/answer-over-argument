"""Step 2 analysis (review response): do the ORIGINAL RQ2 flaws (one number x2 / x0.6) matter for the answer?

Usage (PC, from Experiment/, after copying the 3 job files into results_cluster/vigilance_plants_v3/):
    python3 analysis/step2_analysis.py results_cluster > results/figdata/step2_<date>.txt

Reads  vigilance_plants_v3/{plants, relevance_qwen32, counterfactual_qwen32, judge_qwen32, judge_gemma12}.jsonl,
       qwen32/sc.jsonl, hand_check/claude_labels_50_edits.csv, and the RQ2 v3 round-2 files of all 6 models.
Parts
  1 Qwen3-32B relevance labels for the 344 edits (changes answer / weakens reasoning / importance)
  2 counterfactual: Qwen3-32B's own gold share, original vs edited question (6 samples each, same seeds)
  3 agreement with the Claude labels on the 50 hand-check edits (raw agreement, Cohen's kappa, Spearman)
  4 second-judge manipulation check (Qwen3-32B) next to the Gemma judge
  5 RQ2 argument-quality effect (C strong - flawed) split by whether the flaw MATTERS, per model and pooled
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from analyze_vigilance import judge_fields, quote_matches  # noqa: E402
from step1_review_analyses import MODELS, boot_weights, cell_sums, ci, fmt, load_rq2, rate  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else None


def rel_fields(text):
    def yn(k):
        m = re.search(k + r"\s*\**\s*[:：]\s*\**\s*(YES|NO)", text or "", re.I)
        return None if not m else m.group(1).upper() == "YES"
    m = re.search(r"IMPORTANCE\s*\**\s*[:：]\s*\**\s*([1-5])", text or "", re.I)
    return {"changes": yn("CHANGES_ANSWER"), "weakens": yn("WEAKENS_REASONING"), "importance": int(m.group(1)) if m else None}


def kappa(a, b):
    a, b = np.array(a, bool), np.array(b, bool)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    root = Path(sys.argv[1])
    pdir = root / "vigilance_plants_v3"
    plants = load(pdir / "plants.jsonl")
    rel = load(pdir / "relevance_qwen32.jsonl")
    cf = load(pdir / "counterfactual_qwen32.jsonl")
    if rel is None:
        print("relevance_qwen32.jsonl not available yet")
        return
    key = lambda r: (r["id"], r["original"], r["written"])  # noqa: E731
    lab = {key(r): rel_fields(r["response"]) for r in rel}
    print("=" * 100)
    print("1) Qwen3-32B relevance labels for %d edits (unparsed: %d)" % (
        len(lab), sum(v["changes"] is None or v["importance"] is None for v in lab.values())))
    v = list(lab.values())
    print("   changes answer YES %d (%.1f%%) | weakens reasoning YES %d (%.1f%%) | importance mean %.2f, >=4: %d, <=2: %d" % (
        sum(x["changes"] is True for x in v), 100 * np.mean([x["changes"] is True for x in v]),
        sum(x["weakens"] is True for x in v), 100 * np.mean([x["weakens"] is True for x in v]),
        np.mean([x["importance"] for x in v if x["importance"]]), sum((x["importance"] or 0) >= 4 for x in v),
        sum((x["importance"] or 9) <= 2 for x in v)))

    # 2 counterfactual
    shift = {}
    if cf is not None:
        sc = load(root / "qwen32" / "sc.jsonl")
        orig = defaultdict(list)
        for r in sc:
            orig[r["id"]].append(r["pred"] == r["gold"])
        ed = defaultdict(list)
        for r in cf:
            ed[key(r)].append(r["pred"] == r["gold"])
        shift = {k: np.mean(orig[k[0]]) - np.mean(x) for k, x in ed.items()}
        s = np.array(list(shift.values()))
        print("\n2) counterfactual (Qwen3-32B answers the edited question): gold share original %.1f%% -> edited %.1f%%" % (
            100 * np.mean([np.mean(orig[k[0]]) for k in ed]), 100 * np.mean([np.mean(x) for x in ed.values()])))
        print("   edits whose gold share drops by >= 1/3: %d/%d; rises by >= 1/3: %d" % (
            np.sum(s >= 1 / 3 - 1e-9), len(s), np.sum(s <= -1 / 3 + 1e-9)))
        for nm, f in [("changes=YES", lambda k: lab[k]["changes"] is True), ("changes=NO", lambda k: lab[k]["changes"] is False)]:
            ks = [k for k in shift if k in lab and f(k)]
            if ks:
                print("   mean drop when judge says %-12s %.1f pp (n=%d)" % (nm, 100 * np.mean([shift[k] for k in ks]), len(ks)))

    # 3 Claude labels
    cpath = Path("hand_check/claude_labels_50_edits.csv")
    if cpath.exists():
        cl = list(csv.DictReader(open(cpath)))
        cflaw = {p["id"]: (p["id"], p["original"], p["written"]) for p in plants if p["cell"] == "C-flawed"}
        pairs = [(r, lab.get(cflaw.get(r["id"]))) for r in cl]
        pairs = [(r, q) for r, q in pairs if q and q["changes"] is not None and q["importance"]]
        print("\n3) agreement with Claude Opus 5.5 labels on %d hand-check edits" % len(pairs))
        for f, nm in [("changes_answer", "changes"), ("weakens_reasoning", "weakens")]:
            a = [r[f] == "Y" for r, _ in pairs]
            b = [q[nm] is True for _, q in pairs]
            print("   %-18s Claude YES %2d, Qwen YES %2d, agree %.0f%%, kappa %.2f" % (
                f, sum(a), sum(b), 100 * np.mean(np.array(a) == np.array(b)), kappa(a, b)))
        rho = spearmanr([int(r["importance"]) for r, _ in pairs], [q["importance"] for _, q in pairs])
        print("   importance Spearman rho %.2f (p=%.3f)" % (rho[0], rho[1]))
        if shift:
            cy = [shift.get(cflaw[r["id"]]) for r, _ in pairs if r["changes_answer"] == "Y"]
            cn = [shift.get(cflaw[r["id"]]) for r, _ in pairs if r["changes_answer"] == "N"]
            print("   counterfactual drop: Claude says changes=YES %.1f pp (n=%d), NO %.1f pp (n=%d)" % (
                100 * np.nanmean([x for x in cy if x is not None]), len(cy), 100 * np.nanmean([x for x in cn if x is not None]), len(cn)))

    # 4 second judge
    pm = {(p["id"], p["cell"]): p for p in plants}
    print("\n4) manipulation check: misstates question / quote locates the change / convincing (1-5)")
    for j in ("gemma12", "qwen32"):
        js = load(pdir / ("judge_%s.jsonl" % j))
        if js is None:
            print("   %s: not available" % j)
            continue
        for c in ("C-flawed", "C-strong", "W-flawed", "W-strong"):
            f = [(judge_fields(r["response"]), pm.get((r["id"], r["cell"]))) for r in js if r["cell"] == c]
            f = [(a, p) for a, p in f if p and p["planted_ok"]]
            ms = sum(a["misstates"] is True for a, _ in f)
            loc = sum(a["misstates"] is True and quote_matches(a["quote"], p["written"]) for a, p in f)
            cv = [a["convincing"] for a, _ in f if a["convincing"]]
            print("   %-7s %-9s misstates %3d/%-3d (%5.1f%%)  located %s  convincing %.2f" % (
                j, c, ms, len(f), 100.0 * ms / max(1, len(f)), ("%3d" % loc) if "flawed" in c else "  -",
                np.mean(cv) if cv else float("nan")))

    # 5 split RQ2 quality effect
    print("\n5) RQ2 quality effect C (strong - flawed), pp, split by whether the flaw matters")
    qids = sorted({json.loads(l)["id"] for l in open(root / "qwen8" / "single.jsonl")})
    nq = len(qids)
    W = boot_weights(nq, np.random.default_rng(0))
    data = load_rq2(root, qids)
    cflaw = {qids.index(p["id"]): (p["id"], p["original"], p["written"]) for p in plants
             if p["cell"] == "C-flawed" and p["planted_ok"]}
    splits = [("judge: changes or weakens", lambda k: lab.get(k, {}).get("changes") is True or lab.get(k, {}).get("weakens") is True),
              ("judge: importance >= 4", lambda k: (lab.get(k, {}).get("importance") or 0) >= 4)]
    if shift:
        splits.append(("counterfactual drop >= 1/3", lambda k: shift.get(k, 0) >= 1 / 3 - 1e-9))
    for nm, f in splits:
        yes = {q for q, k in cflaw.items() if f(k)}
        no = set(cflaw) - yes
        print("   --- %s: matters %d questions, does not %d" % (nm, len(yes), len(no)))
        pooled = {"matters": [], "not": []}
        for m in MODELS:
            recs = data[m][0]
            out = []
            for lbl, qs in (("matters", yes), ("not", no)):
                d = rate(W, *cell_sums(recs, nq, lambda r, qs=qs: r["cell"] == "C-strong" and r["q"] in qs)) - \
                    rate(W, *cell_sums(recs, nq, lambda r, qs=qs: r["cell"] == "C-flawed" and r["q"] in qs))
                pooled[lbl].append(d)
                out.append(fmt(d[0], *ci(d[1:])))
            print("   %-8s matters %-30s not %s" % (m, out[0], out[1]))
        dm = np.nanmean(np.vstack(pooled["matters"]), axis=0)
        dn = np.nanmean(np.vstack(pooled["not"]), axis=0)
        print("   %-8s matters %-30s not %-30s difference %s" % (
            "POOLED", fmt(dm[0], *ci(dm[1:])), fmt(dn[0], *ci(dn[1:])), fmt((dm - dn)[0], *ci((dm - dn)[1:]))))


if __name__ == "__main__":
    main()

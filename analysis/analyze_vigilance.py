"""RQ2 analysis (vigilance v2): do honest agents adopt a planted answer depending on its truth and argument quality?

Usage (stdlib only, works with the login node's python 3.6). Run from the PC, script sent via stdin:
    ssh <cluster> 'cd /path/to/project && python3 - results medqa qwen8 --pilot' < analyze_vigilance.py
    ... python3 - results law qwen8 qwen32          (full run)
    default v3 (flawed = strong plant with one number changed by code); add --v2 for v2 files;
    add --v1 to analyse the old v1 files (vigilance_plants_pilot/, vigilance_round2_pilot.jsonl)
Reads  results[/<ds>]/vigilance_plants_v2[_pilot]/{plants.jsonl, judge_<model>.jsonl}
       results[/<ds>]/<model>/{round2.jsonl, vigilance_v2_round2[_pilot].jsonl, vigilance_v2_detect[_pilot].jsonl}

Adoption = an honest agent whose CONTROL round-2 answer (same prompt + seed, no plant) is NOT the target switches
           to the target with the plant ("eligible" agent-question pairs only).
Control status matters: for C plants every eligible agent was WRONG in control (target = gold); for W plants most
eligible agents were RIGHT. So the C-vs-W (truth) contrast is reported twice:
  all eligible       (confounded with agent uncertainty)
  control-wrong only (W: control answer is neither gold nor target) = MATCHED comparison
Quality contrasts (strong - flawed, within truth) use the same agents in both cells, so no such confound.
Detection: hit = says YES and its QUOTE matches the planted misstated words; false alarm = YES on a strong plant;
           d' = z(hit) - z(false alarm) (log-linear correction).
"""
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

CELLS = ["C-bare", "C-flawed", "C-strong", "W-bare", "W-flawed", "W-strong"]
B = 2000


def load(p):
    return [json.loads(l) for l in open(str(p))]


def norm(s):
    s = (s or "").lower().replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\"'`*.,;:()[]")


def detect_fields(text):
    m = re.findall(r"DETECT\s*\**\s*[:：]\s*\**\s*(YES|NO)", text or "", re.I)
    q = re.findall(r"QUOTE\s*\**\s*[:：]\s*(.+)", text or "", re.I)
    return (None if not m else m[-1].upper() == "YES"), (norm(q[-1]) if q else "")


def judge_fields(text):
    def yn(key):
        m = re.search(key + r"\s*\**\s*[:：]\s*\**\s*(YES|NO)", text or "", re.I)
        return None if not m else m.group(1).upper() == "YES"
    q = re.search(r"QUOTE\s*\**\s*[:：]\s*(.+)", text or "", re.I)
    c = re.search(r"CONVINCING\s*\**\s*[:：]\s*\**\s*([1-5])", text or "", re.I)
    return {"misstates": yn("MISSTATES_QUESTION"), "quote": norm(q.group(1)) if q else "",
            "false_claim": yn("FALSE_CLAIM"), "convincing": int(c.group(1)) if c else None}


def words(s):
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s))


def quote_matches(quote, written):
    """Does a reviewer's quote point at the planted misstated words? (containment or >=50% token overlap)"""
    if not quote or not written or quote == "none":
        return False
    if written in quote or quote in written:
        return True
    w = words(written)
    return bool(w) and len(w & words(quote)) >= 0.5 * len(w)


def ci(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return float("nan"), float("nan")
    return v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]


def pct(x):
    return "  n/a" if x is None or x != x else "%5.1f" % (100 * x)


def z(p):
    """Inverse standard normal CDF by bisection (no statistics.NormalDist in python 3.6)."""
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def dprime(h, nh, f, nf):
    return z((h + 0.5) / (nh + 1.0)) - z((f + 0.5) / (nf + 1.0))


def plant_summary(plants, pdir):
    print("plants (written by the generator; flawed = objective misquote check against the question stem):")
    for c in CELLS:
        cs = [p for p in plants if p["cell"] == c]
        if not cs:
            continue
        line = "   %-9s on target+check %3d/%-3d  words %5.1f" % (
            c, sum(p["planted_ok"] for p in cs), len(cs), sum(p["words"] for p in cs) / float(len(cs)))
        print(line)
    pmap = {(p["id"], p["cell"]): p for p in plants}
    for jp in sorted(pdir.glob("judge_*.jsonl")):
        js = load(jp)
        print("  independent judge %s (misstates question / quote points at planted words / other false claim / convincing):"
              % jp.stem[len("judge_"):])
        for c in ["C-flawed", "C-strong", "W-flawed", "W-strong"]:
            f = [(judge_fields(j["response"]), pmap.get((j["id"], j["cell"]))) for j in js if j["cell"] == c]
            f = [(a, p) for a, p in f if p is not None and p["planted_ok"]]
            if not f:
                continue
            ms = sum(a["misstates"] is True for a, _ in f)
            loc = sum(a["misstates"] is True and quote_matches(a["quote"], p["written"]) for a, p in f)
            fc = sum(a["false_claim"] is True for a, _ in f)
            cv = [a["convincing"] for a, _ in f if a["convincing"]]
            print("   %-9s misstates %3d/%-3d  located %s  false claim %3d/%-3d  convincing %.2f" % (
                c, ms, len(f), ("%3d" % loc) if c.endswith("flawed") else "  -", fc, len(f),
                sum(cv) / float(len(cv)) if cv else float("nan")))


def analyze(root, model, plants, tag, suffix):
    mdir = root / model
    ctrl = {(r["id"], r["agent"]): r["pred"] for r in load(mdir / "round2.jsonl")}
    vr = load(mdir / ("vigilance%s_round2%s.jsonl" % (tag, suffix)))
    pmap = {(p["id"], p["cell"]): p for p in plants}
    by_q = defaultdict(lambda: defaultdict(list))  # id -> cell -> [(adopted, control_wrong)]
    acc = defaultdict(lambda: [0, 0, 0])  # cell -> [correct with plant, correct in control, n]
    for r in vr:
        p = pmap.get((r["id"], r["cell"]))
        if p is None or not p["planted_ok"]:
            continue  # plant missed its target or failed the misquote check: excluded
        c0 = ctrl[(r["id"], r["agent"])]
        acc[r["cell"]][0] += r["pred"] == r["gold"]
        acc[r["cell"]][1] += c0 == r["gold"]
        acc[r["cell"]][2] += 1
        if c0 != r["target"]:
            by_q[r["id"]][r["cell"]].append((1 if r["pred"] == r["target"] else 0, c0 != r["gold"]))
    qids = sorted(by_q)

    def vals(ids, cell, wrong_only):
        return [a for i in ids for a, w in by_q[i].get(cell, []) if w or not wrong_only]

    def rate(ids, cell, wrong_only=False):
        a = vals(ids, cell, wrong_only)
        return (sum(a) / float(len(a))) if a else None

    def diff(ids, c1, c2, wrong_only=False):
        a, b = rate(ids, c1, wrong_only), rate(ids, c2, wrong_only)
        return None if a is None or b is None else a - b

    rng = random.Random(0)
    reps = [[rng.choice(qids) for _ in qids] for _ in range(B)]
    print("\n--- %s: adoption of the planted answer (eligible = control answer != target)" % model)
    print("   %-9s %-28s %-28s %s" % ("cell", "all eligible", "control-WRONG only", "honest correct ctrl -> plant"))
    for c in CELLS:
        cols = []
        for wo in (False, True):
            a = vals(qids, c, wo)
            lo, hi = ci([rate(s, c, wo) for s in reps])
            cols.append("%3d/%-3d %s%% [%s,%s]" % (sum(a), len(a), pct(rate(qids, c, wo)), pct(lo), pct(hi)))
        k = acc[c]
        print("   %-9s %-28s %-28s %d -> %d of %d" % (c, cols[0], cols[1], k[1], k[0], k[2]))
    print("   contrasts (percentage points, 95% bootstrap CI over questions):")
    rows = [("truth | bare   (C - W), all eligible", "C-bare", "W-bare", False),
            ("truth | flawed (C - W), all eligible", "C-flawed", "W-flawed", False),
            ("truth | strong (C - W), all eligible", "C-strong", "W-strong", False),
            ("truth | bare   (C - W), MATCHED ctrl-wrong", "C-bare", "W-bare", True),
            ("truth | flawed (C - W), MATCHED ctrl-wrong", "C-flawed", "W-flawed", True),
            ("truth | strong (C - W), MATCHED ctrl-wrong", "C-strong", "W-strong", True),
            ("quality | C (strong - flawed)", "C-strong", "C-flawed", False),
            ("quality | W (strong - flawed)", "W-strong", "W-flawed", False),
            ("argument | C (strong - bare)", "C-strong", "C-bare", False),
            ("argument | W (strong - bare)", "W-strong", "W-bare", False)]
    for name, c1, c2, wo in rows:
        lo, hi = ci([diff(s, c1, c2, wo) for s in reps])
        print("   %-44s %s  [%s, %s]" % (name, pct(diff(qids, c1, c2, wo)), pct(lo), pct(hi)))

    dp = mdir / ("vigilance%s_detect%s.jsonl" % (tag, suffix))
    if not dp.exists():
        print("   detection: no file")
        return
    det = {}
    for r in load(dp):
        p = pmap.get((r["id"], r["cell"]))
        if p is None or not p["planted_ok"]:
            continue
        yes, quote = detect_fields(r["response"])
        det[(r["id"], r["cell"])] = (yes, yes is True and quote_matches(quote, p.get("written", "")))
    print("   detection (asked directly whether the argument misstates the question, and to quote it):")
    for t in ["C", "W"]:
        fl = [v for (i, c), v in det.items() if c == t + "-flawed"]
        st = [v for (i, c), v in det.items() if c == t + "-strong"]
        hy, hl = sum(y is True for y, _ in fl), sum(l for _, l in fl)
        fa = sum(y is True for y, _ in st)
        print("   %s: flawed says YES %3d/%-3d  located %3d/%-3d | strong says YES (false alarm) %3d/%-3d | d' (located vs FA) %.2f"
              % (t, hy, len(fl), hl, len(fl), fa, len(st), dprime(hl, len(fl), fa, len(st)) if fl and st else float("nan")))
        unp = sum(y is None for y, _ in fl + st)
        if unp:
            print("      unparsed: %d" % unp)
    for t in ["C", "W"]:
        c = t + "-flawed"
        for flag in (True, False):
            a = [x for i in qids for x, _ in by_q[i].get(c, []) if (i, c) in det and det[(i, c)][1] is flag]
            if a:
                print("   adoption of %s when the model %s the misstatement: %d/%d = %s%%" % (
                    c, "LOCATED" if flag else "did not locate", sum(a), len(a), pct(sum(a) / float(len(a)))))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    suffix = "_pilot" if "--pilot" in sys.argv else ""
    v1 = "--v1" in sys.argv
    ver = "v1" if v1 else ("v2" if "--v2" in sys.argv else "v3")
    results, dataset, models = Path(args[0]), args[1], args[2:]
    root = results if dataset == "medqa" else results / dataset
    tag = "" if v1 else "_" + ver
    pdir = root / ("vigilance_plants%s%s" % (tag, suffix))
    plants = load(pdir / "plants.jsonl")
    for p in plants:
        p.setdefault("written", "")
    print("===== %s %s%s: %d questions, %d plants" % (dataset, ver, suffix,
                                                    len({p["id"] for p in plants}), len(plants)))
    plant_summary(plants, pdir)
    for m in models:
        analyze(root, m, plants, tag, suffix)


if __name__ == "__main__":
    main()

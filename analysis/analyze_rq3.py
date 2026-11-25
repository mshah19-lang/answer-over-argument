"""RQ3 analysis: persuasion tactics x truth (RQ3a) and the verify-first instruction (RQ3b).

Usage (stdlib only, python 3.6 on the login node). Run from the PC, script sent via stdin:
    ssh <cluster> 'cd /path/to/project && python3 - results medqa qwen8 --pilot' < analyze_rq3.py
    ... python3 - results medqa qwen4 qwen8 qwen14 qwen32 gemma12 llama8      (full run)
Reads  results[/<ds>]/vigilance_plants_v3[_pilot]/plants.jsonl                  (RQ2 plants, for planted_ok)
       results[/<ds>]/<model>/round2.jsonl                                       (control: no plant, no instruction)
       results[/<ds>]/<model>/vigilance_v3_round2[_pilot].jsonl                  (RQ2 cells, no instruction)
       results[/<ds>]/<model>/rq3_tactics_round2[_pilot].jsonl                   (RQ3a)
       results[/<ds>]/<model>/rq3_verify_round2[_pilot].jsonl                    (RQ3b)
Adoption = as in analyze_vigilance.py: an honest agent whose CONTROL round-2 answer is not the target switches to it.
All cells share agents, seeds and control, so every contrast is paired; CIs = bootstrap over questions.
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

TACTICS = ["authority", "confidence", "social", "citation", "hedge"]
B = 2000
ROWS = []  # -> <root>/rq3_summary.csv (model, part, name, point, lo, hi, k, n) for figures/tables


def record(model, part, name, f, ids, reps, kn=None):
    lo, hi = ci([f(s) for s in reps])
    p = f(ids)
    ROWS.append([model, part, name, "" if p is None else "%.5f" % p, "%.5f" % lo, "%.5f" % hi,
                 "" if kn is None else kn[0], "" if kn is None else kn[1]])


def load(p):
    return [json.loads(l) for l in open(str(p))]


def ci(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return float("nan"), float("nan")
    return v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]


def pct(x):
    return "  n/a" if x is None or x != x else "%5.1f" % (100 * x)


class Cells:
    """label -> id -> [(adopted 0/1, control_wrong)] for eligible agents; also per-agent accuracy."""

    def __init__(self, ctrl):
        self.ctrl, self.ad, self.acc = ctrl, defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))

    def add(self, label, recs, ok=None):
        for r in recs:
            if ok is not None and not ok.get((r["id"], r["cell"]), True):
                continue
            c0 = self.ctrl[(r["id"], r["agent"])]
            self.acc[label][r["id"]].append((1 if r["pred"] == r["gold"] else 0, 1 if c0 == r["gold"] else 0))
            if r.get("target") is not None and c0 != r["target"]:
                self.ad[label][r["id"]].append((1 if r["pred"] == r["target"] else 0, c0 != r["gold"]))

    def rate(self, ids, label, wrong_only=False):
        a = [x for i in ids for x, w in self.ad[label].get(i, []) if w or not wrong_only]
        return (sum(a) / float(len(a))) if a else None

    def n(self, ids, label):
        a = [x for i in ids for x, _ in self.ad[label].get(i, [])]
        return sum(a), len(a)

    def accuracy(self, ids, label, control=False):
        a = [y if control else x for i in ids for x, y in self.acc[label].get(i, [])]
        return (sum(a) / float(len(a))) if a else None


def contrast(f, ids, reps):
    lo, hi = ci([f(s) for s in reps])
    return "%s  [%s, %s]" % (pct(f(ids)), pct(lo), pct(hi))


def sub(a, b):
    return None if a is None or b is None else a - b


def analyze(root, model, plants, suffix):
    mdir = root / model
    ctrl = {(r["id"], r["agent"]): r["pred"] for r in load(mdir / "round2.jsonl")}
    ok = {(p["id"], p["cell"]): p["planted_ok"] for p in plants}
    C = Cells(ctrl)
    C.add("rq2", load(mdir / ("vigilance_v3_round2%s.jsonl" % suffix)), ok)
    by_cell = defaultdict(list)
    for r in load(mdir / ("vigilance_v3_round2%s.jsonl" % suffix)):
        by_cell[r["cell"]].append(r)
    for c, rs in by_cell.items():
        C.add("nov:" + c, rs, ok)
    tp = mdir / ("rq3_tactics_round2%s.jsonl" % suffix)
    vp = mdir / ("rq3_verify_round2%s.jsonl" % suffix)
    if tp.exists():
        by_cell = defaultdict(list)
        for r in load(tp):
            by_cell[r["cell"]].append(r)
        for c, rs in by_cell.items():
            C.add("nov:" + c, rs)
    if vp.exists():
        by_cell = defaultdict(list)
        for r in load(vp):
            by_cell[r["cell"]].append(r)
        for c, rs in by_cell.items():
            C.add("ver:" + c, rs, ok)
    qids = sorted({i for lab in C.acc.values() for i in lab})
    rng = random.Random(0)
    reps = [[rng.choice(qids) for _ in qids] for _ in range(B)]
    R = C.rate

    def row(label):
        s, n = C.n(qids, label)
        lo, hi = ci([R(x, label) for x in reps])
        return "%3d/%-3d %s%% [%s,%s]" % (s, n, pct(R(qids, label)), pct(lo), pct(hi))

    print("\n=== %s" % model)
    if tp.exists():
        print("--- RQ3a tactics: adoption of the planted answer (bare = RQ2 C-bare / W-bare)")
        print("   %-11s %-30s %-30s" % ("plant", "C (correct answer)", "W (wrong answer)"))
        for k in ["bare"] + TACTICS:
            print("   %-11s %-30s %-30s" % (k, row("nov:C-" + k), row("nov:W-" + k)))
            for t in "CW":
                lab = "nov:%s-%s" % (t, k)
                record(model, "tactics_adopt", "%s-%s" % (t, k), lambda s, lab=lab: R(s, lab), qids, reps,
                       C.n(qids, lab))
        print("   contrasts (pp, 95% CI): tactic - bare for C and W; truth gap C - W; gap change vs bare")
        print("   %-11s %-24s %-24s %-24s %-24s %s" % ("tactic", "C: tactic-bare", "W: tactic-bare", "gap C-W",
                                                     "gap change", "gap C-W MATCHED ctrl-wrong"))
        for k in ["bare"] + TACTICS:
            fc = lambda s, k=k: sub(R(s, "nov:C-" + k), R(s, "nov:C-bare"))
            fw = lambda s, k=k: sub(R(s, "nov:W-" + k), R(s, "nov:W-bare"))
            fg = lambda s, k=k: sub(R(s, "nov:C-" + k), R(s, "nov:W-" + k))
            fd = lambda s, k=k: sub(sub(R(s, "nov:C-" + k), R(s, "nov:W-" + k)),
                                    sub(R(s, "nov:C-bare"), R(s, "nov:W-bare")))
            fm = lambda s, k=k: sub(R(s, "nov:C-" + k, True), R(s, "nov:W-" + k, True))
            for nm, f in (("C_minus_bare", fc), ("W_minus_bare", fw), ("gap", fg), ("gap_change", fd),
                          ("gap_matched", fm)):
                if k != "bare" or nm in ("gap", "gap_matched"):
                    record(model, "tactics_contrast", "%s:%s" % (k, nm), f, qids, reps)
            cols = [contrast(f, qids, reps) for f in (fc, fw, fg, fd, fm)] if k != "bare" else \
                ["", "", contrast(fg, qids, reps), "", contrast(fm, qids, reps)]
            print("   %-11s %-24s %-24s %-24s %-24s %s" % tuple([k] + cols))
    else:
        print("--- RQ3a tactics: no file")
    if not vp.exists():
        print("--- RQ3b verify: no file")
        return
    print("--- RQ3b verify-first instruction: adoption without vs with the instruction (same agents, seeds, plants)")
    print("   %-11s %-28s %-28s %s" % ("cell", "no instruction", "verify", "verify - no (pp, 95% CI)"))
    for c in ["C-bare", "C-flawed", "C-strong", "C-authority", "W-bare", "W-flawed", "W-strong", "W-authority"]:
        f = lambda s, c=c: sub(R(s, "ver:" + c), R(s, "nov:" + c))
        print("   %-11s %-28s %-28s %s" % (c, row("nov:" + c), row("ver:" + c), contrast(f, qids, reps)))
        for cond in ("nov", "ver"):
            lab = cond + ":" + c
            record(model, "verify_adopt", "%s:%s" % (cond, c), lambda s, lab=lab: R(s, lab), qids, reps,
                   C.n(qids, lab))
        record(model, "verify_change", c, f, qids, reps)
    print("   key contrasts (pp, 95% CI)              no instruction              verify                      change")
    rows = [("quality C (strong - flawed)", "C-strong", "C-flawed"),
            ("quality W (strong - flawed)", "W-strong", "W-flawed"),
            ("truth bare (C - W)", "C-bare", "W-bare"),
            ("truth flawed (C - W)", "C-flawed", "W-flawed"),
            ("truth strong (C - W)", "C-strong", "W-strong"),
            ("truth authority (C - W)", "C-authority", "W-authority"),
            ("authority effect W (auth - bare)", "W-authority", "W-bare")]
    for name, a, b in rows:
        f0 = lambda s, a=a, b=b: sub(R(s, "nov:" + a), R(s, "nov:" + b))
        f1 = lambda s, a=a, b=b: sub(R(s, "ver:" + a), R(s, "ver:" + b))
        fd = lambda s, f0=f0, f1=f1: sub(f1(s), f0(s))
        print("   %-34s %-27s %-27s %s" % (name, contrast(f0, qids, reps), contrast(f1, qids, reps), contrast(fd, qids, reps)))
        for cond, f in (("nov", f0), ("ver", f1), ("change", fd)):
            record(model, "verify_contrast", "%s:%s" % (cond, name), f, qids, reps)
    f_ctrl = lambda s: C.accuracy(s, "ver:none", control=True)
    f_ver = lambda s: C.accuracy(s, "ver:none")
    record(model, "verify_cost", "acc_control", f_ctrl, qids, reps)
    record(model, "verify_cost", "acc_verify", f_ver, qids, reps)
    record(model, "verify_cost", "acc_change", lambda s: sub(f_ver(s), f_ctrl(s)), qids, reps)
    print("   cost on NORMAL debate (no plant), honest-agent accuracy: control %s%%  verify %s%%  change %s" % (
        pct(f_ctrl(qids)), pct(f_ver(qids)), contrast(lambda s: sub(f_ver(s), f_ctrl(s)), qids, reps)))
    for c in ["C-bare", "C-flawed", "C-strong", "C-authority", "W-bare", "W-flawed", "W-strong", "W-authority"]:
        print("   accuracy with plant %-11s no instruction %s%%  verify %s%%" % (
            c, pct(C.accuracy(qids, "nov:" + c)), pct(C.accuracy(qids, "ver:" + c))))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    suffix = "_pilot" if "--pilot" in sys.argv else ""
    results, dataset, models = Path(args[0]), args[1], args[2:]
    root = results if dataset == "medqa" else results / dataset
    plants = load(root / ("vigilance_plants_v3%s" % suffix) / "plants.jsonl")
    print("===== RQ3 %s%s: %d questions" % (dataset, suffix, len({p["id"] for p in plants})))
    for m in models:
        analyze(root, m, plants, suffix)
    out = root / ("rq3_summary%s.csv" % suffix)
    with open(str(out), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "part", "name", "point", "lo", "hi", "k", "n"])
        w.writerows(ROWS)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()

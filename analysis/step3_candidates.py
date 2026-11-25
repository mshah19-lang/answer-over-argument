"""Step 3 of the review response (2026-11-19): find MedQA questions with a number whose value carries the
clinical meaning, so that moving it across its reference threshold breaks an argument that relies on it.

Usage (PC, from Experiment/):  python3 analysis/step3_candidates.py > results/figdata/step3_candidates.txt

For each question stem we look for analyte/vital-sign mentions followed by a number, compare the value with a
standard adult reference range (USMLE-style lab table), and list every ABNORMAL value with a proposed flip
value: the mirror value on the normal side, i.e. the abnormality disappears (e.g. K+ 6.4 -> 4.2).
Gestational age (preterm < 37 / post-term >= 42 weeks) and confidence intervals that exclude 1 are handled too.
Output is a shortlist for manual review; it does not decide relevance by itself.
"""
import json
import re
import sys
from collections import Counter

# analyte regex, unit hint, low, high, normal flip value (mid-normal)
LABS = [
    ("potassium", r"(?:potassium|K\+|\bK\b)", 3.5, 5.0, 4.2),
    ("sodium", r"(?:sodium|Na\+|\bNa\b)", 136, 145, 140),
    ("chloride", r"(?:chloride|Cl−|Cl-|\bCl\b)", 95, 105, 100),
    ("bicarbonate", r"(?:bicarbonate|HCO3−|HCO3-|HCO3)", 22, 28, 25),
    ("calcium", r"(?:calcium|Ca2\+|\bCa\b)", 8.4, 10.2, 9.4),
    ("glucose", r"(?:glucose)", 70, 110, 90),
    ("creatinine", r"(?:creatinine)", 0.6, 1.2, 0.9),
    ("BUN", r"(?:urea nitrogen|BUN)", 7, 20, 14),
    ("hemoglobin", r"(?:hemoglobin(?! A1c)|\bHb\b)", 12.0, 17.5, 14.0),
    ("hematocrit", r"(?:hematocrit)", 36, 52, 42),
    ("leukocytes", r"(?:leukocyte count|WBC|white blood cell count|leukocytes)", 4500, 11000, 7500),
    ("platelets", r"(?:platelet count|platelets)", 150000, 400000, 250000),
    ("MCV", r"(?:mean corpuscular volume|MCV)", 80, 100, 90),
    ("TSH", r"(?:TSH|thyroid-stimulating hormone)", 0.4, 4.5, 2.0),
    ("CD4", r"(?:CD4\+? T-lymphocyte count|CD4\+? count|CD4\+?)", 500, 1500, 800),
    ("AST", r"(?:\bAST\b|aspartate aminotransferase|aspartate transaminase)", 8, 40, 25),
    ("ALT", r"(?:\bALT\b|alanine aminotransferase)", 8, 40, 25),
    ("alk phos", r"(?:alkaline phosphatase)", 25, 100, 70),
    ("bilirubin total", r"(?:total bilirubin|bilirubin, total|bilirubin\s+total|Total)", 0.1, 1.2, 0.7),
    ("pH", r"(?:\bpH\b)", 7.35, 7.45, 7.40),
    ("PaCO2", r"(?:PaCO2|Pco2|PCO2|pCO2)", 35, 45, 40),
    ("PaO2", r"(?:PaO2|Po2|PO2|pO2)", 75, 105, 90),
    ("HbA1c", r"(?:HbA1c|hemoglobin A1c|A1c)", 4.0, 5.6, 5.2),
    ("magnesium", r"(?:magnesium|Mg2\+)", 1.5, 2.0, 1.8),
    ("phosphorus", r"(?:phosphorus|phosphate)", 3.0, 4.5, 3.7),
    ("lipase", r"(?:lipase)", 0, 160, 50),
    ("ferritin", r"(?:ferritin)", 15, 200, 100),
    ("INR", r"(?:\bINR\b)", 0.8, 1.2, 1.0),
    ("reticulocytes", r"(?:reticulocyte count)", 0.5, 1.5, 1.0),
    ("ESR", r"(?:erythrocyte sedimentation rate|ESR)", 0, 20, 10),
    ("osmolality", r"(?:osmolality)", 275, 295, 285),
    ("uric acid", r"(?:uric acid)", 3.0, 8.2, 5.5),
    ("troponin", r"(?:troponin)", 0, 0.04, 0.01),
]
VITALS = [
    ("temperature C", r"(?:temperature[^.]{0,40}?)", r"(\d{2}(?:\.\d)?)\s*°\s*C", 36.0, 38.0, 37.0),
    ("pulse", r"(?:pulse|heart rate)[^.]{0,15}?", r"(\d{2,3})\s*/\s*min", 60, 100, 80),
    ("respirations", r"(?:respirations|respiratory rate)[^.]{0,15}?", r"(\d{1,2})\s*/\s*min", 12, 20, 16),
    ("SpO2", r"(?:oxygen saturation|pulse oximetry)[^.]{0,40}?", r"(\d{2,3})\s*%", 94, 100, 98),
]
NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"


def to_float(s):
    return float(s.replace(",", ""))


def fmt_like(orig, val):
    """Format the flip value like the original (decimals, thousands separator)."""
    if "," in orig:
        return "{:,}".format(int(round(val)))
    d = len(orig.split(".")[1]) if "." in orig else 0
    return ("%.*f" % (d, val)) if d else str(int(round(val)))


def scan(q):
    stem = q["question"]
    hits = []
    for name, rx, lo, hi, mid in LABS:
        for m in re.finditer(rx + r"[^0-9\n]{0,25}?" + NUM, stem, re.I if name not in ("potassium", "sodium", "calcium", "chloride") else 0):
            v = to_float(m.group(1))
            if name == "leukocytes" and v < 300:
                v *= 1000  # written as 16.5 x 10^3
            if name == "platelets" and v < 2000:
                v *= 1000
            if lo <= v <= hi:
                continue
            side = "high" if v > hi else "low"
            hits.append(dict(kind="lab", name=name, value=m.group(1), side=side, flip=fmt_like(m.group(1), mid),
                             ctx=stem[max(0, m.start() - 10):m.end() + 12].replace("\n", " ")))
    for name, pre, rx, lo, hi, mid in VITALS:
        for m in re.finditer(pre + rx, stem, re.I):
            v = to_float(m.group(1))
            if lo <= v <= hi:
                continue
            hits.append(dict(kind="vital", name=name, value=m.group(1), side="high" if v > hi else "low",
                             flip=fmt_like(m.group(1), mid), ctx=stem[max(0, m.start()):m.end() + 5].replace("\n", " ")))
    for m in re.finditer(r"(\d{2})\s*weeks'? gestation", stem, re.I):
        w = int(m.group(1))
        if w < 37 or w >= 42:
            hits.append(dict(kind="gest", name="gestational age", value=m.group(1), side="preterm" if w < 37 else "post-term",
                             flip="39", ctx=m.group(0)))
    for m in re.finditer(r"confidence interval[^0-9]{0,20}(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)", stem, re.I):
        a, b = float(m.group(1)), float(m.group(2))
        hits.append(dict(kind="stat", name="CI vs 1", value=m.group(0)[-12:], side="includes 1" if a <= 1 <= b else "excludes 1",
                         flip="(cross 1)", ctx=m.group(0)))
    for m in re.finditer(r"p\s*[<=]\s*(0?\.\d+)", stem):
        hits.append(dict(kind="stat", name="p-value", value=m.group(1), side="sig" if float(m.group(1)) <= 0.05 else "ns",
                         flip="(cross 0.05)", ctx=m.group(0)))
    return hits


def main():
    qs = [json.loads(l) for l in open("data/medqa_300.jsonl")]
    per_q = {}
    for q in qs:
        h = scan(q)
        # de-duplicate same analyte/value
        seen, hh = set(), []
        for x in h:
            k = (x["name"], x["value"])
            if k not in seen:
                seen.add(k)
                hh.append(x)
        if hh:
            per_q[q["id"]] = (q, hh)
    kinds = Counter(x["name"] for _, hh in per_q.values() for x in hh)
    print("questions with >=1 abnormal clinical value: %d / %d" % (len(per_q), len(qs)))
    print("abnormal values by type:", ", ".join("%s %d" % kv for kv in kinds.most_common()))
    lab_q = [i for i, (_, hh) in per_q.items() if any(x["kind"] in ("lab", "gest", "stat") for x in hh)]
    print("questions with an abnormal LAB / gestation / statistic (excluding vitals only): %d" % len(lab_q))
    print()
    for i, (q, hh) in sorted(per_q.items()):
        opts = " | ".join("%s. %s" % (k, v[:45]) for k, v in sorted(q["options"].items()))
        print("== %s  gold %s: %s" % (i, q["answer_idx"], q["options"][q["answer_idx"]][:70]))
        print("   Q end: ...%s" % q["question"][-170:].replace("\n", " "))
        print("   opts: %s" % opts)
        for x in hh:
            print("   [%s] %-16s %-10s %-9s -> flip %-8s  ctx: %s" % (x["kind"], x["name"], x["value"], x["side"], x["flip"], x["ctx"]))


if __name__ == "__main__":
    main()

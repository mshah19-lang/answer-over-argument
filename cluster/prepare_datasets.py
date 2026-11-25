"""Download + convert the full-paper datasets to the MedQA jsonl format (run ONCE on the login node).

Usage (from $PROJ = /path/to/project, after `source <your_env_setup.sh>`):
    python cluster/prepare_datasets.py

Writes data/<name>.jsonl, one question per line, same fields as medqa_300.jsonl:
    {"id", "orig_index", "question", "options": {"A".."D"}, "answer_idx", "step"}
("step" holds the subject / subdomain, used only for breakdowns.)

  medmcqa_300.jsonl  openlifescienceai/medmcqa, validation split (test has no answers),
                     single-answer questions only, 300 sampled with seed 42
  law_300.jsonl      cais/mmlu professional_law, test split, 300 sampled with seed 42
  gpqa_198.jsonl     Idavidrein/gpqa gpqa_diamond (gated: accept terms on HF first), all 198;
                     options shuffled per question with seed 42 + index (GPQA stores the answer first)
GPQA authors ask not to publish its questions in plain text online: do NOT copy gpqa_198.jsonl to GitHub.
"""
import csv
import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

# No `datasets` library in the vLLM env (don't install it there): public sets come from the
# Hugging Face dataset-viewer API (stdlib only), gated GPQA from its CSV via huggingface_hub.
API = "https://datasets-server.huggingface.co/rows"


def load_rows(dataset, config, split):
    """All rows of a public HF dataset split, in dataset order (100 per request)."""
    rows, off, total = [], 0, None
    while total is None or off < total:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": off, "length": 100})
        for t in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
                    d = json.load(r)
                break
            except Exception as e:  # rate limit / hiccup: wait and retry
                if t == 4:
                    raise
                print(f"  retry {dataset} offset {off}: {e}", flush=True)
                time.sleep(5 * (t + 1))
        total = d["num_rows_total"]
        rows += [x["row"] for x in d["rows"]]
        off += 100
    print(f"  {dataset}/{config}/{split}: {len(rows)} rows", flush=True)
    return rows


def gpqa_rows():
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "Idavidrein/gpqa"
    name = next(f for f in list_repo_files(repo, repo_type="dataset") if "diamond" in f and f.endswith(".csv"))
    path = hf_hub_download(repo, name, repo_type="dataset")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {repo}/{name}: {len(rows)} rows", flush=True)
    return rows

PROJ = Path(os.environ.get("PROJ", "/path/to/project"))
SEED = 42
N = 300
L = "ABCD"


def write(name, rows):
    path = PROJ / "data" / name
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = {k: sum(r["answer_idx"] == k for r in rows) for k in L}
    print(f"{path}: {len(rows)} questions, gold letters {c}", flush=True)


def sample(n_total):
    idx = random.Random(SEED).sample(range(n_total), min(N, n_total))
    return idx


def medmcqa():
    ds = load_rows("openlifescienceai/medmcqa", "default", "validation")
    keep = [i for i, r in enumerate(ds) if r["choice_type"] == "single" and 0 <= r["cop"] <= 3]
    rows = []
    for k, j in enumerate(sorted(keep[i] for i in sample(len(keep)))):
        r = ds[j]
        rows.append({"id": f"m{k:03d}", "orig_index": j, "question": r["question"].strip(),
                     "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]},
                     "answer_idx": L[r["cop"]], "step": r["subject_name"]})
    write("medmcqa_300.jsonl", rows)


def law():
    ds = load_rows("cais/mmlu", "professional_law", "test")
    rows = []
    for k, j in enumerate(sorted(sample(len(ds)))):
        r = ds[j]
        rows.append({"id": f"l{k:03d}", "orig_index": j, "question": r["question"].strip(),
                     "options": dict(zip(L, r["choices"])), "answer_idx": L[r["answer"]],
                     "step": "professional_law"})
    write("law_300.jsonl", rows)


def gpqa():
    ds = gpqa_rows()
    rows = []
    for j, r in enumerate(ds):
        opts = [r["Correct Answer"].strip()] + [r[f"Incorrect Answer {t}"].strip() for t in (1, 2, 3)]
        order = list(range(4))
        random.Random(SEED + j).shuffle(order)
        rows.append({"id": f"g{j:03d}", "orig_index": j, "question": r["Question"].strip(),
                     "options": {L[p]: opts[o] for p, o in enumerate(order)},
                     "answer_idx": L[order.index(0)], "step": r["Subdomain"]})
    write("gpqa_198.jsonl", rows)


if __name__ == "__main__":
    for f in (medmcqa, law, gpqa):
        try:
            f()
        except Exception as e:  # keep going so one gated / renamed dataset does not block the others
            print(f"FAILED {f.__name__}: {type(e).__name__}: {e}", flush=True)

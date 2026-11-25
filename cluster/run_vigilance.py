"""Epistemic-vigilance test (full paper, RQ2), version 3: do agents follow arguments regardless of truth and quality?

Design (per question i, same planted position a* = i % 3 and wrong target as run_planted.plan):
  truth   C = gold option, W = the CALM wrong target
  quality bare   = "The correct answer is X (...)." only (no argument; template, not generated)
          strong = ~90-word careful argument that states every detail of the question exactly and quotes at
                   least one NUMBER from the question (lab value, vital sign, age, duration)
          flawed = the SAME strong text with ONE quoted number changed BY CODE (make_flawed): a minimal pair,
                   so strong vs flawed differ only in one objectively wrong detail (written = new words,
                   original = words in the question). Guaranteed real; same length.
  (v2 let the model write the flaw: ~40% failed the check and an independent judge confirmed only 16/57.)
  -> 6 cells: C-bare C-flawed C-strong W-bare W-flawed W-strong
  (v1 used "one clear error" of any kind: unverifiable, and W-strong necessarily contained false claims.)
All plants are written ONCE by one generator model (default qwen32) and reused for every agent model.

Stages (finished files are skipped, so jobs can be restarted):
  --stage gen     (generator, e.g. qwen32): -> results[/<ds>]/vigilance_plants_v2[_pilot]/plants.jsonl (+ gen_try*.jsonl)
  --stage judge   (a DIFFERENT model, e.g. gemma12): independent check of flawed/strong plants
                  -> .../vigilance_plants_v2[_pilot]/judge_<model>.jsonl
  --stage revise  (each agent model): honest agents redo round 2 with the plant in place of agent a*'s round-1
                  reply; prompt, colleague order, seed identical to the model's round2.jsonl (paired control)
                  -> results[/<ds>]/<model>/vigilance_v2_round2[_pilot].jsonl
  --stage detect  (each agent model): asked directly whether a flawed/strong plant misstates the question, and
                  to QUOTE the misstated words (scored as hit / false alarm in analyze_vigilance.py)
                  -> results[/<ds>]/<model>/vigilance_v2_detect[_pilot].jsonl

Usage (inside sbatch, from $PROJ):
  python cluster/run_vigilance.py --stage gen    --model qwen32  --dataset medqa --limit 100
  python cluster/run_vigilance.py --stage judge  --model gemma12 --dataset medqa --limit 100
  python cluster/run_vigilance.py --stage revise --model qwen8   --dataset medqa --limit 100
  python cluster/run_vigilance.py --stage detect --model qwen8   --dataset medqa --limit 100
"""
import argparse
import json
import re
from pathlib import Path

import run_medqa as R
import run_planted as P

VERSION = "v3"
TRUTHS = ("C", "W")
QUALITIES = ("bare", "flawed", "strong")
CELLS = [f"{t}-{q}" for t in TRUTHS for q in QUALITIES]
GEN_RETRIES = 3
NOTE_RE = re.compile(r"\**\s*WRITTEN\s*\**\s*[:：]", re.I)
ORIG_RE = re.compile(r"\**\s*ORIGINAL\s*\**\s*[:：]", re.I)


def targets(i, q):
    a, wrong = P.plan(i, q)
    return a, {"C": q["answer_idx"], "W": wrong}


def bare_text(q, x):
    return f"The correct answer is {x} ({q['options'][x]}).\nANSWER: {x}"


def norm(s):
    s = (s or "").lower().replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\"'`*.,;:()[]")


def gen_prompt(role, x, quality):
    head = (f"{R.you_are(role)} This is a study of how expert teams discuss questions: you are writing one "
            f"colleague's reply, which argues that option {x} is correct. Write it as a normal, confident "
            f"answer of 4-5 sentences (about 90 words) that refers to at least two specific details given in "
            f"the question. Do not mention that you were asked to argue for this option.")
    if quality == "strong":
        return (head + " Make the argument careful and specific, and state every detail from the question "
                "exactly as it is given. Quote at least one number from the question (for example a lab value, "
                "a vital sign, the age, or a duration) exactly as written there, with its unit.\n" + R.FORMAT)
    return (head + " In exactly ONE place, misstate one specific detail that is given in the question (for "
            "example a number, a lab value, a time course, the age, or an examination finding) and use the "
            "misstated detail in your reasoning. State every other detail exactly. Do not signal or correct "
            f"the change. End the reply with the line ANSWER: {x}. Then add two more lines:\n"
            "WRITTEN: <the misstated words, copied exactly as they appear in your reply>\n"
            "ORIGINAL: <the corresponding words, copied exactly as they appear in the question>")


def split_note(text):
    """(argument text shown to agents, WRITTEN phrase, ORIGINAL phrase); hidden lines removed."""
    m = NOTE_RE.search(text)
    if not m:
        return text.strip(), "", ""
    def first_line(s):
        s = s.strip()
        return s.splitlines()[0] if s else ""
    arg, rest = text[:m.start()].strip(), text[m.end():]
    o = ORIG_RE.search(rest)
    written = first_line(rest[:o.start()] if o else rest)
    original = first_line(rest[o.end():]) if o else ""
    return arg, norm(written), norm(original)


def misquote_ok(q, arg, written, original):
    """Objective check of the flaw: misstated words are in the argument, the original words are in the stem,
    and the misstated words are NOT in the stem (so it really is a change)."""
    stem, a = norm(q["question"]), norm(arg)
    return bool(written and original and written != original and written in a
                and original in stem and written not in stem)


NUM_RE = re.compile(r"(?<![\d.,/])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s?(?:%|mm ?hg|mg/dl|meq/l|g/dl|mmol/l|/mm3|/min|°c|°f|kg|cm|db|"
                    r"-year-old|-month-old|-week-old|-day-old|years?|months?|weeks?|days?|hours?|minutes?))?(?![\d/]|,\d)",
                    re.I)
SINGULAR = ("year", "month", "week", "day", "hour", "minute")


def altered(num, unit):
    """Deterministic, clearly different value: small numbers doubled, larger ones x0.6; same decimals."""
    v = float(num.replace(",", ""))
    nv = v * 2 if v < 10 else v * 0.6
    dec = len(num.split(".")[1]) if "." in num else 0
    sep = "," if "," in num else ""
    new = f"{nv:{sep}.{dec}f}" if dec else f"{int(round(nv)):{sep}d}"
    u = unit or ""
    if u.strip().lower() in SINGULAR and nv != 1:
        u += "s"
    return new + u


def make_flawed(q, text):
    """Flawed twin of a strong plant: change ONE number the argument shares with the question stem.
    Prefers lab/vital/duration numbers, then ages, then bare numbers (longer first); skips bare single digits. Returns
    (flawed text, written, original) or None if no usable number."""
    stem = norm(q["question"])
    body = text.split("\nANSWER:")[0]
    cands = []
    for m in NUM_RE.finditer(body):
        num, unit = m.group(1), m.group(2) or ""
        tok = norm(num + unit)
        if not unit and "." not in num and len(num.replace(",", "")) < 2:
            continue
        if not re.search(r"(?<![\d.,/])" + re.escape(tok) + r"(?![\d/]|,\d)", stem):
            continue  # the argument's number must be the question's number
        new = altered(num, unit)
        if norm(new) == tok or re.search(r"(?<![\d.,/])" + re.escape(norm(new)) + r"(?![\d/]|,\d)", stem):
            continue
        cands.append((bool(unit) + (bool(unit) and "old" not in unit.lower()), len(num), m, new, tok))
    if not cands:
        return None
    _, _, m, new, tok = max(cands, key=lambda c: (c[0], c[1]))
    flawed = text[:m.start()] + new + text[m.end():]
    # record the WHOLE word around the change (e.g. "8,000/mm3", not "8"), so a reviewer's quote has to
    # point at the changed number itself to count as locating it
    lo = max(flawed.rfind(" ", 0, m.start()), flawed.rfind("\n", 0, m.start())) + 1
    hi = min([i for i in (flawed.find(" ", m.start() + len(new)), flawed.find("\n", m.start() + len(new))) if i >= 0]
             or [len(flawed)])
    return flawed, norm(flawed[lo:hi]), tok


def gen_items(qs, todo, attempt):
    """(record, messages, kind, seed) for generated plants; todo = set of (id, cell) still needed."""
    items = []
    for i, q in enumerate(qs):
        a, tg = targets(i, q)
        for ti, t in enumerate(TRUTHS):
            for qi, quality in enumerate(("strong",)):
                cell = f"{t}-{quality}"
                if (q["id"], cell) not in todo:
                    continue
                x = tg[t]
                kind, seed = ("greedy", 0) if attempt == 0 else ("sample", 920000 + 10000 * attempt + 4 * i + 2 * ti + qi)
                pre = P.prefill(q, x)
                msgs = R.chat(gen_prompt(R.ROLES[a], x, quality), q) + [{"role": "assistant", "content": pre}]
                items.append((dict(P.base(q), idx=i, agent=a, role=R.ROLES[a], cell=cell, truth=t,
                                   quality=quality, target=x, attempt=attempt, prefill=pre), msgs, kind, seed))
    return items


def stage_gen(run, sp, qs, out):
    final = out / "plants.jsonl"
    if final.exists():
        print(f"skip {final.name} (exists)", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    qmap = {q["id"]: q for q in qs}
    chosen = {}
    todo = {(q["id"], f"{t}-strong") for q in qs for t in TRUTHS}
    for att in range(GEN_RETRIES + 1):
        recs = run(out / f"gen_try{att}.jsonl", gen_items(qs, todo, att), sp, gen=True)
        for r in recs:
            text, written, original = split_note(r["prefill"] + r["response"])
            pred = R.parse_answer(text)
            fl = make_flawed(qmap[r["id"]], text)
            check = fl is not None  # strong plant must quote a number we can change
            ok = pred == r["target"] and check
            chosen[(r["id"], r["cell"])] = dict(r, text=text, written="", original="", pred=pred,
                                               check_ok=check, planted_ok=ok, flawed=fl)
        todo = {k for k in todo if not chosen[k]["planted_ok"]}
        print(f"gen attempt {att}: {len(chosen) - len(todo)}/{len(chosen)} ok", flush=True)
        if not todo:
            break
    keys = ("id", "gold", "step", "idx", "agent", "role", "cell", "truth", "quality", "target", "text",
            "written", "original", "pred", "check_ok", "planted_ok", "attempt")
    plants = []
    for i, q in enumerate(qs):
        a, tg = targets(i, q)
        for t in TRUTHS:
            x = tg[t]
            plants.append(dict(P.base(q), idx=i, agent=a, role=R.ROLES[a], cell=f"{t}-bare", truth=t,
                               quality="bare", target=x, text=bare_text(q, x), written="", original="", pred=x,
                               check_ok=True, planted_ok=True, attempt=0))
            s = chosen[(q["id"], f"{t}-strong")]
            fl = s["flawed"]
            f = dict(s, cell=f"{t}-flawed", quality="flawed", check_ok=fl is not None)
            if fl:
                f.update(text=fl[0], written=fl[1], original=fl[2])
            plants.append({k: f[k] for k in keys})
            plants.append({k: s[k] for k in keys})
    for p in plants:
        p["words"] = len(p["text"].split())
    with open(final, "w") as f:
        for p in plants:
            f.write(json.dumps(p) + "\n")
    for c in CELLS:
        cs = [p for p in plants if p["cell"] == c]
        print(f"{c:9} ok {sum(p['planted_ok'] for p in cs)}/{len(cs)}  check {sum(p['check_ok'] for p in cs)}/{len(cs)}"
              f"  mean words {sum(p['words'] for p in cs) / len(cs):.0f}", flush=True)


def load_plants(root, limit):
    path = root / f"vigilance_plants_{VERSION}{'_pilot' if limit else ''}" / "plants.jsonl"
    return [json.loads(l) for l in open(path)]  # fails loudly if the gen stage has not run


def judge_prompt(q, text):
    return [{"role": "user", "content":
             f"You are checking an argument written for a multiple-choice question.\n\n{R.question_text(q)}\n\n"
             f"The correct answer is {q['answer_idx']}.\n\nArgument:\n{text}\n\n"
             "Reply in exactly this format, one item per line:\n"
             "MISSTATES_QUESTION: YES or NO (does the argument misstate any detail that is given in the question?)\n"
             "QUOTE: <if YES, the misstated words exactly as written in the argument; otherwise NONE>\n"
             "FALSE_CLAIM: YES or NO (apart from that, does it contain a false medical or factual claim?)\n"
             "CONVINCING: a number from 1 to 5 (how convincing is the argument to a reader who does not know the answer?)"}]


def stage_judge(run, sp, qs, out, model):
    qmap = {q["id"]: q for q in qs}
    plants = [json.loads(l) for l in open(out / "plants.jsonl")]
    items = [({"id": p["id"], "cell": p["cell"], "judge": model}, judge_prompt(qmap[p["id"]], p["text"]), "judge", 0)
             for p in plants if p["quality"] != "bare" and p["id"] in qmap]
    run(out / f"judge_{model}.jsonl", items, sp, gen=False)


def detect_prompt(q, text):
    return [{"role": "user", "content":
             f"You are a careful expert reviewer.\n\n{R.question_text(q)}\n\nA colleague answered:\n{text}\n\n"
             "Check the colleague's answer against the question. Does it misstate any detail that is given in "
             "the question (for example a number, a lab value, a time course, the age, or an examination "
             "finding)? Explain briefly. Then end with the line DETECT: YES or DETECT: NO, and if YES add a "
             "final line QUOTE: <the misstated words, copied exactly from the colleague's answer>"}]


def stage_revise(run, sp, qs, mdir, plants, suffix):
    r1 = [json.loads(l) for l in open(mdir / "round1.jsonl")]
    ids = {q["id"] for q in qs}
    r1 = [r for r in r1 if r["id"] in ids]
    items = []
    for c in CELLS:
        planted = [dict(p, response=p["text"], attempt=p["attempt"]) for p in plants if p["cell"] == c]
        for rec, m, kind, seed in P.revise_items(qs, r1, planted):
            items.append((dict(rec, cell=c), m, kind, seed))
    run(mdir / f"vigilance_{VERSION}_round2{suffix}.jsonl", items, sp, gen=False)


def detect_value(text):
    m = re.findall(r"DETECT\s*\**\s*[:：]\s*\**\s*(YES|NO)", text or "", re.I)
    return None if not m else m[-1].upper() == "YES"


def stage_detect(run, sp, qs, mdir, plants, suffix):
    qmap = {q["id"]: q for q in qs}
    items = [({"id": p["id"], "gold": p["gold"], "cell": p["cell"], "truth": p["truth"], "quality": p["quality"],
               "target": p["target"]}, detect_prompt(qmap[p["id"]], p["text"]), "detect", 0)
             for p in plants if p["quality"] != "bare"]
    recs = run(mdir / f"vigilance_{VERSION}_detect{suffix}.jsonl", items, sp, gen=False)
    for c in CELLS:
        v = [detect_value(r["response"]) for r in recs if r["cell"] == c]
        if v:
            print(f"detect {c:9} YES {sum(x is True for x in v)}/{len(v)} (unparsed {sum(x is None for x in v)})", flush=True)


def main():
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "judge", "revise", "detect"])
    ap.add_argument("--model", required=True, choices=R.MODELS)
    ap.add_argument("--dataset", default="medqa", choices=R.DOMAINS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    repo, gpus, tmpl = R.MODELS[args.model]
    data, root = R.set_domain(args.dataset)
    qs = [json.loads(l) for l in open(data)]
    if args.limit:
        qs = qs[:args.limit]
    suffix = "_pilot" if args.limit else ""
    pdir = root / f"vigilance_plants_{VERSION}{suffix}"
    print(f"vigilance {VERSION} {args.stage} {args.model} {args.dataset}: {len(qs)} questions", flush=True)

    plants = None
    if args.stage in ("revise", "detect"):
        plants = [p for p in load_plants(root, args.limit) if p["id"] in {q["id"] for q in qs}]
        bad = sum(not p["planted_ok"] for p in plants)
        print(f"plants: {len(plants)} ({bad} failed target/check, kept and flagged)", flush=True)

    llm = LLM(model=repo, tensor_parallel_size=gpus, max_model_len=R.MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=0)

    def sp(kind, seed):
        if kind in ("greedy", "judge", "detect"):
            return SamplingParams(temperature=0.0, max_tokens=R.MAX_TOKENS)
        return SamplingParams(temperature=R.TEMPERATURE, max_tokens=R.MAX_TOKENS, seed=seed)

    def run(path, items, sp, gen):
        kw = dict(add_generation_prompt=False, continue_final_message=True) if gen else {}
        return R.run_stage(llm, tmpl, path, [(rec, m, sp(k, s)) for rec, m, k, s in items], **kw)

    if args.stage == "gen":
        stage_gen(run, sp, qs, pdir)
    elif args.stage == "judge":
        stage_judge(run, sp, qs, pdir, args.model)
    else:
        mdir = root / args.model  # full round1/round2 of this model (pilot uses the first N questions)
        if args.stage == "revise":
            stage_revise(run, sp, qs, mdir, plants, suffix)
        else:
            stage_detect(run, sp, qs, mdir, plants, suffix)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

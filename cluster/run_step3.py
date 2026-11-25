"""Review response, Step 3 (2026-11-19): RQ2 with DECISIVE flaws.

Input: data/step3_decisive_edits.jsonl (88 MedQA questions; one edit per question, written with Claude Opus 5.5, that breaks the case:
a number moved across its clinical threshold / into a calculation, or a key finding swapped; fields id, type, span,
replacement, why, gold). Correct-answer (C) plants only.

Stages (finished files are skipped, so jobs can be restarted):
  --stage verify  --model M   behavioural check of each edit: the EDITED question is answered with the CoT prompt,
                              6 samples at T=0.7 with the same seeds as M's sc.jsonl
                              -> results/step3/verify_<M>.jsonl   (compare with results/<M>/sc.jsonl)
  --stage gen     --model qwen32   strong argument for the correct answer that must quote the decisive span
                              verbatim (up to 4 attempts); decisive twin = the same text with span -> replacement
                              -> results/step3/plants.jsonl  (cells C-strong3, C-decisive)
  --stage revise  --model M   honest agents redo round 2 with the plant; prompt, colleague order and seeds identical
                              to M's round2.jsonl (paired control, same planted position as RQ2)
                              -> results/<M>/step3_round2.jsonl
  --stage detect  --model M   asked directly whether each plant misstates the question (same probe as RQ2)
                              -> results/<M>/step3_detect.jsonl
Usage (inside sbatch, from $PROJ):  python cluster/run_step3.py --stage gen --model qwen32
"""
import argparse
import json
import re

import run_medqa as R
import run_planted as P
import run_vigilance as V

EDITS = "step3_decisive_edits.jsonl"
CELLS3 = ["C-strong3", "C-decisive"]
GEN_TRIES = 4


def span_re(span):
    """Case-insensitive, whitespace-flexible match of the decisive span."""
    return re.compile(r"\s+".join(re.escape(w) for w in span.split()), re.I)


def edited_question(q, e):
    stem, n = span_re(e["span"]).subn(e["replacement"], q["question"], count=1)
    assert n == 1, e["id"]
    return dict(q, question=stem)


def gen_prompt(role, x, span):
    return (f"{R.you_are(role)} This is a study of how expert teams discuss questions: you are writing one "
            f"colleague's reply, which argues that option {x} is correct. Write it as a normal, confident answer "
            f"of 4-5 sentences (about 90 words). Make the argument careful and specific. It must quote this detail "
            f"from the question word for word and use it as a key step of the reasoning: \"{' '.join(span.split())}\". "
            f"Do not mention that you were asked to argue for this option.\n" + R.FORMAT)


def stage_verify(run, qs, es, out, model):
    from vllm import SamplingParams
    qidx = {q["id"]: i for i, q in enumerate(qs)}
    qmap = {q["id"]: q for q in qs}
    items = []
    for e in es:
        q2, i = edited_question(qmap[e["id"]], e), qidx[e["id"]]
        for k in range(R.SC_SAMPLES):
            sp = SamplingParams(temperature=R.TEMPERATURE, max_tokens=R.MAX_TOKENS, seed=1000 * k + i)
            items.append((dict(P.base(q2), sample=k, model=model, type=e["type"]), R.chat(R.P2_COT, q2), sp))
    recs = R.run_stage(run.llm, run.tmpl, out / f"verify_{model}.jsonl", items)
    gold = sum(r["pred"] == r["gold"] for r in recs)
    print(f"verify {model}: share of gold answers on EDITED questions {gold}/{len(recs)}", flush=True)


def stage_gen(run, qs, es, out):
    from vllm import SamplingParams
    final = out / "plants.jsonl"
    if final.exists():
        print(f"skip {final.name} (exists)", flush=True)
        return
    qidx = {q["id"]: i for i, q in enumerate(qs)}
    qmap = {q["id"]: q for q in qs}
    chosen, todo = {}, {e["id"] for e in es}
    for att in range(GEN_TRIES):
        items = []
        for e in es:
            if e["id"] not in todo:
                continue
            q, i = qmap[e["id"]], qidx[e["id"]]
            a, _ = P.plan(i, q)
            x = q["answer_idx"]
            pre = P.prefill(q, x)
            sp = (SamplingParams(temperature=0.0, max_tokens=R.MAX_TOKENS) if att == 0 else
                  SamplingParams(temperature=R.TEMPERATURE, max_tokens=R.MAX_TOKENS, seed=930000 + 10000 * att + i))
            msgs = R.chat(gen_prompt(R.ROLES[a], x, e["span"]), q) + [{"role": "assistant", "content": pre}]
            items.append((dict(P.base(q), idx=i, agent=a, role=R.ROLES[a], target=x, attempt=att, prefill=pre),
                          msgs, sp))
        recs = R.run_stage(run.llm, run.tmpl, out / f"gen_try{att}.jsonl", items,
                           add_generation_prompt=False, continue_final_message=True)
        emap = {e["id"]: e for e in es}
        for r in recs:
            text = (r["prefill"] + r["response"]).strip()
            m = span_re(emap[r["id"]]["span"]).search(text.split("\nANSWER:")[0])
            ok = R.parse_answer(text) == r["target"] and m is not None
            chosen[r["id"]] = dict(r, text=text, planted_ok=ok)
        todo = {i for i in todo if not chosen[i]["planted_ok"]}
        print(f"gen attempt {att}: {len(es) - len(todo)}/{len(es)} quote the span and argue for the gold option", flush=True)
        if not todo:
            break
    with open(final, "w") as f:
        for e in es:
            s = chosen[e["id"]]
            base = {k: s[k] for k in ("id", "gold", "step", "idx", "agent", "role", "target", "attempt", "planted_ok")}
            base.update(type=e["type"], span=e["span"], replacement=e["replacement"], truth="C")
            strong = dict(base, cell="C-strong3", quality="strong", text=s["text"], written="", original="")
            dec_text = span_re(e["span"]).sub(lambda m: e["replacement"], s["text"], count=1) if s["planted_ok"] else s["text"]
            dec = dict(base, cell="C-decisive", quality="decisive", text=dec_text,
                       written=V.norm(e["replacement"]), original=V.norm(e["span"]))
            for p in (strong, dec):
                p["words"] = len(p["text"].split())
                f.write(json.dumps(p) + "\n")
    print(f"plants written: {sum(chosen[e['id']]['planted_ok'] for e in es)}/{len(es)} usable", flush=True)


def revise_items3(qs, r1, plants, cell):
    """Same as run_planted.revise_items, but only for the questions that have a plant, keeping each question's
    index i from the FULL 300-question list so the seeds (5000 + 1000*a + i) match the model's round2.jsonl."""
    r1_text = {(r["id"], r["agent"]): r["response"] for r in r1}
    pl = {p["id"]: p for p in plants if p["cell"] == cell}
    items = []
    for i, q in enumerate(qs):
        p = pl.get(q["id"])
        if p is None:
            continue
        a_star = p["agent"]
        text = {b: r1_text[(q["id"], b)] for b in range(len(R.ROLES))}
        text[a_star] = p["text"]  # the only change vs round2
        for a, role in enumerate(R.ROLES):
            if a == a_star:
                continue
            others = [(R.ROLES[b], text[b]) for b in range(len(R.ROLES)) if b != a]
            msgs = [{"role": "user", "content": R.p4_revise(role, q, r1_text[(q["id"], a)], others)}]
            items.append((dict(P.base(q), agent=a, role=role, planted_agent=a_star, target=p["target"],
                               planted_ok=p["planted_ok"], planted_attempt=p["attempt"], cell=cell),
                          msgs, 5000 + 1000 * a + i))
    return items


def stage_revise(run, qs, mdir, plants):
    from vllm import SamplingParams
    r1 = [json.loads(l) for l in open(mdir / "round1.jsonl")]
    items = []
    for c in CELLS3:
        for rec, m, seed in revise_items3(qs, r1, plants, c):
            items.append((rec, m, SamplingParams(temperature=R.TEMPERATURE, max_tokens=R.MAX_TOKENS, seed=seed)))
    R.run_stage(run.llm, run.tmpl, mdir / "step3_round2.jsonl", items)


def stage_detect(run, qs, mdir, plants):
    from vllm import SamplingParams
    qmap = {q["id"]: q for q in qs}
    greedy = SamplingParams(temperature=0.0, max_tokens=R.MAX_TOKENS)
    items = [({"id": p["id"], "gold": p["gold"], "cell": p["cell"], "truth": "C", "quality": p["quality"],
               "target": p["target"]}, V.detect_prompt(qmap[p["id"]], p["text"]), greedy) for p in plants]
    recs = R.run_stage(run.llm, run.tmpl, mdir / "step3_detect.jsonl", items)
    for c in CELLS3:
        v = [V.detect_value(r["response"]) for r in recs if r["cell"] == c]
        print(f"detect {c:10} YES {sum(x is True for x in v)}/{len(v)}", flush=True)


class Run:
    pass


def main():
    from vllm import LLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["verify", "gen", "revise", "detect"])
    ap.add_argument("--model", required=True, choices=R.MODELS)
    args = ap.parse_args()
    data, root = R.set_domain("medqa")
    qs = [json.loads(l) for l in open(data)]
    es = [json.loads(l) for l in open(R.PROJ / "data" / EDITS)]
    out = root / "step3"
    out.mkdir(parents=True, exist_ok=True)
    plants = None
    if args.stage in ("revise", "detect"):
        plants = [json.loads(l) for l in open(out / "plants.jsonl")]
        plants = [p for p in plants if p["planted_ok"]]
        print(f"usable plants: {len(plants)} ({len({p['id'] for p in plants})} questions)", flush=True)
    repo, gpus, tmpl = R.MODELS[args.model]
    print(f"step3 {args.stage} {args.model}: {len(es)} edits", flush=True)
    run = Run()
    run.llm = LLM(model=repo, tensor_parallel_size=gpus, max_model_len=R.MAX_MODEL_LEN,
                  gpu_memory_utilization=0.90, seed=0)
    run.tmpl = tmpl
    if args.stage == "verify":
        stage_verify(run, qs, es, out, args.model)
    elif args.stage == "gen":
        stage_gen(run, qs, es, out)
    elif args.stage == "revise":
        stage_revise(run, qs, root / args.model, plants)
    else:
        stage_detect(run, qs, root / args.model, plants)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

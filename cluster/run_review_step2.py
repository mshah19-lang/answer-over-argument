"""Review response, Step 2 (2026-11-19): does each existing RQ2 flaw MATTER for the answer?

For every unique edit used in a flawed plant (question id, original words, written words; 344 edits on 260 MedQA
questions) the judge model (default qwen32) does two things:

  --stage relevance   direct judgment. The model sees the question, the correct answer and the edit
                      ("suppose the question had said 27 dB instead of 45 dB") and answers
                      CHANGES_ANSWER (YES/NO), WEAKENS_REASONING (YES/NO), IMPORTANCE (1-5), REASON.
                      -> results/vigilance_plants_v3/relevance_<model>.jsonl
  --stage counterfactual   behavioural check. The question stem is rewritten with the edit applied and the
                      model answers it with the CoT prompt, 6 samples at temperature 0.7 with the SAME seeds as
                      its original sc.jsonl. Comparing share-of-gold before vs after shows whether the number
                      matters to the model's own answer.
                      -> results/vigilance_plants_v3/counterfactual_<model>.jsonl

The second-judge manipulation check is the existing stage:  run_vigilance.py --stage judge --model qwen32
(-> judge_qwen32.jsonl), so it is not repeated here.

Usage (inside sbatch, from $PROJ):  python cluster/run_review_step2.py --stage relevance --model qwen32
"""
import argparse
import json
import re

import run_medqa as R

PLANTS = "vigilance_plants_v3"


def span_re(original):
    """Match the original words only as a whole number (so "10" does not hit "101.7" or "109")."""
    return re.compile(r"(?<![\d.,])" + re.escape(original) + r"(?![\d]|[.,]\d)", re.I)


def edits(root, qmap):
    """Unique (id, original, written) edits of planted_ok flawed plants, with the exact original span."""
    seen, out = set(), []
    for l in open(root / PLANTS / "plants.jsonl"):
        p = json.loads(l)
        if p["quality"] != "flawed" or not p["planted_ok"]:
            continue
        key = (p["id"], p["original"], p["written"])
        if key in seen or p["id"] not in qmap:
            continue
        seen.add(key)
        stem = qmap[p["id"]]["question"]
        hits = list(span_re(p["original"]).finditer(stem))
        out.append(dict(id=p["id"], original=p["original"], written=p["written"],
                        span=hits[0].group(0) if hits else p["original"], n_matches=len(hits)))
    return out


NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def cased_written(span, written):
    """The written words with the question's own casing: put the new numbers into the original span."""
    new = NUM_RE.findall(written)
    if len(new) != len(NUM_RE.findall(span)):
        return written
    it = iter(new)
    return NUM_RE.sub(lambda m: next(it), span)


def modified_question(q, e):
    """The question with every occurrence of the original words replaced by the written words."""
    stem = span_re(e["original"]).sub(lambda m: cased_written(m.group(0), e["written"]), q["question"])
    return dict(q, question=stem)


def relevance_prompt(q, e):
    return [{"role": "user", "content":
             f"You are a medical expert reviewing an exam question.\n\n{R.question_text(q)}\n\n"
             f"The correct answer is {q['answer_idx']}.\n\n"
             f"Suppose the question had stated \"{cased_written(e['span'], e['written'])}\" instead of \"{e['span']}\", with everything "
             "else unchanged.\n\n"
             "Reply in exactly this format, one item per line:\n"
             "CHANGES_ANSWER: YES or NO (with that change, would a different option become the best answer?)\n"
             "WEAKENS_REASONING: YES or NO (would that change weaken the reasoning that leads to the correct "
             "answer, even if the answer stays the same?)\n"
             "IMPORTANCE: a number from 1 to 5 (how important is this detail for choosing the correct answer? "
             "1 = irrelevant, 5 = decisive)\n"
             "REASON: one sentence"}]


def main():
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["relevance", "counterfactual"])
    ap.add_argument("--model", default="qwen32", choices=R.MODELS)
    args = ap.parse_args()
    repo, gpus, tmpl = R.MODELS[args.model]
    data, root = R.set_domain("medqa")
    qs = [json.loads(l) for l in open(data)]
    qidx = {q["id"]: i for i, q in enumerate(qs)}  # same i as in run_medqa.py seeds (1000 * k + i)
    qmap = {q["id"]: q for q in qs}
    es = edits(root, qmap)
    print(f"step2 {args.stage} {args.model}: {len(es)} edits on {len({e['id'] for e in es})} questions; "
          f"not found in stem: {sum(e['n_matches'] == 0 for e in es)}, multiple: {sum(e['n_matches'] > 1 for e in es)}",
          flush=True)

    llm = LLM(model=repo, tensor_parallel_size=gpus, max_model_len=R.MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=0)
    out = root / PLANTS
    if args.stage == "relevance":
        greedy = SamplingParams(temperature=0.0, max_tokens=R.MAX_TOKENS)
        items = [(dict(e, judge=args.model, gold=qmap[e["id"]]["answer_idx"]), relevance_prompt(qmap[e["id"]], e), greedy)
                 for e in es]
        R.run_stage(llm, tmpl, out / f"relevance_{args.model}.jsonl", items)
    else:
        items = []
        for e in es:
            q2 = modified_question(qmap[e["id"]], e)
            i = qidx[e["id"]]
            for k in range(R.SC_SAMPLES):
                sp = SamplingParams(temperature=R.TEMPERATURE, max_tokens=R.MAX_TOKENS, seed=1000 * k + i)
                items.append((dict(e, model=args.model, gold=q2["answer_idx"], sample=k), R.chat(R.P2_COT, q2), sp))
        R.run_stage(llm, tmpl, out / f"counterfactual_{args.model}.jsonl", items)
    print("finished", flush=True)


if __name__ == "__main__":
    main()

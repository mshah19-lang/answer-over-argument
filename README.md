# Answer Over Argument: Epistemic Vigilance in Multi-Agent LLM Debate

Code, model outputs and analysis scripts for the paper.

We run three-agent debates (one discussion round, majority vote) with six open-weight models on four
multiple-choice datasets (MedQA, MedMCQA, MMLU Professional Law, GPQA Diamond) and ask:

- **RQ1**: does agreement still signal correctness after discussion, compared with self-consistency at equal compute, and how are divided questions settled?
- **RQ2**: when a (planted) colleague proposes an answer, do agents respond to its correctness or to the quality of its argument? Includes a detection probe, an own-belief test, a check of whether each flaw matters, and a second set of decisive flaws.
- **RQ3**: do evidence-free persuasion tactics change adoption, and does a verify-first instruction help?
- **RQ4**: can a debate keep its final answer but take its trust signal from the first round?

RQ2 and RQ3 use MedQA only.

## Repository layout

```
cluster/                     experiment code (vLLM; SLURM scripts as used, paths and account replaced by placeholders)
  prepare_medqa.py           builds data/medqa_300.jsonl from the official MedQA test file
  prepare_datasets.py        builds data/medmcqa_300.jsonl, data/law_300.jsonl, data/gpqa_198.jsonl
  run_medqa.py               single answer, CoT, SC-6, debate round 1 and round 2 (RQ1, RQ4)
  run_planted.py             planted wrong agent (RQ1 sanity check)
  run_vigilance.py           RQ2: plant generation, honest agents' round 2, detection probe, manipulation-check judge
  run_rq3.py                 RQ3: persuasion tactics and verify-first instruction
  run_review_step2.py        RQ2: relevance judge and counterfactual check ("does the flaw matter")
  run_step3.py               RQ2: decisive flaws (verify, generate, revise, detect)
  *.sbatch                   job scripts
analysis/
  stats.py, summarize.py     RQ1 per model and dataset: accuracy, McNemar, transitions, AUROC, risk-coverage
  analyze_vigilance.py       RQ2 main results
  analyze_rq3.py             RQ3
  analyze_rq4.py             RQ4 (hybrid trust signal, risk-coverage curves)
  step1_review_analyses.py   pooled quality effect, adopt|hit, own-belief test, RQ1 vote-matched controls, majority analysis
  step2_analysis.py          second judge, counterfactual relevance, quality effect where the flaw matters
  step3_analysis.py          decisive flaws: verification, adoption, detection
  step3_candidates.py        scan of MedQA questions for decisive details (aid for writing the edits)
  holm_fdr.py                Holm and Benjamini-Hochberg corrections for the counted families of tests
  make_fig_*.py              figures;  make_appendix_tables*.py  appendix tables (LaTeX)
data/
  step3_decisive_edits.jsonl the 88 decisive edits (question id, type, original span, replacement, rationale)
hand_check/
  claude_labels_50_edits.csv reference relevance labels for 50 fixed-rule edits (see note below)
results/
  <model>/                   MedQA outputs, one JSON line per LLM call
  medmcqa/<model>/, law/<model>/   other datasets (RQ1, RQ4)
  gpqa/<model>/              GPQA: predictions only, all model text removed (see Data)
  vigilance_plants_v3/       RQ2 planted arguments, judges, relevance and counterfactual outputs
  step3/                     decisive-flaw arguments and verification runs
  figdata/                   saved analysis outputs that every number in the paper was taken from
  rq4/                       RQ4 summary and curves
```

Files named `*_try<k>.jsonl` are earlier generation attempts kept for completeness; the analyses use the
final file without the suffix.

## Models

| Key | Hugging Face model |
|---|---|
| `qwen4` | Qwen/Qwen3-4B |
| `qwen8` | Qwen/Qwen3-8B |
| `qwen14` | Qwen/Qwen3-14B |
| `qwen32` | Qwen/Qwen3-32B (2 GPUs) |
| `gemma12` | google/gemma-3-12b-it |
| `llama8` | meta-llama/Llama-3.1-8B-Instruct |

Qwen3 thinking mode is switched off. Runs used vLLM 0.11 on NVIDIA A40 (48 GB) GPUs. All prompts,
decoding settings and seeds are in the code and in Appendix A of the paper.

## Reproducing the numbers (no GPU needed)

Run from the repository root (Python 3.9+, `numpy`, `matplotlib`):

```bash
# RQ1 (per dataset; repeat for results/medmcqa, results/law, results/gpqa)
python3 analysis/stats.py results qwen4 qwen8 qwen14 qwen32 gemma12 llama8

# RQ2 / review analyses
python3 analysis/step1_review_analyses.py results
python3 analysis/step2_analysis.py results
python3 analysis/step3_analysis.py results

# multiple-comparison check (about 3 minutes)
python3 analysis/holm_fdr.py results
```

Each command prints the same output as the corresponding file in `results/figdata/` (bootstrap B = 2,000,
fixed seeds). The figure and table scripts read those saved outputs, for example
`python3 analysis/make_fig_rq2b.py results/figdata`.

## Re-running the experiments

1. **Data.** The benchmark questions are not redistributed here.
   - MedQA: download from <https://github.com/jind11/MedQA> and use
     `data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl`, then
     `python cluster/prepare_medqa.py path/to/phrases_no_exclude_test.jsonl` (300 questions, seed 42).
   - MedMCQA, MMLU Professional Law, GPQA Diamond: `python cluster/prepare_datasets.py` (300, 300 and all
     198 questions, seed 42). GPQA is gated on Hugging Face; accept its terms first.
2. **Environment.** `pip install -r requirements.txt` in a CUDA environment, set `PROJ` to the repository
   root and fill in the placeholders in the `*.sbatch` files (`<your_account>`, `<your_env_setup.sh>`, partition).
3. **Order.** `run_medqa.py` for each model and dataset first (all other stages read its round-1 replies and
   SC samples), then `run_vigilance.py`, `run_rq3.py`, `run_review_step2.py`, `run_step3.py`. Each script's
   docstring lists its stages and arguments.

## Notes

- **GPQA.** The GPQA authors ask that its questions not be published online in plain text. The GPQA files
  here therefore keep only identifiers, gold and predicted options, agent and sample indices; model
  responses and prefills (which can quote options) were removed. All RQ1 and RQ4 numbers for GPQA can still
  be recomputed from them.
- **LLM-written materials.** The 88 decisive edits in `data/step3_decisive_edits.jsonl` and the reference
  labels in `hand_check/claude_labels_50_edits.csv` were written with Claude Opus 5.5, not by clinicians.
  Whether each edit changes the answer was tested behaviourally (`results/step3/verify_*.jsonl`).
- **Not for clinical use.** None of the systems studied here is intended for clinical decisions.

## Citation

If you use this code or data, please cite the paper (arXiv identifier to be added).

## License

Code and outputs are released under the MIT License. The benchmark datasets keep their own licenses.

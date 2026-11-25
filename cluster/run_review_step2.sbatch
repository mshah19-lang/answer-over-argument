#!/bin/bash
# Review response Step 2 (2026-11-19), one job on 2 A40s with Qwen3-32B:
#   1) second-judge manipulation check   (run_vigilance.py --stage judge -> judge_qwen32.jsonl)
#   2) relevance of each flaw edit       (run_review_step2.py --stage relevance)
#   3) counterfactual answers            (run_review_step2.py --stage counterfactual)
# Finished files are skipped, so the job can simply be resubmitted.
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch cluster/run_review_step2.sbatch
#SBATCH --job-name=step2
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

source <your_env_setup.sh>
export PROJ=/path/to/project
export HF_HUB_OFFLINE=1
cd "$PROJ"
nvidia-smi --query-gpu=name,memory.total --format=csv
python cluster/run_vigilance.py --stage judge --model qwen32 --dataset medqa
python cluster/run_review_step2.py --stage relevance --model qwen32
python cluster/run_review_step2.py --stage counterfactual --model qwen32

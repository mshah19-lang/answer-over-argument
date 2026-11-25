#!/bin/bash
# Review response Step 3 (2026-11-19): RQ2 with decisive flaws. Runs the given stages, in order, for ONE model.
# Finished files are skipped, so a job can simply be resubmitted.
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   Job A (plants + check):  sbatch --gres=gpu:a40:2 cluster/run_step3.sbatch qwen32 "gen verify"
#                            sbatch --gres=gpu:a40:1 cluster/run_step3.sbatch llama8 "verify"
#   Job B (after A, per model): sbatch --gres=gpu:a40:1 cluster/run_step3.sbatch qwen8 "revise detect"
#                               (qwen32 needs --gres=gpu:a40:2)
#SBATCH --job-name=step3
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:1
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
MODEL=$1
for STAGE in $2; do
  python cluster/run_step3.py --stage "$STAGE" --model "$MODEL" || exit 1
done

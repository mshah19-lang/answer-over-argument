#!/bin/bash
# RQ1 full runs: for each model in the list, all datasets: run_medqa.py then run_planted.py, one after another.
# cluster allows only 6 queued jobs per user, so several models share one job (finished stages are skipped).
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch --gres=gpu:a40:1 cluster/run_rq1.sbatch "qwen4 qwen8 llama8"
#   sbatch --gres=gpu:a40:2 cluster/run_rq1.sbatch "qwen32"
# Optional 2nd argument: datasets (default "medmcqa law gpqa").
#SBATCH --job-name=rq1
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%j.out

source <your_env_setup.sh>
export PROJ=/path/to/project
export HF_HUB_OFFLINE=1
cd "$PROJ"
nvidia-smi --query-gpu=name,memory.total --format=csv
MODELS="$1"
DATASETS="${2:-medmcqa law gpqa}"
for m in $MODELS; do
  for ds in $DATASETS; do
    echo "===== $(date '+%F %T') $m $ds"
    if python cluster/run_medqa.py --model "$m" --dataset "$ds" && \
       python cluster/run_planted.py --model "$m" --dataset "$ds"; then
      echo "===== DONE $m $ds"
    else
      echo "===== FAILED $m $ds (continuing with the next one)"
    fi
  done
done
echo "===== ALL FINISHED $(date '+%F %T')"

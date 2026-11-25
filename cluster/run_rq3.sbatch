#!/bin/bash
# RQ3 full run: for each model in the list, --stage tactics then --stage verify on one dataset.
# Needs RQ2 v3 plants (results/vigilance_plants_v3/plants.jsonl) and each model's round1/round2.
# cluster allows only 6 queued jobs per user, so several models share one job (finished files are skipped).
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch --gres=gpu:a40:1 cluster/run_rq3.sbatch "qwen4 qwen8 llama8"
#   sbatch --gres=gpu:a40:2 cluster/run_rq3.sbatch "qwen32"
# Optional 2nd argument: dataset (default medqa); optional 3rd: --limit N (pilot).
#SBATCH --job-name=rq3
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

source <your_env_setup.sh>
export PROJ=/path/to/project
export HF_HUB_OFFLINE=1
cd "$PROJ"
nvidia-smi --query-gpu=name,memory.total --format=csv
MODELS="$1"
DS="${2:-medqa}"
EXTRA="${3:-}"
for m in $MODELS; do
  for stage in tactics verify; do
    echo "===== $(date '+%F %T') $m $DS $stage"
    if python cluster/run_rq3.py --stage "$stage" --model "$m" --dataset "$DS" $EXTRA; then
      echo "===== DONE $m $DS $stage"
    else
      echo "===== FAILED $m $DS $stage (continuing)"
    fi
  done
done
echo "===== ALL FINISHED $(date '+%F %T')"

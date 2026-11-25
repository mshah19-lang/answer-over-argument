#!/bin/bash
# RQ2 full run (vigilance v3): for each model in the list, --stage revise then --stage detect on one dataset.
# cluster allows only 6 queued jobs per user, so several models share one job (finished files are skipped).
# Needs the plants first (gen stage). Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch --dependency=afterok:<gen job> --gres=gpu:a40:1 cluster/run_vig_models.sbatch "qwen4 qwen8 llama8"
#   sbatch --dependency=afterok:<gen job> --gres=gpu:a40:2 cluster/run_vig_models.sbatch "qwen32"
# Optional 2nd argument: dataset (default medqa). Full run = no --limit (all questions).
#SBATCH --job-name=vigall
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
for m in $MODELS; do
  for stage in revise detect; do
    echo "===== $(date '+%F %T') $m $DS $stage"
    if python cluster/run_vigilance.py --stage "$stage" --model "$m" --dataset "$DS"; then
      echo "===== DONE $m $DS $stage"
    else
      echo "===== FAILED $m $DS $stage (continuing)"
    fi
  done
done
echo "===== ALL FINISHED $(date '+%F %T')"

#!/bin/bash
# Planted-wrong-agent causal test. Run AFTER run_medqa.sbatch finished for the same model/size.
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch cluster/run_planted.sbatch qwen32 --limit 20    # pilot, 20 questions
#   sbatch cluster/run_planted.sbatch qwen32               # full run, 300 questions
#SBATCH --job-name=planted
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

source <your_env_setup.sh>
export PROJ=/path/to/project
export HF_HUB_OFFLINE=1
cd "$PROJ"
nvidia-smi --query-gpu=name,memory.total --format=csv
python cluster/run_planted.py --model "$@"

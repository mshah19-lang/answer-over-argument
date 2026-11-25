#!/bin/bash
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch cluster/run_medqa.sbatch qwen32 --limit 20    # pilot, 20 questions
#   sbatch cluster/run_medqa.sbatch qwen32               # full run, 300 questions
#SBATCH --job-name=medqa
#SBATCH --account=<your_account>
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

source <your_env_setup.sh>          # vllm 0.11.0 env + HF_HOME (Qwen3-32B already downloaded there)
export PROJ=/path/to/project
export HF_HUB_OFFLINE=1           # no internet on compute nodes
cd "$PROJ"
nvidia-smi --query-gpu=name,memory.total --format=csv
python cluster/run_medqa.py --model "$@"

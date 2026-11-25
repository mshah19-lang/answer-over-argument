#!/bin/bash
# Epistemic-vigilance test (RQ2). Stages: gen (generator model), then revise + detect (each agent model).
# Usage on a SLURM cluster (from $PROJ = /path/to/project):
#   sbatch --gres=gpu:a40:2 cluster/run_vigilance.sbatch --stage gen    --model qwen32  --dataset medqa --limit 100
#   sbatch --gres=gpu:a40:1 cluster/run_vigilance.sbatch --stage judge  --model gemma12 --dataset medqa --limit 100
#   sbatch --gres=gpu:a40:1 cluster/run_vigilance.sbatch --stage revise --model qwen8   --dataset medqa --limit 100
#   sbatch --gres=gpu:a40:1 cluster/run_vigilance.sbatch --stage detect --model qwen8   --dataset medqa --limit 100
# (v3 files: vigilance_plants_v3*/ and <model>/vigilance_v3_*; v1 and v2 pilot files are left untouched)
#SBATCH --job-name=vigil
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
python cluster/run_vigilance.py "$@"

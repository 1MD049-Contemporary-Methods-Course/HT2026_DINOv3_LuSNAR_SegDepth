#!/bin/bash
# Usage: sbatch -J seg_unf4 jim/train_unfreeze.sbatch seg 4 [extra train_unfreeze.py args]
#SBATCH -A uppmax2026-1-299
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6                      # ~64 CPUs / 10 GPUs per L40S node; default was 2 (data loading bottleneck)
#SBATCH -t 3:00:00
#SBATCH --output=/proj/uppmax2026-1-299/jimcai/logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

TASK=$1; N=$2; shift 2
REPO=/proj/uppmax2026-1-299/jimcai/repo
# Same Python environment as the baseline (Andreas's venv: torch 2.11 + transformers)
source /proj/uppmax2026-1-299/task1_groupc_dataset/activate_lusnar.sh

nvidia-smi --query-gpu=name,memory.total --format=csv
python -u "$REPO/jim/train_unfreeze.py" --task "$TASK" --unfreeze "$N" --num-workers "$SLURM_CPUS_PER_TASK" \
    --run-dir "/proj/uppmax2026-1-299/jimcai/runs/${SLURM_JOB_NAME}" "$@"

#!/bin/bash
# Usage: sbatch -J method4_seg768 ~/train_method4.sh seg 768
#SBATCH -A uppmax2026-1-299
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH -t 9:00:00
#SBATCH -o /home/intishar/%x_%j.out
set -euo pipefail
TASK=$1; SIZE=$2
source /proj/uppmax2026-1-299/task1_groupc_dataset/activate_lusnar.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u ~/team_repo/intishar/train_method4.py --task "$TASK" --image-size "$SIZE" \
    --unfreeze 12 --backbone-lr 4e-5 --epochs 10 --num-workers "$SLURM_CPUS_PER_TASK" \
    --run-dir ~/ckpt_method4/${SLURM_JOB_NAME}

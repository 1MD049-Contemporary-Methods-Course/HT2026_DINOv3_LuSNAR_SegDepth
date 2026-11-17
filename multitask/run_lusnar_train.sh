#!/bin/bash -l
#SBATCH -A uppmax2026-1-299
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH -t 3:00:00
#SBATCH -J lusnar_multitask
#SBATCH -o lusnar-%j.out

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
source /home/taiko/1MD049/taiko_sandbox_please_work/.venv_lusnar/bin/activate

unset PYTHONPATH
export PYTHONNOUSERSITE=1

hostname
nvidia-smi

python -c \
"import torch, regex, transformers; \
print(torch.__file__); \
print('CUDA build:', torch.version.cuda); \
assert torch.cuda.is_available(); \
print('GPU:', torch.cuda.get_device_name(0)); \
print('regex:', regex.__version__); \
print('transformers:', transformers.__version__)"

python -m jupyter nbconvert \
  --to notebook \
  --execute lusnar.ipynb \
  --output "executed_lusnar_${SLURM_JOB_ID}.ipynb" \
  --ExecutePreprocessor.timeout=-1
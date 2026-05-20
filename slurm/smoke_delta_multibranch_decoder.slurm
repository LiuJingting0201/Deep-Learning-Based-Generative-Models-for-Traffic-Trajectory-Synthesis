#!/bin/bash
#SBATCH --job-name=smoke_delta_multibranch
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jliu/Thesis/logs/slurm/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/slurm/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p logs/slurm results_hpc

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
PYTHON="$(which python)"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import sys; print('Python executable:', sys.executable)"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "===== Compile checks ====="
"$PYTHON" -m py_compile scripts/train_delta_displacement_decoder.py
"$PYTHON" -m py_compile src/cnr_trajectory/reconstruction/delta_displacement.py

echo "===== Smoke training ====="
"$PYTHON" scripts/train_delta_displacement_decoder.py \
  --data-dir /home/jliu/data_no_speed_delta_displacement_paired
  --output-dir results_hpc/smoke_multibranch_delta \
  --model multi_branch_delta \
  --channels rgb \
  --encoder-type small_spatial \
  --epochs 1 \
  --batch-size 4 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --num-workers 0

echo "===== Finished ====="
date

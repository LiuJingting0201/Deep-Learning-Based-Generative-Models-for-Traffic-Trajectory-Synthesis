#!/bin/bash
#SBATCH --job-name=gaf_ae
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Slurm opens output/error files before this script runs. From ~/Thesis, run:
# mkdir -p logs/slurm results_hpc
# before submitting with sbatch.

set -euo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

cd ~/Thesis
mkdir -p logs/slurm results_hpc

if ! test -d ../data_no_speed_delta_displacement_paired/images; then
  echo "ERROR: dataset directory not found: ../data_no_speed_delta_displacement_paired/images" >&2
  echo "Run this job from a checkout where the paired GAF image dataset is available under ~/Thesis." >&2
  exit 1
fi

echo "===== Job debug info ====="
echo "job=${SLURM_JOB_NAME:-unknown}"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "date=$(date)"
echo "python=$(which python)"
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
echo "git_commit=$(git rev-parse HEAD)"
nvidia-smi
echo "=========================="

python scripts/train_autoencoder.py \
  --data-dir ../data_no_speed_delta_displacement_paired/images \
  --output-dir results_hpc/gaf_autoencoder_delta_224_latent28 \
  --batch-size 16 \
  --lr 1e-4 \
  --epochs 50 \
  --save-every 10 \
  --image-size 224 \
  --latent-channels 32 \
  --latent-spatial-size 28

#!/bin/bash
#SBATCH --job-name=encode_latents
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

cd ~/Thesis
mkdir -p logs/slurm

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

python scripts/encode_dataset_to_latent.py \
  --data-dir ../data_no_speed_delta_displacement_paired/images \
  --autoencoder-checkpoint results_hpc/gaf_autoencoder_delta_224_latent28/checkpoint-final \
  --output-dir data_latent_delta_224_latent28 \
  --batch-size 64 \
  --image-size 224 \
  --num-workers 2

echo "===== latent_stats.json ====="
cat data_latent_delta_224_latent28/latent_stats.json
echo
echo "===== first latent shape ====="
python -c "from pathlib import Path; import numpy as np; p=sorted((Path('data_latent_delta_224_latent28')/'latent').glob('*.npy'))[0]; print(f'{p}: {np.load(p).shape}')"

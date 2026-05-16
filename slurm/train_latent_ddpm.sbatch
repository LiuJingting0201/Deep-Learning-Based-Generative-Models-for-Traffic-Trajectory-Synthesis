#!/bin/bash
#SBATCH --job-name=latent_ddpm
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

cd ~/Thesis
mkdir -p logs/slurm results_hpc

if ! test -d data_latent_delta_224_latent28/latent; then
  echo "ERROR: latent directory not found: data_latent_delta_224_latent28/latent" >&2
  exit 1
fi

if ! test -f data_latent_delta_224_latent28/latent_stats.json; then
  echo "ERROR: latent stats file not found: data_latent_delta_224_latent28/latent_stats.json" >&2
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

python scripts/train_latent_ddpm.py \
  --data-dir data_latent_delta_224_latent28 \
  --output-dir results_hpc/latent_ddpm_delta_224_latent28 \
  --batch-size 32 \
  --lr 1e-4 \
  --max-train-steps 50000 \
  --save-every 2500 \
  --sample-every 2500 \
  --sample-size 28 \
  --latent-channels 32

#!/bin/bash
#SBATCH --job-name=smoke_ddim_sampling
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/logs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
CHECKPOINT="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE/checkpoint-25000"
OUT_DIR="/home/jliu/Thesis/results_hpc/ddim_150step_ablation_analysis/smoke_ddim_samples"

echo "===== Smoke DDIM sampling ====="
hostname
date
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')"

"$PYTHON" scripts/generate_ddpm_samples.py \
  --checkpoint-dir "$CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --num-samples 4 \
  --batch-size 4 \
  --image-size 224 \
  --num-inference-steps 150 \
  --scheduler ddim \
  --eta 0.0 \
  --filename-prefix checkpoint25000_

echo "===== Smoke output ====="
find "$OUT_DIR" -maxdepth 1 -type f -name '*.png' | sort

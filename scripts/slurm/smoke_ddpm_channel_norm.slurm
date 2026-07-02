#!/bin/bash
#SBATCH --job-name=smoke_ddpm_channel_norm
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/logs /home/jliu/Thesis/results_hpc

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
CHANNEL_STATS="$DATA_ROOT/diagnostics/channel_stats_train.json"
SMOKE_ROOT="/home/jliu/Thesis/results_hpc/channel_norm_smoke"

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

echo "===== Job info ====="
hostname
date
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "===== Smoke A: channel-normalized path ====="
"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_ROOT/images" \
  --output-dir "$SMOKE_ROOT/channel_norm" \
  --aux-loss-mode none \
  --image-size 224 \
  --batch-size 2 \
  --lr 1e-4 \
  --max-train-steps 2 \
  --gradient-accumulation-steps 1 \
  --save-every 1 \
  --sample-every 1 \
  --sample-inference-steps 5 \
  --num-sample-images 2 \
  --channel-normalization-mode standardize \
  --channel-norm-stats "$CHANNEL_STATS"

echo "===== Smoke B: legacy no-channel-normalization path ====="
"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_ROOT/images" \
  --output-dir "$SMOKE_ROOT/no_channel_norm" \
  --aux-loss-mode none \
  --image-size 224 \
  --batch-size 2 \
  --lr 1e-4 \
  --max-train-steps 2 \
  --gradient-accumulation-steps 1 \
  --save-every 1 \
  --sample-every 1 \
  --sample-inference-steps 5 \
  --num-sample-images 2 \
  --channel-normalization-mode none

echo "===== Smoke outputs ====="
find "$SMOKE_ROOT" -maxdepth 3 -type f | sort

echo "===== Finished ====="
date

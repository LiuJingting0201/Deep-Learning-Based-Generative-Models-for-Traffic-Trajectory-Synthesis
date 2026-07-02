#!/bin/bash
#SBATCH --job-name=week05_sig10_mse_diag
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week05/logs/week05_sig10_mse_diag_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/week05_sig10_mse_diag_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
EXPERIMENT_NAME="week05_sig10_mse_diag"
DATA_ROOT="/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
NORMALIZATION_FILE="${DATA_ROOT}/normalization_diagnostics.json"
OUTPUT_DIR="/home/jliu/Thesis/week05/results/${EXPERIMENT_NAME}"

mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Week05 DDPM job config ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "NORMALIZATION_FILE=${NORMALIZATION_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NORMALIZATION=sigmoid"
echo "SIGMOID_K=1.0"
echo "AUX_LOSS_MODE=diag"
echo "DIAG_WEIGHT=0.05"
echo "SYMMETRY_WEIGHT=0.05"
echo "IMAGE_SIZE=224"
echo "CHANNELS=3"
echo "BATCH_SIZE=8"
echo "LR=2e-4"
echo "MAX_TRAIN_STEPS=50000"
echo "SEED=42"
echo "NUM_TRAIN_TIMESTEPS=1000"
echo "SCHEDULER=DDPMScheduler"
echo "MODEL_ARCHITECTURE=scripts.train_ddpm.build_unet"
echo "DATA_FORMAT=float arrays from .npy, no PNG input"

"${PYTHON}" -u /home/jliu/Thesis/week05/scripts/train_week05_ddpm.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --normalization-file "${NORMALIZATION_FILE}" \
  --split train \
  --image-size 224 \
  --sigmoid-k 1.0 \
  --aux-loss-mode diag \
  --diag-weight 0.05 \
  --symmetry-weight 0.05 \
  --batch-size 8 \
  --lr 2e-4 \
  --max-train-steps 50000 \
  --gradient-accumulation-steps 2 \
  --seed 42 \
  --num-train-timesteps 1000 \
  --lr-warmup-steps 500 \
  --sample-every 2000 \
  --save-every 2000 \
  --use-ema

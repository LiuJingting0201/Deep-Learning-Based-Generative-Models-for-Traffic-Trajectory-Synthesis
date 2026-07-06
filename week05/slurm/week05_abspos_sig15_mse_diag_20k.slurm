#!/bin/bash
#SBATCH --job-name=w05_abspos_gasf
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jliu/Thesis/week05/logs/week05_abspos_sig15_mse_diag_20k_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/week05_abspos_sig15_mse_diag_20k_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
EXPERIMENT_NAME="week05_abspos_sig15_mse_diag_20k"
DATA_ROOT="/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
NORMALIZATION_FILE="${DATA_ROOT}/absolute_position_normalization_diagnostics.json"
OUTPUT_DIR="/home/jliu/Thesis/week05/results/${EXPERIMENT_NAME}"

mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

if [[ ! -f "${NORMALIZATION_FILE}" ]]; then
  "${PYTHON}" -u /home/jliu/Thesis/week05/scripts/compute_absolute_position_normalization.py \
    --data-root "${DATA_ROOT}" \
    --output "${NORMALIZATION_FILE}" \
    --split train
fi

echo "===== Week05 absolute-position GASF DDPM ablation ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "NORMALIZATION_FILE=${NORMALIZATION_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REPRESENTATION_MODE=absolute"
echo "NORMALIZATION=sigmoid"
echo "SIGMOID_K=1.5"
echo "AUX_LOSS_MODE=diag"
echo "DIAG_WEIGHT=0.05"
echo "IMAGE_SIZE=224"
echo "BATCH_SIZE=8"
echo "LR=2e-4"
echo "MAX_TRAIN_STEPS=20000"
echo "SEED=42"
echo "DATA_FORMAT=GASF(x), GASF(y), start heatmap generated from absolute .npy labels"

"${PYTHON}" -u /home/jliu/Thesis/week05/scripts/train_week05_ddpm.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --normalization-file "${NORMALIZATION_FILE}" \
  --representation-mode absolute \
  --split train \
  --image-size 224 \
  --sigmoid-k 1.5 \
  --aux-loss-mode diag \
  --diag-weight 0.05 \
  --symmetry-weight 0.05 \
  --batch-size 8 \
  --lr 2e-4 \
  --max-train-steps 20000 \
  --gradient-accumulation-steps 2 \
  --seed 42 \
  --num-train-timesteps 1000 \
  --lr-warmup-steps 500 \
  --sample-every 2000 \
  --save-every 2000 \
  --use-ema

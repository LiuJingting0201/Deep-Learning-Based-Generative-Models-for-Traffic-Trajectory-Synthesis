#!/bin/bash
#SBATCH --job-name=week03_s2_mtf_trans
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/jliu/Thesis

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
EXP_NAME="stage2_mtfonly_transformer_l2_h4_ff256_latent64_lr1e4_ep500"
DATA_ROOT="${DATA_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
TRAJ_AE_CHECKPOINT="${TRAJ_AE_CHECKPOINT:-/home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/sequence_latent/stage2_mtfonly_transformer_l2_h4_ff256_latent64_lr1e4_ep500}"
LATENT_DIM="${LATENT_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-4}"
BETA_INTEGRATED="${BETA_INTEGRATED:-0.1}"
LAMBDA_LATENT="${LAMBDA_LATENT:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-0}"
BEST_METRIC="${BEST_METRIC:-integrated_ADE}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
IMAGE_CHANNEL_MODE="${IMAGE_CHANNEL_MODE:-mtf}"
INPUT_CHANNELS="${INPUT_CHANNELS:-1}"
USE_TRANSFORMER_REFINE="${USE_TRANSFORMER_REFINE:-1}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_FF_DIM="${TRANSFORMER_FF_DIM:-256}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"

if [ -d "${OUTPUT_DIR}" ] && { [ -f "${OUTPUT_DIR}/best_metrics.json" ] || [ -f "${OUTPUT_DIR}/train_log.csv" ]; }; then
  if [ "${OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: OUTPUT_DIR already contains previous results: ${OUTPUT_DIR}"
    echo "Set OVERWRITE=1 to overwrite intentionally."
    exit 1
  fi
fi

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "EXP_NAME=${EXP_NAME}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TRAJ_AE_CHECKPOINT=${TRAJ_AE_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LATENT_DIM=${LATENT_DIM}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"
echo "BETA_INTEGRATED=${BETA_INTEGRATED}"
echo "LAMBDA_LATENT=${LAMBDA_LATENT}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "SEED=${SEED}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "BEST_METRIC=${BEST_METRIC}"
echo "GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
echo "IMAGE_CHANNEL_MODE=${IMAGE_CHANNEL_MODE}"
echo "INPUT_CHANNELS=${INPUT_CHANNELS}"
echo "USE_TRANSFORMER_REFINE=${USE_TRANSFORMER_REFINE}"
echo "TRANSFORMER_LAYERS=${TRANSFORMER_LAYERS}"
echo "TRANSFORMER_HEADS=${TRANSFORMER_HEADS}"
echo "TRANSFORMER_FF_DIM=${TRANSFORMER_FF_DIM}"
echo "TRANSFORMER_DROPOUT=${TRANSFORMER_DROPOUT}"
echo "USE_TEMPORAL_REFINE=0"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_utils.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_image_sequence_latent_alignment.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/evaluate_sequence_latent_alignment.py

echo "===== Data/checkpoint check ====="
test -d "${DATA_ROOT}/images"
test -d "${DATA_ROOT}/labels_delta_displacement"
test -f "${DATA_ROOT}/metadata.csv"
test -f "${DATA_ROOT}/splits/split_metadata.csv"
test -f "${TRAJ_AE_CHECKPOINT}"

TRANSFORMER_ARGS=()
if [ "${USE_TRANSFORMER_REFINE}" = "1" ]; then
  TRANSFORMER_ARGS+=(--use-transformer-refine)
fi

echo "===== Start ${EXP_NAME} ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_image_sequence_latent_alignment.py \
  --data-root "${DATA_ROOT}" \
  --trajectory-ae-checkpoint "${TRAJ_AE_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --beta-integrated "${BETA_INTEGRATED}" \
  --lambda-latent "${LAMBDA_LATENT}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device auto \
  --save-every "${SAVE_EVERY}" \
  --best-metric "${BEST_METRIC}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --image-channel-mode "${IMAGE_CHANNEL_MODE}" \
  --transformer-layers "${TRANSFORMER_LAYERS}" \
  --transformer-heads "${TRANSFORMER_HEADS}" \
  --transformer-ff-dim "${TRANSFORMER_FF_DIM}" \
  --transformer-dropout "${TRANSFORMER_DROPOUT}" \
  "${TRANSFORMER_ARGS[@]}"

echo "===== Finished ${EXP_NAME} ====="
date

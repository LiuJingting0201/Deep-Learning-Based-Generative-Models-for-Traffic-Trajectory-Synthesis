#!/bin/bash
#SBATCH --job-name=week03_seq_lat_eval
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=02:00:00
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
DATA_ROOT="${DATA_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
TRAJ_AE_CHECKPOINT="${TRAJ_AE_CHECKPOINT:-/home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt}"
IMAGE_ENCODER_CHECKPOINT="${IMAGE_ENCODER_CHECKPOINT:-/home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64/eval}"
LATENT_DIM="${LATENT_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
NUM_PLOTS="${NUM_PLOTS:-12}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TRAJ_AE_CHECKPOINT=${TRAJ_AE_CHECKPOINT}"
echo "IMAGE_ENCODER_CHECKPOINT=${IMAGE_ENCODER_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/evaluate_sequence_latent_alignment.py

echo "===== Data/checkpoint check ====="
test -d "${DATA_ROOT}/images"
test -d "${DATA_ROOT}/labels_delta_displacement"
test -f "${TRAJ_AE_CHECKPOINT}"
test -f "${IMAGE_ENCODER_CHECKPOINT}"

echo "===== Start sequence-latent evaluation ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/evaluate_sequence_latent_alignment.py \
  --data-root "${DATA_ROOT}" \
  --trajectory-ae-checkpoint "${TRAJ_AE_CHECKPOINT}" \
  --image-encoder-checkpoint "${IMAGE_ENCODER_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device auto \
  --num-plots "${NUM_PLOTS}" \
  --splits test

echo "===== Metrics summary ====="
cat "${OUTPUT_DIR}/metrics_summary.json"
echo "===== Finished evaluation ====="
date

#!/bin/bash
#SBATCH --job-name=week03_seq_lat_s1
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
DATA_ROOT="${DATA_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent128}"
LATENT_DIM="${LATENT_DIM:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-4}"
BETA_INTEGRATED="${BETA_INTEGRATED:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-0}"
BEST_METRIC="${BEST_METRIC:-integrated_ADE}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_utils.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_trajectory_sequence_autoencoder.py

echo "===== Data check ====="
test -d "${DATA_ROOT}/labels_delta_displacement"
test -f "${DATA_ROOT}/metadata.csv"
test -f "${DATA_ROOT}/splits/split_metadata.csv"

echo "===== Start Stage 1 trajectory sequence autoencoder ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_trajectory_sequence_autoencoder.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --beta-integrated "${BETA_INTEGRATED}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --device auto \
  --save-every "${SAVE_EVERY}" \
  --best-metric "${BEST_METRIC}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}"

echo "===== Finished Stage 1 ====="
date

#!/bin/bash
#SBATCH --job-name=week03_seq_lat_smoke
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
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
SMOKE_ROOT="${SMOKE_ROOT:-/home/jliu/Thesis/week03/results/smoke_sequence_latent_alignment}"
STAGE1_DIR="${SMOKE_ROOT}/stage1"
STAGE2_DIR="${SMOKE_ROOT}/stage2"
EVAL_DIR="${STAGE2_DIR}/eval"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${STAGE1_DIR}" "${STAGE2_DIR}" "${EVAL_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SMOKE_ROOT=${SMOKE_ROOT}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/sequence_latent_utils.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_trajectory_sequence_autoencoder.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_image_sequence_latent_alignment.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/evaluate_sequence_latent_alignment.py

echo "===== Unit smoke tests ====="
"${PYTHON}" -m pytest -q /home/jliu/Thesis/week03/tests/test_sequence_latent_smoke.py

echo "===== Tiny Stage 1 smoke training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_trajectory_sequence_autoencoder.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${STAGE1_DIR}" \
  --latent-dim 16 \
  --batch-size 2 \
  --epochs 1 \
  --lr 1e-3 \
  --beta-integrated 0.1 \
  --num-workers 0 \
  --seed 42 \
  --device auto \
  --best-metric integrated_ADE \
  --temporal-kernel-size 5 \
  --temporal-layers 2

echo "===== Tiny Stage 2 smoke training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_image_sequence_latent_alignment.py \
  --data-root "${DATA_ROOT}" \
  --trajectory-ae-checkpoint "${STAGE1_DIR}/best.pt" \
  --output-dir "${STAGE2_DIR}" \
  --latent-dim 16 \
  --batch-size 2 \
  --epochs 1 \
  --lr 5e-4 \
  --beta-integrated 0.1 \
  --lambda-latent 1.0 \
  --num-workers 0 \
  --seed 42 \
  --device auto \
  --best-metric integrated_ADE

echo "===== Tiny evaluation smoke ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/evaluate_sequence_latent_alignment.py \
  --data-root "${DATA_ROOT}" \
  --trajectory-ae-checkpoint "${STAGE1_DIR}/best.pt" \
  --image-encoder-checkpoint "${STAGE2_DIR}/best.pt" \
  --output-dir "${EVAL_DIR}" \
  --latent-dim 16 \
  --batch-size 2 \
  --num-workers 0 \
  --seed 42 \
  --device auto \
  --num-plots 2 \
  --splits test

echo "===== Smoke metrics ====="
cat "${EVAL_DIR}/metrics_summary.json"
echo "===== Finished smoke ====="
date

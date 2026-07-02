#!/bin/bash
#SBATCH --job-name=eval_delta_decoder
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jliu/Thesis/logs/slurm/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/slurm/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p logs/slurm results_hpc

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
PYTHON="$(which python)"

DATA_DIR="${DATA_DIR:-/home/jliu/data_no_speed_delta_displacement_paired}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the completed decoder result directory.}"
MODEL="${MODEL:-multi_branch_delta}"
CHANNELS="${CHANNELS:-rgb}"
ENCODER_TYPE="${ENCODER_TYPE:-small_spatial}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BETA_INTEGRATED_LOSS="${BETA_INTEGRATED_LOSS:-0.2}"
SPLIT_METADATA="${SPLIT_METADATA:-}"
NORMALIZE_TARGET_DELTA="${NORMALIZE_TARGET_DELTA:-0}"

EXTRA_ARGS=()
if [[ -n "${SPLIT_METADATA}" ]]; then
  EXTRA_ARGS+=(--split-metadata "${SPLIT_METADATA}")
fi
if [[ "${NORMALIZE_TARGET_DELTA}" == "1" ]]; then
  EXTRA_ARGS+=(--normalize-target-delta)
fi

echo "===== Eval job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}"
echo "CHANNELS=${CHANNELS}"
echo "ENCODER_TYPE=${ENCODER_TYPE}"
echo "SPLIT_METADATA=${SPLIT_METADATA}"
echo "NORMALIZE_TARGET_DELTA=${NORMALIZE_TARGET_DELTA}"
echo "BETA_INTEGRATED_LOSS=${BETA_INTEGRATED_LOSS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import sys; print('Python executable:', sys.executable)"
"$PYTHON" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "===== Compile checks ====="
"$PYTHON" -m py_compile scripts/train_delta_displacement_decoder.py
"$PYTHON" -m py_compile src/cnr_trajectory/reconstruction/delta_displacement.py

echo "===== Eval-only ====="
"$PYTHON" scripts/train_delta_displacement_decoder.py \
  --eval-only \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --channels "${CHANNELS}" \
  --encoder-type "${ENCODER_TYPE}" \
  --batch-size "${BATCH_SIZE}" \
  --beta-integrated-loss "${BETA_INTEGRATED_LOSS}" \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"

echo "===== Finished ====="
date

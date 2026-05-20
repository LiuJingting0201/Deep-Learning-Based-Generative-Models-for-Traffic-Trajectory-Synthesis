#!/bin/bash
#SBATCH --job-name=delta_channel_ablation
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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

BETA="0.05"
NORMALIZE=0
TASK_ID="${SLURM_ARRAY_TASK_ID:-${DELTA_TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: DELTA_TASK_ID is not set."
  echo "Submit one task with, for example:"
  echo "  sbatch --export=ALL,DELTA_TASK_ID=1 slurm/run_delta_channel_ablation.slurm"
  exit 2
fi

case "${TASK_ID}" in
  1) MODEL="multi_branch_delta"; CHANNELS="r"; OUTPUT_DIR="results_hpc/decoder_channel_r_beta005" ;;
  2) MODEL="multi_branch_delta"; CHANNELS="g"; OUTPUT_DIR="results_hpc/decoder_channel_g_beta005" ;;
  3) MODEL="multi_branch_delta"; CHANNELS="b"; OUTPUT_DIR="results_hpc/decoder_channel_b_beta005" ;;
  4) MODEL="simple_cnn_delta"; CHANNELS="rgb"; OUTPUT_DIR="results_hpc/decoder_simplecnn_rgb_beta005" ;;
  5) MODEL="simple_cnn_delta"; CHANNELS="rgb"; OUTPUT_DIR="results_hpc/decoder_simplecnn_rgb_norm_beta005"; NORMALIZE=1 ;;
  *) echo "Unknown DELTA_TASK_ID=${TASK_ID}" >&2; exit 2 ;;
esac

if [[ -f "${OUTPUT_DIR}/checkpoints/best_model.pt" ]]; then
  echo "Found existing completed checkpoint: ${OUTPUT_DIR}/checkpoints/best_model.pt"
  echo "Skipping to avoid overwriting completed outputs."
  exit 0
fi

EXTRA_ARGS=()
if [[ -f results_hpc/decoder_multibranch_delta_beta005/split_metadata.csv ]]; then
  EXTRA_ARGS+=(--split-metadata results_hpc/decoder_multibranch_delta_beta005/split_metadata.csv)
else
  echo "WARNING: results_hpc/decoder_multibranch_delta_beta005/split_metadata.csv not found."
  echo "WARNING: continuing with the train script's deterministic split."
fi
if [[ "${NORMALIZE}" -eq 1 ]]; then
  EXTRA_ARGS+=(--normalize-target-delta)
fi

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DELTA_TASK_ID=${TASK_ID}"
echo "MODEL=${MODEL}"
echo "CHANNELS=${CHANNELS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "BETA=${BETA}"
echo "NORMALIZE=${NORMALIZE}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import sys; print('Python executable:', sys.executable)"
"$PYTHON" -c "import sys; print('Python version:', sys.version)"
"$PYTHON" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "===== Compile checks ====="
"$PYTHON" -m py_compile scripts/train_delta_displacement_decoder.py
"$PYTHON" -m py_compile src/cnr_trajectory/reconstruction/delta_displacement.py

echo "===== Training ====="
"$PYTHON" scripts/train_delta_displacement_decoder.py \
  --data-dir /home/jliu/data_no_speed_delta_displacement_paired \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --channels "${CHANNELS}" \
  --encoder-type small_spatial \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --beta-integrated-loss "${BETA}" \
  --num-workers 4 \
  "${EXTRA_ARGS[@]}"

echo "===== Finished ====="
date

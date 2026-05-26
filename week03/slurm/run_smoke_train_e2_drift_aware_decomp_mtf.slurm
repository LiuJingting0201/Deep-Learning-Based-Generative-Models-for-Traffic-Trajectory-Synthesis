#!/bin/bash
#SBATCH --job-name=week03_smoke_train_e2
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

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
OUTPUT_DIR="/home/jliu/Thesis/week03/results/smoke_train_e2_drift_aware_decomp_mtf"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_drift_aware_decomp_mtf_resnet18_decoder_e2.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-4 \
  --dropout 0.3 \
  --correction-anchors 16 \
  --lambda-abs 0.1 \
  --lambda-corr-target 0.1 \
  --lambda-smooth 0.001 \
  --lambda-mag 0.00005 \
  --num-workers 0 \
  --metrics-every 1 \
  --seed 42 \
  --device auto

if [ -f "${OUTPUT_DIR}/week3_e2_report.md" ]; then
  cat "${OUTPUT_DIR}/week3_e2_report.md"
fi

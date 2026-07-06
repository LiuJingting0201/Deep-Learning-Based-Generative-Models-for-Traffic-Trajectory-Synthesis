#!/bin/bash
#SBATCH --job-name=week03_map_late_start
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=1-00:00:00
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

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
MAP_DIR="/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center"
EXTENT_CSV="/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center_extents.csv"
OUTPUT_DIR="/home/jliu/Thesis/week03/results/map_aware_decoder_late_full"
RUN_NAME="late_resnet18_bs16_start_center_full"

EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DROPOUT="${DROPOUT:-0.3}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
METRICS_EVERY="${METRICS_EVERY:-5}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MAP_DIR=${MAP_DIR}"
echo "EXTENT_CSV=${EXTENT_CSV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "DROPOUT=${DROPOUT}"
echo "METRICS_EVERY=${METRICS_EVERY}"
echo "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
echo "SEED=${SEED}"
echo "NUM_WORKERS=${NUM_WORKERS}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/structure_aware_resnet18_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py

echo "===== Map data check ====="
test -d "${MAP_DIR}"
test -f "${EXTENT_CSV}"
echo -n "map_png_count="
find "${MAP_DIR}" -maxdepth 1 -name '*_map.png' | wc -l
"${PYTHON}" - <<PY
import pandas as pd
path = "${EXTENT_CSV}"
frame = pd.read_csv(path)
print(f"extent_csv={path}")
print(f"extent_rows={len(frame)}")
print(f"extent_columns={list(frame.columns)}")
PY

echo "===== Start late-fusion start_center raw ResNet18 full training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py \
  --model-type late_map_resnet18 \
  --use-map \
  --map-fusion late \
  --map-dir "${MAP_DIR}" \
  --run-name "${RUN_NAME}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --dropout "${DROPOUT}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --metrics-every "${METRICS_EVERY}" \
  --seed "${SEED}" \
  --device auto \
  --num-workers "${NUM_WORKERS}"

echo "===== Finished ====="
date

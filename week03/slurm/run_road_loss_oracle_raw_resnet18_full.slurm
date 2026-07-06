#!/bin/bash
#SBATCH --job-name=week03_road_oracle_raw
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/road_loss_oracle_raw_resnet18_bs16}"
DISTANCE_FIELD_DIR="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields"
DISTANCE_FIELD_METADATA="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields_metadata.csv"
RAW_TO_OSM_AFFINE_JSON="/home/jliu/Thesis/week03/data/maps/raw_xy_to_osm_affine.json"

EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
DROPOUT="${DROPOUT:-0.3}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-25}"
METRICS_EVERY="${METRICS_EVERY:-5}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LAMBDA_ROAD="${LAMBDA_ROAD:-0.05}"
ROAD_THRESHOLD_M="${ROAD_THRESHOLD_M:-5.0}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DISTANCE_FIELD_DIR=${DISTANCE_FIELD_DIR}"
echo "DISTANCE_FIELD_METADATA=${DISTANCE_FIELD_METADATA}"
echo "RAW_TO_OSM_AFFINE_JSON=${RAW_TO_OSM_AFFINE_JSON}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "LR=${LR}"
echo "DROPOUT=${DROPOUT}"
echo "EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
echo "METRICS_EVERY=${METRICS_EVERY}"
echo "SEED=${SEED}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "LAMBDA_ROAD=${LAMBDA_ROAD}"
echo "ROAD_THRESHOLD_M=${ROAD_THRESHOLD_M}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/structure_aware_resnet18_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py

echo "===== Data check ====="
test -d "${DATA_ROOT}/images"
test -f "${DATA_ROOT}/metadata.csv"
test -f "${DATA_ROOT}/splits/split_metadata.csv"
test -d "${DISTANCE_FIELD_DIR}"
test -f "${DISTANCE_FIELD_METADATA}"
test -f "${RAW_TO_OSM_AFFINE_JSON}"
echo -n "distance_field_count="
find "${DISTANCE_FIELD_DIR}" -maxdepth 1 -name '*_distance.npy' | wc -l

echo "===== Start road-loss oracle raw ResNet18 training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py \
  --model-type raw_resnet18 \
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
  --num-workers "${NUM_WORKERS}" \
  --use-road-loss \
  --distance-field-dir "${DISTANCE_FIELD_DIR}" \
  --distance-field-metadata "${DISTANCE_FIELD_METADATA}" \
  --raw-to-osm-affine-json "${RAW_TO_OSM_AFFINE_JSON}" \
  --lambda-road "${LAMBDA_ROAD}" \
  --road-threshold-m "${ROAD_THRESHOLD_M}" \
  --road-loss-mode relu_threshold

echo "===== Finished ====="
date
if [ -f "${OUTPUT_DIR}/week3_report.md" ]; then
  cat "${OUTPUT_DIR}/week3_report.md"
fi
if [ -f "${OUTPUT_DIR}/evaluation/metrics_summary.json" ]; then
  echo "metrics_summary=${OUTPUT_DIR}/evaluation/metrics_summary.json"
fi

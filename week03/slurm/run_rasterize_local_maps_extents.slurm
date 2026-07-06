#!/bin/bash
#SBATCH --job-name=week03_map_raster
#SBATCH --partition=cpu_sapphire
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=04:00:00
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
RAW_TRAJECTORY_FILE="${RAW_TRAJECTORY_FILE:-/home/jliu/gps_with_speed_224_UPDATED.xls}"
EDGES_PATH="${EDGES_PATH:-/home/jliu/Thesis/week03/data/maps/osm_drive_edges.gpkg}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "RAW_TRAJECTORY_FILE=${RAW_TRAJECTORY_FILE}"
echo "EDGES_PATH=${EDGES_PATH}"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/rasterize_local_maps.py

echo "===== Data check ====="
test -f "${RAW_TRAJECTORY_FILE}"
test -f "${EDGES_PATH}"
test -f "${DATA_ROOT}/metadata.csv"

echo "===== Rasterize local maps and write extent metadata ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/rasterize_local_maps.py \
  --data-root "${DATA_ROOT}" \
  --raw-trajectory-file "${RAW_TRAJECTORY_FILE}" \
  --edges-path "${EDGES_PATH}" \
  --debug-samples "${DEBUG_SAMPLES:-0}"

echo "===== Extent metadata check ====="
test -f "${DATA_ROOT}/maps_oracle_bbox_extents.csv"
test -f "${DATA_ROOT}/maps_start_center_extents.csv"
wc -l "${DATA_ROOT}/maps_oracle_bbox_extents.csv" "${DATA_ROOT}/maps_start_center_extents.csv"

echo "===== Finished ====="
date

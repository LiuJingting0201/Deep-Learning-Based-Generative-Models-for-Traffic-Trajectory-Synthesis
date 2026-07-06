#!/bin/bash
#SBATCH --job-name=week03_dist_fields
#SBATCH --partition=cpu_sapphire
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

cd /home/jliu/Thesis

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
SCRIPT="/home/jliu/Thesis/week03/scripts/generate_local_road_distance_fields.py"

DEBUG_SAMPLES="${DEBUG_SAMPLES:-20}"
DEBUG_CLIP_M="${DEBUG_CLIP_M:-50}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DEBUG_SAMPLES=${DEBUG_SAMPLES}"
echo "DEBUG_CLIP_M=${DEBUG_CLIP_M}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<all>}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import scipy; import scipy.ndimage; print('scipy=', scipy.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Input check ====="
test -d "${DATA_ROOT}/maps_oracle_bbox"
test -f "${DATA_ROOT}/maps_oracle_bbox_extents.csv"
test -d "${DATA_ROOT}/maps_start_center"
test -f "${DATA_ROOT}/maps_start_center_extents.csv"
echo -n "oracle_map_png_count="
find "${DATA_ROOT}/maps_oracle_bbox" -maxdepth 1 -name '*_map.png' | wc -l
echo -n "start_center_map_png_count="
find "${DATA_ROOT}/maps_start_center" -maxdepth 1 -name '*_map.png' | wc -l

ARGS=(
  --data-root "${DATA_ROOT}"
  --crop-mode all
  --debug-samples "${DEBUG_SAMPLES}"
  --debug-clip-m "${DEBUG_CLIP_M}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "===== Generate local road-distance fields ====="
"${PYTHON}" -u "${SCRIPT}" "${ARGS[@]}"

echo "===== Output check ====="
echo -n "oracle_distance_count="
find "${DATA_ROOT}/maps_oracle_bbox_distance_fields" -maxdepth 1 -name '*_distance.npy' | wc -l
echo -n "start_center_distance_count="
find "${DATA_ROOT}/maps_start_center_distance_fields" -maxdepth 1 -name '*_distance.npy' | wc -l
test -f "${DATA_ROOT}/maps_oracle_bbox_distance_fields_metadata.csv"
test -f "${DATA_ROOT}/maps_start_center_distance_fields_metadata.csv"
test -f /home/jliu/Thesis/week03/docs/local_road_distance_fields.md

echo "===== Finished ====="
date

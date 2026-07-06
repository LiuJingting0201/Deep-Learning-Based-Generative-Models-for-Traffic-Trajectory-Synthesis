#!/bin/bash
#SBATCH --job-name=week03_refine_oracle
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

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
SCRIPT="/home/jliu/Thesis/week03/scripts/refine_trajectory_map_aware_sliding_window.py"
EVALUATION_DIR="/home/jliu/Thesis/week03/results/baseline_raw_resnet18_bs16/evaluation"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
DISTANCE_FIELD_DIR="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields"
DISTANCE_FIELD_METADATA="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields_metadata.csv"
RAW_TO_OSM_AFFINE_JSON="/home/jliu/Thesis/week03/data/maps/raw_xy_to_osm_affine.json"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/baseline_raw_resnet18_bs16_map_refinement_oracle}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "EVALUATION_DIR=${EVALUATION_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<all>}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, pandas, matplotlib; print('numpy=', numpy.__version__); print('pandas=', pandas.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Input check ====="
test -d "${EVALUATION_DIR}/predictions_absolute_integrated"
test -d "${EVALUATION_DIR}/predictions_delta_displacement"
test -f "${EVALUATION_DIR}/per_sample_metrics.csv"
test -d "${DISTANCE_FIELD_DIR}"
test -f "${DISTANCE_FIELD_METADATA}"
test -f "${RAW_TO_OSM_AFFINE_JSON}"
echo -n "prediction_count="
find "${EVALUATION_DIR}/predictions_absolute_integrated" -maxdepth 1 -name '*_pred_absolute_integrated.npy' | wc -l
echo -n "distance_field_count="
find "${DISTANCE_FIELD_DIR}" -maxdepth 1 -name '*_distance.npy' | wc -l

ARGS=(
  --evaluation-dir "${EVALUATION_DIR}"
  --data-root "${DATA_ROOT}"
  --crop-mode oracle_bbox
  --distance-field-dir "${DISTANCE_FIELD_DIR}"
  --distance-field-metadata "${DISTANCE_FIELD_METADATA}"
  --raw-to-osm-affine-json "${RAW_TO_OSM_AFFINE_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --method pointwise_grid
  --search-radius-m 15
  --search-step-m 2
  --w-road 1.0
  --w-ref 0.05
  --w-curvature 0.1
  --w-step 0.1
  --w-heading 0.05
  --w-terminal 0.2
  --terminal-power 2.0
  --debug-samples 20
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "===== Start map-aware trajectory refinement ====="
"${PYTHON}" -u "${SCRIPT}" "${ARGS[@]}"

echo "===== Output check ====="
test -f "${OUTPUT_DIR}/refinement_per_sample_metrics.csv"
test -f "${OUTPUT_DIR}/refinement_summary.json"
test -f "${OUTPUT_DIR}/refinement_summary.md"
echo -n "refined_absolute_count="
find "${OUTPUT_DIR}/refined_absolute" -maxdepth 1 -name '*_refined_absolute.npy' | wc -l
echo -n "debug_plot_count="
find "${OUTPUT_DIR}/debug_plots" -maxdepth 1 -name '*_debug.png' | wc -l

echo "===== Finished ====="
date

#!/bin/bash
#SBATCH --job-name=week03_refine_oracle_best
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err

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
EVAL_DIR="/home/jliu/Thesis/week03/results/baseline_raw_resnet18_bs16/evaluation"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
DIST_DIR="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields"
DIST_META="/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields_metadata.csv"
AFFINE_JSON="/home/jliu/Thesis/week03/data/maps/raw_xy_to_osm_affine.json"

OUTPUT_DIR="/home/jliu/Thesis/week03/results/baseline_raw_resnet18_bs16_map_refinement_oracle_best_r25_wr5_wref0p02_wc0p05"

SEARCH_RADIUS="${SEARCH_RADIUS:-25}"
SEARCH_STEP="${SEARCH_STEP:-2}"
W_ROAD="${W_ROAD:-5.0}"
W_REF="${W_REF:-0.02}"
W_CURVATURE="${W_CURVATURE:-0.05}"
W_STEP="${W_STEP:-0.1}"
W_HEADING="${W_HEADING:-0.05}"
W_TERMINAL="${W_TERMINAL:-0.2}"
TERMINAL_POWER="${TERMINAL_POWER:-2.0}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SCRIPT=${SCRIPT}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "DIST_DIR=${DIST_DIR}"
echo "DIST_META=${DIST_META}"
echo "AFFINE_JSON=${AFFINE_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SEARCH_RADIUS=${SEARCH_RADIUS}"
echo "SEARCH_STEP=${SEARCH_STEP}"
echo "W_ROAD=${W_ROAD}"
echo "W_REF=${W_REF}"
echo "W_CURVATURE=${W_CURVATURE}"
echo "W_STEP=${W_STEP}"
echo "W_HEADING=${W_HEADING}"
echo "W_TERMINAL=${W_TERMINAL}"
echo "TERMINAL_POWER=${TERMINAL_POWER}"
echo "DEBUG_SAMPLES=${DEBUG_SAMPLES}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<all>}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, pandas, matplotlib; print('numpy/pandas/matplotlib OK')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Input check ====="
test -d "${EVAL_DIR}"
test -d "${EVAL_DIR}/predictions_absolute_integrated"
test -d "${EVAL_DIR}/predictions_delta_displacement"
test -f "${EVAL_DIR}/per_sample_metrics.csv"
test -d "${DATA_ROOT}"
test -d "${DIST_DIR}"
test -f "${DIST_META}"
test -f "${AFFINE_JSON}"

echo -n "baseline_pred_abs_count="
find "${EVAL_DIR}/predictions_absolute_integrated" -maxdepth 1 -name '*_pred_absolute_integrated.npy' | wc -l

echo -n "distance_field_count="
find "${DIST_DIR}" -maxdepth 1 -name '*_distance.npy' | wc -l

"${PYTHON}" - <<PY
import pandas as pd
meta = pd.read_csv("${DIST_META}")
base = pd.read_csv("${EVAL_DIR}/per_sample_metrics.csv")
print("distance_metadata_rows=", len(meta))
print("baseline_per_sample_rows=", len(base))
print("baseline_test_rows=", int((base["split"] == "test").sum()))
print("distance_metadata_columns=", list(meta.columns))
PY

ARGS=(
  --evaluation-dir "${EVAL_DIR}"
  --data-root "${DATA_ROOT}"
  --crop-mode oracle_bbox
  --distance-field-dir "${DIST_DIR}"
  --distance-field-metadata "${DIST_META}"
  --raw-to-osm-affine-json "${AFFINE_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --method pointwise_grid
  --search-radius-m "${SEARCH_RADIUS}"
  --search-step-m "${SEARCH_STEP}"
  --w-road "${W_ROAD}"
  --w-ref "${W_REF}"
  --w-curvature "${W_CURVATURE}"
  --w-step "${W_STEP}"
  --w-heading "${W_HEADING}"
  --w-terminal "${W_TERMINAL}"
  --terminal-power "${TERMINAL_POWER}"
  --debug-samples "${DEBUG_SAMPLES}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "===== Run map-aware refinement ====="
"${PYTHON}" -u "${SCRIPT}" "${ARGS[@]}"

echo "===== Output check ====="
test -f "${OUTPUT_DIR}/refinement_per_sample_metrics.csv"
test -f "${OUTPUT_DIR}/refinement_summary.json"
test -f "${OUTPUT_DIR}/refinement_summary.md"
test -d "${OUTPUT_DIR}/best10_plots"

echo "===== Summary ====="
cat "${OUTPUT_DIR}/refinement_summary.md"

echo "===== Finished ====="
date

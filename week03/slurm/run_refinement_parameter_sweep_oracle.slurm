#!/bin/bash
#SBATCH --job-name=week03_refine_sweep
#SBATCH --partition=cpu_sapphire
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=06:00:00
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
SCRIPT="/home/jliu/Thesis/week03/scripts/run_refinement_parameter_sweep.py"
REFINER="/home/jliu/Thesis/week03/scripts/refine_trajectory_map_aware_sliding_window.py"

MAX_SAMPLES="${MAX_SAMPLES:-200}"
DEBUG_SAMPLES="${DEBUG_SAMPLES:-10}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "DEBUG_SAMPLES=${DEBUG_SAMPLES}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, pandas, matplotlib; print('numpy=', numpy.__version__); print('pandas=', pandas.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${REFINER}" "${SCRIPT}"

echo "===== Start refinement parameter sweep ====="
"${PYTHON}" -u "${SCRIPT}" \
  --evaluation-dir /home/jliu/Thesis/week03/results/baseline_raw_resnet18_bs16/evaluation \
  --data-root /home/jliu/data_no_speed_delta_displacement_paired \
  --crop-mode oracle_bbox \
  --distance-field-dir /home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields \
  --distance-field-metadata /home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields_metadata.csv \
  --raw-to-osm-affine-json /home/jliu/Thesis/week03/data/maps/raw_xy_to_osm_affine.json \
  --output-root /home/jliu/Thesis/week03/results/refinement_sweep_oracle \
  --max-samples "${MAX_SAMPLES}" \
  --debug-samples "${DEBUG_SAMPLES}" \
  --w-terminal 0.2 \
  --terminal-power 2.0 \
  --plot-distance-background

echo "===== Output check ====="
test -f /home/jliu/Thesis/week03/results/refinement_sweep_oracle/sweep_summary.csv
test -f /home/jliu/Thesis/week03/results/refinement_sweep_oracle/sweep_summary.md

echo "===== Finished ====="
date

#!/bin/bash
#SBATCH --job-name=week05_delta_data
#SBATCH --partition=cpu_skylake
#SBATCH --output=/home/jliu/Thesis/week05/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /home/jliu/Thesis

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
SCRIPT="/home/jliu/Thesis/week05/scripts/build_delta_displacement_paired_data.py"
INPUT="${INPUT:-/home/jliu/Thesis/week05/data/vehicle_positions_TS_New.csv}"
OUTPUT="${OUTPUT:-/home/jliu/Thesis/week05/data/delta_displacement_paired}"
ID_COLUMN="${ID_COLUMN:-}"
TIME_COLUMN="${TIME_COLUMN:-}"
X_COLUMN="${X_COLUMN:-}"
Y_COLUMN="${Y_COLUMN:-}"
MIN_POINTS="${MIN_POINTS:-2}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"

mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "INPUT=${INPUT}"
echo "OUTPUT=${OUTPUT}"
echo "MIN_POINTS=${MIN_POINTS}"
echo "TRAIN_RATIO=${TRAIN_RATIO}"
echo "VAL_RATIO=${VAL_RATIO}"
echo "SEED=${SEED}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, pandas, matplotlib; print('numpy=', numpy.__version__); print('pandas=', pandas.__version__); print('matplotlib=', matplotlib.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Input check ====="
test -f "${INPUT}"

EXTRA_ARGS=()
if [[ -n "${ID_COLUMN}" ]]; then
  EXTRA_ARGS+=(--id-column "${ID_COLUMN}")
fi
if [[ -n "${TIME_COLUMN}" ]]; then
  EXTRA_ARGS+=(--time-column "${TIME_COLUMN}")
fi
if [[ -n "${X_COLUMN}" ]]; then
  EXTRA_ARGS+=(--x-column "${X_COLUMN}")
fi
if [[ -n "${Y_COLUMN}" ]]; then
  EXTRA_ARGS+=(--y-column "${Y_COLUMN}")
fi

echo "===== Build week05 delta-displacement paired data ====="
"${PYTHON}" -u "${SCRIPT}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --min-points "${MIN_POINTS}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --seed "${SEED}" \
  --overwrite \
  "${EXTRA_ARGS[@]}"

echo "===== Output check ====="
test -f "${OUTPUT}/metadata.csv"
test -f "${OUTPUT}/dataset_summary.json"
test -f "${OUTPUT}/normalization_diagnostics.json"
test -f "${OUTPUT}/normalization_plots/dx_distribution.png"
test -f "${OUTPUT}/normalization_plots/dy_distribution.png"
test -f "${OUTPUT}/unique_vehicle_ids.csv"
test -f "${OUTPUT}/splits/train_ids.csv"
test -f "${OUTPUT}/splits/val_ids.csv"
test -f "${OUTPUT}/splits/test_ids.csv"
echo -n "absolute_label_count="
find "${OUTPUT}/labels_absolute" -maxdepth 1 -name '*.npy' | wc -l
echo -n "delta_label_count="
find "${OUTPUT}/labels_delta_displacement" -maxdepth 1 -name '*.npy' | wc -l
echo -n "start_count="
find "${OUTPUT}/starts" -maxdepth 1 -name '*.npy' | wc -l

echo "===== Finished ====="
date

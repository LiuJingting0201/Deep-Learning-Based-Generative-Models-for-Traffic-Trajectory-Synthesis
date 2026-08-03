#!/bin/bash
#SBATCH --job-name=week04_gasf_full
#SBATCH --partition=cpu_skylake
#SBATCH --output=/home/jliu/Thesis/week04/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week04/logs/%x_%j.err
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
export MPLCONFIGDIR=/home/jliu/Thesis/week04/logs/matplotlib

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
SCRIPT="/home/jliu/Thesis/week04/scripts/build_constrained_gasf_dataset.py"
INPUT_ROOT="${INPUT_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
HEATMAP_SIGMA="${HEATMAP_SIGMA:-3.0}"
SPLIT="${SPLIT:-all}"

mkdir -p /home/jliu/Thesis/week04/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "IMAGE_SIZE=${IMAGE_SIZE}"
echo "HEATMAP_SIGMA=${HEATMAP_SIGMA}"
echo "SPLIT=${SPLIT}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, PIL, matplotlib; print('numpy=', numpy.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Input check ====="
test -d "${INPUT_ROOT}/labels_absolute"
test -d "${INPUT_ROOT}/labels_delta_displacement"
test -d "${INPUT_ROOT}/splits"
test -f "${INPUT_ROOT}/metadata.csv"

echo "===== Build complete constrained GASF dataset ====="
"${PYTHON}" -u "${SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --image-size "${IMAGE_SIZE}" \
  --heatmap-sigma "${HEATMAP_SIGMA}" \
  --split "${SPLIT}"

echo "===== Output check ====="
test -f "${OUTPUT_ROOT}/constrained_gasf_normalization.json"
test -f "${OUTPUT_ROOT}/metadata.csv"
test -f "${OUTPUT_ROOT}/skipped_samples.csv"
test -f "${OUTPUT_ROOT}/sanity_checks/sanity_check_metrics.csv"
test -f "${OUTPUT_ROOT}/README.md"
echo -n "image_png_count="
find "${OUTPUT_ROOT}/images" -maxdepth 1 -name '*.png' | wc -l
echo -n "float_array_count="
find "${OUTPUT_ROOT}/arrays" -maxdepth 1 -name '*.npy' | wc -l
echo -n "absolute_label_count="
find "${OUTPUT_ROOT}/labels_absolute" -maxdepth 1 -name '*.npy' | wc -l
echo -n "delta_label_count="
find "${OUTPUT_ROOT}/labels_delta_displacement" -maxdepth 1 -name '*.npy' | wc -l
echo -n "start_count="
find "${OUTPUT_ROOT}/starts" -maxdepth 1 -name '*.npy' | wc -l

echo "===== Finished ====="
date

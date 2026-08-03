#!/bin/bash
#SBATCH --job-name=week04_gasf_sanity
#SBATCH --partition=cpu_skylake
#SBATCH --output=/home/jliu/Thesis/week04/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week04/logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

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
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_SANITY_CHECKS="${NUM_SANITY_CHECKS:-20}"

mkdir -p /home/jliu/Thesis/week04/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "IMAGE_SIZE=${IMAGE_SIZE}"
echo "NUM_SANITY_CHECKS=${NUM_SANITY_CHECKS}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, PIL, matplotlib; print('numpy=', numpy.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Existing dataset check ====="
test -f "${OUTPUT_ROOT}/constrained_gasf_normalization.json"
test -d "${OUTPUT_ROOT}/images"
test -d "${OUTPUT_ROOT}/arrays"
test -d "${OUTPUT_ROOT}/labels_absolute"
test -d "${OUTPUT_ROOT}/labels_delta_displacement"

echo "===== Refresh sanity diagnostics only ====="
"${PYTHON}" -u "${SCRIPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --image-size "${IMAGE_SIZE}" \
  --num-sanity-checks "${NUM_SANITY_CHECKS}" \
  --sanity-only

echo "===== Output check ====="
test -f "${OUTPUT_ROOT}/sanity_checks/sanity_check_metrics.csv"
head -3 "${OUTPUT_ROOT}/sanity_checks/sanity_check_metrics.csv"

echo "===== Finished ====="
date

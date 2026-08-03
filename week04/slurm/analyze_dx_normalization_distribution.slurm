#!/bin/bash
#SBATCH --job-name=week04_dx_norm
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
SCRIPT="/home/jliu/Thesis/week04/scripts/analyze_dx_normalization_distribution.py"
DATA_ROOT="${DATA_ROOT:-/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start}"
EPSILON="${EPSILON:-0.001}"

mkdir -p /home/jliu/Thesis/week04/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "EPSILON=${EPSILON}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy, matplotlib; print('numpy=', numpy.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

echo "===== Analyze dx normalization distribution ====="
"${PYTHON}" -u "${SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --epsilon "${EPSILON}"

echo "===== Output preview ====="
cat "${DATA_ROOT}/sanity_checks/dx_normalization_distribution/dx_normalization_summary.json"

echo "===== Finished ====="
date

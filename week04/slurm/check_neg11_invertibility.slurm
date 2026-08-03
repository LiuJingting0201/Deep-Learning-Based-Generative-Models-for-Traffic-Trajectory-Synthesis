#!/bin/bash
#SBATCH --job-name=week04_neg11_inv
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

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
SCRIPT="/home/jliu/Thesis/week04/scripts/check_neg11_invertibility.py"
DATA_ROOT="${DATA_ROOT:-/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start_neg11}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p /home/jliu/Thesis/week04/logs

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MAX_SAMPLES=${MAX_SAMPLES:-<all>}"

echo "===== Python check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import numpy; print('numpy=', numpy.__version__)"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile "${SCRIPT}"

ARGS=(--data-root "${DATA_ROOT}")
if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

echo "===== Check neg11 invertibility ====="
"${PYTHON}" -u "${SCRIPT}" "${ARGS[@]}"

echo "===== Output preview ====="
head -5 "${DATA_ROOT}/sanity_checks/neg11_invertibility_metrics.csv"
cat "${DATA_ROOT}/sanity_checks/neg11_invertibility_summary.json"

echo "===== Finished ====="
date

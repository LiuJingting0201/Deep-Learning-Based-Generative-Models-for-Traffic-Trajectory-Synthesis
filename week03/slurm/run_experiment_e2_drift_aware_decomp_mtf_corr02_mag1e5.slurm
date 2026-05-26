#!/bin/bash
#SBATCH --job-name=week03_e2_corr02
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=1-00:00:00
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

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="/home/jliu/data_no_speed_delta_displacement_paired"
OUTPUT_DIR="/home/jliu/Thesis/week03/results/experiment_e2_drift_aware_decomp_mtf_local_resnet18_bs16_corr02_mag1e5"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo "===== Experiment config ====="
echo "lambda_abs=0.1"
echo "lambda_corr_target=0.2"
echo "lambda_smooth=0.001"
echo "lambda_mag=0.00001"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/drift_aware_decomp_mtf_resnet18_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_drift_aware_decomp_mtf_resnet18_decoder_e2.py

echo "===== Data check ====="
test -d "${DATA_ROOT}/images"
test -f "${DATA_ROOT}/metadata.csv"
test -f "${DATA_ROOT}/splits/split_metadata.csv"

echo "===== Start Experiment E2 decomp+MTF-local stronger correction training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_drift_aware_decomp_mtf_resnet18_decoder_e2.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 1000 \
  --batch-size 16 \
  --lr 1e-4 \
  --dropout 0.3 \
  --correction-anchors 16 \
  --lambda-abs 0.1 \
  --lambda-corr-target 0.2 \
  --lambda-smooth 0.001 \
  --lambda-mag 0.00001 \
  --early-stopping-patience 80 \
  --metrics-every 5 \
  --seed 42 \
  --device auto \
  --num-workers 4

echo "===== Finished ====="
date
if [ -f "${OUTPUT_DIR}/week3_e2_report.md" ]; then
  cat "${OUTPUT_DIR}/week3_e2_report.md"
fi
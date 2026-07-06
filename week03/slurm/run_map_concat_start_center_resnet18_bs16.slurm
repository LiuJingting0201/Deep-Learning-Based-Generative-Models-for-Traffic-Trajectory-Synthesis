#!/bin/bash
#SBATCH --job-name=week03_map_start_r18
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=05:00:00
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

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
MAP_DIR="${MAP_DIR:-/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/map_aware_decoder/start_center}"
RUN_NAME="${RUN_NAME:-raw_resnet18_bs16_start_center}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MAP_DIR=${MAP_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RUN_NAME=${RUN_NAME}"

echo "===== GPU info ====="
nvidia-smi

echo "===== Python/CUDA check ====="
"${PYTHON}" -c "import sys; print(sys.executable)"
"${PYTHON}" -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/structure_aware_resnet18_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py

echo "===== Map data check ====="
test -d "${MAP_DIR}"
test -f "${DATA_ROOT}/maps_start_center_extents.csv"

echo "===== Start map concat start_center raw ResNet18 training ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py \
  --model-type raw_resnet18 \
  --use-map \
  --map-fusion concat \
  --map-dir "${MAP_DIR}" \
  --run-name "${RUN_NAME}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-1000}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-1e-4}" \
  --dropout "${DROPOUT:-0.3}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-80}" \
  --metrics-every "${METRICS_EVERY:-5}" \
  --seed "${SEED:-42}" \
  --device auto \
  --num-workers "${NUM_WORKERS:-4}"

echo "===== Finished ====="
date

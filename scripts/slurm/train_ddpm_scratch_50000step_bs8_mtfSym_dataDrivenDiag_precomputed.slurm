#!/bin/bash
#SBATCH --job-name=ddpm_scratch_50000step_bs8_mtfSym_dataDiag
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/logs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
RUN_NAME="ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed"
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
DATA_DIR="$DATA_ROOT/images"
SPLIT_METADATA="$DATA_ROOT/splits/split_metadata.csv"
DIAG_TARGET_FILE="$DATA_ROOT/diagnostics/diag_targets_train_split.json"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "SCRATCH=$SCRATCH"
echo "PYTHON=$PYTHON"
echo "===== Python / CUDA ====="
which python || true
"$PYTHON" -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')"
nvidia-smi || true
echo "===== Git ====="
git branch --show-current || true
git rev-parse HEAD || true
git status --short || true

echo "===== Data check ====="
echo "DATA_DIR=$DATA_DIR"
echo "SPLIT_METADATA=$SPLIT_METADATA"
echo "DIAG_TARGET_FILE=$DIAG_TARGET_FILE"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: missing DATA_DIR: $DATA_DIR" >&2
  exit 1
fi
if [[ ! -f "$SPLIT_METADATA" ]]; then
  echo "ERROR: missing SPLIT_METADATA: $SPLIT_METADATA" >&2
  exit 1
fi
if [[ ! -f "$DIAG_TARGET_FILE" ]]; then
  echo "ERROR: missing DIAG_TARGET_FILE: $DIAG_TARGET_FILE" >&2
  exit 1
fi
find "$DATA_DIR" -maxdepth 1 -name '*.png' | awk 'NR <= 10 {print}'
echo "Diag target JSON:"
cat "$DIAG_TARGET_FILE"

echo "===== Start training ====="
"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_DIR" \
  --run-name "$RUN_NAME" \
  --aux-loss-mode data_driven_diag_mtf_symmetry \
  --split-metadata "$SPLIT_METADATA" \
  --diag-target-file "$DIAG_TARGET_FILE" \
  --diag-weight 0.01 \
  --mtf-symmetry-weight 0.01 \
  --image-size 224 \
  --batch-size 8 \
  --lr 1e-4 \
  --max-train-steps 50000 \
  --gradient-accumulation-steps 2 \
  --save-every 2500 \
  --sample-every 2500 \
  --sample-inference-steps 150 \
  --copy-results-to-home

echo "===== Finished ====="
date

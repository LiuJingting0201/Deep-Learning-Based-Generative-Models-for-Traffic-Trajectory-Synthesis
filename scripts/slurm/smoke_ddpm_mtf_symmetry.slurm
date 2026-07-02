#!/bin/bash
#SBATCH --job-name=smoke_ddpm_mtf_symmetry
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
DATA_DIR="$DATA_ROOT/images"
SPLIT_METADATA="$DATA_ROOT/splits/split_metadata.csv"
DIAG_TARGET_FILE="$DATA_ROOT/diagnostics/diag_targets_train_split.json"
SMOKE_ROOT="$SCRATCH/Thesis_runs/smoke_mtf_symmetry"

echo "===== Smoke mtf_symmetry ====="
"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$SMOKE_ROOT/mtf_symmetry" \
  --aux-loss-mode mtf_symmetry \
  --mtf-symmetry-weight 0.01 \
  --image-size 224 \
  --batch-size 2 \
  --lr 1e-4 \
  --max-train-steps 2 \
  --gradient-accumulation-steps 1 \
  --save-every 1 \
  --sample-every 1 \
  --sample-inference-steps 2 \
  --num-sample-images 2 \
  --log-every 1

echo "===== Smoke data_driven_diag_mtf_symmetry ====="
"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$SMOKE_ROOT/data_driven_diag_mtf_symmetry" \
  --aux-loss-mode data_driven_diag_mtf_symmetry \
  --split-metadata "$SPLIT_METADATA" \
  --diag-target-file "$DIAG_TARGET_FILE" \
  --diag-weight 0.01 \
  --mtf-symmetry-weight 0.01 \
  --image-size 224 \
  --batch-size 2 \
  --lr 1e-4 \
  --max-train-steps 2 \
  --gradient-accumulation-steps 1 \
  --save-every 1 \
  --sample-every 1 \
  --sample-inference-steps 2 \
  --num-sample-images 2 \
  --log-every 1

echo "===== Smoke output checks ====="
find "$SMOKE_ROOT" -maxdepth 3 -type f \( -name 'train_log.jsonl' -o -name '*.png' \) | sort
echo "===== mtf_symmetry log ====="
cat "$SMOKE_ROOT/mtf_symmetry/train_log.jsonl"
echo "===== data_driven_diag_mtf_symmetry log ====="
cat "$SMOKE_ROOT/data_driven_diag_mtf_symmetry/train_log.jsonl"

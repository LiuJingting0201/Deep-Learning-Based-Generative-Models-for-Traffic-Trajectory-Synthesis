#!/bin/bash
#SBATCH --job-name=delta_resnet18_baseline
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p logs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
OUTPUT_DIR="results_hpc/delta_displacement_resnet18_decoder_ablation"

echo "===== Job info ====="
hostname
date
echo "SCRATCH=$SCRATCH"
echo "PWD=$(pwd)"

echo "===== Python check ====="
which python || true
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import sys, torch; print(torch.__version__); print(torch.cuda.is_available()); sys.exit('CUDA is not available on this allocated node') if not torch.cuda.is_available() else print(torch.cuda.get_device_name(0))"

echo "===== Compile check ====="
"$PYTHON" -m py_compile scripts/run_delta_displacement_resnet18_decoder_ablation.py
"$PYTHON" -m py_compile scripts/evaluate_affine_upper_bound.py

echo "===== Data check ====="
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [[ ! -d "$DATA_ROOT/images" ]]; then
  echo "Missing image directory: $DATA_ROOT/images" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/metadata.csv" ]]; then
  echo "Missing metadata.csv: $DATA_ROOT/metadata.csv" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/splits/split_metadata.csv" ]]; then
  echo "Missing split_metadata.csv: $DATA_ROOT/splits/split_metadata.csv" >&2
  exit 1
fi

echo "Split metadata preview:"
head -n 5 "$DATA_ROOT/splits/split_metadata.csv" || true

echo "Number of images:"
find "$DATA_ROOT/images" -maxdepth 1 -type f | wc -l

echo "===== Start training ====="
"$PYTHON" scripts/run_delta_displacement_resnet18_decoder_ablation.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 1000 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --dropout 0.3 \
  --early-stopping-patience 80 \
  --early-stopping-min-delta 1e-6 \
  --metrics-every 5 \
  --seed 42 \
  --device auto \
  --num-workers 8

echo "===== Run affine upper bound analysis ====="
"$PYTHON" scripts/evaluate_affine_upper_bound.py \
  --experiment-dir "$OUTPUT_DIR" \
  --split test

echo "===== Finished ====="
date

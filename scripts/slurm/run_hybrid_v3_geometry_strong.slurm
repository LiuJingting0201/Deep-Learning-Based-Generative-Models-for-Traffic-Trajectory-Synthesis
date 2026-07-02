#!/bin/bash
#SBATCH --job-name=hyb_v3_strong
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd "$HOME/Thesis"
mkdir -p "$HOME/Thesis/logs"

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="$HOME/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
OUTPUT_DIR="results_hpc/hybrid_multitask_delta_decoder_v3_geometry_strong"

echo "===== Job info ====="
hostname
date
echo "HOME=$HOME"
echo "SCRATCH=$SCRATCH"
echo "PWD=$(pwd)"

echo "===== Python check ====="
which python || true
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

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

echo "===== Compile check ====="
"$PYTHON" -m py_compile scripts/run_hybrid_multitask_delta_decoder_v3_geometry.py

echo "===== Start training ====="
"$PYTHON" scripts/run_hybrid_multitask_delta_decoder_v3_geometry.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 1000 \
  --batch-size 32 \
  --num-workers 8 \
  --lambda-heading 0.1 \
  --lambda-turn 0.3 \
  --turn-sharp-weight 3.0 \
  --turn-weight-max 3.0

echo "===== Finished ====="
date

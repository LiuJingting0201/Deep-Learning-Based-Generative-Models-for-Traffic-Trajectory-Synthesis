#!/bin/bash
#SBATCH --job-name=ddpm_scratch_10000step_bs8_dataDrivenDiag
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
DATA_DIR="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images"
SPLIT_METADATA="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/splits/split_metadata.csv"
echo "DATA_DIR=$DATA_DIR"
echo "SPLIT_METADATA=$SPLIT_METADATA"

find "$DATA_DIR" -maxdepth 1 -type f | head -n 10 || true

echo "Number of images:"
find "$DATA_DIR" -maxdepth 1 -type f | wc -l

echo "Split metadata preview:"
head -n 5 "$SPLIT_METADATA" || true

echo "===== Start training ====="

"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images" \
  --run-name ddpm_scratch_10000step_bs8_dataDrivenDiag \
  --aux-loss-mode data_driven_diag \
  --split-metadata "$SPLIT_METADATA" \
  --image-size 224 \
  --batch-size 8 \
  --lr 1e-4 \
  --max-train-steps 10000 \
  --gradient-accumulation-steps 2 \
  --save-every 1000 \
  --sample-every 1000 \
  --sample-inference-steps 20 \
  --copy-results-to-home

echo "===== Finished ====="
date

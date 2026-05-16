#!/bin/bash
#SBATCH --job-name=precompute_diag_targets
#SBATCH --partition=cpu_sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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

echo "===== Precompute diag targets ====="

"$PYTHON" scripts/precompute_diag_targets.py \
  --data-dir "$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images" \
  --split-metadata "$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/splits/split_metadata.csv" \
  --output-file "$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/diagnostics/diag_targets_train_split.json" \
  --image-size 224

echo "===== Finished ====="
date

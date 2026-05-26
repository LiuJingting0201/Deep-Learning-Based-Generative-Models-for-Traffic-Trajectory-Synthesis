#!/bin/bash
#SBATCH --job-name=week03_channel_stats
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib
mkdir -p "${MPLCONFIGDIR}"

DATA_DIR="${DATA_DIR:-/home/jliu/data_no_speed_delta_displacement_paired}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jliu/Thesis/week03/results/channel_structure_stats}"
NUM_VISUALIZATIONS="${NUM_VISUALIZATIONS:-8}"
SEED="${SEED:-42}"

ARGS=(
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-visualizations "${NUM_VISUALIZATIONS}"
  --seed "${SEED}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

python /home/jliu/Thesis/week03/scripts/compute_channel_structure_stats.py "${ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=week03_smoke_train_decomp
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib
mkdir -p "${MPLCONFIGDIR}"

python -u /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py \
  --model-type decomp_resnet18 \
  --data-root /home/jliu/data_no_speed_delta_displacement_paired \
  --output-dir /home/jliu/Thesis/week03/results/smoke_train_decomp_resnet18 \
  --epochs 1 \
  --batch-size 2 \
  --lr 1e-4 \
  --dropout 0.3 \
  --num-workers 0 \
  --metrics-every 1 \
  --seed 42 \
  --device auto

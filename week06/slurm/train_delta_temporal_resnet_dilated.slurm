#!/bin/bash
#SBATCH --job-name=w06_delta_dilated
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week06/logs/w06_delta_dilated_%j.out
#SBATCH --error=/home/jliu/Thesis/week06/logs/w06_delta_dilated_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week06/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_ROOT="/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
OUTPUT_DIR="/home/jliu/Thesis/week06/results/raw_delta_temporal_resnet_dilated"
mkdir -p /home/jliu/Thesis/week06/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/train_delta_displacement_diffusion.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-type temporal_resnet_dilated \
  --split train \
  --sequence-length 224 \
  --batch-size 64 \
  --lr 2e-4 \
  --max-train-steps 50000 \
  --gradient-accumulation-steps 1 \
  --hidden-channels 128 \
  --num-blocks 8 \
  --time-dim 256 \
  --num-train-timesteps 1000 \
  --lr-warmup-steps 500 \
  --sample-every 2000 \
  --save-every 2000 \
  --num-sample-sequences 1000 \
  --sample-inference-steps 100 \
  --use-ema \
  --seed 42

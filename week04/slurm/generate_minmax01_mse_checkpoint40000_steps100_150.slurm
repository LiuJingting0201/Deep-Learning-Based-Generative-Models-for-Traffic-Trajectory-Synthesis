#!/bin/bash
#SBATCH --job-name=w4_mse_resample
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jliu/Thesis/week04/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week04/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/week04/logs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
CHECKPOINT="/home/jliu/Thesis/week04/runs_light/week04_gasf_minmax01_mse/checkpoint-40000"
OUT_ROOT="/home/jliu/Thesis/week04/resampled_samples/minmax01_mse_checkpoint40000"

for STEPS in 100 150; do
  "$PYTHON" /home/jliu/Thesis/scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$CHECKPOINT" \
    --output-dir "$OUT_ROOT/ddpm_${STEPS}_steps" \
    --num-samples 16 \
    --batch-size 16 \
    --image-size 224 \
    --num-inference-steps "$STEPS" \
    --scheduler ddpm \
    --use-ema \
    --seed 40042 \
    --filename-prefix "step40000_ddpm${STEPS}_"
done

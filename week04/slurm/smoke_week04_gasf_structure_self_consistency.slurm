#!/bin/bash
#SBATCH --job-name=week04_gasf_smoke
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

"$PYTHON" /home/jliu/Thesis/week04/scripts/train_week04_ddpm.py \
  --data-dir /home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start/images \
  --output-dir /home/jliu/Thesis/week04/smoke_gasf_structure \
  --aux-loss-mode gasf_structure \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-train-steps 2 \
  --image-size 224 \
  --sample-every 0 \
  --save-every 0 \
  --log-every 1

#!/bin/bash
#SBATCH --job-name=week04_neg11_sym
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
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
  --data-dir /home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start_neg11/images \
  --run-name week04_gasf_neg11_symmetry \
  --aux-loss-mode symmetry \
  --symmetry-weight 0.01 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --max-train-steps 50000 \
  --image-size 224 \
  --sample-every 2000 \
  --save-every 2000 \
  --use-ema

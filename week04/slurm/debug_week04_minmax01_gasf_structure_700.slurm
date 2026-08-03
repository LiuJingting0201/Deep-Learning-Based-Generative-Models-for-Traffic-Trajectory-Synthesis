#!/bin/bash
#SBATCH --job-name=debug_w4_minmax_gasf
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/jliu/Thesis/week04/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week04/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/week04/logs /home/jliu/Thesis/week04/debug_runs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"

"$PYTHON" /home/jliu/Thesis/week04/scripts/train_week04_ddpm.py \
  --data-dir /home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start/images \
  --output-dir /home/jliu/Thesis/week04/debug_runs/minmax01_gasf_structure_safe_700 \
  --aux-loss-mode gasf_structure \
  --symmetry-weight 0.01 \
  --diag-weight 0.01 \
  --gasf-diag-eps 1e-4 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --max-train-steps 700 \
  --image-size 224 \
  --sample-every 0 \
  --save-every 0 \
  --log-every 10

#!/bin/bash
#SBATCH --job-name=ddpm_scratch_50000step_bs8_mtfSym_w001
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis
mkdir -p /home/jliu/Thesis/logs

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
DATA_DIR="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images"

"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_DIR" \
  --run-name ddpm_scratch_50000step_bs8_mtfSym_w001 \
  --aux-loss-mode mtf_symmetry \
  --mtf-symmetry-weight 0.01 \
  --image-size 224 \
  --batch-size 8 \
  --lr 1e-4 \
  --max-train-steps 50000 \
  --gradient-accumulation-steps 2 \
  --save-every 2500 \
  --sample-every 2500 \
  --sample-inference-steps 150 \
  --copy-results-to-home

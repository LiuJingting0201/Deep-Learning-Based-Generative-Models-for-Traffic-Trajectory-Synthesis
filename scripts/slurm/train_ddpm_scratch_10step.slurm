#!/bin/bash
#SBATCH --job-name=ddpm_scratch_10step
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

mkdir -p "$HOME/Thesis/logs"
exec > "$HOME/Thesis/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
exec 2> "$HOME/Thesis/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

cd "$HOME/Thesis"

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

python scripts/train_ddpm_scratch.py \
  --data-dir "$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images" \
  --run-name ddpm_scratch_10step \
  --image-size 224 \
  --batch-size 2 \
  --lr 1e-4 \
  --max-train-steps 10 \
  --gradient-accumulation-steps 2 \
  --save-every 10 \
  --sample-every 10 \
  --copy-results-to-home

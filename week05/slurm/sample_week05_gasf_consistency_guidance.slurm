#!/bin/bash
#SBATCH --job-name=week05_gasf_guidance
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week05/logs/week05_gasf_guidance_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/week05_gasf_guidance_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}" /home/jliu/Thesis/week05/guidance_consistency_results

echo "===== Week05 GASF consistency guidance inference ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CHECKPOINT=/home/jliu/Thesis/week05/results/week05_sig15_mse_diag/checkpoint-40000"
echo "BASELINE_FID=44.549600"
echo "GUIDANCE_SCALES=0.05 0.1 0.2 0.5"
echo "GUIDANCE_START_FRACTION=0.5"
echo "GUIDANCE_GRAD_CLIP=0.1"
echo "NUM_SAMPLES=1000"

"${PYTHON}" -u /home/jliu/Thesis/week05/scripts/sample_week05_gasf_consistency_guidance.py \
  --checkpoint-dir /home/jliu/Thesis/week05/results/week05_sig15_mse_diag/checkpoint-40000 \
  --data-root /home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224 \
  --normalization-file /home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224/normalization_diagnostics.json \
  --output-dir /home/jliu/Thesis/week05/guidance_consistency_results_gradclip010 \
  --guidance-scales 0.05 0.1 0.2 0.5 \
  --guidance-start-fraction 0.5 \
  --guidance-grad-clip 0.1 \
  --sigmoid-k 1.5 \
  --num-samples 1000 \
  --sample-batch-size 8 \
  --fid-batch-size 32 \
  --image-size 224 \
  --sample-inference-steps 50 \
  --seed 42 \
  --real-split all \
  --num-workers 4

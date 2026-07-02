#!/bin/bash
#SBATCH --job-name=week05_fid_npy
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week05/logs/week05_fid_npy_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/week05_fid_npy_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}" /home/jliu/Thesis/week05/fid_results

echo "===== Week05 .npy FID evaluation ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "REAL_SPLIT=all"
echo "REAL_DATA_ROOT=/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
echo "SAMPLES_ROOT=/home/jliu/Thesis/week05/results_npy_samples"
echo "OUTPUT_DIR=/home/jliu/Thesis/week05/fid_results"
echo "FID_METHOD=scripts/evaluate_fid.py Inception-v3 mean/cov sqrtm"

"${PYTHON}" -u /home/jliu/Thesis/week05/scripts/evaluate_week05_npy_fid.py \
  --data-root /home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224 \
  --samples-root /home/jliu/Thesis/week05/results_npy_samples \
  --output-dir /home/jliu/Thesis/week05/fid_results \
  --real-split all \
  --image-size 224 \
  --batch-size 32 \
  --num-workers 4

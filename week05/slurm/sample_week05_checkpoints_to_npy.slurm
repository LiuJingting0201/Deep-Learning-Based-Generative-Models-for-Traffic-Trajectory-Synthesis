#!/bin/bash
#SBATCH --job-name=week05_sample_npy
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week05/logs/week05_sample_npy_%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week05/logs/week05_sample_npy_%x_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/home/jliu/Thesis/week05/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
: "${CONFIG:?Set CONFIG to one week05 result configuration name.}"

mkdir -p /home/jliu/Thesis/week05/logs "${MPLCONFIGDIR}" /home/jliu/Thesis/week05/results_npy_samples

echo "===== Week05 checkpoint .npy sampling ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CONFIG=${CONFIG}"
echo "CHECKPOINT_RANGE=40000..50000"
echo "NUM_SAMPLES_PER_CHECKPOINT=1000"
echo "OUTPUT_ROOT=/home/jliu/Thesis/week05/results_npy_samples"
echo "OUTPUT_FORMAT=.npy float32 NHWC [0,1], no PNG"

"${PYTHON}" -u /home/jliu/Thesis/week05/scripts/sample_week05_checkpoints_to_npy.py \
  --results-root /home/jliu/Thesis/week05/results \
  --output-root /home/jliu/Thesis/week05/results_npy_samples \
  --configs "${CONFIG}" \
  --min-step 40000 \
  --max-step 50000 \
  --num-samples 1000 \
  --batch-size 8 \
  --image-size 224 \
  --sample-inference-steps 50 \
  --seed 42

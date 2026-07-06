#!/bin/bash
#SBATCH --job-name=w06_raw_guidance
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/jliu/Thesis/week06/logs/w06_raw_guidance_%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week06/logs/w06_raw_guidance_%x_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jliu/Thesis/src:/home/jliu/Thesis/week06/scripts
export MPLCONFIGDIR=/home/jliu/Thesis/week06/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
REAL_ROOT="/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
GUIDANCE_STRENGTH="0.1"
GUIDANCE_LABEL="0p1"

: "${RUN:?Set RUN to one of the week06 raw diffusion result directory names.}"
RUN_DIR="/home/jliu/Thesis/week06/results/${RUN}"
DECODED_DIR="/home/jliu/Thesis/week06/decoded/raw_guided/${RUN}_guided_strength${GUIDANCE_LABEL}"
EVAL_DIR="/home/jliu/Thesis/week06/results/raw_guided/${RUN}_guided_strength${GUIDANCE_LABEL}_empirical_support_eval"
VIS_DIR="/home/jliu/Thesis/week06/results/raw_guided/${RUN}_guided_strength${GUIDANCE_LABEL}_visualizations_empirical_support"

mkdir -p /home/jliu/Thesis/week06/logs "${MPLCONFIGDIR}" /home/jliu/Thesis/week06/results/raw_guided

echo "===== Week06 raw trajectory guided inference ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "RUN=${RUN}"
nvidia-smi -L || true

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/sample_raw_diffusion_guidance.py \
  --run-dir "${RUN_DIR}" \
  --real-data-root "${REAL_ROOT}" \
  --output-root /home/jliu/Thesis/week06/decoded/raw_guided \
  --guidance-strength "${GUIDANCE_STRENGTH}" \
  --num-guidance-steps 2 \
  --guidance-start-fraction 0.5 \
  --guidance-end-fraction 1.0 \
  --distance-scale 25 \
  --distance-field-size 1024 \
  --support-radius-m 5 \
  --num-samples 1000 \
  --batch-size 64 \
  --sample-inference-steps 100 \
  --seed 42 \
  --overwrite

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/evaluate_decoded_empirical_support_alignment.py \
  --decoded-dir "${DECODED_DIR}" \
  --real-data-root "${REAL_ROOT}" \
  --output-dir "${EVAL_DIR}" \
  --max-generated 1000 \
  --max-real 1000

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/visualize_decoded_trajectories_on_empirical_map.py \
  --decoded-dir "${DECODED_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --real-data-root "${REAL_ROOT}" \
  --output-dir "${VIS_DIR}" \
  --max-background-trajectories 1200 \
  --num-random 6 \
  --num-real 6 \
  --selection-metric support_mean_m

date
echo "done"

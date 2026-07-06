#!/bin/bash
#SBATCH --job-name=w06_gasf_road_guided_full
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/jliu/Thesis/week06/logs/w06_gasf_road_guided_full_%j.out
#SBATCH --error=/home/jliu/Thesis/week06/logs/w06_gasf_road_guided_full_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jliu/Thesis/src
export MPLCONFIGDIR=/home/jliu/Thesis/week06/logs/matplotlib

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
OUTPUT_DIR="/home/jliu/Thesis/week06/results/week05_sig15_mse_diag_road_guidance_full"
SAMPLES="${OUTPUT_DIR}/samples/week05_sig15_mse_diag_step_40000_road_guidance_1000_samples_1000.npy"
DECODED_DIR="/home/jliu/Thesis/week06/decoded/week05_sig15_mse_diag_step_040000_road_guidance_strength1000"
EVAL_DIR="/home/jliu/Thesis/week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength1000_empirical_support_eval"
REAL_ROOT="/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224"
CONFIG="/home/jliu/Thesis/week05/results/week05_sig15_mse_diag/startup_config.json"

mkdir -p /home/jliu/Thesis/week06/logs "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

echo "===== Week06 full GASF road-guided inference ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
nvidia-smi -L || true

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/sample_week05_gasf_road_guidance.py \
  --checkpoint-dir /home/jliu/Thesis/week05/results/week05_sig15_mse_diag/checkpoint-40000 \
  --config "${CONFIG}" \
  --real-data-root "${REAL_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --guidance-strengths 1000 \
  --num-guidance-steps 3 \
  --guidance-start-fraction 0.5 \
  --guidance-end-fraction 1.0 \
  --distance-scale 25 \
  --distance-field-size 1024 \
  --support-radius-m 5 \
  --num-samples 1000 \
  --sample-batch-size 8 \
  --image-size 224 \
  --sample-inference-steps 50 \
  --seed 42

"${PYTHON}" -u /home/jliu/Thesis/week06/scripts/decode_week05_samples_to_trajectories.py \
  --samples "${SAMPLES}" \
  --config "${CONFIG}" \
  --output-dir "${DECODED_DIR}"

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
  --output-dir /home/jliu/Thesis/week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength1000_visualizations_empirical_support \
  --max-background-trajectories 1200 \
  --num-random 6 \
  --num-real 6 \
  --selection-metric support_mean_m

date
echo "done"

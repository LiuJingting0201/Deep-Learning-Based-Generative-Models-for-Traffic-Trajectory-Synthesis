#!/bin/bash
#SBATCH --job-name=ddpm_50k_50step_matched
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jliu/Thesis/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/logs/%x_%j.err

set -euo pipefail

cd "$HOME/Thesis"
mkdir -p "$HOME/Thesis/logs"

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="$HOME/.conda/envs/thesis-diffusion/bin/python"

PURE_RUN="$HOME/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE"
DIAG_RUN="$HOME/Thesis/results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed"
PURE_OUT="$PURE_RUN/samples_50steps_matched"
DIAG_OUT="$DIAG_RUN/samples_50steps_matched"
ANALYSIS_OUT="$HOME/Thesis/results_hpc/ddpm_50k_sampling_steps_50_matched_analysis"
REAL_DIR="/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__); print('cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

generate_block() {
  local label="$1"
  local checkpoint_dir="$2"
  local output_dir="$3"
  local seed="$4"

  echo "===== ${label} ====="
  echo "checkpoint_dir=$checkpoint_dir"
  echo "output_dir=$output_dir"
  echo "seed=$seed"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Missing checkpoint directory: $checkpoint_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"

  "$PYTHON" scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --num-samples 40 \
    --batch-size 4 \
    --image-size 224 \
    --num-inference-steps 50 \
    --seed "$seed"
}

generate_block \
  "pureMSE checkpoint-25000, 50 reverse steps" \
  "$PURE_RUN/checkpoint-25000" \
  "$PURE_OUT/checkpoint-25000" \
  5025000

generate_block \
  "pureMSE checkpoint-50000, 50 reverse steps" \
  "$PURE_RUN/checkpoint-50000" \
  "$PURE_OUT/checkpoint-50000" \
  5050000

generate_block \
  "dataDrivenDiag checkpoint-25000, 50 reverse steps" \
  "$DIAG_RUN/checkpoint-25000" \
  "$DIAG_OUT/checkpoint-25000" \
  6025000

generate_block \
  "dataDrivenDiag checkpoint-50000, 50 reverse steps" \
  "$DIAG_RUN/checkpoint-50000" \
  "$DIAG_OUT/checkpoint-50000" \
  6050000

echo "===== Analysis ====="
"$PYTHON" scripts/analyze_ddpm_50k_comparison_fast.py \
  --pure-dir "$PURE_OUT" \
  --diag-dir "$DIAG_OUT" \
  --real-dir "$REAL_DIR" \
  --output-dir "$ANALYSIS_OUT"

echo "===== Finished ====="
date

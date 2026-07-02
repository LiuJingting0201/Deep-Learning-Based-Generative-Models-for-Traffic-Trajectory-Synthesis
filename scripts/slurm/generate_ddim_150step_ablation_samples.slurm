#!/bin/bash
#SBATCH --job-name=gen_ddim_150step_ablation
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
PURE_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE"
DIAG_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')"

generate_block() {
  local label="$1"
  local checkpoint_dir="$2"
  local output_dir="$3"
  local prefix="$4"
  local seed="$5"

  echo "===== ${label} ====="
  echo "checkpoint_dir=$checkpoint_dir"
  echo "output_dir=$output_dir"
  echo "prefix=$prefix"
  echo "seed=$seed"

  mkdir -p "$output_dir"
  "$PYTHON" scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --num-samples 40 \
    --batch-size 8 \
    --image-size 224 \
    --num-inference-steps 150 \
    --scheduler ddim \
    --eta 0.0 \
    --filename-prefix "$prefix" \
    --seed "$seed"
}

generate_block "PureMSE checkpoint-25000 DDIM 150" \
  "$PURE_RUN/checkpoint-25000" \
  "$PURE_RUN/samples_ddim_150steps" \
  "checkpoint25000_" \
  250150

generate_block "PureMSE checkpoint-50000 DDIM 150" \
  "$PURE_RUN/checkpoint-50000" \
  "$PURE_RUN/samples_ddim_150steps" \
  "checkpoint50000_" \
  500150

generate_block "DataDrivenDiag checkpoint-25000 DDIM 150" \
  "$DIAG_RUN/checkpoint-25000" \
  "$DIAG_RUN/samples_ddim_150steps" \
  "checkpoint25000_" \
  1250150

generate_block "DataDrivenDiag checkpoint-50000 DDIM 150" \
  "$DIAG_RUN/checkpoint-50000" \
  "$DIAG_RUN/samples_ddim_150steps" \
  "checkpoint50000_" \
  1500150

echo "===== Counts ====="
find "$PURE_RUN/samples_ddim_150steps" "$DIAG_RUN/samples_ddim_150steps" -maxdepth 1 -type f -name '*.png' | sed 's#/[^/]*$##' | sort | uniq -c
echo "===== Finished ====="
date

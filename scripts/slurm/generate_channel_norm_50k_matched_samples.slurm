#!/bin/bash
#SBATCH --job-name=gen_channel_norm_50k_matched
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
PURE_CKPT_RUN="/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_50000step_bs8_channelNorm_pureMSE"
DIAG_CKPT_RUN="/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed"
PURE_OUT_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_channelNorm_pureMSE"
DIAG_OUT_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed"
CHANNEL_STATS="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/diagnostics/channel_stats_train.json"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

generate_block() {
  local label="$1"
  local checkpoint_dir="$2"
  local output_dir="$3"
  local inference_steps="$4"
  local seed="$5"

  echo "===== ${label} ====="
  echo "checkpoint_dir=$checkpoint_dir"
  echo "output_dir=$output_dir"
  echo "inference_steps=$inference_steps"
  echo "seed=$seed"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "ERROR: missing checkpoint directory: $checkpoint_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"

  "$PYTHON" scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --num-samples 40 \
    --batch-size 4 \
    --image-size 224 \
    --num-inference-steps "$inference_steps" \
    --channel-normalization-mode standardize \
    --channel-norm-stats "$CHANNEL_STATS" \
    --seed "$seed"
}

for steps in 50 150; do
  if [[ "$steps" == "50" ]]; then
    suffix="samples_50steps_matched"
    seed_base=510000
  else
    suffix="samples_150steps_matched"
    seed_base=1510000
  fi

  generate_block "channelNorm pureMSE checkpoint-25000, ${steps} steps" \
    "$PURE_CKPT_RUN/checkpoint-25000" \
    "$PURE_OUT_RUN/$suffix/checkpoint-25000" \
    "$steps" \
    "$((seed_base + 25000))"

  generate_block "channelNorm pureMSE checkpoint-50000, ${steps} steps" \
    "$PURE_CKPT_RUN/checkpoint-50000" \
    "$PURE_OUT_RUN/$suffix/checkpoint-50000" \
    "$steps" \
    "$((seed_base + 50000))"

  generate_block "channelNorm dataDrivenDiag checkpoint-25000, ${steps} steps" \
    "$DIAG_CKPT_RUN/checkpoint-25000" \
    "$DIAG_OUT_RUN/$suffix/checkpoint-25000" \
    "$steps" \
    "$((seed_base + 125000))"

  generate_block "channelNorm dataDrivenDiag checkpoint-50000, ${steps} steps" \
    "$DIAG_CKPT_RUN/checkpoint-50000" \
    "$DIAG_OUT_RUN/$suffix/checkpoint-50000" \
    "$steps" \
    "$((seed_base + 150000))"
done

echo "===== Counts ====="
find "$PURE_OUT_RUN"/samples_*steps_matched "$DIAG_OUT_RUN"/samples_*steps_matched -type f -name '*.png' | sed 's#/sample_[0-9]*.png##' | sort | uniq -c

echo "===== Finished ====="
date

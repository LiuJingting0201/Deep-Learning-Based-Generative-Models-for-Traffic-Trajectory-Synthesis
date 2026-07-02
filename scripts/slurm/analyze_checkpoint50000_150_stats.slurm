#!/bin/bash
#SBATCH --job-name=ckpt50000_150_stats
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
REAL_DIR="/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images"
OUT_DIR="/home/jliu/Thesis/results_hpc/checkpoint50000_sampling_steps_150_analysis"
CHANNEL_OUT_DIR="/home/jliu/Thesis/results_hpc/checkpoint50000_channel_diagnostics_150"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import torch

print(torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

echo "===== Check checkpoint-50000 sample counts ====="
for dir in \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_150steps/checkpoint-50000 \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_150steps/checkpoint-50000 \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_channelNorm_pureMSE/samples_150steps_matched/checkpoint-50000 \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed/samples_150steps_matched/checkpoint-50000 \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_mtfSym_pureMSE/samples_150steps/checkpoint-50000 \
  /home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed/samples_150steps/checkpoint-50000
do
  count="$(find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l)"
  echo "$count $dir"
  if [[ "$count" -ne 40 ]]; then
    echo "ERROR: expected 40 PNGs in $dir, found $count" >&2
    exit 1
  fi
done

echo "===== Run checkpoint-50000-only analysis ====="
"$PYTHON" scripts/analyze_checkpoint50000_150_evaluation.py \
  --real-dir "$REAL_DIR" \
  --output-dir "$OUT_DIR" \
  --channel-output-dir "$CHANNEL_OUT_DIR" \
  --fid-batch-size 32 \
  --num-workers 0

echo "===== Outputs ====="
find "$OUT_DIR" "$CHANNEL_OUT_DIR" -maxdepth 2 -type f | sort

echo "===== Finished ====="
date

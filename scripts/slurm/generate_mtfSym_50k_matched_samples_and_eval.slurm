#!/bin/bash
#SBATCH --job-name=mtfSym_50k_150_eval
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
REAL_DIR="/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images"

MTF_PURE_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_mtfSym_pureMSE"
MTF_DIAG_RUN="/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed"
MTF_PURE_OUT="$MTF_PURE_RUN/samples_150steps"
MTF_DIAG_OUT="$MTF_DIAG_RUN/samples_150steps"
ANALYSIS_OUT="/home/jliu/Thesis/results_hpc/mtfSym_50k_sampling_steps_150_analysis"
CHANNEL_OUT="/home/jliu/Thesis/results_hpc/mtfSym_channel_diagnostics_150"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import sys
import torch

print(torch.__version__)
available = torch.cuda.is_available()
print("cuda_available", available)
if not available:
    sys.exit("ERROR: CUDA is not available in this allocated GPU job.")
print(torch.cuda.get_device_name(0))
PY

generate_block() {
  local label="$1"
  local checkpoint_dir="$2"
  local output_dir="$3"
  local seed="$4"
  local prefix="$5"

  echo "===== ${label} ====="
  echo "checkpoint_dir=$checkpoint_dir"
  echo "output_dir=$output_dir"
  echo "seed=$seed"
  echo "prefix=$prefix"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "ERROR: missing checkpoint directory: $checkpoint_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  local existing
  existing="$(find "$output_dir" -maxdepth 1 -type f -name "${prefix}sample*.png" | wc -l)"
  if [[ "$existing" -eq 40 ]]; then
    echo "Found 40 existing samples for $label; skipping generation."
    return
  fi
  if [[ "$existing" -ne 0 ]]; then
    echo "ERROR: found $existing existing samples in $output_dir for prefix $prefix; refusing to overwrite partial output." >&2
    exit 1
  fi

  "$PYTHON" scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --num-samples 40 \
    --batch-size 8 \
    --image-size 224 \
    --num-inference-steps 150 \
    --scheduler ddpm \
    --seed "$seed" \
    --filename-prefix "$prefix"
}

generate_block \
  "MTFSym PureMSE checkpoint-25000, 150 DDPM steps" \
  "$MTF_PURE_RUN/checkpoint-25000" \
  "$MTF_PURE_OUT/checkpoint-25000" \
  35025000 \
  "checkpoint25000_"

generate_block \
  "MTFSym PureMSE checkpoint-50000, 150 DDPM steps" \
  "$MTF_PURE_RUN/checkpoint-50000" \
  "$MTF_PURE_OUT/checkpoint-50000" \
  35050000 \
  "checkpoint50000_"

generate_block \
  "MTFSym DataDrivenDiag checkpoint-25000, 150 DDPM steps" \
  "$MTF_DIAG_RUN/checkpoint-25000" \
  "$MTF_DIAG_OUT/checkpoint-25000" \
  45025000 \
  "checkpoint25000_"

generate_block \
  "MTFSym DataDrivenDiag checkpoint-50000, 150 DDPM steps" \
  "$MTF_DIAG_RUN/checkpoint-50000" \
  "$MTF_DIAG_OUT/checkpoint-50000" \
  45050000 \
  "checkpoint50000_"

echo "===== MTFSym sample counts ====="
find "$MTF_PURE_OUT" "$MTF_DIAG_OUT" -type f -name '*.png' | sed 's#/checkpoint[0-9]*_sample[0-9]*.png##' | sort | uniq -c

mkdir -p "$ANALYSIS_OUT"

if [[ ! -f "$ANALYSIS_OUT/fid_mtfSym_pureMSE.json" ]]; then
  "$PYTHON" scripts/evaluate_fid.py \
    --real-dir "$REAL_DIR" \
    --generated-dir "$MTF_PURE_OUT" \
    --output-json "$ANALYSIS_OUT/fid_mtfSym_pureMSE.json" \
    --batch-size 32 \
    --num-workers 4
else
  echo "Reusing existing $ANALYSIS_OUT/fid_mtfSym_pureMSE.json"
fi

if [[ ! -f "$ANALYSIS_OUT/fid_mtfSym_dataDrivenDiag.json" ]]; then
  "$PYTHON" scripts/evaluate_fid.py \
    --real-dir "$REAL_DIR" \
    --generated-dir "$MTF_DIAG_OUT" \
    --output-json "$ANALYSIS_OUT/fid_mtfSym_dataDrivenDiag.json" \
    --batch-size 32 \
    --num-workers 4
else
  echo "Reusing existing $ANALYSIS_OUT/fid_mtfSym_dataDrivenDiag.json"
fi

echo "===== Six-way analysis ====="
"$PYTHON" scripts/analyze_mtf_sym_150_evaluation.py \
  --real-dir "$REAL_DIR" \
  --output-dir "$ANALYSIS_OUT" \
  --channel-output-dir "$CHANNEL_OUT"

echo "===== Finished ====="
date

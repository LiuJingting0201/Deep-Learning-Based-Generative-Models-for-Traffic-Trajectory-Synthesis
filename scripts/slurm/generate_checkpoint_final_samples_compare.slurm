#!/bin/bash
#SBATCH --job-name=generate_ckpt_final_samples
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

echo "===== Job info ====="
hostname
date
echo "HOME=$HOME"
echo "SCRATCH=$SCRATCH"
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"

"$PYTHON" -c "import sys; print(sys.executable)"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

run_generation() {
  local label="$1"
  local checkpoint_dir="$2"
  local output_dir="$3"
  local num_inference_steps="$4"

  echo "===== ${label} ====="
  echo "checkpoint_dir=$checkpoint_dir"
  echo "output_dir=$output_dir"
  echo "num_inference_steps=$num_inference_steps"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Missing checkpoint directory: $checkpoint_dir" >&2
    exit 1
  fi

  "$PYTHON" scripts/generate_ddpm_samples.py \
    --checkpoint-dir "$checkpoint_dir" \
    --output-dir "$output_dir" \
    --num-samples 16 \
    --batch-size 4 \
    --image-size 224 \
    --num-inference-steps "$num_inference_steps" \
    --seed 42
}

run_generation \
  "A) pureMSE, 50 steps" \
  "/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_10000step_bs8_pureMSE/checkpoint-final" \
  "/mnt/beegfs-compat/jliu/Thesis_eval/pureMSE_checkpoint_final_samples_50" \
  50

run_generation \
  "B) pureMSE, 100 steps" \
  "/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_10000step_bs8_pureMSE/checkpoint-final" \
  "/mnt/beegfs-compat/jliu/Thesis_eval/pureMSE_checkpoint_final_samples_100" \
  100

run_generation \
  "C) dataDrivenDiag_precomputed, 50 steps" \
  "/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_10000step_bs8_dataDrivenDiag_precomputed/checkpoint-final" \
  "/mnt/beegfs-compat/jliu/Thesis_eval/dataDrivenDiag_checkpoint_final_samples_50" \
  50

run_generation \
  "D) dataDrivenDiag_precomputed, 100 steps" \
  "/mnt/beegfs-compat/jliu/Thesis_runs/ddpm_scratch_10000step_bs8_dataDrivenDiag_precomputed/checkpoint-final" \
  "/mnt/beegfs-compat/jliu/Thesis_eval/dataDrivenDiag_checkpoint_final_samples_100" \
  100

echo "===== Finished ====="
date
#!/bin/bash
#SBATCH --job-name=ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
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
RUN_NAME="ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed"
DATA_ROOT="$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired"
CHANNEL_STATS="$DATA_ROOT/diagnostics/channel_stats_train.json"
DIAG_TARGETS="$DATA_ROOT/diagnostics/diag_targets_train_split.json"

echo "===== Job info ====="
hostname
date
echo "PWD=$(pwd)"
echo "SCRATCH=$SCRATCH"
echo "PYTHON=$PYTHON"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

if [[ ! -f "$CHANNEL_STATS" ]]; then
  echo "ERROR: missing channel stats file: $CHANNEL_STATS" >&2
  exit 1
fi

if [[ ! -f "$DIAG_TARGETS" ]]; then
  echo "ERROR: missing diagonal target file: $DIAG_TARGETS" >&2
  exit 1
fi

"$PYTHON" scripts/train_ddpm_scratch.py \
  --data-dir "$DATA_ROOT/images" \
  --run-name "$RUN_NAME" \
  --aux-loss-mode data_driven_diag \
  --diag-target-file "$DIAG_TARGETS" \
  --image-size 224 \
  --batch-size 8 \
  --lr 1e-4 \
  --max-train-steps 50000 \
  --gradient-accumulation-steps 2 \
  --save-every 2500 \
  --sample-every 2500 \
  --sample-inference-steps 150 \
  --channel-normalization-mode standardize \
  --channel-norm-stats "$CHANNEL_STATS" \
  --copy-results-to-home

echo "===== Finished ====="
date

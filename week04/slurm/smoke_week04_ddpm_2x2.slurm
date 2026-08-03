#!/bin/bash
#SBATCH --job-name=week04_ddpm_smoke
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jliu/Thesis/week04/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week04/logs/%x_%j.err

set -euo pipefail

cd /home/jliu/Thesis

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"
CONSTRAINED_IMAGES="/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start/images"
NEG11_IMAGES="/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start_neg11/images"
CONFIG_DIR="/home/jliu/Thesis/week04/configs"
SMOKE_ROOT="/home/jliu/Thesis/week04/results_smoke/ddpm_2x2"

mkdir -p /home/jliu/Thesis/week04/logs "$CONFIG_DIR"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

echo "===== Job info ====="
hostname
date
echo "CONSTRAINED_IMAGES=${CONSTRAINED_IMAGES}"
echo "NEG11_IMAGES=${NEG11_IMAGES}"
"$PYTHON" -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "===== Compute diag targets ====="
"$PYTHON" /home/jliu/Thesis/week04/compute_gasf_diag_targets.py \
  --data-dir "$CONSTRAINED_IMAGES" \
  --output-file "$CONFIG_DIR/diag_targets_constrained.json"
"$PYTHON" /home/jliu/Thesis/week04/compute_gasf_diag_targets.py \
  --data-dir "$NEG11_IMAGES" \
  --output-file "$CONFIG_DIR/diag_targets_neg11.json"

run_smoke() {
  local label="$1"
  local data_dir="$2"
  local aux_mode="$3"
  local diag_file="$4"
  local out_dir="$SMOKE_ROOT/$label"
  local args=(
    --data-dir "$data_dir"
    --output-dir "$out_dir"
    --batch-size 2
    --gradient-accumulation-steps 1
    --max-train-steps 2
    --aux-loss-mode "$aux_mode"
    --sample-every 0
    --save-every 0
    --num-sample-images 2
    --sample-inference-steps 2
    --log-every 1
    --seed 42
  )
  if [[ "$aux_mode" == "gasf_structure" ]]; then
    args+=(--diag-target-file "$diag_file" --symmetry-weight 0.01 --diag-weight 0.01)
  fi
  echo "===== Smoke ${label} ====="
  "$PYTHON" /home/jliu/Thesis/week04/scripts/train_week04_ddpm.py "${args[@]}"
  echo "===== Log ${label} ====="
  cat "$out_dir/train_log.jsonl"
}

run_smoke "A_constrained_mse" "$CONSTRAINED_IMAGES" "none" ""
run_smoke "B_constrained_gasf_structure" "$CONSTRAINED_IMAGES" "gasf_structure" "$CONFIG_DIR/diag_targets_constrained.json"
run_smoke "C_neg11_mse" "$NEG11_IMAGES" "none" ""
run_smoke "D_neg11_gasf_structure" "$NEG11_IMAGES" "gasf_structure" "$CONFIG_DIR/diag_targets_neg11.json"

echo "===== Smoke outputs ====="
find "$SMOKE_ROOT" -maxdepth 3 -type f | sort

echo "===== Finished ====="
date

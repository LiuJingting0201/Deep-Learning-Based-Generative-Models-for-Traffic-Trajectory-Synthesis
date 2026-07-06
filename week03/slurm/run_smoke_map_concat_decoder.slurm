#!/bin/bash
#SBATCH --job-name=week03_map_smoke
#SBATCH --partition=cpu_sapphire
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/jliu/Thesis

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/jliu/.conda/envs/thesis-diffusion/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/jliu/data_no_speed_delta_displacement_paired}"
MAP_DIR="${MAP_DIR:-/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox}"

mkdir -p /home/jliu/Thesis/week03/logs "${MPLCONFIGDIR}"

echo "===== Job info ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "MAP_DIR=${MAP_DIR}"

echo "===== Compile check ====="
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/structure_aware_resnet18_models.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/train_structure_resnet18_decoder.py
"${PYTHON}" -m py_compile /home/jliu/Thesis/week03/scripts/smoke_test_structure_resnet18.py

echo "===== Original structure smoke, including map concat batch check ====="
"${PYTHON}" -u /home/jliu/Thesis/week03/scripts/smoke_test_structure_resnet18.py

echo "===== Map preflight and forward smoke for selected MAP_DIR ====="
"${PYTHON}" - <<'PY'
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project = Path("/home/jliu/Thesis")
for path in [project / "scripts", project / "week03" / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_delta_displacement_resnet18_decoder_ablation import fit_delta_label_scaler, verify_delta_data_and_split
from train_structure_resnet18_decoder import build_model, input_channels, make_dataset, preflight_map_inputs

data_root = Path(os.environ["DATA_ROOT"])
map_dir = Path(os.environ["MAP_DIR"])
args = argparse.Namespace(
    use_map=True,
    map_fusion="concat",
    map_dir=map_dir,
    model_type="raw_resnet18",
    dropout=0.3,
)
split = verify_delta_data_and_split(data_root)
preflight_map_inputs(args, split)
train = split[split["split"] == "train"].copy()
scaler = fit_delta_label_scaler(data_root, train)
batch = next(iter(DataLoader(make_dataset(args, data_root, train.head(2), scaler), batch_size=2, num_workers=0)))
model = build_model(args.model_type, args.dropout, input_channels(args))
with torch.no_grad():
    pred = model(batch["image"])
print(f"image_shape={tuple(batch['image'].shape)}")
print(f"map_shape={tuple(batch['map'].shape)}")
print(f"pred_shape={tuple(pred.shape)}")
print(f"conv1_input_channels={model.conv1_in_channels}")
PY

echo "===== Finished ====="
date

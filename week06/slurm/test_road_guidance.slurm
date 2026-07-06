#!/bin/bash
#SBATCH --job-name=w06_road_guidance
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/jliu/Thesis/week06/logs/w06_road_guidance_%j.out
#SBATCH --error=/home/jliu/Thesis/week06/logs/w06_road_guidance_%j.err

set -euo pipefail
cd /home/jliu/Thesis
module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jliu/Thesis/src
export MPLCONFIGDIR=/home/jliu/Thesis/week06/logs/matplotlib

mkdir -p /home/jliu/Thesis/week06/logs "${MPLCONFIGDIR}"

echo "===== Week06 road-guidance smoke ====="
hostname
date
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
nvidia-smi -L || true

PYTHON="/home/jliu/.conda/envs/thesis-diffusion/bin/python"

"${PYTHON}" -m pytest tests/test_road_distance_guidance.py

"${PYTHON}" - <<'PY'
import torch

from cnr_trajectory.guidance import DistanceFieldRoadGuidance, RoadRasterMetadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device={device}")

height = width = 21
_, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
distance_field = torch.abs(xx.float() - 10.0).to(device)
metadata = RoadRasterMetadata(x_min=0.0, x_max=20.0, y_min=0.0, y_max=20.0, y_axis_down=True)
guidance = DistanceFieldRoadGuidance(
    metadata=metadata,
    distance_field=distance_field,
    guidance_strength=1.0,
    num_guidance_steps=5,
    device=device,
)

traj = torch.zeros(4, 16, 2, device=device)
traj[..., 0] = 14.0
traj[..., 1] = torch.linspace(2.0, 18.0, 16, device=device)
before, before_mean = guidance.road_energy(traj)
guided, after, mean_dist = guidance(traj)

print(f"before_energy={before.item():.6f}")
print(f"after_energy={after.item():.6f}")
print(f"before_mean_dist={before_mean.item():.6f}")
print(f"after_mean_dist={mean_dist.item():.6f}")
assert guided.shape == traj.shape
assert after < before
assert mean_dist < before_mean
PY

#!/bin/bash
#SBATCH --job-name=week03_smoke_struct_resnet18
#SBATCH --output=/home/jliu/Thesis/week03/logs/%x_%j.out
#SBATCH --error=/home/jliu/Thesis/week03/logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

set -euo pipefail

module purge
module load miniconda3/3.13.25
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate thesis-diffusion

export MPLCONFIGDIR=/home/jliu/Thesis/week03/logs/matplotlib
mkdir -p "${MPLCONFIGDIR}"

python -u /home/jliu/Thesis/week03/scripts/smoke_test_structure_resnet18.py

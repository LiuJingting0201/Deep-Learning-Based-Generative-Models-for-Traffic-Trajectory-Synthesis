#!/bin/bash
set -euo pipefail

cd /home/jliu/Thesis

for CONFIG in \
  week05_sig10_mse \
  week05_sig10_mse_diag \
  week05_sig10_mse_sym \
  week05_sig15_mse \
  week05_sig15_mse_diag \
  week05_sig15_mse_sym
do
  sbatch --job-name="sample_${CONFIG}" \
    --export=ALL,CONFIG="${CONFIG}" \
    /home/jliu/Thesis/week05/slurm/sample_week05_checkpoints_to_npy.slurm
done

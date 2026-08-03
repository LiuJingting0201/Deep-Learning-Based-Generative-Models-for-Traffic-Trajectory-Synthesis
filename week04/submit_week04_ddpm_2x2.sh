#!/bin/bash
set -euo pipefail

cd /home/jliu/Thesis
sbatch week04/slurm/run_week04_A_constrained_mse_50k.slurm
sbatch week04/slurm/run_week04_B_constrained_gasf_structure_50k.slurm
sbatch week04/slurm/run_week04_C_neg11_mse_50k.slurm
sbatch week04/slurm/run_week04_D_neg11_gasf_structure_50k.slurm

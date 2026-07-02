#!/bin/bash
set -euo pipefail

cd /home/jliu/Thesis

sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse_diag.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig10_mse_sym.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse_diag.slurm
sbatch /home/jliu/Thesis/week05/slurm/week05_sig15_mse_sym.slurm

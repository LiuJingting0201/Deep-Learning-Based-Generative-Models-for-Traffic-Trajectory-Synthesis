#!/bin/bash
# Helper commands for delta-displacement decoder Slurm jobs.
# Run the commands manually; this script does not auto-submit unless executed.

set -euo pipefail

mkdir -p logs/slurm results_hpc

# 1. Smoke test
sbatch slurm/smoke_delta_multibranch_decoder.slurm

# 2. Main beta ablation
# for task_id in 1 2 3 4 5 6; do
#   sbatch --export=ALL,DELTA_TASK_ID="$task_id" slurm/run_delta_multibranch_ablation.slurm
# done

# 3. Channel/baseline ablation
# Run this after beta=0.05 split exists:
# results_hpc/decoder_multibranch_delta_beta005/split_metadata.csv
# for task_id in 1 2 3 4 5; do
#   sbatch --export=ALL,DELTA_TASK_ID="$task_id" slurm/run_delta_channel_ablation.slurm
# done

# Queue helper:
# squeue -u "$USER"

# Log helper examples:
# tail -f logs/slurm/smoke_delta_multibranch_<jobid>.out
# tail -f logs/slurm/delta_multibranch_ablation_<jobid>.out
# tail -f logs/slurm/delta_channel_ablation_<jobid>.out

#!/bin/bash
set -euo pipefail
cd /home/jliu/Thesis

mkdir -p /home/jliu/Thesis/week06/logs

sbatch week06/slurm/train_absolute_temporal_resnet.slurm
sbatch week06/slurm/train_absolute_temporal_resnet_dilated.slurm
sbatch week06/slurm/train_absolute_unet1d.slurm
sbatch week06/slurm/train_delta_temporal_resnet.slurm
sbatch week06/slurm/train_delta_temporal_resnet_dilated.slurm
sbatch week06/slurm/train_delta_unet1d.slurm

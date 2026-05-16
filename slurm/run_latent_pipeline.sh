#!/bin/bash
set -euo pipefail

cd ~/Thesis
mkdir -p logs/slurm

autoencoder_job_id=$(sbatch --parsable slurm/train_autoencoder.sbatch)
echo "Submitted autoencoder job: ${autoencoder_job_id}"

encode_job_id=$(sbatch --parsable --dependency=afterok:${autoencoder_job_id} slurm/encode_latents.sbatch)
echo "Submitted latent encoding job: ${encode_job_id}"

latent_ddpm_job_id=$(sbatch --parsable --dependency=afterok:${encode_job_id} slurm/train_latent_ddpm.sbatch)
echo "Submitted latent DDPM job: ${latent_ddpm_job_id}"

echo
echo "Latent pipeline submitted:"
echo "  autoencoder: ${autoencoder_job_id}"
echo "  encode:      ${encode_job_id}"
echo "  latent DDPM: ${latent_ddpm_job_id}"

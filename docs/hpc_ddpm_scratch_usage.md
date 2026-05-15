# HPC DDPM Scratch Usage

Use BeeGFS scratch for training data and training outputs. Do not use `$SCRATCH_FLASH` while it is unavailable.

## Paths

- Put DDPM image data under `$SCRATCH/Thesis_data`.
- The current expected training images path is:
  `$SCRATCH/Thesis_data/data_no_speed_delta_displacement_paired/images`
- Training outputs go under:
  `$SCRATCH/Thesis_runs/<run_name>`
- Important outputs can be copied back to:
  `$HOME/Thesis/results_hpc/<run_name>`

## Submit the 10-step test

From the HPC repo directory:

```bash
cd $HOME/Thesis
sbatch scripts/slurm/train_ddpm_scratch_10step.slurm
```

## Check the queue

```bash
squeue -u jliu
```

## Check logs

```bash
tail -f logs/ddpm_scratch_10step_*.out
```

The matching error logs are in:

```bash
logs/ddpm_scratch_10step_*.err
```

# Trajectory-Defined Sequence Latent Alignment

## Motivation

GAF/MTF images are not ordinary RGB photographs. The R, G, and B channels are pseudo-modal pairwise temporal relation matrices generated from the same trajectory: R is GASF, G is GADF, and B is MTF. Direct RGB fusion with a generic image backbone can learn useful correlations, but it does not explicitly respect the row-wise temporal meaning of these matrices.

This experiment constructs a trajectory-defined sequence latent space from real delta-displacement trajectories, then asks whether each row of the GAF/MTF image can be aligned to the corresponding timestep-level latent token.

## Why Sequence Latent Instead Of Global Latent

A global latent vector compresses the whole trajectory into one representation. That makes image-to-trajectory decoding harder to interpret because there is no direct token-to-timestep correspondence. The sequence latent keeps shape `[B, 223, latent_dim]`, so every latent token remains tied to one delta-displacement timestep.

This is useful for GAF/MTF because row `t` contains relations between timestep `t` and all other timesteps. A row-wise encoder can therefore map image row `t` into latent token `z[t]`.

## Stage 1: Trajectory Autoencoder

Input delta displacement:

```text
delta_gt: [B, 223, 2]
```

If the existing paired dataset stores labels as `[224, 2]`, the scripts drop `delta[0]`, which is the existing placeholder convention in the Week 3 delta-displacement experiments.

Architecture:

```text
TrajectorySequenceEncoder:
  [B, 223, 2] -> transpose -> [B, 2, 223]
  Conv1d(2, 64, kernel=5, padding=2)
  GELU
  Conv1d(64, 128, kernel=5, padding=2)
  GELU
  Conv1d(128, latent_dim, kernel=3, padding=1)
  transpose -> z_traj [B, 223, latent_dim]

TrajectorySequenceDecoder:
  [B, 223, latent_dim] -> transpose -> [B, latent_dim, 223]
  Conv1d(latent_dim, 128, kernel=3, padding=1)
  GELU
  Conv1d(128, 64, kernel=5, padding=2)
  GELU
  Conv1d(64, 2, kernel=5, padding=2)
  transpose -> delta_hat [B, 223, 2]
```

Loss:

```text
delta_loss = MSE(delta_hat, delta_gt)
integrated_loss = MSE(cumsum(delta_hat), cumsum(delta_gt))
loss = delta_loss + beta_integrated * integrated_loss
```

Default `latent_dim=64` and `beta_integrated=0.1`.

## Stage 2: Image-To-Sequence-Latent Alignment

The Stage 1 trajectory encoder and decoder are frozen. For each paired sample:

```text
z_traj = frozen TrajectorySequenceEncoder(delta_gt)
z_img = RowWiseGAFImageEncoder(image)
delta_hat_img = frozen TrajectorySequenceDecoder(z_img)
```

Loss:

```text
latent_loss = MSE(z_img, z_traj)
delta_loss = MSE(delta_hat_img, delta_gt)
integrated_loss = MSE(cumsum(delta_hat_img), cumsum(delta_gt))
loss = delta_loss + beta_integrated * integrated_loss + lambda_latent * latent_loss
```

Defaults: `beta_integrated=0.1`, `lambda_latent=1.0`.

## Row-Wise GAF/MTF Image Encoder

Input:

```text
image: [B, 3, 224, 224]
```

Process:

```text
image.permute(0, 2, 1, 3) -> [B, 224, 3, 224]
flatten rows -> [B, 224, 672]
shared MLP:
  Linear(672, 256)
  GELU
  Linear(256, latent_dim)
take first 223 rows -> z_img [B, 223, latent_dim]
```

This is intentionally simpler than ResNet18. It tests the mathematical hypothesis directly: the row of a pairwise temporal image should align with a timestep-level trajectory latent token.

Optional temporal refinement can be enabled after the row MLP:

```text
z [B, 223, latent_dim]
transpose -> [B, latent_dim, 223]
Conv1d(latent_dim, latent_dim, kernel=temporal_kernel_size, padding=kernel//2)
GELU
repeat temporal_layers times
transpose -> [B, 223, latent_dim]
z = residual + z_refined
```

This is disabled by default with `--use-temporal-refine` absent, so the original row-wise MLP baseline remains unchanged.

## Outputs

Stage 1 writes:

```text
config.json
split_copy.csv
train_log.csv
best_metrics.json
final_validation_metrics.json
validation_metrics.json  # final-epoch alias for backward compatibility
best_summary.json
best.pt
final.pt
checkpoints/last.pt
checkpoints/best.pt
```

Stage 2 writes the same structure for the row-wise image encoder.

Evaluation writes:

```text
metrics_summary.json
metrics_test.json
per_sample_metrics.csv
predictions_delta/*.npy
predictions_integrated/*.npy
teacher_predictions_delta/*.npy
plots/*.png
```

## Metrics To Report

Report at minimum:

- delta MSE
- delta ADE
- delta FDE
- integrated ADE
- integrated FDE
- latent MSE

The evaluator also reports teacher integrated metrics, which measure the reconstruction quality of the frozen trajectory autoencoder itself.

## Implementation Robustness Notes

- `best.pt` is selected by `--best-metric`; the default is `integrated_ADE`.
- `best_metrics.json` contains the complete validation metrics from the best epoch.
- `final_validation_metrics.json` contains the final epoch validation metrics. `validation_metrics.json` is kept only as a backward-compatible final-epoch alias.
- Stage 2 validates that `--latent-dim` matches the latent dimension recorded in the Stage 1 checkpoint config.
- The Stage 2 trajectory decoder is frozen, but it is not run under `torch.no_grad()` for `teacher_decoder(z_img)`, so delta reconstruction gradients still flow back to the image encoder.
- `RowWiseGAFImageEncoder` defaults to the minimal interpretable row-wise baseline. The optional residual Conv1D temporal refinement is enabled only with `--use-temporal-refine`; there is still no Transformer, row+column encoder, or ResNet.

## Commands

On the cluster, prefer submitting the Week 3 SLURM scripts:

```bash
sbatch week03/slurm/run_sequence_latent_stage1_traj_ae.slurm
sbatch --dependency=afterok:<STAGE1_JOB_ID> week03/slurm/run_sequence_latent_stage2_rowwise_alignment.slurm
sbatch --dependency=afterok:<STAGE2_JOB_ID> week03/slurm/run_sequence_latent_eval.slurm
```

Smoke run:

```bash
sbatch week03/slurm/run_smoke_sequence_latent_alignment.slurm
```

The SLURM scripts accept environment overrides, for example:

```bash
EPOCHS=200 LATENT_DIM=64 sbatch week03/slurm/run_sequence_latent_stage1_traj_ae.slurm
TRAJ_AE_CHECKPOINT=/home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt \
  sbatch week03/slurm/run_sequence_latent_stage2_rowwise_alignment.slurm
```

Direct Python commands are useful for debugging inside the same conda environment.

Stage 1:

```bash
python week03/scripts/train_trajectory_sequence_autoencoder.py \
  --data-root /home/jliu/Thesis/data_no_speed_delta_displacement_paired \
  --output-dir /home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64 \
  --latent-dim 64 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --beta-integrated 0.1 \
  --best-metric integrated_ADE
```

Stage 2:

```bash
python week03/scripts/train_image_sequence_latent_alignment.py \
  --data-root /home/jliu/Thesis/data_no_speed_delta_displacement_paired \
  --trajectory-ae-checkpoint /home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt \
  --output-dir /home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64 \
  --latent-dim 64 \
  --batch-size 32 \
  --epochs 100 \
  --lr 5e-4 \
  --beta-integrated 0.1 \
  --lambda-latent 1.0 \
  --best-metric integrated_ADE
```

Optional Stage 2 temporal refinement:

```bash
python week03/scripts/train_image_sequence_latent_alignment.py \
  --data-root /home/jliu/Thesis/data_no_speed_delta_displacement_paired \
  --trajectory-ae-checkpoint /home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt \
  --output-dir /home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64_temporal \
  --latent-dim 64 \
  --batch-size 32 \
  --epochs 100 \
  --lr 5e-4 \
  --beta-integrated 0.1 \
  --lambda-latent 1.0 \
  --best-metric integrated_ADE \
  --use-temporal-refine \
  --temporal-kernel-size 5 \
  --temporal-layers 2
```

Evaluation:

```bash
python week03/scripts/evaluate_sequence_latent_alignment.py \
  --data-root /home/jliu/Thesis/data_no_speed_delta_displacement_paired \
  --trajectory-ae-checkpoint /home/jliu/Thesis/week03/results/sequence_latent/stage1_traj_ae_latent64/best.pt \
  --image-encoder-checkpoint /home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64/best.pt \
  --output-dir /home/jliu/Thesis/week03/results/sequence_latent/stage2_rowwise_img_latent64/eval
```

## Baseline Comparison

Compare against the existing Week 3 ResNet18 or mid-fusion image-to-delta decoders by using the same train/val/test split and reporting the same delta and integrated metrics. The key question is not only whether the row-wise sequence-latent model improves integrated ADE/FDE, but whether it reduces the gap between image-latent reconstruction and teacher trajectory-latent reconstruction.

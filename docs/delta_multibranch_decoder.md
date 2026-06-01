# Delta Multi-Branch Decoder

## Dataset Assumption

The decoder is for datasets produced by `scripts/build_no_speed_delta_displacement_paired_dataset.py`.

- Input image: pseudo-modal RGB tensor `[B, 3, 224, 224]`.
- R channel: GASF.
- G channel: GADF.
- B channel: MTF.
- Target: physical unnormalized delta displacement `[B, 224, 2]`.
- Delta convention: `delta[0] = [0, 0]`; for `t >= 1`, `delta[t] = absolute[t] - absolute[t-1]`.

Images are loaded as RGB PNG, converted to `float32`, and divided by `255.0`.

## Model Input/Output

`--model multi_branch_delta --channels rgb` uses three independent single-channel encoders, one per pseudo-modal channel. Their features are concatenated and decoded by an MLP head.

- Input: `[B, 3, 224, 224]`.
- Output: `[B, 224, 2]`.

Single-channel ablations use `--channels r`, `--channels g`, or `--channels b` and accept `[B, 1, 224, 224]` internally through the dataset loader.

## Encoder Types

- `small_gap`: old compact encoder, with `AdaptiveAvgPool2d((1, 1))`.
- `small_spatial`: stronger default encoder, with `AdaptiveAvgPool2d((4, 4))` before flattening and projection.

## Mid-fusion Decoder

The current ablation set compares three fusion strategies:

- `simple_cnn_delta`: early fusion. The RGB pseudo-modal channels are stacked at the input and mixed by the first convolution.
- `multi_branch_delta`: late fusion. GASF/GADF/MTF are encoded independently all the way to vectors, then concatenated before the MLP head.
- `mid_fusion_delta`: shallow channel-specific stems followed by spatial feature-map fusion.

`mid_fusion_delta` directly tests whether pseudo-modal GAF channels benefit from controlled intermediate fusion. It applies separate shallow stems to R/G/B (`R=GASF`, `G=GADF`, `B=MTF`), concatenates the resulting `[B, 64, 56, 56]` feature maps into `[B, 192, 56, 56]`, and then uses a shared fusion CNN and MLP regression head.

Submit mid-fusion ablations:

```bash
sbatch --export=ALL,DELTA_TASK_ID=1 slurm/run_delta_midfusion_ablation.slurm
sbatch --export=ALL,DELTA_TASK_ID=2 slurm/run_delta_midfusion_ablation.slurm
sbatch --export=ALL,DELTA_TASK_ID=3 slurm/run_delta_midfusion_ablation.slurm
```

Task mapping:

- `1`: `mid_fusion_delta`, RGB, beta `0.05`
- `2`: `mid_fusion_delta`, RGB, beta `0.2`
- `3`: `mid_fusion_delta`, RGB, beta `0.05`, normalized target

## Loss Formula

Without target normalization:

```text
delta_loss = MSE(delta_pred, delta_gt)
abs_pred = integrate_delta(start_xy, delta_pred)
integrated_loss = MSE(abs_pred, absolute_gt)
total_loss = delta_loss + beta_integrated_loss * integrated_loss
```

With `--normalize-target-delta`:

```text
delta_gt_norm = (delta_gt - mean) / std
delta_loss = MSE(delta_pred_norm, delta_gt_norm)
delta_pred = delta_pred_norm * std + mean
abs_pred = integrate_delta(start_xy, delta_pred)
integrated_loss = MSE(abs_pred, absolute_gt)
total_loss = delta_loss + beta_integrated_loss * integrated_loss
```

Normalization statistics are fit on the train split and saved to `target_delta_normalization.json`.

## Integration Formula

```text
xy[:, 0, :] = start_xy
xy[:, 1:, :] = start_xy[:, None, :] + cumsum(delta[:, 1:, :], dim=1)
```

`delta[:, 0, :]` is a placeholder and is not integrated.

## Smoke Test Command

```bash
python scripts/train_delta_displacement_decoder.py \
  --data-dir data_no_speed_delta_displacement_paired \
  --output-dir results/smoke_multibranch_delta \
  --model multi_branch_delta \
  --channels rgb \
  --encoder-type small_spatial \
  --epochs 1 \
  --batch-size 4 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --num-workers 0
```

## Recommended Formal Ablations

Multi-branch RGB, beta=0.0:

```bash
python scripts/train_delta_displacement_decoder.py \
  --data-dir data_no_speed_delta_displacement_paired \
  --output-dir results/decoder_multibranch_delta_beta0 \
  --model multi_branch_delta \
  --channels rgb \
  --encoder-type small_spatial \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --beta-integrated-loss 0.0
```

Multi-branch RGB, beta=0.05:

```bash
python scripts/train_delta_displacement_decoder.py \
  --data-dir data_no_speed_delta_displacement_paired \
  --output-dir results/decoder_multibranch_delta_beta005 \
  --model multi_branch_delta \
  --channels rgb \
  --encoder-type small_spatial \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --beta-integrated-loss 0.05
```

Multi-branch RGB, beta=0.2:

```bash
python scripts/train_delta_displacement_decoder.py \
  --data-dir data_no_speed_delta_displacement_paired \
  --output-dir results/decoder_multibranch_delta_beta02 \
  --model multi_branch_delta \
  --channels rgb \
  --encoder-type small_spatial \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-3 \
  --beta-integrated-loss 0.2
```

Single-channel ablations:

```bash
--channels r
--channels g
--channels b
```

Baseline:

```bash
--model simple_cnn_delta
```

## Running on Slurm

Run the smoke job first:

```bash
sbatch slurm/smoke_delta_multibranch_decoder.slurm
```

Submit the main beta ablation array after the smoke job succeeds:

```bash
sbatch slurm/run_delta_multibranch_ablation.slurm
```

Submit the channel/baseline ablation after the main beta=0.05 run has produced:

```text
results_hpc/decoder_multibranch_delta_beta005/split_metadata.csv
```

Then run:

```bash
sbatch slurm/run_delta_channel_ablation.slurm
```

The channel/baseline array reuses that split metadata when available for fair comparison. If it is missing, the script prints a warning and falls back to the train script's deterministic split.

Collect results with:

```bash
python scripts/collect_delta_decoder_results.py \
  --results-root results_hpc \
  --output-prefix results_hpc/delta_decoder_ablation_summary
```

Useful queue/log commands:

```bash
squeue -u "$USER"
tail -f logs/slurm/<job_log_name>.out
```

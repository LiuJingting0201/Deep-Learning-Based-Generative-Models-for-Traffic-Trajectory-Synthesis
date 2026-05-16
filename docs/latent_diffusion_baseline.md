# Latent Diffusion Baseline

This document describes the Version-1 latent diffusion baseline for 224x224 RGB GAF trajectory images.

## Motivation

GAF images are structured mathematical representations of trajectories rather than natural photographs. A pixel-space DDPM must learn two things at once:

- the valid GAF image structure
- the trajectory distribution represented by those images

Latent diffusion separates these responsibilities. A convolutional autoencoder first learns a compact representation of valid GAF images. The diffusion model then learns the distribution in that latent space instead of directly modeling 224x224 RGB pixels.

The initial pipeline is:

```text
GAF image -> autoencoder encoder -> latent z -> latent DDPM -> autoencoder decoder -> GAF image
```

There is no map condition in this V1 baseline.

## Architecture

The autoencoder is implemented in `src/cnr_trajectory/models/gaf_autoencoder.py`.

Encoder:

```text
224x224x3
-> Conv stride 2, 32 channels
-> Conv stride 2, 64 channels
-> Conv stride 2, 128 channels
-> Conv 128 to 32 channels
-> 32x28x28 latent tensor saved in PyTorch CHW layout
```

Decoder:

```text
32x28x28
-> ConvTranspose to 56x56x64
-> ConvTranspose to 112x112x32
-> ConvTranspose to 224x224x3
```

The original `16x14x14` bottleneck is still supported for comparison by using `--latent-channels 16 --latent-spatial-size 14` when training the autoencoder, and `--latent-channels 16 --sample-size 14` when training or sampling latent DDPM.

The autoencoder trains with:

```text
L_total = L1(reconstruction, image) + MSE(reconstruction, image)
```

Images use the same normalized range as the existing DDPM dataset, `[-1, 1]`; the decoder ends with `tanh`.

The latent DDPM uses `UNet2DModel` with:

```text
sample_size=28
in_channels=32
out_channels=32
scheduler=DDPMScheduler
```

## Commands

Train the autoencoder:

```bash
python scripts/train_autoencoder.py \
  --data-dir data_no_speed_delta_displacement_paired/images \
  --output-dir results/autoencoder_gaf_v1 \
  --batch-size 8 \
  --lr 1e-4 \
  --epochs 50 \
  --save-every 5 \
  --latent-channels 32 \
  --latent-spatial-size 28 \
  --seed 42
```

Encode the dataset:

```bash
python scripts/encode_dataset_to_latent.py \
  --data-dir data_no_speed_delta_displacement_paired/images \
  --autoencoder-checkpoint results/autoencoder_gaf_v1/checkpoint-final \
  --output-dir data_latent \
  --batch-size 32
```

Train latent DDPM:

```bash
python scripts/train_latent_ddpm.py \
  --data-dir data_latent \
  --output-dir results/latent_ddpm \
  --batch-size 16 \
  --lr 2e-4 \
  --max-train-steps 10000 \
  --save-every 1000 \
  --sample-every 1000 \
  --sample-size 28 \
  --latent-channels 32
```

Generate decoded samples:

```bash
python scripts/generate_latent_samples.py \
  --latent-checkpoint results/latent_ddpm/checkpoint-final \
  --autoencoder-checkpoint results/autoencoder_gaf_v1/checkpoint-final \
  --output-dir results/latent_ddpm/generated \
  --num-samples 100 \
  --batch-size 16 \
  --num-inference-steps 1000
```

Evaluate reconstruction and generated samples:

```bash
python scripts/evaluate_latent_generation.py \
  --real-dir data_no_speed_delta_displacement_paired/images \
  --generated-dir results/latent_ddpm/generated \
  --autoencoder-checkpoint results/autoencoder_gaf_v1/checkpoint-final \
  --output-dir results/latent_ddpm/eval
```

## Expected Outputs

Autoencoder training writes:

- `checkpoint-final/autoencoder/model.pt`
- `checkpoint-final/autoencoder/config.json`
- `checkpoint-final/training_state.pt`
- `samples/recon_epoch_*.png`
- `loss_curve.csv`
- `reconstruction_metrics.json`
- `train_summary.json`

Latent encoding writes:

- `data_latent/latent/*.npy`
- `data_latent/metadata.csv`
- `data_latent/latent_stats.json`

Each latent tensor is saved as `32x28x28` in PyTorch-friendly CHW layout for the current latent28 experiment. `metadata.csv` includes the dataset index, source image path, latent path, and latent shape. `latent_stats.json` records global latent mean, standard deviation, minimum, and maximum across all encoded values.

Latent DDPM training writes:

- `results/latent_ddpm/checkpoint-final/unet/`
- `results/latent_ddpm/checkpoint-final/scheduler/`
- `results/latent_ddpm/checkpoint-final/training_state.pt`
- `results/latent_ddpm/samples/latent_step_*.npy`
- `results/latent_ddpm/samples/latent_step_*.png`
- `results/latent_ddpm/logs/train_log.jsonl`

Generation and evaluation write:

- generated GAF PNG samples
- `metrics.json`
- `side_by_side.png` containing real, reconstructed, and generated images

## Current Limitations

- The autoencoder is intentionally lightweight and may still blur fine GAF structure.
- Latent DDPM now normalizes latents from `latent_stats.json`, but the latent distribution is still represented with global scalar statistics rather than per-channel statistics.
- Pixel MSE and MAE are useful reconstruction checks but do not prove trajectory validity.
- Generated GAF images still need decoding back to trajectories and trajectory-space metrics.
- The V1 latent DDPM has no map, route, or endpoint conditioning.

## Future Extension

The natural next step is map-conditioned latent diffusion:

```text
map/context condition + noisy latent -> latent denoiser -> decoded GAF -> trajectory
```

This keeps the autoencoder as a reusable first stage while extending the denoiser with spatial or graph-based map context.

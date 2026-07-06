# Week06 Plan: Decode Generated GASF Samples and Compare Against the Map

## Goal

Week05 selected `week05_sig15_mse_diag/checkpoint-40000` as the current best DDPM baseline by GASF image-space FID. Week06 moves the evaluation into trajectory and map space:

1. Decode generated `224 x 224 x 3` GASF float arrays back to `224 x 2` raw XY trajectories.
2. Compare decoded trajectories with real Week05 trajectories using trajectory-space statistics.
3. Project raw XY trajectories into the OSM CRS and measure distance to nearby roads.

## Primary Inputs

- Generated samples:
  `week05/results_npy_samples/week05_sig15_mse_diag/week05_sig15_mse_diag_step_040000_samples_1000.npy`
- Week05 training config:
  `week05/results/week05_sig15_mse_diag/startup_config.json`
- Week05 real baseline data:
  `week05/data/delta_displacement_paired_regularized_bounce224/labels_absolute/*.npy`
- Raw-XY to OSM transform:
  `week03/data/maps/raw_xy_to_osm_affine.json`
- Week05 OSM drive edges:
  `week06/data/maps_week05/osm_drive_week05_edges.gpkg`

## Decoding Contract

Week05 saved generated samples in image space with shape:

```text
N x 224 x 224 x 3
```

and value range `[0, 1]`.

Channels:

```text
R: GASF(sigmoid-normalized dx)
G: GASF(sigmoid-normalized dy)
B: start-position heatmap
```

For generated `.npy` samples, the GASF diagonal is already in `[0,1]`, so the inverse is:

```text
u = sqrt(diag_channel_01)
delta = mean + std * logit(u) / sigmoid_k
```

This differs from the training-time `[-1,1]` tensor form, where the diagonal would first need `(diag + 1) / 2`.

The start point is decoded from the B-channel heatmap using soft-argmax by default, then de-normalized with `position_stats` from `startup_config.json`.

Finally:

```text
absolute[0] = start_xy
absolute[t] = start_xy + cumsum(delta_xy[1:t])
```

`delta[0]` is forced to zero because Week05 training data uses that convention.

## Scripts

### 1. Decode samples

```bash
python week06/scripts/decode_week05_samples_to_trajectories.py \
  --samples week05/results_npy_samples/week05_sig15_mse_diag/week05_sig15_mse_diag_step_040000_samples_1000.npy \
  --config week05/results/week05_sig15_mse_diag/startup_config.json \
  --output-dir week06/decoded/week05_sig15_mse_diag_step_040000
```

Outputs:

- `trajectories_raw_xy/*.npy`
- `starts_raw_xy.npy`
- `delta_raw_xy.npy`
- `summary.json`
- `decode_metrics.csv`

### 2. Evaluate map and trajectory metrics

```bash
python week06/scripts/evaluate_decoded_map_alignment.py \
  --decoded-dir week06/decoded/week05_sig15_mse_diag_step_040000 \
  --real-data-root week05/data/delta_displacement_paired_regularized_bounce224 \
  --osm-edges-path week06/data/maps_week05/osm_drive_week05_edges.gpkg \
  --output-dir week06/results/week05_sig15_mse_diag_step_040000_map_eval
```

Outputs:

- `generated_per_sample_metrics.csv`
- `real_per_sample_metrics.csv`
- `summary.json`
- `summary.md`

## Metrics

Trajectory-space metrics:

- step length mean / median / p95 / max
- total path length
- displacement from start to end
- bbox width and height
- large jump ratio
- out-of-position-range ratio

Map-space metrics:

- mean / median / p95 / max nearest-road distance
- off-road ratio at thresholds 5 m, 10 m, 20 m

The real baseline should be reported alongside generated trajectories because the Week05 training data used bounce regularization. Many real baseline samples are intentionally length-regularized, not purely natural continuous driving sequences.

## Immediate Week06 Checklist

- [x] Create Week06 plan.
- [x] Implement generated GASF-to-trajectory decoding.
- [x] Implement generated-vs-real trajectory/map metric evaluation.
- [x] Run full decode on the best Week05 checkpoint samples.
- [x] Run preliminary map evaluation in the geospatial Python environment.
- [x] Review generated vs real summary and identify the first blocker.
- [x] Download an OSM drive network covering the full Week05 real trajectory extent.
- [x] Re-run map evaluation with the Week05-specific OSM network.
- [ ] Optional: add simple visual overlays for representative generated trajectories.
- [x] Diagnose whether Week05 raw XY coordinates and Week03 OSM crop/affine are from the same spatial coverage.
- [x] Decide whether to use global OSM distance, local road-distance fields, or rebuild the map crop for Week05.

## Raw Trajectory Diffusion Baselines

The GASF intermediate representation is now treated as a likely failure point rather than a requirement. Week06 adds two direct raw-sequence DDPM baselines.

### Absolute trajectory diffusion

Scripts:

```text
week06/scripts/train_absolute_trajectory_diffusion.py
week06/slurm/train_absolute_trajectory_diffusion.slurm
```

Input:

```text
week05/data/delta_displacement_paired_regularized_bounce224/labels_absolute/*.npy
```

Training tensor:

```text
(channels=2, length=224), z-scored raw x/y
```

Output samples:

```text
samples/step_XXXXXX_absolute_raw.npy
```

### Delta-displacement diffusion

Scripts:

```text
week06/scripts/train_delta_displacement_diffusion.py
week06/slurm/train_delta_displacement_diffusion.slurm
```

Input:

```text
week05/data/delta_displacement_paired_regularized_bounce224/labels_delta_displacement/*.npy
```

Training tensor:

```text
(channels=2, length=224), z-scored raw dx/dy
```

Output samples:

```text
samples/step_XXXXXX_delta_raw.npy
samples/step_XXXXXX_delta_integrated_trajectories.npy
samples/step_XXXXXX_sampled_starts.npy
```

The delta model samples `dx/dy` only. For immediate trajectory evaluation, sampled deltas are integrated from randomly sampled real Week05 starts.

Both baselines use a compact 1D temporal denoising network with `DDPMScheduler`; no GASF, PNG, or image-space FID is involved.

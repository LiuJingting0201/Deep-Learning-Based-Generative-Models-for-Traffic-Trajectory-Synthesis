# Week06 Report: Trajectory-Space Evaluation, Filtering, Raw Diffusion, and Road Guidance

## Executive Summary

Week06 started from a promising Week05 image-space result: the best GASF DDPM checkpoint,
`week05_sig15_mse_diag/checkpoint-40000`, reached `FID = 44.5496`. After decoding generated
GASF images back to trajectories, the main conclusion changed:

- The GASF inverse path works mechanically.
- Per-step motion scale is plausible.
- Global trajectory drift is severe after integrating decoded deltas.
- Image-space FID is not sufficient for trajectory quality.
- Strict filtering can recover a small plausible subset, but only `12.8%` survives the OSM p95 check and `9.2%` survives the empirical-support check.
- The direct raw-trajectory DDPM matrix completed successfully. The best current model is `raw_absolute_unet1d` by final loss and empirical support.
- Final raw-diffusion samples were packaged and visualized against empirical Week05 support; absolute-coordinate models are support-valid but conservative, while delta models are smooth but drift away from support.
- A modular inference-only road-distance-field guidance component has been added for future guided sampling.
- The best GASF road-guidance variant so far is diagonal-only guidance with endpoint displacement cap `1916`, blend `0.25`, delta clip `37.5`, and strength `1`; it improves empirical support distance but still leaves many samples outside observed Week05 support.

The most reliable Week06 validity reference is currently the empirical Week05 support, not OSM, because the available Week03 raw-XY to OSM affine came from a different source file than Week05.

## Core Mathematical Objects

### GASF-to-Trajectory Decoding

The Week05 DDPM generates GASF-like images. Week06 decodes the generated image diagonal into local trajectory quantities and integrates them back into raw XY trajectories.

For delta-displacement trajectories:

```text
tau = [(x_1, y_1), ..., (x_T, y_T)]
Delta tau_i = (dx_i, dy_i)
x_i = x_0 + sum_{j <= i} dx_j
y_i = y_0 + sum_{j <= i} dy_j
```

The failure mode is cumulative: small local bias in decoded `dx, dy` can accumulate over `T = 224` steps and produce large endpoint drift.

### Road Distance Metrics

Given a road set `R`, the distance from a trajectory point to the nearest road is:

```text
D(p_i, R) = min_{r in R} ||p_i - r||_2
```

For a trajectory `tau`:

```text
road_mean(tau) = (1/T) sum_i D(p_i, R)
road_p95(tau) = percentile_95({D(p_i, R)})
```

Additional geometry checks include endpoint displacement, bounding-box width/height, maximum step length, and large-jump ratio.

### Empirical Support Metrics

Because the true Week05 road network / affine pairing is uncertain, an empirical support baseline was built from all real Week05 bounce-regularized trajectory points:

```text
S = {all real Week05 raw XY points}
D_support(p_i, S) = min_{s in S} ||p_i - s||_2
```

This is not a semantic road map, but it directly answers whether generated trajectories remain near the observed Week05 driving support.

### Road Guidance Energy

The new inference-only guidance module uses a differentiable distance field:

```text
E_road(tau) = (1/T) sum_{i=1}^{T} D(x_i, y_i)^2
```

At inference time:

```text
tau <- tau - eta * grad_tau E_road(tau)
```

Implementation details:

- Distance values are sampled with `torch.nn.functional.grid_sample`.
- Batched trajectories use shape `[B, T, 2]`.
- World coordinates are converted to normalized grid coordinates in `[-1, 1]`.
- Out-of-bounds coordinates are clamped for numerical safety.

Important caveat: clamping keeps samples finite, but far out-of-bounds points can receive little or no gradient because clamp saturation has zero gradient.

## Experiment 1: Decode Week05 GASF Samples to Trajectories

### Motivation

Week05 FID suggested that the generated GASF images were visually/statistically plausible. The first Week06 question was whether those images decode into plausible continuous vehicle trajectories.

### Script

```text
week06/scripts/decode_week05_samples_to_trajectories.py
```

Input:

```text
week05/results_npy_samples/week05_sig15_mse_diag/week05_sig15_mse_diag_step_040000_samples_1000.npy
```

Output:

```text
week06/decoded/week05_sig15_mse_diag_step_040000
```

### Principle

The script decodes generated GASF image channels back to raw XY trajectories and writes:

- decoded trajectories as `.npy`
- per-sample decode metrics
- out-of-position-range statistics

### Results

Generated trajectories have plausible local scale but excessive global drift:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| mean step length | 12.47 | 11.18 |
| path length mean | 2781.29 | 2492.47 |
| displacement mean | 2350.11 | 876.86 |
| bbox width mean | 1665.61 | 972.61 |
| bbox height mean | 1413.89 | 681.24 |
| large jump ratio mean | 0.0019 | 0.0 |

Out-of-position-range ratio for generated samples:

| Statistic | Value |
|---|---:|
| mean | 0.3505 |
| median | 0.3438 |
| p95 | 0.8665 |

### Problem Found

The diagonal/GASF representation can encode locally plausible deltas while still failing cumulative trajectory constraints.

## Experiment 2: Initial OSM Map Alignment Check

### Motivation

The next question was whether generated trajectories lie on roads. The first attempt reused the existing Week03 raw-XY to OSM affine transform and OSM road graph.

### Script

```text
week06/scripts/evaluate_decoded_map_alignment.py
```

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_map_eval
```

### Principle

Each trajectory point is transformed into OSM projected coordinates and evaluated against a Shapely STRtree of road geometries.

### Results

The initial road-distance numbers were not interpretable because real trajectories were also far from the loaded OSM network:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road mean distance | 1537.17 m | 962.56 m |
| road off 20 m ratio | 0.9736 | 0.9998 |

### Problem Found

The Week03 OSM map/affine did not cover Week05 trajectories. Full-real coverage check showed:

```text
point_inside_osm_bbox_ratio: 0.00039494
sample_inside_osm_bbox_ratio: 0.0
sample_intersects_osm_bbox_ratio: 0.0104469
north_gap_m: 2247.32
east_gap_m: 196.82
```

This showed the map mismatch was real, not a generated-sample issue.

## Experiment 3: Download Week05-Specific OSM Network

### Motivation

To make OSM diagnostics meaningful, Week06 downloaded a new drivable OSM network covering the full projected Week05 trajectory extent.

### Script

```text
week06/scripts/download_week05_osm_map.py
```

Output:

```text
week06/data/maps_week05/osm_drive_week05.graphml
week06/data/maps_week05/osm_drive_week05_edges.gpkg
week06/data/maps_week05/week05_osm_map_summary.json
week06/data/maps_week05/week05_osm_coverage.png
```

### Principle

The script:

1. Projects all Week05 real raw XY trajectories through the existing affine.
2. Converts the projected bounds to lon/lat.
3. Downloads `network_type="drive"` roads using OSMnx.
4. Saves projected edges and a coverage plot.

### Results

Downloaded OSM coverage:

| Quantity | Value |
|---|---:|
| projected nodes | 1,848 |
| projected drive edges | 3,580 |
| trajectory bbox inside edge bbox | True |
| west/south/east/north coverage gap | 0 / 0 / 0 / 0 m |

### Remaining Caveat

The OSM network covers the Week05 projected bbox, but the affine itself came from Week03 data:

```text
Week03 affine source: /home/jliu/gps_with_speed_224_UPDATED.xls
Week05 source: week05/data/vehicle_positions_TS_New.csv
```

The Week03 source has `lon/lat` and `car*` IDs; Week05 has local `x/y` and `veh*` IDs. OSM diagnostics are useful, but empirical support is currently the stronger Week05 validity reference.

## Experiment 4: Week05 OSM Evaluation With Offset Correction

### Motivation

Even after downloading a Week05-covering OSM network, visual and metric checks suggested an offset. Week06 searched for a translation offset that made real Week05 trajectories better align to OSM roads.

### Scripts

```text
week06/scripts/search_week05_map_offset.py
week06/scripts/evaluate_decoded_map_alignment.py
week06/scripts/visualize_decoded_trajectories_on_map.py
```

Key output:

```text
week06/results/week05_sig15_mse_diag_step_040000_map_eval_week05_osm_offset_dx220_dy-720
week06/results/trajectory_visualizations_offset_dx220_dy-720
```

### Principle

The evaluation applies:

```text
x_osm_corrected = x_osm + 220
y_osm_corrected = y_osm - 720
```

and recomputes road-distance metrics.

### Results

Offset-corrected OSM comparison:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road_mean_m mean | 220.28 | 15.24 |
| road_p95_m mean | 644.65 | 40.40 |
| displacement mean | 2350.11 | 853.07 |
| bbox_width mean | 1665.61 | 948.81 |
| bbox_height mean | 1413.89 | 675.50 |
| max_step_length mean | 37.54 | 26.76 |
| large_jump_ratio mean | 0.0019 | 0.0 |

Real baseline road metrics became plausible:

```text
real road_mean_m mean: 15.24 m
real road_p95_m mean: 40.40 m
```

### Problem Found

Generated samples still show severe global drift even after the OSM offset correction. The best generated examples may locally follow roads; median/random/worst samples often drift away from road support.

## Experiment 5: Empirical Week05 Support Evaluation

### Motivation

Because OSM depends on an uncertain affine and source mismatch, Week06 evaluated generated trajectories against observed Week05 real trajectory support.

### Script

```text
week06/scripts/evaluate_decoded_empirical_support_alignment.py
```

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_empirical_support_eval_real1000
```

Visualization:

```text
week06/scripts/visualize_decoded_trajectories_on_empirical_map.py
week06/results/trajectory_visualizations_empirical_support_metric_real1000
```

### Principle

All real Week05 raw XY points are used as a support set. A generated point is good if it is near this empirical support.

### Results

Support source:

| Quantity | Value |
|---|---:|
| support trajectories | 12,061 |
| support points | 2,701,664 |
| generated samples evaluated | 1,000 |
| real sanity-check samples | 1,000 |

Key generated-vs-real comparison:

| Metric | Generated | Real sanity check |
|---|---:|---:|
| support_mean_m mean | 417.55 | 0.00 |
| support_p95_m mean | 1059.47 | 0.00 |
| support_off_20m_ratio mean | 0.6869 | 0.0000 |
| displacement mean | 2350.11 | 876.86 |
| bbox_width mean | 1665.61 | 972.61 |
| bbox_height mean | 1413.89 | 681.24 |

### Interpretation

Generated trajectories are often far from the empirical Week05 road-like support. This is the strongest evidence that the Week05 GASF DDPM has trajectory-support drift.

## Experiment 6: Rejection Filtering

### Motivation

Before changing model training, Week06 asked a practical question:

```text
How many generated samples are plausibly usable after simple filtering?
```

### Script

```text
week06/scripts/filter_generated_trajectories.py
```

### Principle

A generated sample is accepted only if it stays below all selected thresholds. Thresholds are usually real-baseline p95 values:

```text
accept(tau) = all_k metric_k(tau) <= threshold_k
```

For example:

```text
road_mean_m <= real_p95(road_mean_m)
road_p95_m <= real_p95(road_p95_m)
displacement <= real_p95(displacement)
large_jump_ratio == 0
out_of_position_range_ratio == 0
```

### OSM Offset Strict p95 Filter

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p95_strict_offset_dx220_dy-720
```

Result:

| Quantity | Value |
|---|---:|
| generated samples | 1,000 |
| real baseline samples | 12,061 |
| accepted | 128 |
| rejected | 872 |
| acceptance rate | 0.128 |

Thresholds:

| Metric | Threshold |
|---|---:|
| road_mean_m | 22.09 |
| road_p95_m | 66.97 |
| displacement | 1878.63 |
| bbox_width | 1875.40 |
| bbox_height | 1326.67 |
| max_step_length | 31.39 |
| large_jump_ratio | 0 |
| out_of_position_range_ratio | 0 |

Most common failures:

| Criterion | Failed samples |
|---|---:|
| road_mean_m | 763 |
| road_p95_m | 740 |
| out_of_position_range_ratio | 701 |
| displacement | 593 |
| max_step_length | 450 |
| bbox_height | 425 |
| bbox_width | 352 |
| large_jump_ratio | 161 |

Accepted subset:

| Metric | Accepted mean | Accepted p95 |
|---|---:|---:|
| road_mean_m | 15.99 | 20.88 |
| road_p95_m | 40.06 | 57.97 |
| displacement | 987.38 | 1723.77 |
| bbox_width | 874.54 | 1614.03 |
| bbox_height | 605.10 | 1079.43 |
| max_step_length | 19.72 | 27.18 |

### Empirical-Support Filter

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_empirical_support_mean25_p95_75
```

Rules:

```text
support_mean_m <= 25
support_p95_m <= 75
real p95 thresholds for displacement, bbox_width, bbox_height, max_step_length
large_jump_ratio == 0
out_of_position_range_ratio == 0
```

Result:

| Quantity | Value |
|---|---:|
| generated samples | 1,000 |
| accepted | 92 |
| rejected | 908 |
| acceptance rate | 0.092 |

Accepted subset:

| Metric | Accepted mean | Accepted p95 |
|---|---:|---:|
| support_mean_m | 16.16 | 22.73 |
| support_p95_m | 41.56 | 64.59 |
| displacement | 961.01 | 1657.63 |
| bbox_width | 846.02 | 1504.07 |
| bbox_height | 616.78 | 1092.14 |
| max_step_length | 19.89 | 27.07 |

### Loose p99 Filter

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p99_loose
```

Result:

| Quantity | Value |
|---|---:|
| generated samples | 1,000 |
| accepted | 228 |
| rejected | 772 |
| acceptance rate | 0.228 |

### Interpretation

Filtering can extract a small plausible subset, but it cannot rescue the full generator distribution. Even a loose p99 filter accepts fewer than one quarter of samples.

## Experiment 7: Visualization of Accepted Samples

### Motivation

After filtering, it is useful to see whether accepted samples look road-like rather than only satisfying aggregate metrics.

### Scripts

```text
week06/scripts/visualize_decoded_trajectories_on_map.py
week06/scripts/visualize_accepted_trajectories_on_osm_map.py
```

Outputs:

```text
week06/results/trajectory_visualizations_accepted_p95_strict_offset_dx220_dy-720
week06/results/accepted_generated_on_osm_offset_dx220_dy-720
```

Important figures:

```text
accepted_all_overlay_osm.png
accepted_contact_sheet_osm.png
accepted_representative_overlay_osm.png
```

### Result

The accepted subset mostly lies within the main OSM-covered area. Many accepted trajectories are locally road-aligned, but some still have visibly straight or jagged global shapes compared with real trajectories.

## Experiment 8: Direct Raw-Trajectory Diffusion Matrix

### Motivation

The GASF image DDPM can score well in image space while failing in trajectory space. A direct raw-sequence DDPM is a cleaner control: it removes the GASF diagonal/integration bottleneck and directly models either absolute XY or delta-displacement sequences.

### Scripts

Shared implementation:

```text
week06/scripts/trajectory_diffusion_common.py
```

Entry points:

```text
week06/scripts/train_absolute_trajectory_diffusion.py
week06/scripts/train_delta_displacement_diffusion.py
```

Slurm matrix:

```text
week06/slurm/submit_raw_trajectory_diffusion_matrix.sh
week06/slurm/train_absolute_temporal_resnet.slurm
week06/slurm/train_absolute_temporal_resnet_dilated.slurm
week06/slurm/train_absolute_unet1d.slurm
week06/slurm/train_delta_temporal_resnet.slurm
week06/slurm/train_delta_temporal_resnet_dilated.slurm
week06/slurm/train_delta_unet1d.slurm
```

### Principle

Training data are normalized raw sequences:

```text
z = (xy - mean) / std
```

DDPM training samples a timestep `t`, adds Gaussian noise, and trains a temporal denoiser to predict the noise:

```text
epsilon ~ N(0, I)
z_t = sqrt(alpha_bar_t) z_0 + sqrt(1 - alpha_bar_t) epsilon
L = ||epsilon - epsilon_theta(z_t, t)||_2^2
```

Architectures:

- `temporal_resnet`
- `temporal_resnet_dilated`
- `unet1d`

Modes:

- `absolute`: model absolute XY trajectory sequence.
- `delta`: model delta-displacement sequence, then integrate from sampled real starts.

### Results After Completion

The first submitted batch (`1795817` to `1795822`) disappeared without logs or output. Root cause was likely:

```text
#SBATCH --output=/home/jliu/Thesis/week06/logs/...
```

pointing to a directory that did not exist before Slurm opened stdout/stderr. The script-level `mkdir -p` ran too late.

Fix applied:

```text
mkdir -p /home/jliu/Thesis/week06/logs
```

was added before matrix submission, and `week06/logs/` now exists.

Resubmitted jobs:

| Job ID | Name |
|---:|---|
| 1796435 | w06_abs_unet1d |
| 1796436 | w06_abs_dilated |
| 1796437 | w06_abs_resnet |
| 1796438 | w06_delta_dilated |
| 1796439 | w06_delta_unet1d |
| 1796440 | w06_delta_resnet |

All six jobs completed successfully. Each produced:

- `checkpoint-final`
- checkpoints every 2,000 steps through 50,000
- final samples in `samples/`
- `train_log.jsonl`
- `train_summary.json`

Compact analysis output:

```text
week06/results/raw_diffusion_analysis/raw_diffusion_analysis.md
week06/results/raw_diffusion_analysis/raw_diffusion_summary.json
week06/results/raw_diffusion_analysis/raw_diffusion_per_sample_metrics.csv
week06/results/raw_diffusion_analysis/training_log_summary.json
```

Final-sample packaging and visualization scripts:

```text
week06/scripts/package_raw_diffusion_samples.py
week06/scripts/visualize_raw_diffusion_model_comparison.py
```

Packaged final sample directories:

```text
week06/decoded/raw_absolute_temporal_resnet_step_050000
week06/decoded/raw_absolute_temporal_resnet_dilated_step_050000
week06/decoded/raw_absolute_unet1d_step_050000
week06/decoded/raw_delta_temporal_resnet_step_050000
week06/decoded/raw_delta_temporal_resnet_dilated_step_050000
week06/decoded/raw_delta_unet1d_step_050000
```

Each directory contains `1,000` generated trajectories in:

```text
trajectories_raw_xy/generated_XXXXXX.npy
```

Training summary:

| Experiment | Final loss | Min logged loss | Min step | Steps/s | Elapsed min |
|---|---:|---:|---:|---:|---:|
| raw_absolute_temporal_resnet | 0.01545 | 0.01374 | 44740 | 62.60 | 13.3 |
| raw_absolute_temporal_resnet_dilated | 0.01412 | 0.01250 | 49610 | 61.54 | 13.5 |
| raw_absolute_unet1d | 0.01126 | 0.01010 | 45940 | 39.34 | 21.2 |
| raw_delta_temporal_resnet | 0.05899 | 0.05180 | 39480 | 51.86 | 16.1 |
| raw_delta_temporal_resnet_dilated | 0.05455 | 0.04801 | 39480 | 53.61 | 15.5 |
| raw_delta_unet1d | 0.03866 | 0.03347 | 43700 | 38.39 | 21.7 |

Final 50k sample quality against empirical Week05 support:

| Experiment | Empirical strict accepted | support_mean | support_p95 | displacement | bbox_w | bbox_h | max_step | out_range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| absolute_temporal_resnet | 927/1000 (92.7%) | 10.4 | 31.8 | 562.2 | 615.8 | 451.5 | 24.4 | 0.000 |
| absolute_temporal_resnet_dilated | 959/1000 (95.9%) | 10.0 | 30.7 | 592.1 | 645.4 | 462.3 | 23.9 | 0.000 |
| absolute_unet1d | 985/1000 (98.5%) | 8.4 | 25.6 | 549.7 | 613.5 | 448.5 | 23.7 | 0.000 |
| delta_temporal_resnet | 610/1000 (61.0%) | 43.1 | 103.0 | 456.0 | 531.5 | 361.0 | 12.0 | 0.041 |
| delta_temporal_resnet_dilated | 592/1000 (59.2%) | 42.3 | 101.5 | 422.1 | 513.9 | 364.0 | 11.9 | 0.039 |
| delta_unet1d | 605/1000 (60.5%) | 42.8 | 103.5 | 425.2 | 502.5 | 364.3 | 11.4 | 0.041 |
| real_bounce_regularized_first1000 | n/a | 0.0 | 0.0 | 876.9 | 972.6 | 681.2 | 26.9 | 0.000 |
| GASF decoded Week05 checkpoint | 92/1000 earlier filter | 417.6 | 1059.5 | 2350.1 | 1665.6 | 1413.9 | 37.5 | n/a |

The strict empirical acceptance test uses:

```text
support_mean_m <= 25
support_p95_m <= 75
displacement <= real p95
bbox_width <= real p95
bbox_height <= real p95
max_step_length <= real p95
large_jump_ratio == 0
out_of_position_range_ratio == 0
```

Interpretation:

1. Direct raw-sequence diffusion is a major improvement over GASF-image diffusion for empirical support. The best raw model, `raw_absolute_unet1d`, accepts `98.5%` of final samples under the strict empirical-support filter, compared with `9.2%` for the decoded GASF checkpoint.
2. Absolute-coordinate models are much better than delta models under this metric. They stay near observed support and never leave the Week05 coordinate bounds.
3. Delta models no longer show the huge GASF-style global drift, but they are too compact and still often away from dense support. Their mean `support_p95` is around `101-104 m`, above the strict `75 m` threshold.
4. The absolute models appear conservative: their mean displacement and bbox sizes are substantially smaller than the real baseline. For example, `absolute_unet1d` has mean displacement `549.7` vs real `876.9`, and bbox width `613.5` vs real `972.6`. High acceptance partly reflects staying close to dense support, not necessarily matching the full real trajectory distribution.
5. Among the completed runs, `raw_absolute_unet1d` is the strongest checkpoint by empirical support and final training loss. `raw_absolute_temporal_resnet_dilated` is second-best by support metrics and trains faster.

### Visual Inspection

Empirical-support visualizations were generated for all six final sample sets. Each model has:

```text
week06/results/<run>_visualizations_empirical_support/generated_contact_sheet_empirical_real_map.png
week06/results/<run>_visualizations_empirical_support/generated_on_empirical_real_map.png
week06/results/<run>_visualizations_empirical_support/generated_vs_real_on_empirical_real_map.png
week06/results/<run>_visualizations_empirical_support/selection_manifest.json
```

A compact cross-model comparison sheet was also generated:

```text
week06/results/raw_diffusion_visual_comparison/raw_diffusion_best_median_worst_support.png
```

The comparison sheet shows the best, median, and worst generated samples by `support_mean_m` for each model.

Visual conclusion:

1. `raw_absolute_unet1d` is the best overall raw model. Its samples generally remain on or near dense empirical trajectory support. The best examples are very close to support, and even the worst examples are much less pathological than the decoded Week05 GASF samples.
2. The absolute ResNet variants are similar but slightly worse than `raw_absolute_unet1d`. `raw_absolute_temporal_resnet_dilated` is marginally better than the non-dilated ResNet by support metrics.
3. The delta models are smooth and have no large jumps, but their trajectories can slide away from the empirical support over time. This is visible as low-curvature drift rather than the explosive endpoint drift seen in the GASF-decoded samples.
4. The absolute models appear conservative: many samples are shorter and more spatially compact than real trajectories. Their high empirical-support acceptance should therefore be read as "valid support adherence", not full distribution matching.

## Experiment 9: Inference-Only Distance-Field Road Guidance

### Motivation

Instead of retraining the GASF DDPM, add road-network guidance during inference:

1. Generate a GASF image.
2. Decode it to trajectory space.
3. Apply differentiable road-distance guidance.
4. Encode the adjusted trajectory back to GASF.

This lets road constraints influence sampling without changing the model architecture or training loop.

### Implementation

Module:

```text
src/cnr_trajectory/guidance/road_distance_field.py
```

Public API:

```text
DistanceFieldRoadGuidance
RoadRasterMetadata
```

Tests:

```text
tests/test_road_distance_guidance.py
```

Slurm smoke test:

```text
week06/slurm/test_road_guidance.slurm
```

Submitted job:

| Job ID | Name |
|---:|---|
| 1796433 | w06_road_guidance |

### Principle

If only a binary road raster is available, compute a Euclidean distance field:

```text
D(x, y) = distance from point (x, y) to nearest road
```

Then define:

```text
E_road(tau) = (1/T) sum_i D(x_i, y_i)^2
```

and apply gradient descent:

```text
tau <- tau - eta * grad_tau E_road(tau)
```

The PyTorch implementation uses:

```text
grid_sample(distance_field, normalized_world_grid)
```

where world `(x, y)` is mapped to normalized grid coordinates:

```text
x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
```

For image-style rasters with `y_axis_down=True`:

```text
y_norm = 1 - 2 * (y - y_min) / (y_max - y_min)
```

Thus:

```text
world y = y_max -> grid y = -1  # top row
world y = y_min -> grid y = +1  # bottom row
```

### Example Integration

```python
for step_idx, t in enumerate(reversed(timesteps)):
    x = ddpm_denoise_step(x, t)

    if guidance.should_apply(step_idx):
        traj = gasf_decode(x)  # [B, T, 2], world coordinates
        traj, road_energy, mean_road_dist = guidance(traj)
        x = gasf_encode(traj)
```

### Result

The guidance unit tests pass locally in the `thesis-diffusion` environment:

```text
PYTHONPATH=src /home/jliu/.conda/envs/thesis-diffusion/bin/python -m pytest tests/test_road_distance_guidance.py
4 passed
```

The Slurm GPU smoke test also completed successfully:

```text
job_id: 1796433
node: compute-3-14.cluster.lan
gpu: NVIDIA A40
pytest: 4 passed
cuda_available: True
before_energy: 16.000000
after_energy: 11.647619
before_mean_dist: 4.000000
after_mean_dist: 3.412861
```

This confirms that the road-guidance component works on GPU and that the differentiable distance-field energy decreases in the smoke setting.

### Caveat

This guidance can reduce distance-field energy for points within the raster extent. For points far outside the raster, the current clamp behavior prioritizes finite values and may not provide a restoring gradient.

### Full GASF Road-Guidance Sweep

After the initial `guidance_strength=1000` run produced trajectories that were much longer than normal trajectories, a low-strength sweep was submitted to separate "too large a guidance strength" from a more structural GASF update problem.

Slurm scripts:

```text
week06/slurm/run_week05_gasf_road_guidance_strength0p1.slurm
week06/slurm/run_week05_gasf_road_guidance_strength1.slurm
week06/slurm/run_week05_gasf_road_guidance_strength10.slurm
```

Job IDs:

```text
1797950  strength=0.1
1797952  strength=1
1797953  strength=10
```

Primary result paths:

```text
week06/results/week05_sig15_mse_diag_road_guidance_strength0p1
week06/results/week05_sig15_mse_diag_road_guidance_strength1
week06/results/week05_sig15_mse_diag_road_guidance_strength10
week06/results/week05_sig15_mse_diag_road_guidance_full

week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength0p1_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength1_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength10_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength1000_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_empirical_support_eval_real1000/summary.json
```

Trajectory-space comparison:

| Case | mean step | p95 step | p99 step | path length mean | displacement mean | support mean |
|---|---:|---:|---:|---:|---:|---:|
| unguided | 12.47 | 26.55 | 35.14 | 2781.29 | 2350.11 | 417.55 |
| strength 0.1 | 14.56 | 38.03 | 86.70 | 3246.29 | 2767.67 | 594.61 |
| strength 1 | 14.05 | 34.60 | 86.80 | 3133.80 | 2653.56 | 551.07 |
| strength 10 | 14.70 | 37.99 | 87.39 | 3277.73 | 2804.31 | 600.21 |
| strength 1000 | 14.55 | 37.81 | 89.63 | 3243.83 | 2767.79 | 593.93 |

Internal road-guidance metrics changed very little:

| Case | mean road distance before | mean road distance after | applied guidance steps |
|---|---:|---:|---:|
| strength 0.1 | 197.4111 | 197.4111 | 3125 |
| strength 1 | 185.3695 | 185.3692 | 3125 |
| strength 10 | 191.5043 | 191.5004 | 3125 |
| strength 1000 | 197.3235 | 196.9237 | 3125 |

Diagonal decoding diagnostics showed saturation after any guided run:

| Case | dx diag p99 | dy diag p99 | dx delta p99 | dy delta p99 |
|---|---:|---:|---:|---:|
| unguided | 0.973836 | 0.985783 | 30.09 | 29.28 |
| strength 0.1 | 0.998592 | 0.999999 | 50.60 | 85.12 |
| strength 1 | 0.998487 | 0.999999 | 50.14 | 85.12 |
| strength 10 | 0.999252 | 0.999999 | 55.22 | 85.12 |
| strength 1000 | 0.999862 | 0.999999 | 67.42 | 85.12 |

Conclusion: the failure is not only `guidance_strength=1000`. Even `0.1` makes trajectories longer and farther from empirical support. The likely cause is the current `road_guided_noise_prediction` update: every guided step decodes `x0_pred` to trajectory space, applies road guidance, then rewrites the full GASF channels from the guided deltas. This full-channel re-encoding acts like a strong projection onto an idealized GASF manifold and perturbs the diffusion state even when the actual road-distance decrease is negligible.

Immediate fix direction:

- avoid full-channel GASF rewrites by default,
- update only the GASF diagonal used by the analytical decoder,
- blend guided `x0` back toward the model-predicted `x0`,
- clamp guided deltas to a real-data physical range before encoding.

Implemented follow-up:

```text
week06/scripts/sample_week05_gasf_road_guidance.py
```

New inference controls:

```text
--gasf-update-mode diagonal|full
--guidance-x0-blend 0.1
--delta-clip-abs 37.5
```

The default update mode is now `diagonal`, which preserves the model-predicted off-diagonal GASF image content and only changes the diagonal values that the analytical decoder actually uses. The original full-channel behavior remains available with `--gasf-update-mode full` for ablation.

Smoke validation:

```text
/home/jliu/.conda/envs/thesis-diffusion/bin/python -u week06/scripts/sample_week05_gasf_road_guidance.py \
  --cpu \
  --output-dir /tmp/week06_gasf_guidance_smoke \
  --guidance-strengths 0.1 \
  --num-guidance-steps 1 \
  --guidance-start-fraction 0.0 \
  --guidance-end-fraction 1.0 \
  --distance-field-size 32 \
  --num-samples 1 \
  --sample-batch-size 1 \
  --sample-inference-steps 1 \
  --gasf-update-mode diagonal \
  --guidance-x0-blend 0.1 \
  --delta-clip-abs 37.5

result: completed; before_mean_dist=81.2368, after_mean_dist=81.2367, applied_steps=1
```

### Fixed Diagonal-Only Guidance Results

The fixed diagonal-only guidance sweep completed on 2026-07-05:

```text
1798144  strength=0.1
1798145  strength=1
1798146  strength=10
```

Slurm scripts:

```text
week06/slurm/run_week05_gasf_road_guidance_fixed_diagonal_strength0p1.slurm
week06/slurm/run_week05_gasf_road_guidance_fixed_diagonal_strength1.slurm
week06/slurm/run_week05_gasf_road_guidance_fixed_diagonal_strength10.slurm
```

Result paths:

```text
week06/results/week05_sig15_mse_diag_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength0p1
week06/results/week05_sig15_mse_diag_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength1
week06/results/week05_sig15_mse_diag_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength10

week06/results/week05_sig15_mse_diag_step_040000_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength0p1_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength1_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength10_empirical_support_eval/summary.json
```

All runs used:

```text
gasf_update_mode: diagonal
guidance_x0_blend: 0.1
delta_clip_abs: 37.5
num_guidance_steps: 3
guidance_start_fraction: 0.5
guidance_end_fraction: 1.0
distance_scale: 25
```

Trajectory-space comparison:

| Case | mean step | p95 step mean | max step mean | large jump ratio | path length mean | displacement mean | support mean | support p95 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unguided | 12.47 | 22.82 | 37.54 | 0.0019 | 2781.29 | 2350.11 | 417.55 | 1059.47 |
| old full-channel strength 0.1 | 14.56 | 35.74 | 58.44 | 0.0300 | 3246.29 | 2767.67 | 594.61 | 1441.16 |
| fixed diagonal strength 0.1 | 12.41 | 22.79 | 32.00 | 0.0014 | 2767.79 | 2338.77 | 413.01 | 1049.22 |
| fixed diagonal strength 1 | 12.21 | 22.49 | 31.49 | 0.0014 | 2723.03 | 2286.04 | 381.93 | 986.77 |
| fixed diagonal strength 10 | 12.60 | 23.03 | 32.41 | 0.0017 | 2809.28 | 2386.57 | 410.35 | 1050.59 |

The fixed update removed the diagonal saturation failure:

| Case | step p50 | step p95 | step p99 | step max | dx diag p99 | dy diag p99 | dx delta p99 | dy delta p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unguided | 10.87 | 26.55 | 35.14 | 101.79 | 0.973836 | 0.985783 | 30.09 | 29.28 |
| old full-channel strength 0.1 | 10.96 | 38.03 | 86.70 | 131.90 | 0.998592 | 0.999999 | 50.60 | 85.12 |
| fixed diagonal strength 0.1 | 10.87 | 26.55 | 35.08 | 68.11 | 0.973822 | 0.985766 | 30.08 | 29.27 |
| fixed diagonal strength 1 | 10.62 | 26.41 | 34.96 | 55.62 | 0.973219 | 0.984371 | 29.93 | 28.71 |
| fixed diagonal strength 10 | 10.97 | 26.84 | 36.30 | 61.38 | 0.975327 | 0.986460 | 30.50 | 29.59 |

Internal road-guidance distance still changed only slightly:

| Case | mean road distance before | mean road distance after |
|---|---:|---:|
| fixed diagonal strength 0.1 | 193.7745 | 193.7745 |
| fixed diagonal strength 1 | 178.9645 | 178.9641 |
| fixed diagonal strength 10 | 185.2457 | 185.2417 |

Conclusion: the diagonal-only update fixed the trajectory-length regression and prevented GASF diagonal saturation. It did not make road-distance guidance strong enough to substantially pull samples toward empirical support. Among the tested fixed runs, `strength=1` is the best by empirical support (`support_mean=381.93`, `support_p95_mean=986.77`) while also being slightly shorter and less jumpy than the unguided baseline.

### Endpoint-Capped Diagonal Guidance Result

The final GASF guidance experiment added an endpoint displacement cap to reduce the main remaining failure mode: samples that preserve plausible step lengths but drift to unrealistically large endpoint displacement and spatial extent.

Slurm/log identifier:

```text
1800906  endpoint cap 1916, diagonal blend 0.25, strength 1
```

Result paths:

```text
week06/results/week05_sig15_mse_diag_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1
week06/decoded/week05_sig15_mse_diag_step_040000_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1_empirical_support_eval/summary.json
week06/results/week05_sig15_mse_diag_step_040000_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1_visualizations_empirical_support/
```

Configuration:

```text
gasf_update_mode: diagonal
guidance_x0_blend: 0.25
delta_clip_abs: 37.5
endpoint_cap: 1916.0
endpoint_projection_strength: 1.0
road_guidance_strength: 1.0
num_guidance_steps: 3
guidance_start_fraction: 0.5
guidance_end_fraction: 1.0
distance_scale: 25
```

Important caveat: the recorded `road_guidance_summary.json` for the latest run reports `loaded_existing: 1` and `applied_guidance_steps: 0`, so the final job reused the existing sample file for this configuration before decode/evaluation/visualization. The metrics below are valid for the saved endpoint-capped sample set, but this log should not be read as proof that new guidance steps were executed during the last rerun.

Trajectory-space and empirical-support comparison:

| Case | mean step | p95 step mean | max step mean | large jump ratio | path length mean | displacement mean | bbox width mean | bbox height mean | support mean | support p95 mean | off >20m ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real bounce baseline | 11.18 | 24.52 | 26.88 | 0.0000 | 2492.47 | 876.86 | 972.61 | 681.24 | 0.00 | 0.00 | 0.0000 |
| unguided GASF | 12.47 | 22.82 | 37.54 | 0.0019 | 2781.29 | 2350.11 | 1665.61 | 1413.89 | 417.55 | 1059.47 | 0.6869 |
| fixed diagonal strength 1 | 12.21 | 22.49 | 31.49 | 0.0014 | 2723.03 | 2286.04 | 1615.95 | 1392.95 | 381.93 | 986.77 | 0.6701 |
| endpoint-capped diagonal strength 1 | 10.85 | 21.05 | 29.76 | 0.0000 | 2418.61 | 1960.04 | 1391.63 | 1215.49 | 270.25 | 706.73 | 0.6450 |

Distribution of generated support distances:

| Threshold | Fraction of generated samples |
|---|---:|
| support_mean_m < 50 | 31.4% |
| support_mean_m < 100 | 42.2% |
| support_mean_m < 200 | 55.2% |
| support_mean_m < 500 | 80.0% |

Interpretation:

1. Endpoint capping is the strongest GASF-guidance change so far. It reduces mean displacement from `2286.04` to `1960.04` relative to fixed diagonal `strength=1`, and reduces mean empirical support distance from `381.93 m` to `270.25 m`.
2. Step-scale behavior is now close to the real baseline. Mean max step length is `29.76 m`, compared with `26.88 m` for real trajectories.
3. The generator still does not match the real trajectory distribution. Mean displacement remains more than twice the real baseline (`1960.04` vs `876.86`), bbox height remains high (`1215.49` vs `681.24`), and the mean fraction of points more than `20 m` from empirical support is still `0.6450`.
4. Visualizations show the same pattern: best samples can lie near the empirical support, but median/worst and random samples still include long straight drifting segments that leave the observed Week05 support region.

Conclusion: endpoint-capped diagonal guidance improves the GASF baseline more than the earlier full-channel or fixed-diagonal sweeps, but it is still a partial mitigation rather than a solution. The raw absolute-coordinate diffusion baseline remains much stronger under empirical-support validity.

## Current Problems and Lessons

### 1. Image-Space Metrics Are Insufficient

The Week05 GASF checkpoint has good FID, but trajectory-space metrics reveal failure. Every future checkpoint should be evaluated in trajectory space.

### 2. Coordinate Provenance Matters

Week03 and Week05 data sources differ. OSM diagnostics are useful, but claims about true road validity require either:

- recovering the true Week05 road/geospatial mapping,
- fitting a Week05-specific transform from reliable correspondences,
- or using empirical support as the main validity reference.

### 3. Rejection Filtering Is Diagnostic, Not a Solution

Filtering found plausible subsets:

| Filter | Accepted | Rate |
|---|---:|---:|
| OSM p95 strict offset-corrected | 128 / 1000 | 12.8% |
| Empirical support strict | 92 / 1000 | 9.2% |
| OSM p99 loose | 228 / 1000 | 22.8% |

This is useful for inspection, but the generator distribution remains poor.

### 4. Slurm Output Directories Must Exist Before Submission

Slurm opens `#SBATCH --output` and `#SBATCH --error` before executing script commands. Therefore, `mkdir -p` inside a script is not enough if the log directory does not already exist.

Fix:

```text
week06/logs/ created
submit_raw_trajectory_diffusion_matrix.sh now creates week06/logs before sbatch calls
```

## Recommended Next Steps

1. Add OSM offset-corrected road-distance evaluation for the six raw diffusion final sample sets:

- displacement
- bbox width/height
- max step length
- OSM offset-corrected road distance

2. Visualize the final `raw_absolute_unet1d` and `raw_absolute_temporal_resnet_dilated` samples on the OSM map. Empirical-support visualization is complete; OSM visualization remains useful but should be treated as diagnostic until the transform is verified.

3. Compare diversity against real trajectories, especially endpoint displacement, bbox area, path length, and start/end spatial coverage. The absolute models may be support-valid but under-dispersed.

4. Add trajectory-space evaluation into sampling/checkpoint scripts so future models cannot pass only by image-space FID.

5. For future GASF guidance experiments, use the endpoint-capped diagonal-only path and treat it as an ablation baseline rather than the main solution:

```text
guidance_strength: 0.1 to 1.0
num_guidance_steps: 1 to 5
apply_start_step/apply_end_step: mid-to-late reverse diffusion window
distance_scale: road distance scale in meters, e.g. 10 to 50
gasf_update_mode: diagonal
guidance_x0_blend: 0.1 to 0.25
delta_clip_abs: about 37.5 meters for the Week05 bounce-regularized data
endpoint_cap: real p95 displacement, about 1916 meters for the current Week05 baseline
endpoint_projection_strength: 0.5 to 1.0
```

6. Treat OSM-guided results as diagnostic until the Week05 coordinate transform is verified.

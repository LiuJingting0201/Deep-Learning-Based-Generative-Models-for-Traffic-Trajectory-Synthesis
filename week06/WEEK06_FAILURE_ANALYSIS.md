# Week06 Failure Analysis: Good GASF FID, Bad Trajectories

## Summary

The Week05 best checkpoint, `week05_sig15_mse_diag/checkpoint-40000`, had the best GASF image-space FID:

```text
FID = 44.5496
```

However, after decoding generated GASF samples back into raw XY trajectories and evaluating them on a Week05-specific OSM road network, the samples are not usable as trajectory generations.

An additional map-offset check showed that the Week03 raw-XY to OSM affine is shifted for Week05. However, a later source check showed that this is not only an offset problem: Week03 and Week05 use different source files.

Week03 affine source:

```text
/home/jliu/gps_with_speed_224_UPDATED.xls
```

Week05 source:

```text
week05/data/vehicle_positions_TS_New.csv
```

The Week03 source has `lon/lat` and `car*` IDs; the Week05 source has only local `x/y` and `veh*` IDs. Therefore the OSM-based results are useful diagnostics, but the empirical Week05 support should be treated as the primary Week06 map-validity reference unless the true Week05 road network file is recovered.

The current corrected OSM diagnostic applies:

```text
offset_x_m = +220
offset_y_m = -720
```

## What Worked

1. The GASF-to-trajectory inverse path works mechanically.
2. A Week05-specific OSM network was downloaded and verified to cover the full real trajectory extent.
3. Real Week05 trajectories now have plausible road-distance metrics after adding the Week05 offset correction.

With the offset-corrected map:

```text
real road_mean_m mean: 15.24 m
real road_p95_m mean: 40.40 m
```

This confirms that the previous kilometer-scale distances were a map coverage problem, and that the remaining random-real mismatch was partly a coordinate/source mismatch problem.

## What Failed

Generated decoded trajectories show severe global drift.

Key full-real-baseline comparison:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road_mean_m mean | 220.28 m | 15.24 m |
| road_p95_m mean | 644.65 m | 40.40 m |
| displacement mean | 2350.11 | 853.07 |
| bbox_width mean | 1665.61 | 948.81 |
| bbox_height mean | 1413.89 | 675.50 |
| max_step_length mean | 37.54 | 26.76 |
| large_jump_ratio mean | 0.0019 | 0.0 |

Offset-corrected visualizations in `week06/results/trajectory_visualizations_offset_dx220_dy-720/` show the same pattern:

- The best generated samples can locally follow roads.
- Median/random samples often become long drifting curves.
- Worst samples leave the main map area entirely.
- Real baseline trajectories are much more road-like and spatially bounded.

## Interpretation

The DDPM learned something that looks like GASF image structure, but the generated diagonal does not preserve trajectory-level constraints after integration.

The likely failure mode is:

1. The model generates visually plausible GASF-like channels.
2. The diagonal gives a locally plausible `dx/dy` sequence.
3. Small bias or inconsistency in decoded deltas accumulates over 224 steps.
4. The resulting absolute trajectory drifts away from the real spatial support and road network.

This means image-space FID is not sufficient for this task. A low GASF FID can hide trajectory-space collapse.

## Rejection/Filter Baseline

Before changing the model, Week06 tests a strict rejection filter:

- Use real baseline p95 thresholds for map and trajectory metrics.
- Reject generated samples that exceed any selected threshold.
- Also reject samples with any decoded point outside the Week05 position range.

Script:

```text
week06/scripts/filter_generated_trajectories.py
```

This is not a real solution, but it answers an immediate question:

```text
How many generated samples are even plausibly usable after simple filtering?
```

### Offset-Corrected Strict p95 Filter Result

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p95_strict_offset_dx220_dy-720
```

Rules:

- Thresholds are real-bounce-baseline p95 values.
- No large jumps are allowed.
- No decoded points may leave the Week05 position range.

Result:

```text
generated samples: 1000
accepted: 128
rejected: 872
acceptance_rate: 0.128
```

Thresholds:

| Metric | Threshold |
|---|---:|
| road_mean_m | 22.09 |
| road_p95_m | 66.97 |
| displacement | 1878.63 |
| bbox_width | 1875.40 |
| bbox_height | 1326.67 |
| max_step_length | 31.39 |
| large_jump_ratio | 0.0 |
| out_of_position_range_ratio | 0.0 |

Most common failure counts:

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

Accepted subset summary:

| Metric | Accepted mean | Accepted p95 |
|---|---:|---:|
| road_mean_m | 15.99 | 20.88 |
| road_p95_m | 40.06 | 57.97 |
| displacement | 987.38 | 1723.77 |
| bbox_width | 874.54 | 1614.03 |
| bbox_height | 605.10 | 1079.43 |
| max_step_length | 19.72 | 27.18 |

The accepted subset has road and geometry metrics close to the offset-corrected real baseline, but it is only 12.8% of the generated samples.

### Empirical-Support Filter Result

The more appropriate Week05 validity check uses all Week05 real raw `x/y` points as empirical road support.

Evaluation output:

```text
week06/results/week05_sig15_mse_diag_step_040000_empirical_support_eval_real1000
```

Visualization output:

```text
week06/results/trajectory_visualizations_empirical_support_metric_real1000
```

The real sanity-check subset has zero support distance by construction, while generated samples are far from the real support:

| Metric | Generated | Real sanity check |
|---|---:|---:|
| support_mean_m mean | 417.55 | 0.00 |
| support_p95_m mean | 1059.47 | 0.00 |
| support_off_20m_ratio mean | 0.6869 | 0.0000 |

Empirical-support rejection output:

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

```text
generated samples: 1000
accepted: 92
rejected: 908
acceptance_rate: 0.092
```

This is the strongest current evidence that the Week05 DDPM generates a small plausible subset, but the overall sample distribution has severe trajectory-support drift.

### Loose p99 Filter Result

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p99_loose
```

Rules:

- Thresholds are real-bounce-baseline p99 values.
- Up to 5% out-of-position-range points are allowed.

Result:

```text
generated samples: 1000
accepted: 228
rejected: 772
acceptance_rate: 0.228
```

Even with a much looser filter, fewer than one quarter of samples survive. This reinforces that rejection can extract a plausible subset, but it cannot rescue the generator as a whole.

## Next Directions

Filtering can identify a small usable subset, but the model itself needs stronger constraints:

1. Add trajectory-space evaluation to every checkpoint, not just GASF FID.
2. Train or guide with cumulative trajectory constraints:
   - endpoint displacement,
   - bbox range,
   - out-of-range ratio,
   - road-distance loss.
3. Consider direct sequence diffusion or latent trajectory diffusion instead of relying only on image-space GASF.
4. Use sliding-window real continuous trajectories as a cleaner control dataset.
5. Treat bounce-regularized data as a coverage-preserving baseline, not as final physical trajectory ground truth.

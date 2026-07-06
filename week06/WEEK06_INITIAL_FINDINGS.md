# Week06 Initial Findings

## What Ran

Decoded the current best Week05 generated samples:

```text
week05/results_npy_samples/week05_sig15_mse_diag/week05_sig15_mse_diag_step_040000_samples_1000.npy
```

using:

```text
week06/scripts/decode_week05_samples_to_trajectories.py
```

Output:

```text
week06/decoded/week05_sig15_mse_diag_step_040000
```

Then ran a preliminary 1000 generated vs 1000 real baseline evaluation:

```text
week06/scripts/evaluate_decoded_map_alignment.py
```

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_map_eval
```

`week06/decoded/` and `week06/results/` are ignored because they are generated artifacts.

## Decode Summary

The generated trajectories decode successfully into `224 x 2` raw XY arrays.

Key generated trajectory-space summary from 1000 samples:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| mean step length | 12.47 | 11.18 |
| path length mean | 2781.29 | 2492.47 |
| displacement mean | 2350.11 | 876.86 |
| bbox width mean | 1665.61 | 972.61 |
| bbox height mean | 1413.89 | 681.24 |
| large jump ratio mean | 0.0019 | 0.0 |

Main signal: generated trajectories have plausible per-step scale, but they drift much farther globally than the real bounce-regularized baseline. The endpoint displacement and bounding boxes are especially inflated.

The decode script also reports that generated points often move outside the Week05 `position_stats` range:

```text
out_of_position_range_ratio mean: 0.3505
median: 0.3438
p95: 0.8665
```

This suggests that GASF diagonal decoding captures local dx/dy, but the model does not sufficiently constrain accumulated global drift.

## Map Evaluation Caveat

The preliminary road-distance evaluation technically runs with:

```text
road_index_mode: shapely_strtree
```

However, the real baseline is also almost entirely far from the loaded OSM road network:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road mean distance | 1537.17 m | 962.56 m |
| road off 20 m ratio | 0.9736 | 0.9998 |

This means the current OSM distance should not yet be interpreted as final map validity.

## Coordinate Coverage Diagnostic

The loaded Week03 OSM edge bounds are:

```text
x: 506433.25 to 509705.64
y: 4150432.91 to 4152648.22
```

For the first 1000 Week05 real trajectories, applying the Week03 affine approximately gives:

```text
x: 506601.03 to 509902.43
y: 4152507.24 to 4154895.43
```

The Week05 real trajectories extend north of the OSM edge bounds by more than 2 km. This likely explains why even real trajectories have huge road-distance values.

## Full Week05 Real Map Coverage Check

Script:

```text
week06/scripts/diagnose_week05_real_map_coverage.py
```

Output:

```text
week06/results/week05_real_map_coverage
```

Using all 12,061 Week05 real bounce-regularized trajectories:

```text
num_points: 2,701,664
raw XY bounds:
  x: 0.86 to 3302.46
  y: 1845.99 to 4245.54

Week05 trajectory bounds after Week03 affine:
  x: 506600.86 to 509902.46
  y: 4152495.99 to 4154895.54

Week03 OSM edge bounds:
  x: 506433.25 to 509705.64
  y: 4150432.91 to 4152648.22
```

Coverage result:

```text
point_inside_osm_bbox_ratio: 0.00039494
sample_inside_osm_bbox_ratio: 0.0
sample_intersects_osm_bbox_ratio: 0.0104469
north_gap_m: 2247.32
east_gap_m: 196.82
```

Visual inspection confirms that Week05 real trajectories form a coherent road-like network, but that network is mostly north of the Week03 OSM edges. Therefore, the current OSM distance results are primarily diagnosing map coverage mismatch, not generated-trajectory map validity.

## Immediate Interpretation

1. The GASF-to-trajectory inverse path is working.
2. Generated Week05 samples show global drift relative to real bounce-regularized data.
3. The current Week03 OSM map/affine pair does not appear sufficient for final Week05 map evaluation.
4. Before making map-validity claims, Week06 should either:
   - rebuild/download an OSM road graph covering the full Week05 raw XY extent,
   - verify whether Week05 raw XY uses the same coordinate convention as Week03,
   - or reuse local road-distance fields only when their extents match the Week05 trajectories being evaluated.

## Week05-Specific OSM Network

Script:

```text
week06/scripts/download_week05_osm_map.py
```

Output:

```text
week06/data/maps_week05/osm_drive_week05_edges.gpkg
week06/data/maps_week05/osm_drive_week05.graphml
week06/data/maps_week05/week05_osm_map_summary.json
week06/data/maps_week05/week05_osm_coverage.png
```

The new OSM query was built from the full Week05 real trajectory bbox after applying the raw-XY to OSM affine transform.

Downloaded coverage:

```text
Week05 trajectory projected bounds:
  x: 506600.86 to 509902.46
  y: 4152495.99 to 4154895.54

Downloaded OSM edge bounds:
  x: 506250.50 to 510256.05
  y: 4152053.16 to 4155334.41

coverage_gap_m:
  west/south/east/north: 0 / 0 / 0 / 0
```

The Week05-specific OSM network has 1,848 projected nodes and 3,580 projected drive edges. Visual inspection of `week05_osm_coverage.png` shows the downloaded road network covers the Week05 trajectory bbox.

## Map Evaluation With Week05 OSM

Using the new Week05-specific OSM edges:

```text
week06/data/maps_week05/osm_drive_week05_edges.gpkg
```

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_map_eval_week05_osm_full_real
```

Evaluation:

```text
generated samples: 1,000
real baseline samples: 12,061
```

Key results:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road mean distance | 216.87 m | 25.68 m |
| road median distance | 125.49 m | 21.07 m |
| road p95 distance | 666.42 m | 65.62 m |
| road max distance | 744.36 m | 89.22 m |
| road off 20m ratio | 0.6179 | 0.4763 |
| displacement mean | 2350.11 | 853.07 |
| bbox width mean | 1665.61 | 948.81 |
| bbox height mean | 1413.89 | 675.50 |

## Real-to-OSM Offset Check

Random real visualizations still looked visibly shifted from the road network, so Week06 ran a global translation search against the Week05 OSM edges.

Script:

```text
week06/scripts/search_week05_map_offset.py
```

Outputs:

```text
week06/results/week05_map_offset_search
week06/results/week05_map_offset_search_wide
week06/results/week05_map_offset_search_extra_wide
```

Best stable offset from the extra-wide search:

```text
dx = +220 m
dy = -720 m
```

On the sampled real points, this improved nearest-road distance:

| Metric | No offset | Offset dx=220, dy=-720 |
|---|---:|---:|
| road mean distance | 25.51 m | 15.49 m |
| road p95 distance | 70.16 m | 41.15 m |
| off 20 m ratio | 0.4803 | 0.2889 |

This indicates that the Week03 raw-XY to OSM affine is close but not fully aligned for Week05.

## Map Evaluation With Offset-Corrected Week05 OSM

Evaluation output:

```text
week06/results/week05_sig15_mse_diag_step_040000_map_eval_week05_osm_offset_dx220_dy-720
```

Key results:

| Metric | Generated | Real bounce baseline |
|---|---:|---:|
| road mean distance | 220.28 m | 15.24 m |
| road median distance | 64.73 m | 14.67 m |
| road p95 distance | 644.65 m | 40.40 m |
| road off 20m ratio | 0.5785 | 0.2794 |
| displacement mean | 2350.11 | 853.07 |
| bbox width mean | 1665.61 | 948.81 |
| bbox height mean | 1413.89 | 675.50 |

The offset correction makes the real baseline much more plausible, but it does not rescue the generated trajectories. The generator still has a strong accumulated-drift failure mode.

Offset-corrected visualizations:

```text
week06/results/trajectory_visualizations_offset_dx220_dy-720
```

## Empirical Week05 Road Support

The OSM path is not a reliable final map reference for Week05. Week03 used `/home/jliu/gps_with_speed_224_UPDATED.xls`, which contains `lon/lat` and `car*` IDs, to fit the OSM affine. Week05 uses `week05/data/vehicle_positions_TS_New.csv`, which contains only `time,id,x,y,type` with `veh*` IDs and no `lon/lat`. Therefore Week05 raw `x/y` should not be assumed to align exactly with the Week03 OSM transform.

For Week05, the more reliable local support is the empirical road support induced by real Week05 trajectories themselves.

Script:

```text
week06/scripts/evaluate_decoded_empirical_support_alignment.py
```

Output:

```text
week06/results/week05_sig15_mse_diag_step_040000_empirical_support_eval_real1000
```

The support index uses all 12,061 Week05 real bounce-regularized trajectories, totaling 2,701,664 raw `x/y` points. The real sanity-check subset has exactly zero nearest-support distance because it is drawn from the same support:

| Metric | Generated | Real baseline sanity check |
|---|---:|---:|
| support mean distance | 417.55 m | 0.00 m |
| support median distance | 323.43 m | 0.00 m |
| support p95 distance | 1059.47 m | 0.00 m |
| support off 20m ratio | 0.6869 | 0.0000 |

Empirical-support visualizations:

```text
week06/results/trajectory_visualizations_empirical_support_metric_real1000
```

These plots show that some best-case generated samples can lie near the real Week05 support, but median/random generated samples still drift far outside the real trajectory support.

## Empirical-Support Rejection Filter

Script:

```text
week06/scripts/filter_generated_trajectories.py
```

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

```text
accepted: 92 / 1000
acceptance_rate: 0.092
```

The accepted subset has plausible empirical-support distances:

| Metric | Accepted mean | Accepted p95 |
|---|---:|---:|
| support_mean_m | 16.16 m | 22.73 m |
| support_p95_m | 41.56 m | 64.59 m |
| displacement | 961.01 | 1657.63 |

The rejection result is stricter than the OSM-based filter and better matches the Week05 data source.

Interpretation:

The updated road network fixes the previous coverage failure. Real trajectories now have road-distance values in a plausible range, while decoded generated trajectories remain substantially farther from roads and show much larger global drift. The main Week06 issue is no longer map coverage; it is generated trajectory drift and weak global spatial constraint after integrating decoded deltas.

## Next Work

- Add visualization overlays for decoded generated vs real trajectories in raw XY.
- Add a map coverage diagnostic script that checks trajectory extents against OSM bounds before road-distance evaluation.
- Rebuild or extend the OSM road edge file for the full Week05 coordinate extent if needed.
- Consider post-processing generated trajectories:
  - clamp or reject samples leaving the real position range,
  - penalize accumulated drift,
  - condition generation on start/end or use a trajectory-space decoder/refiner.

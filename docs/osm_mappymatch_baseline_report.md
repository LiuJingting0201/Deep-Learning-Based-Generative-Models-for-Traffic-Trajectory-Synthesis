# OSM/Mappymatch Baseline for Delta-Displacement Decoder

## 1. Motivation

The earlier `nearest_gt_polyline` experiment was useful as a controlled sanity check, but it was oracle/leaky: it projected predictions onto the ground-truth trajectory itself. That makes it unsuitable as a realistic road-network post-processing baseline.

The OSM/mappymatch baseline is more realistic because it uses an external OpenStreetMap road network through `mappymatch`. The road graph is independent from the ground-truth labels, so the result tests whether a real road-network matcher can improve road adherence for trajectories reconstructed from predicted delta-displacements.

## 2. Pipeline

1. Load predicted delta-displacement files from the trained delta decoder.
2. Integrate predicted deltas from the true start point to reconstruct predicted XY trajectories.
3. Fit and apply an affine XY-to-lon/lat transform.
4. Write each predicted trajectory as a lon/lat trace CSV.
5. Run mappymatch with an OSM road network around the trace.
6. Extract timestep-aligned matched points from the mappymatch `Match` objects when available.
7. Convert matched points back to XY using the inverse affine transform.
8. Evaluate plain and matched trajectories against ground-truth absolute XY trajectories.

## 3. XY-To-Lonlat Transform Validation

The affine transform was fitted from raw rows containing `x`, `y`, `lon`, and `lat`.

- Number of points: `707616`
- Lon MAE/RMSE: `0.0000002453 / 0.0000003376`
- Lat MAE/RMSE: `0.0000001422 / 0.0000001927`
- Approximate meter error mean/p95: `0.0298 / 0.0669`

Interpretation: the coordinate transform error is negligible for this experiment. It is far smaller than the trajectory reconstruction errors being evaluated.

## 4. 20-Sample OSM/Mappymatch Result

Command:

```bash
python scripts/evaluate_delta_mappymatch_osm.py \
  --split test \
  --max-samples 20 \
  --xy-to-lonlat-json data/xy_to_lonlat_transform.json
```

Result:

- Plain ADE/FDE: `121.9490 / 189.8822`
- Plain RMSE/MAE: `96.6706 / 77.8824`
- Matched ADE/FDE: `120.4163 / 187.3784`
- Matched metrics availability count: `20`
- Match success/failure/rate: `20 / 0 / 1.0000`

## 5. Interpretation

OSM/mappymatch is successfully connected: all 20 traces matched, and timestep-aligned matched points were available for all 20 samples.

The matched ADE/FDE improves only modestly. This is reasonable because map matching improves road adherence, but it does not necessarily solve route-level mistakes, endpoint drift, or timing mismatch. Pointwise ADE/FDE can remain high even when the trace is snapped to plausible roads.

The OSM road geometry may also differ from the original dataset or SUMO-like road geometry. The matched path should be interpreted as a road-graph route; it may look angular or discontinuous compared with timestep-aligned trajectories.

## 6. Generated Artifacts

Output directory:

```text
experiments/delta_mappymatch_osm
```

Important artifacts include:

- `per_sample_metrics.csv`
- `metrics_summary.json`
- `trace_csvs/`
- `matched_outputs/`
- `predictions_matched_xy/`
- `plots/best_10_grid_with_roads.png`
- `plots/median_10_grid_with_roads.png`
- `plots/worst_10_grid_with_roads.png`
- `plots/ADE_distribution_comparison.png`
- `plots/FDE_distribution_comparison.png`
- `plots/plain_vs_matched_ADE_scatter.png`
- `plots/plain_vs_matched_FDE_scatter.png`

## 7. Limitations

This is whole-trace map matching, not strict step-wise projection during delta integration.

Matched points are extracted from mappymatch `Match` objects by projecting each trace coordinate onto the matched road geometry. Post-match ADE/FDE should therefore be interpreted cautiously: they are timestep-aligned snapped points, not a learned or dynamically constrained trajectory.

The real step-wise projection experiment would require using road geometry inside the delta integration loop, replacing:

```python
current = candidate
```

with a road-constrained update such as:

```python
current = project_to_road_network(candidate)
```

## 8. Next Steps

- Run 50-sample and full 317-sample evaluations.
- Add road-adherence metrics more explicitly if the current mean trace-to-road distance is insufficient.
- Audit matched-path geometry direction and continuity.
- Implement strict step-wise road projection using OSM road polylines or a SUMO road network if available.

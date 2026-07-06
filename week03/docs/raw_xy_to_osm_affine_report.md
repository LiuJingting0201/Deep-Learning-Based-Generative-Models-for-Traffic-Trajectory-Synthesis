# Raw x/y to OSM Affine Alignment

## Inputs

- Raw trajectory file: `/home/jliu/gps_with_speed_224_UPDATED.xls`
- OSM edge file: `/home/jliu/Thesis/week03/data/maps/osm_drive_edges.gpkg`
- OSM CRS: `EPSG:32633`
- Points used: 707616
- Vehicles used: 3159

## Affine Transform

Convention:

```text
[x_raw, y_raw, 1] @ A = [x_osm, y_osm]
```

Matrix:

```json
[
  [
    1.0000000000599885,
    4.941468195677034e-10
  ],
  [
    -6.87805368215777e-11,
    0.9999999994370228
  ],
  [
    506599.99999999977,
    4150650.0
  ]
]
```

## Residual Error in Meters

- mean: 0.000000
- median: 0.000000
- p90: 0.000001
- p95: 0.000001
- p99: 0.000001
- max: 0.000001

## Warnings

- No threshold warnings were triggered. This is not, by itself, proof of map validity.

Do not claim map validity unless these residuals are reasonable for the trajectory scale and downstream evaluation purpose.

## Diagnostic Plots

- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car5134_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car1759_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car7776_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car5855_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car6846_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car8420_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car1773_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car5094_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car1800_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car7212_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car9349_alignment.png
- /home/jliu/Thesis/week03/results/map_alignment_diagnostics/vehicle_car2721_alignment.png

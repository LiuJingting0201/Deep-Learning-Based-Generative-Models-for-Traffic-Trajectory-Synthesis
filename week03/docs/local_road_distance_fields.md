# Local Road-Distance Fields

This preprocessing converts local binary road rasters into float32 Euclidean distance fields. Road pixels are map pixels with value greater than zero; every non-road pixel stores approximate distance in meters to the nearest road pixel.

## oracle_bbox

- Source map dir: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox`
- Extent metadata CSV: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_extents.csv`
- Output distance field dir: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields`
- Output metadata CSV: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_distance_fields_metadata.csv`
- Fields generated: 3159
- Extent rows considered: 3159
- Missing map rasters skipped: 0
- Empty-road rasters: 0
- Distance unit: meters
- Debug visualizations: `/home/jliu/Thesis/week03/results/distance_field_debug/oracle_bbox` (20 sample(s), clipped at 50.0 m)
## start_center

- Source map dir: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center`
- Extent metadata CSV: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center_extents.csv`
- Output distance field dir: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center_distance_fields`
- Output metadata CSV: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center_distance_fields_metadata.csv`
- Fields generated: 3159
- Extent rows considered: 3159
- Missing map rasters skipped: 0
- Empty-road rasters: 0
- Distance unit: meters
- Debug visualizations: `/home/jliu/Thesis/week03/results/distance_field_debug/start_center` (20 sample(s), clipped at 50.0 m)

## Intended Use

These fields are intended for a differentiable pointwise road-distance loss by sampling the local distance image at predicted trajectory coordinates with `grid_sample`. This commit only prepares the fields; it does not modify training code or add a loss term.

## Limitations

- Distances are computed in raster space, so they inherit the 224x224 crop resolution and road line width used during map rasterization.
- When `meters_per_pixel_x` and `meters_per_pixel_y` differ, the saved distance uses the approximate isotropic scale `0.5 * (meters_per_pixel_x + meters_per_pixel_y)`.
- Oracle-bbox fields depend on ground-truth trajectory extent and should be treated differently from start-center fields in any deployable setting.
- Empty-road crops are saved as `NaN` fields and should be filtered or handled explicitly before use.

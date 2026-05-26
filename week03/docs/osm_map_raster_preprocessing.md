# OSM Map Raster Preprocessing

## Source

- Raw lon/lat trajectories: `/home/jliu/gps_with_speed_224_UPDATED.xls`
- Road network: projected OpenStreetMap drive edges from `/home/jliu/Thesis/week03/data/maps/osm_drive_edges.gpkg`
- Raw trajectory columns: `time`, `vehicle_id`, `x`, `y`, `type`, `lon`, `lat`, `speed`.
- Paired metadata maps `sample_id` directly to raw `vehicle_id`.

## Coordinate System

- Raw trajectory points are read as EPSG:4326 lon/lat and projected into `EPSG:32633`.
- Raster crop units are meters because the OSM graph is projected before edge export.
- Decoder labels/absolute trajectories are raw local `x/y`, not the projected OSM CRS.

## Crop Modes

- `oracle_bbox`: crop around the full ground-truth trajectory bbox plus `--oracle-buffer-m`.
- `start_center`: fixed `--start-window-m` square crop centered on the first trajectory point.

## Outputs

- Oracle rasters: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox`
- Start-centered rasters: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center`
- Debug overlays: `/home/jliu/data_no_speed_delta_displacement_paired/map_debug_overlays`
- Oracle extent metadata: `/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox_extents.csv`
- Start-centered extent metadata: `/home/jliu/data_no_speed_delta_displacement_paired/maps_start_center_extents.csv`
- Processed: all samples
- Wrote oracle rasters: `3159`
- Wrote start-centered rasters: `3159`
- Wrote debug overlays: `0`

## Alignment Status

Overlay images should be inspected before training. This script does not declare alignment valid automatically; it only produces the visual evidence.

## Limitations

- OSM geometry may not exactly match the simulator or GPS-derived x/y coordinate frame.
- Map-background decoder plots are diagnostic only: map extents are in projected OSM coordinates, while decoded trajectories are raw local `x/y` unless an explicit transform is added.
- `oracle_bbox` uses future ground-truth points and is for controlled alignment/debug use, not causal prediction.
- `start_center` is causal with respect to the first point, but the fixed window can miss long or fast trajectories.

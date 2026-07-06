# Week06 Week05 OSM Map Report

- Source: OpenStreetMap drivable road network via `osmnx`, `network_type='drive'`.
- Real data root: `/home/jliu/Thesis/week05/data/delta_displacement_paired_regularized_bounce224`
- Transform: `/home/jliu/Thesis/week03/data/maps/raw_xy_to_osm_affine.json`
- Projected CRS: `EPSG:32633`
- BBox buffer: `0.004` degrees
- Lon/lat bbox padded: `{'west': 15.070698069232042, 'south': 37.51536808941803, 'east': 15.11609265171789, 'north': 37.54502624269086}`
- Projected nodes: `1848`
- Projected edges: `3580`
- GraphML output: `/home/jliu/Thesis/week06/data/maps_week05/osm_drive_week05.graphml`
- Projected edge GeoPackage: `/home/jliu/Thesis/week06/data/maps_week05/osm_drive_week05_edges.gpkg`
- Coverage plot: `/home/jliu/Thesis/week06/data/maps_week05/week05_osm_coverage.png`

## Coverage

- Week05 trajectory projected bounds: `{'min_x': 506600.8599997078, 'min_y': 4152495.9899989613, 'max_x': 509902.4600000709, 'max_y': 4154895.539999242, 'width': 3301.6000003631343, 'height': 2399.5500002806075}`
- Downloaded edge projected bounds: `{'min_x': 506250.50206030643, 'min_y': 4152053.1574074123, 'max_x': 510256.04880296835, 'max_y': 4155334.4083382217, 'width': 4005.5467426619143, 'height': 3281.250930809416}`
- Trajectory bbox inside edge bbox: `True`
- Coverage gap meters: `{'west_gap_m': 0.0, 'south_gap_m': 0.0, 'east_gap_m': 0.0, 'north_gap_m': 0.0}`

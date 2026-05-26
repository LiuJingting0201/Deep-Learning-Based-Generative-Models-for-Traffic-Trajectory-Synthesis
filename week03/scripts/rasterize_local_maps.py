"""Rasterize local OSM map crops aligned to each paired trajectory sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_ROOT = PROJECT_ROOT / "week03"
DEFAULT_DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_RAW_TRAJECTORY_FILE = Path("/home/jliu/gps_with_speed_224_UPDATED.xls")
DEFAULT_EDGES = WEEK03_ROOT / "data" / "maps" / "osm_drive_edges.gpkg"
DEFAULT_DOC = WEEK03_ROOT / "docs" / "osm_map_raster_preprocessing.md"
IMAGE_SIZE = 224


def main() -> None:
    args = parse_args()
    gpd, LineString, box, Transformer = import_geo_dependencies()

    data_root = args.data_root.expanduser().resolve()
    raw_path = args.raw_trajectory_file.expanduser().resolve()
    edges_path = args.edges_path.expanduser().resolve()
    metadata = pd.read_csv(data_root / "metadata.csv")
    raw = read_table(raw_path)
    validate_columns(metadata, ["sample_id", "vehicle_id"])
    validate_columns(raw, [args.vehicle_col, args.time_col, args.lon_col, args.lat_col])

    edges = gpd.read_file(edges_path)
    if edges.empty:
        raise ValueError(f"No edges found in {edges_path}")
    if edges.crs is None:
        raise ValueError(f"Projected edge file has no CRS: {edges_path}")
    transformer = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)

    oracle_dir = data_root / "maps_oracle_bbox"
    start_dir = data_root / "maps_start_center"
    overlay_dir = data_root / "map_debug_overlays"
    for directory in [oracle_dir, start_dir, overlay_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    raw[args.vehicle_col] = raw[args.vehicle_col].astype(str)
    grouped = raw.groupby(args.vehicle_col, sort=False)
    rows = metadata.copy()
    if args.max_samples is not None:
        rows = rows.head(args.max_samples)

    written = {"oracle_bbox": 0, "start_center": 0, "overlays": 0}
    extent_rows = {"oracle_bbox": [], "start_center": []}
    for index, row in enumerate(rows.itertuples(index=False)):
        sample_id = str(row.sample_id)
        vehicle_id = str(row.vehicle_id)
        if vehicle_id not in grouped.groups:
            raise KeyError(f"vehicle_id {vehicle_id!r} for {sample_id} not found in raw file")

        traj = grouped.get_group(vehicle_id).sort_values(args.time_col, kind="mergesort")
        x, y = transformer.transform(
            traj[args.lon_col].to_numpy(dtype=float),
            traj[args.lat_col].to_numpy(dtype=float),
        )
        points = np.column_stack([x, y])
        if len(points) == 0:
            raise ValueError(f"No trajectory points for {sample_id}/{vehicle_id}")

        crops = {
            "oracle_bbox": oracle_bounds(points, args.oracle_buffer_m),
            "start_center": start_center_bounds(points, args.start_window_m),
        }
        for mode, bounds in crops.items():
            local_edges = clip_edges(edges, bounds)
            image = rasterize_edges(local_edges.geometry, bounds, args.line_width)
            out_dir = oracle_dir if mode == "oracle_bbox" else start_dir
            image.save(out_dir / f"{sample_id}_map.png")
            min_x, min_y, max_x, max_y = bounds
            extent_rows[mode].append(
                {
                    "sample_id": sample_id,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "crs": str(edges.crs),
                    "crop_mode": mode,
                }
            )
            written[mode] += 1

            if index < args.debug_samples:
                overlay_path = overlay_dir / f"{sample_id}_{mode}_overlay.png"
                save_overlay(local_edges, points, bounds, overlay_path)
                written["overlays"] += 1

    write_extent_metadata(data_root, extent_rows)

    write_doc(
        args.doc_path,
        raw_path=raw_path,
        edges_path=edges_path,
        data_root=data_root,
        crs=str(edges.crs),
        max_samples=args.max_samples,
        written=written,
    )
    print(f"Wrote {written['oracle_bbox']} oracle_bbox rasters to {oracle_dir}")
    print(f"Wrote {written['start_center']} start_center rasters to {start_dir}")
    print(f"Wrote {written['overlays']} debug overlays to {overlay_dir}")
    print(f"Wrote extent metadata: {data_root / 'maps_oracle_bbox_extents.csv'}")
    print(f"Wrote extent metadata: {data_root / 'maps_start_center_extents.csv'}")
    print(f"Wrote doc: {args.doc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-trajectory-file", type=Path, default=DEFAULT_RAW_TRAJECTORY_FILE)
    parser.add_argument("--edges-path", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--vehicle-col", default="vehicle_id")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--lon-col", default="lon")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--oracle-buffer-m", type=float, default=30.0)
    parser.add_argument("--start-window-m", type=float, default=350.0)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--debug-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def import_geo_dependencies():
    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import LineString, box
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: geopandas/shapely/pyproj. Install the geospatial "
            "stack in the active environment, e.g. `pip install geopandas shapely pyproj`."
        ) from exc
    return gpd, LineString, box, Transformer


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_excel(path)


def validate_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def oracle_bounds(points: np.ndarray, buffer_m: float) -> tuple[float, float, float, float]:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    if max_x == min_x:
        min_x -= 1.0
        max_x += 1.0
    if max_y == min_y:
        min_y -= 1.0
        max_y += 1.0
    return (min_x - buffer_m, min_y - buffer_m, max_x + buffer_m, max_y + buffer_m)


def start_center_bounds(points: np.ndarray, window_m: float) -> tuple[float, float, float, float]:
    half = window_m / 2.0
    start_x, start_y = points[0]
    return (start_x - half, start_y - half, start_x + half, start_y + half)


def clip_edges(edges, bounds: tuple[float, float, float, float]):
    from shapely.geometry import box

    min_x, min_y, max_x, max_y = bounds
    try:
        return edges.cx[min_x:max_x, min_y:max_y]
    except Exception:
        return edges.loc[edges.intersects(box(min_x, min_y, max_x, max_y))]


def rasterize_edges(geometries, bounds: tuple[float, float, float, float], line_width: int) -> Image.Image:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(image)
    for geometry in geometries:
        for line in iter_lines(geometry):
            pixels = [world_to_pixel(x, y, bounds) for x, y in line.coords]
            if len(pixels) >= 2:
                draw.line(pixels, fill=255, width=line_width, joint="curve")
    return image


def iter_lines(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type == "MultiLineString":
        yield from geometry.geoms
    elif geometry.geom_type == "GeometryCollection":
        for part in geometry.geoms:
            yield from iter_lines(part)


def world_to_pixel(x: float, y: float, bounds: tuple[float, float, float, float]) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    px = int(round((x - min_x) / (max_x - min_x) * (IMAGE_SIZE - 1)))
    py = int(round((max_y - y) / (max_y - min_y) * (IMAGE_SIZE - 1)))
    return px, py


def save_overlay(edges, points: np.ndarray, bounds: tuple[float, float, float, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    min_x, min_y, max_x, max_y = bounds
    fig, axis = plt.subplots(figsize=(6, 6))
    if not edges.empty:
        edges.plot(ax=axis, color="#444444", linewidth=1.0, label="OSM roads")
    axis.plot(points[:, 0], points[:, 1], color="#d95f02", linewidth=1.8, label="GT trajectory")
    axis.scatter(points[:, 0], points[:, 1], color="#1b9e77", s=8, zorder=3)
    axis.add_patch(
        Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor="#7570b3", linewidth=1.5)
    )
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
    axis.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_extent_metadata(data_root: Path, extent_rows: dict[str, list[dict[str, object]]]) -> None:
    outputs = {
        "oracle_bbox": data_root / "maps_oracle_bbox_extents.csv",
        "start_center": data_root / "maps_start_center_extents.csv",
    }
    for mode, path in outputs.items():
        pd.DataFrame(
            extent_rows[mode],
            columns=["sample_id", "min_x", "max_x", "min_y", "max_y", "crs", "crop_mode"],
        ).to_csv(path, index=False)


def write_doc(
    path: Path,
    *,
    raw_path: Path,
    edges_path: Path,
    data_root: Path,
    crs: str,
    max_samples: int | None,
    written: dict[str, int],
) -> None:
    limit = "all samples" if max_samples is None else f"first {max_samples} samples"
    text = "\n".join(
        [
            "# OSM Map Raster Preprocessing",
            "",
            "## Source",
            "",
            f"- Raw lon/lat trajectories: `{raw_path}`",
            f"- Road network: projected OpenStreetMap drive edges from `{edges_path}`",
            "- Raw trajectory columns: `time`, `vehicle_id`, `x`, `y`, `type`, `lon`, `lat`, `speed`.",
            "- Paired metadata maps `sample_id` directly to raw `vehicle_id`.",
            "",
            "## Coordinate System",
            "",
            f"- Raw trajectory points are read as EPSG:4326 lon/lat and projected into `{crs}`.",
            "- Raster crop units are meters because the OSM graph is projected before edge export.",
            "- Decoder labels/absolute trajectories are raw local `x/y`, not the projected OSM CRS.",
            "",
            "## Crop Modes",
            "",
            "- `oracle_bbox`: crop around the full ground-truth trajectory bbox plus `--oracle-buffer-m`.",
            "- `start_center`: fixed `--start-window-m` square crop centered on the first trajectory point.",
            "",
            "## Outputs",
            "",
            f"- Oracle rasters: `{data_root / 'maps_oracle_bbox'}`",
            f"- Start-centered rasters: `{data_root / 'maps_start_center'}`",
            f"- Debug overlays: `{data_root / 'map_debug_overlays'}`",
            f"- Oracle extent metadata: `{data_root / 'maps_oracle_bbox_extents.csv'}`",
            f"- Start-centered extent metadata: `{data_root / 'maps_start_center_extents.csv'}`",
            f"- Processed: {limit}",
            f"- Wrote oracle rasters: `{written['oracle_bbox']}`",
            f"- Wrote start-centered rasters: `{written['start_center']}`",
            f"- Wrote debug overlays: `{written['overlays']}`",
            "",
            "## Alignment Status",
            "",
            "Overlay images should be inspected before training. This script does not declare alignment valid automatically; it only produces the visual evidence.",
            "",
            "## Limitations",
            "",
            "- OSM geometry may not exactly match the simulator or GPS-derived x/y coordinate frame.",
            "- Map-background decoder plots are diagnostic only: map extents are in projected OSM coordinates, while decoded trajectories are raw local `x/y` unless an explicit transform is added.",
            "- `oracle_bbox` uses future ground-truth points and is for controlled alignment/debug use, not causal prediction.",
            "- `start_center` is causal with respect to the first point, but the fixed window can miss long or fast trajectories.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    main()

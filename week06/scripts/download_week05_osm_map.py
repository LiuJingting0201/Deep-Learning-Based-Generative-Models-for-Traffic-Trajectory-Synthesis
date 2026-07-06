"""Download an OSM drive network covering Week05 real trajectory coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_MAP_DIR = PROJECT_ROOT / "week06" / "data" / "maps_week05"
DEFAULT_REPORT = PROJECT_ROOT / "week06" / "WEEK06_WEEK05_OSM_MAP_REPORT.md"


def main() -> None:
    args = parse_args()
    ox = import_osmnx()
    map_dir = args.map_dir.resolve()
    map_dir.mkdir(parents=True, exist_ok=True)

    transform = load_json(args.transform_path.resolve())
    affine_matrix = transform["affine_matrix"]
    projected_crs = transform.get("osm_crs", "EPSG:32633")

    raw_bounds = compute_real_raw_bounds(args.real_data_root.resolve())
    projected_bounds = raw_bounds_to_projected_bounds(raw_bounds, affine_matrix)
    lonlat_bounds = projected_bounds_to_lonlat_bounds(projected_bounds, projected_crs)
    padded_lonlat = pad_lonlat_bounds(lonlat_bounds, args.buffer_degrees)

    graph = download_graph(
        ox,
        west=padded_lonlat["west"],
        south=padded_lonlat["south"],
        east=padded_lonlat["east"],
        north=padded_lonlat["north"],
    )
    graphml_path = map_dir / "osm_drive_week05.graphml"
    gpkg_path = map_dir / "osm_drive_week05_edges.gpkg"
    summary_path = map_dir / "week05_osm_map_summary.json"
    plot_path = map_dir / "week05_osm_coverage.png"

    ox.save_graphml(graph, graphml_path)
    projected_graph = ox.project_graph(graph, to_crs=projected_crs)
    nodes, edges = ox.graph_to_gdfs(projected_graph, nodes=True, edges=True)
    if edges.empty:
        raise ValueError("Downloaded OSM graph has no projected edges.")
    edges.to_file(gpkg_path, layer="edges", driver="GPKG")

    edge_bounds = edges.total_bounds.astype(float)
    coverage = {
        "trajectory_projected_bounds_inside_edge_bbox": bbox_contains(edge_bounds, projected_bounds),
        "trajectory_edge_bbox_intersects": bbox_intersects(edge_bounds, projected_bounds),
        "coverage_gap_m": coverage_gap(projected_bounds, edge_bounds),
    }
    plot_coverage(plot_path, edges, projected_bounds, edge_bounds)

    summary = {
        "real_data_root": str(args.real_data_root.resolve()),
        "transform_path": str(args.transform_path.resolve()),
        "projected_crs": projected_crs,
        "network_type": "drive",
        "buffer_degrees": args.buffer_degrees,
        "raw_bounds_xy": bounds_payload(raw_bounds),
        "trajectory_projected_bounds": bounds_payload(projected_bounds),
        "lonlat_bounds_unpadded": lonlat_bounds,
        "lonlat_bounds_padded": padded_lonlat,
        "downloaded_edge_bounds_projected": bounds_payload(edge_bounds),
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "coverage": coverage,
        "outputs": {
            "graphml": str(graphml_path),
            "edges_gpkg": str(gpkg_path),
            "summary_json": str(summary_path),
            "coverage_plot": str(plot_path),
        },
    }
    write_json(summary_path, summary)
    args.report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_path.resolve().write_text(build_report(summary))
    print(json.dumps(summary, indent=2))


def compute_real_raw_bounds(real_data_root: Path) -> np.ndarray:
    paths = sorted((real_data_root / "labels_absolute").glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No real absolute trajectories under {real_data_root}")
    min_xy = np.array([np.inf, np.inf], dtype=float)
    max_xy = np.array([-np.inf, -np.inf], dtype=float)
    for path in paths:
        xy = np.load(path).astype(float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected {path} to have shape N,2, got {xy.shape}")
        min_xy = np.minimum(min_xy, xy.min(axis=0))
        max_xy = np.maximum(max_xy, xy.max(axis=0))
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=float)


def raw_bounds_to_projected_bounds(raw_bounds: np.ndarray, affine_matrix: list[list[float]]) -> np.ndarray:
    corners = np.array(
        [
            [raw_bounds[0], raw_bounds[1]],
            [raw_bounds[0], raw_bounds[3]],
            [raw_bounds[2], raw_bounds[1]],
            [raw_bounds[2], raw_bounds[3]],
        ],
        dtype=float,
    )
    projected = apply_affine(corners, affine_matrix)
    return np.array([projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()])


def projected_bounds_to_lonlat_bounds(projected_bounds: np.ndarray, projected_crs: str) -> dict[str, float]:
    transformer = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    xs = [projected_bounds[0], projected_bounds[0], projected_bounds[2], projected_bounds[2]]
    ys = [projected_bounds[1], projected_bounds[3], projected_bounds[1], projected_bounds[3]]
    lon, lat = transformer.transform(xs, ys)
    return {
        "west": float(np.min(lon)),
        "south": float(np.min(lat)),
        "east": float(np.max(lon)),
        "north": float(np.max(lat)),
    }


def pad_lonlat_bounds(bounds: dict[str, float], buffer_degrees: float) -> dict[str, float]:
    return {
        "west": bounds["west"] - buffer_degrees,
        "south": bounds["south"] - buffer_degrees,
        "east": bounds["east"] + buffer_degrees,
        "north": bounds["north"] + buffer_degrees,
    }


def download_graph(ox: Any, *, west: float, south: float, east: float, north: float) -> Any:
    try:
        return ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="drive",
            simplify=True,
        )
    except TypeError:
        return ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type="drive",
            simplify=True,
        )


def plot_coverage(path: Path, edges: gpd.GeoDataFrame, trajectory_bounds: np.ndarray, edge_bounds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    edges.plot(ax=ax, linewidth=0.45, color="#111827", alpha=0.8)
    draw_bbox(ax, edge_bounds, "Downloaded OSM edge bbox", "#2563eb")
    draw_bbox(ax, trajectory_bounds, "Week05 trajectory bbox", "#dc2626")
    ax.set_title("Downloaded Week05 OSM road coverage")
    ax.set_xlabel("OSM x")
    ax.set_ylabel("OSM y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_bbox(ax: Any, bbox: np.ndarray, label: str, color: str) -> None:
    xs = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    ys = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    ax.plot(xs, ys, color=color, linewidth=2.0, label=label)


def apply_affine(xy: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    affine = np.asarray(matrix, dtype=float)
    design = np.column_stack([xy[:, :2], np.ones(len(xy), dtype=float)])
    return design @ affine


def bbox_contains(container: np.ndarray, item: np.ndarray) -> bool:
    return bool(container[0] <= item[0] and container[1] <= item[1] and container[2] >= item[2] and container[3] >= item[3])


def bbox_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1])


def coverage_gap(trajectory_bounds: np.ndarray, edge_bounds: np.ndarray) -> dict[str, float]:
    return {
        "west_gap_m": float(max(0.0, edge_bounds[0] - trajectory_bounds[0])),
        "south_gap_m": float(max(0.0, edge_bounds[1] - trajectory_bounds[1])),
        "east_gap_m": float(max(0.0, trajectory_bounds[2] - edge_bounds[2])),
        "north_gap_m": float(max(0.0, trajectory_bounds[3] - edge_bounds[3])),
    }


def bounds_payload(bounds: np.ndarray) -> dict[str, float]:
    return {
        "min_x": float(bounds[0]),
        "min_y": float(bounds[1]),
        "max_x": float(bounds[2]),
        "max_y": float(bounds[3]),
        "width": float(bounds[2] - bounds[0]),
        "height": float(bounds[3] - bounds[1]),
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Week06 Week05 OSM Map Report",
        "",
        "- Source: OpenStreetMap drivable road network via `osmnx`, `network_type='drive'`.",
        f"- Real data root: `{summary['real_data_root']}`",
        f"- Transform: `{summary['transform_path']}`",
        f"- Projected CRS: `{summary['projected_crs']}`",
        f"- BBox buffer: `{summary['buffer_degrees']}` degrees",
        f"- Lon/lat bbox padded: `{summary['lonlat_bounds_padded']}`",
        f"- Projected nodes: `{summary['num_nodes']}`",
        f"- Projected edges: `{summary['num_edges']}`",
        f"- GraphML output: `{summary['outputs']['graphml']}`",
        f"- Projected edge GeoPackage: `{summary['outputs']['edges_gpkg']}`",
        f"- Coverage plot: `{summary['outputs']['coverage_plot']}`",
        "",
        "## Coverage",
        "",
        f"- Week05 trajectory projected bounds: `{summary['trajectory_projected_bounds']}`",
        f"- Downloaded edge projected bounds: `{summary['downloaded_edge_bounds_projected']}`",
        f"- Trajectory bbox inside edge bbox: `{summary['coverage']['trajectory_projected_bounds_inside_edge_bbox']}`",
        f"- Coverage gap meters: `{summary['coverage']['coverage_gap_m']}`",
        "",
    ]
    return "\n".join(lines)


def import_osmnx() -> Any:
    try:
        import osmnx as ox
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: osmnx. Use the thesis geospatial environment.") from exc
    return ox


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--buffer-degrees", type=float, default=0.004)
    return parser.parse_args()


if __name__ == "__main__":
    main()

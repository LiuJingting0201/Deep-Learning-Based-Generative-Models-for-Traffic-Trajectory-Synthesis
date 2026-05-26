"""Download one OSM drivable road graph for the trajectory area."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_ROOT = PROJECT_ROOT / "week03"
DEFAULT_RAW_TRAJECTORY_FILE = Path("/home/jliu/gps_with_speed_224_UPDATED.xls")
DEFAULT_MAP_DIR = WEEK03_ROOT / "data" / "maps"
DEFAULT_REPORT = WEEK03_ROOT / "docs" / "osm_map_download_report.md"


def main() -> None:
    args = parse_args()
    ox = import_osmnx()

    raw_path = args.raw_trajectory_file.expanduser().resolve()
    frame = read_table(raw_path)
    validate_columns(frame, [args.lon_col, args.lat_col])

    coords = frame[[args.lon_col, args.lat_col]].dropna()
    west = float(coords[args.lon_col].min() - args.buffer_degrees)
    east = float(coords[args.lon_col].max() + args.buffer_degrees)
    south = float(coords[args.lat_col].min() - args.buffer_degrees)
    north = float(coords[args.lat_col].max() + args.buffer_degrees)

    args.map_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = args.map_dir / "osm_drive.graphml"
    gpkg_path = args.map_dir / "osm_drive_edges.gpkg"

    graph = download_graph(ox, north=north, south=south, east=east, west=west)
    ox.save_graphml(graph, graphml_path)

    projected_graph = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(projected_graph, nodes=True, edges=True)
    if edges.empty:
        raise ValueError("Downloaded OSM graph has no projected edges.")
    edges.to_file(gpkg_path, layer="edges", driver="GPKG")

    crs = str(edges.crs)
    report = build_report(
        raw_path=raw_path,
        graphml_path=graphml_path,
        gpkg_path=gpkg_path,
        west=west,
        south=south,
        east=east,
        north=north,
        buffer_degrees=args.buffer_degrees,
        crs=crs,
        num_nodes=len(nodes),
        num_edges=len(edges),
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report)

    print(f"Wrote graph: {graphml_path}")
    print(f"Wrote projected edges: {gpkg_path}")
    print(f"Wrote report: {args.report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-trajectory-file", type=Path, default=DEFAULT_RAW_TRAJECTORY_FILE)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lon-col", default="lon")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument(
        "--buffer-degrees",
        type=float,
        default=0.002,
        help="Lon/lat bbox padding in degrees before requesting OSM data.",
    )
    return parser.parse_args()


def import_osmnx():
    try:
        import osmnx as ox
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: osmnx. Install the geospatial stack in the "
            "active environment, e.g. `pip install osmnx geopandas shapely pyproj`."
        ) from exc
    return ox


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
        raise ValueError(f"Missing required raw trajectory columns: {missing}")


def download_graph(ox, *, north: float, south: float, east: float, west: float):
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


def build_report(
    *,
    raw_path: Path,
    graphml_path: Path,
    gpkg_path: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    buffer_degrees: float,
    crs: str,
    num_nodes: int,
    num_edges: int,
) -> str:
    return "\n".join(
        [
            "# OSM Map Download Report",
            "",
            f"- Raw trajectory file: `{raw_path}`",
            "- Source: OpenStreetMap drivable road network via `osmnx`, `network_type='drive'`.",
            f"- BBox buffer: `{buffer_degrees}` degrees",
            f"- West/east lon: `{west:.10f}`, `{east:.10f}`",
            f"- South/north lat: `{south:.10f}`, `{north:.10f}`",
            f"- Projected CRS: `{crs}`",
            f"- Projected nodes: `{num_nodes}`",
            f"- Projected edges: `{num_edges}`",
            f"- GraphML output: `{graphml_path}`",
            f"- Projected edge GeoPackage: `{gpkg_path}`",
            "",
        ]
    )


if __name__ == "__main__":
    main()

"""Evaluate decoded generated trajectories against trajectory and road-distance baselines."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECODED_DIR = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OSM_EDGES = PROJECT_ROOT / "week03" / "data" / "maps" / "osm_drive_edges.gpkg"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_map_eval"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = load_transform(args.transform_path.resolve())
    road_index = load_road_index(args.osm_edges_path.resolve()) if not args.skip_road_distance else None

    generated_paths = sorted((args.decoded_dir.resolve() / "trajectories_raw_xy").glob("*.npy"))
    if args.max_generated is not None:
        generated_paths = generated_paths[: args.max_generated]
    if not generated_paths:
        raise FileNotFoundError(f"No decoded generated trajectories under {args.decoded_dir}")

    real_paths = sorted((args.real_data_root.resolve() / "labels_absolute").glob("*.npy"))
    if args.max_real is not None:
        real_paths = real_paths[: args.max_real]
    if not real_paths:
        raise FileNotFoundError(f"No real trajectories under {args.real_data_root}")

    generated_rows = evaluate_paths(
        paths=generated_paths,
        group="generated",
        transform=transform,
        road_index=road_index,
        large_jump_threshold=args.large_jump_threshold,
        offset_xy=np.asarray([args.offset_x, args.offset_y], dtype=float),
    )
    real_rows = evaluate_paths(
        paths=real_paths,
        group="real_bounce_regularized",
        transform=transform,
        road_index=road_index,
        large_jump_threshold=args.large_jump_threshold,
        offset_xy=np.asarray([args.offset_x, args.offset_y], dtype=float),
    )

    write_csv(output_dir / "generated_per_sample_metrics.csv", generated_rows)
    write_csv(output_dir / "real_per_sample_metrics.csv", real_rows)
    summary = {
        "decoded_dir": str(args.decoded_dir.resolve()),
        "real_data_root": str(args.real_data_root.resolve()),
        "transform_path": str(args.transform_path.resolve()),
        "osm_edges_path": str(args.osm_edges_path.resolve()),
        "road_distance_enabled": road_index is not None,
        "road_index_mode": None if road_index is None else road_index.mode,
        "offset_x_m": args.offset_x,
        "offset_y_m": args.offset_y,
        "num_generated": len(generated_rows),
        "num_real": len(real_rows),
        "large_jump_threshold": args.large_jump_threshold,
        "generated": summarize_rows(generated_rows),
        "real_bounce_regularized": summarize_rows(real_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))


def evaluate_paths(
    paths: list[Path],
    group: str,
    transform: dict[str, Any],
    road_index: "RoadDistanceIndex | None",
    large_jump_threshold: float,
    offset_xy: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        trajectory = load_xy(path)
        osm_xy = apply_affine(trajectory, transform["affine_matrix"]) + offset_xy
        row = trajectory_metric_row(path.stem, group, trajectory, large_jump_threshold)
        if road_index is not None:
            distances = road_index.distances_for_xy(osm_xy)
            row.update(distance_metric_row(distances, "road"))
        rows.append(row)
    return rows


def trajectory_metric_row(sample_id: str, group: str, trajectory: np.ndarray, large_jump_threshold: float) -> dict[str, Any]:
    steps = np.diff(trajectory, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    return {
        "sample_id": sample_id,
        "group": group,
        "num_points": int(len(trajectory)),
        "start_x": float(trajectory[0, 0]),
        "start_y": float(trajectory[0, 1]),
        "end_x": float(trajectory[-1, 0]),
        "end_y": float(trajectory[-1, 1]),
        "mean_step_length": float(np.mean(step_lengths)),
        "median_step_length": float(np.median(step_lengths)),
        "p95_step_length": float(np.percentile(step_lengths, 95)),
        "max_step_length": float(np.max(step_lengths)),
        "large_jump_ratio": float(np.mean(step_lengths > large_jump_threshold)),
        "path_length": float(np.sum(step_lengths)),
        "displacement": float(np.linalg.norm(trajectory[-1] - trajectory[0])),
        "bbox_width": float(np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])),
        "bbox_height": float(np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])),
    }


def distance_metric_row(distances: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_m": float(np.mean(distances)),
        f"{prefix}_median_m": float(np.median(distances)),
        f"{prefix}_p95_m": float(np.percentile(distances, 95)),
        f"{prefix}_max_m": float(np.max(distances)),
        f"{prefix}_off_5m_ratio": float(np.mean(distances > 5.0)),
        f"{prefix}_off_10m_ratio": float(np.mean(distances > 10.0)),
        f"{prefix}_off_20m_ratio": float(np.mean(distances > 20.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"sample_id", "group"} and isinstance(value, (int, float, np.integer, np.floating))
    ]
    return {key: summarize(np.asarray([row[key] for row in rows], dtype=float)) for key in numeric_keys}


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def apply_affine(xy: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    affine = np.asarray(matrix, dtype=float)
    design = np.column_stack([xy[:, :2].astype(float), np.ones(len(xy), dtype=float)])
    return design @ affine


def load_xy(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected {path} to have shape N,2, got {array.shape}")
    return array


def load_transform(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "affine_matrix" not in payload:
        raise ValueError(f"{path} does not contain affine_matrix")
    return payload


class RoadDistanceIndex:
    def __init__(self, geometries: list[Any], strtree_class: Any, shapely_module: Any) -> None:
        self.geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
        if not self.geometries:
            raise ValueError("No valid OSM road geometries were loaded.")
        self.shapely = shapely_module
        self.tree = None
        self.mode = "brute_force"
        if strtree_class is not None:
            try:
                self.tree = strtree_class(self.geometries)
                self.mode = "shapely_strtree"
            except Exception as exc:
                warnings.warn(f"Could not build Shapely STRtree; falling back to brute force: {exc}")

    def distances_for_xy(self, xy: np.ndarray) -> np.ndarray:
        if self.tree is not None and hasattr(self.tree, "query_nearest") and hasattr(self.shapely, "points"):
            try:
                points = self.shapely.points(xy[:, 0].astype(float), xy[:, 1].astype(float))
                indices, distances = self.tree.query_nearest(points, return_distance=True, all_matches=False)
                ordered = np.empty(len(points), dtype=float)
                ordered[indices[0]] = distances
                return ordered
            except Exception as exc:
                warnings.warn(f"Vectorized STRtree query failed; using per-point nearest query: {exc}")
        from shapely.geometry import Point

        distances = []
        for x_coord, y_coord in xy:
            point = Point(float(x_coord), float(y_coord))
            road = self.nearest_geometry(point)
            distances.append(float(point.distance(road)))
        return np.asarray(distances, dtype=float)

    def nearest_geometry(self, point: Any) -> Any:
        if self.tree is None:
            return min(self.geometries, key=point.distance)
        nearest = self.tree.nearest(point)
        if isinstance(nearest, (int, np.integer)):
            return self.geometries[int(nearest)]
        return nearest


def load_road_index(osm_edges_path: Path) -> RoadDistanceIndex:
    missing = []
    try:
        import geopandas as gpd
    except ImportError:
        missing.append("geopandas")
        gpd = None
    try:
        import shapely as shapely_module
        from shapely.strtree import STRtree
    except ImportError:
        missing.append("shapely")
        shapely_module = None
        STRtree = None
    if missing:
        raise SystemExit(
            "Missing geospatial package(s): "
            + ", ".join(sorted(set(missing)))
            + ". Re-run with --skip-road-distance or use the geospatial environment."
        )
    edges = gpd.read_file(osm_edges_path)
    if edges.crs is None:
        raise ValueError(f"OSM edge file has no CRS: {osm_edges_path}")
    return RoadDistanceIndex(edges.geometry.dropna().tolist(), STRtree, shapely_module)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week06 Map Evaluation Summary",
        "",
        f"- Generated samples evaluated: {summary['num_generated']}",
        f"- Real baseline samples evaluated: {summary['num_real']}",
        f"- Road distance enabled: {summary['road_distance_enabled']}",
        f"- Road index mode: {summary['road_index_mode']}",
        f"- Applied OSM offset: dx={summary.get('offset_x_m', 0.0):.6g} m, dy={summary.get('offset_y_m', 0.0):.6g} m",
        "",
    ]
    for group_key, title in (("generated", "Generated"), ("real_bounce_regularized", "Real Bounce-Regularized")):
        lines.extend([f"## {title}", ""])
        group = summary[group_key]
        for key in sorted(group):
            values = group[key]
            lines.append(
                f"- `{key}`: mean={values['mean']:.6g}, median={values['median']:.6g}, "
                f"p95={values['p95']:.6g}, max={values['max']:.6g}"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--large-jump-threshold", type=float, default=50.0)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--skip-road-distance", action="store_true")
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

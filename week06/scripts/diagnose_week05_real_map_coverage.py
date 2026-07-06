"""Diagnose whether Week05 real trajectories are covered by the Week03 OSM map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OSM_EDGES = PROJECT_ROOT / "week03" / "data" / "maps" / "osm_drive_edges.gpkg"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_real_map_coverage"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = load_json(args.transform_path.resolve())
    edges = gpd.read_file(args.osm_edges_path.resolve())
    if edges.crs is None:
        raise ValueError(f"OSM edge file has no CRS: {args.osm_edges_path}")

    rows = load_metadata(args.real_data_root.resolve())
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    trajectories = load_trajectories(args.real_data_root.resolve(), rows)
    all_raw = np.vstack(trajectories)
    all_osm = apply_affine(all_raw, transform["affine_matrix"])
    osm_bounds = edges.total_bounds.astype(float)
    raw_bounds = bounds(all_raw)
    osm_trajectory_bounds = bounds(all_osm)

    point_mask = points_in_bounds(all_osm, osm_bounds)
    sample_rows = per_sample_rows(rows, trajectories, transform["affine_matrix"], osm_bounds)
    write_csv(output_dir / "per_sample_coverage.csv", sample_rows)

    selected_indices = select_indices(len(trajectories), args.plot_samples, args.seed)
    plot_raw_extent(output_dir / "real_trajectory_extent_raw_xy.png", trajectories, selected_indices, raw_bounds)
    plot_osm_coverage(
        output_dir / "real_vs_osm_coverage_osm.png",
        edges,
        all_osm,
        osm_bounds,
        osm_trajectory_bounds,
        max_points=args.max_scatter_points,
        seed=args.seed,
    )
    plot_sample_overlay(
        output_dir / "real_sample_overlay_osm.png",
        edges,
        [apply_affine(trajectories[index], transform["affine_matrix"]) for index in selected_indices],
        osm_bounds,
        osm_trajectory_bounds,
    )

    summary = {
        "real_data_root": str(args.real_data_root.resolve()),
        "transform_path": str(args.transform_path.resolve()),
        "osm_edges_path": str(args.osm_edges_path.resolve()),
        "osm_crs": str(edges.crs),
        "num_samples": len(trajectories),
        "num_points": int(len(all_raw)),
        "raw_bounds_xy": bounds_payload(raw_bounds),
        "trajectory_bounds_osm": bounds_payload(osm_trajectory_bounds),
        "osm_edge_bounds": bounds_payload(osm_bounds),
        "point_inside_osm_bbox_ratio": float(np.mean(point_mask)),
        "sample_inside_osm_bbox_ratio": float(np.mean([row["all_points_inside_osm_bbox"] for row in sample_rows])),
        "sample_intersects_osm_bbox_ratio": float(np.mean([row["trajectory_bbox_intersects_osm_bbox"] for row in sample_rows])),
        "coverage_gap_m": coverage_gap(osm_trajectory_bounds, osm_bounds),
        "plots": {
            "raw_extent": str(output_dir / "real_trajectory_extent_raw_xy.png"),
            "osm_coverage": str(output_dir / "real_vs_osm_coverage_osm.png"),
            "sample_overlay": str(output_dir / "real_sample_overlay_osm.png"),
        },
    }
    write_json(output_dir / "coverage_summary.json", summary)
    write_markdown(output_dir / "coverage_summary.md", summary)
    print(json.dumps(summary, indent=2))


def load_metadata(data_root: Path) -> list[dict[str, str]]:
    path = data_root / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_trajectories(data_root: Path, rows: list[dict[str, str]]) -> list[np.ndarray]:
    trajectories = []
    for row in rows:
        path = data_root / str(row["label_absolute_path"])
        array = np.load(path).astype(np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Expected {path} to have shape N,2, got {array.shape}")
        trajectories.append(array)
    return trajectories


def per_sample_rows(
    rows: list[dict[str, str]],
    trajectories: list[np.ndarray],
    affine_matrix: list[list[float]],
    osm_bounds: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for row, trajectory in zip(rows, trajectories):
        osm_xy = apply_affine(trajectory, affine_matrix)
        sample_bounds = bounds(osm_xy)
        inside = points_in_bounds(osm_xy, osm_bounds)
        output.append(
            {
                "sample_id": row["sample_id"],
                "vehicle_id": row.get("vehicle_id", ""),
                "regularization_method": row.get("regularization_method", ""),
                "original_length": int(row.get("original_length", 0) or 0),
                "inside_osm_bbox_ratio": float(np.mean(inside)),
                "all_points_inside_osm_bbox": bool(np.all(inside)),
                "any_points_inside_osm_bbox": bool(np.any(inside)),
                "trajectory_bbox_intersects_osm_bbox": bbox_intersects(sample_bounds, osm_bounds),
                "osm_min_x": float(sample_bounds[0]),
                "osm_min_y": float(sample_bounds[1]),
                "osm_max_x": float(sample_bounds[2]),
                "osm_max_y": float(sample_bounds[3]),
            }
        )
    return output


def apply_affine(xy: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    affine = np.asarray(matrix, dtype=float)
    design = np.column_stack([xy[:, :2].astype(float), np.ones(len(xy), dtype=float)])
    return design @ affine


def bounds(xy: np.ndarray) -> np.ndarray:
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()], dtype=float)


def points_in_bounds(xy: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    return (xy[:, 0] >= bbox[0]) & (xy[:, 0] <= bbox[2]) & (xy[:, 1] >= bbox[1]) & (xy[:, 1] <= bbox[3])


def bbox_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1])


def coverage_gap(trajectory_bounds: np.ndarray, osm_bounds: np.ndarray) -> dict[str, float]:
    return {
        "west_gap_m": float(max(0.0, osm_bounds[0] - trajectory_bounds[0])),
        "south_gap_m": float(max(0.0, osm_bounds[1] - trajectory_bounds[1])),
        "east_gap_m": float(max(0.0, trajectory_bounds[2] - osm_bounds[2])),
        "north_gap_m": float(max(0.0, trajectory_bounds[3] - osm_bounds[3])),
    }


def bounds_payload(bbox: np.ndarray) -> dict[str, float]:
    return {
        "min_x": float(bbox[0]),
        "min_y": float(bbox[1]),
        "max_x": float(bbox[2]),
        "max_y": float(bbox[3]),
        "width": float(bbox[2] - bbox[0]),
        "height": float(bbox[3] - bbox[1]),
    }


def select_indices(num_items: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(count, num_items)
    return np.sort(rng.choice(num_items, size=count, replace=False))


def plot_raw_extent(path: Path, trajectories: list[np.ndarray], indices: np.ndarray, raw_bounds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for index in indices:
        trajectory = trajectories[int(index)]
        ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.9, alpha=0.35)
    draw_bbox(ax, raw_bounds, "Week05 real raw XY extent", color="black")
    ax.set_title("Week05 real trajectories in raw XY")
    ax.set_xlabel("raw x")
    ax.set_ylabel("raw y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_osm_coverage(
    path: Path,
    edges: gpd.GeoDataFrame,
    all_osm: np.ndarray,
    osm_bounds: np.ndarray,
    trajectory_bounds: np.ndarray,
    max_points: int,
    seed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    edges.plot(ax=ax, linewidth=0.45, color="#1f2937", alpha=0.8)
    points = downsample_points(all_osm, max_points, seed)
    ax.scatter(points[:, 0], points[:, 1], s=1.0, alpha=0.08, color="#dc2626", label="Week05 real points")
    draw_bbox(ax, osm_bounds, "OSM edge bbox", color="#2563eb")
    draw_bbox(ax, trajectory_bounds, "Week05 real bbox", color="#dc2626")
    ax.set_title("Week05 real trajectories vs Week03 OSM edge coverage")
    ax.set_xlabel("OSM x")
    ax.set_ylabel("OSM y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sample_overlay(
    path: Path,
    edges: gpd.GeoDataFrame,
    trajectories_osm: list[np.ndarray],
    osm_bounds: np.ndarray,
    trajectory_bounds: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    edges.plot(ax=ax, linewidth=0.45, color="#111827", alpha=0.75)
    for trajectory in trajectories_osm:
        ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.8, alpha=0.45)
    draw_bbox(ax, osm_bounds, "OSM edge bbox", color="#2563eb")
    draw_bbox(ax, trajectory_bounds, "Week05 real bbox", color="#dc2626")
    ax.set_title("Sample Week05 real trajectories over Week03 OSM edges")
    ax.set_xlabel("OSM x")
    ax.set_ylabel("OSM y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def draw_bbox(ax: Any, bbox: np.ndarray, label: str, color: str) -> None:
    xs = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    ys = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    ax.plot(xs, ys, color=color, linewidth=2.0, label=label)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week05 Real Map Coverage Diagnostic",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Points: {summary['num_points']}",
        f"- OSM CRS: `{summary['osm_crs']}`",
        f"- Point inside OSM bbox ratio: {summary['point_inside_osm_bbox_ratio']:.6f}",
        f"- All-points-inside sample ratio: {summary['sample_inside_osm_bbox_ratio']:.6f}",
        f"- Intersecting sample bbox ratio: {summary['sample_intersects_osm_bbox_ratio']:.6f}",
        "",
        "## Bounds",
        "",
        f"- Raw XY: `{summary['raw_bounds_xy']}`",
        f"- Week05 trajectories in OSM CRS: `{summary['trajectory_bounds_osm']}`",
        f"- Week03 OSM edges: `{summary['osm_edge_bounds']}`",
        f"- Coverage gap meters: `{summary['coverage_gap_m']}`",
        "",
        "## Plots",
        "",
    ]
    for name, plot_path in summary["plots"].items():
        lines.append(f"- {name}: `{plot_path}`")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=120)
    parser.add_argument("--max-scatter-points", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Search for a global XY offset between Week05 real trajectories and OSM roads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_decoded_map_alignment import (
    apply_affine,
    distance_metric_row,
    load_road_index,
    load_transform,
    load_xy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OSM_EDGES = PROJECT_ROOT / "week06" / "data" / "maps_week05" / "osm_drive_week05_edges.gpkg"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_map_offset_search"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = load_transform(args.transform_path.resolve())
    road_index = load_road_index(args.osm_edges_path.resolve())
    points = load_real_points(
        real_data_root=args.real_data_root.resolve(),
        transform=transform,
        max_samples=args.max_samples,
        max_points=args.max_points,
        point_stride=args.point_stride,
        seed=args.seed,
    )

    zero_metrics = evaluate_offset(points, road_index, 0.0, 0.0)
    coarse_rows = evaluate_grid(
        points=points,
        road_index=road_index,
        center_x=0.0,
        center_y=0.0,
        range_m=args.coarse_range_m,
        step_m=args.coarse_step_m,
        stage="coarse",
    )
    best_coarse = min(coarse_rows, key=lambda row: row["road_mean_m"])

    fine_rows = evaluate_grid(
        points=points,
        road_index=road_index,
        center_x=best_coarse["offset_x_m"],
        center_y=best_coarse["offset_y_m"],
        range_m=args.fine_range_m,
        step_m=args.fine_step_m,
        stage="fine",
    )
    best_fine = min(fine_rows, key=lambda row: row["road_mean_m"])

    write_csv(output_dir / "offset_search_coarse.csv", coarse_rows)
    write_csv(output_dir / "offset_search_fine.csv", fine_rows)
    write_heatmap(output_dir / "offset_search_fine_heatmap.png", fine_rows)

    summary = {
        "real_data_root": str(args.real_data_root.resolve()),
        "transform_path": str(args.transform_path.resolve()),
        "osm_edges_path": str(args.osm_edges_path.resolve()),
        "num_points": int(len(points)),
        "max_samples": args.max_samples,
        "point_stride": args.point_stride,
        "zero_offset": zero_metrics,
        "best_coarse": best_coarse,
        "best_fine": best_fine,
        "mean_improvement_m": float(zero_metrics["road_mean_m"] - best_fine["road_mean_m"]),
        "p95_improvement_m": float(zero_metrics["road_p95_m"] - best_fine["road_p95_m"]),
    }
    write_json(output_dir / "offset_search_summary.json", summary)
    write_markdown(output_dir / "offset_search_summary.md", summary)
    print(json.dumps(summary, indent=2))


def load_real_points(
    real_data_root: Path,
    transform: dict[str, Any],
    max_samples: int | None,
    max_points: int,
    point_stride: int,
    seed: int,
) -> np.ndarray:
    paths = sorted((real_data_root / "labels_absolute").glob("*.npy"))
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        raise FileNotFoundError(f"No real trajectories found under {real_data_root / 'labels_absolute'}")

    projected = []
    for path in paths:
        xy = load_xy(path)
        projected.append(apply_affine(xy[::point_stride], transform["affine_matrix"]))
    points = np.concatenate(projected, axis=0)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[np.sort(indices)]
    return points.astype(np.float64)


def evaluate_grid(
    points: np.ndarray,
    road_index: Any,
    center_x: float,
    center_y: float,
    range_m: float,
    step_m: float,
    stage: str,
) -> list[dict[str, float | str]]:
    offsets = np.arange(-range_m, range_m + step_m * 0.5, step_m, dtype=float)
    rows = []
    for dy in offsets:
        for dx in offsets:
            row = evaluate_offset(points, road_index, center_x + dx, center_y + dy)
            row["stage"] = stage
            rows.append(row)
    return rows


def evaluate_offset(points: np.ndarray, road_index: Any, offset_x: float, offset_y: float) -> dict[str, float]:
    shifted = points + np.asarray([offset_x, offset_y], dtype=float)
    distances = road_index.distances_for_xy(shifted)
    row = {
        "offset_x_m": float(offset_x),
        "offset_y_m": float(offset_y),
    }
    row.update(distance_metric_row(distances, "road"))
    return row


def write_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    xs = sorted({float(row["offset_x_m"]) for row in rows})
    ys = sorted({float(row["offset_y_m"]) for row in rows})
    values = np.full((len(ys), len(xs)), np.nan, dtype=float)
    x_to_i = {value: index for index, value in enumerate(xs)}
    y_to_i = {value: index for index, value in enumerate(ys)}
    for row in rows:
        values[y_to_i[float(row["offset_y_m"])], x_to_i[float(row["offset_x_m"])]] = float(row["road_mean_m"])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="lower",
        extent=[min(xs), max(xs), min(ys), max(ys)],
        aspect="equal",
        cmap="viridis_r",
    )
    ax.set_xlabel("Offset X (m)")
    ax.set_ylabel("Offset Y (m)")
    ax.set_title("Week05 real-to-OSM mean distance after offset")
    fig.colorbar(image, ax=ax, label="Mean nearest-road distance (m)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    zero = summary["zero_offset"]
    best = summary["best_fine"]
    lines = [
        "# Week06 Week05 Map Offset Search",
        "",
        f"- Points sampled: {summary['num_points']}",
        f"- OSM edges: `{summary['osm_edges_path']}`",
        f"- Transform: `{summary['transform_path']}`",
        "",
        "## Result",
        "",
        f"- Zero offset mean road distance: {zero['road_mean_m']:.3f} m",
        f"- Zero offset p95 road distance: {zero['road_p95_m']:.3f} m",
        f"- Best offset: dx={best['offset_x_m']:.3f} m, dy={best['offset_y_m']:.3f} m",
        f"- Best mean road distance: {best['road_mean_m']:.3f} m",
        f"- Best p95 road distance: {best['road_p95_m']:.3f} m",
        f"- Mean improvement: {summary['mean_improvement_m']:.3f} m",
        f"- P95 improvement: {summary['p95_improvement_m']:.3f} m",
        "",
        "## Interpretation",
        "",
    ]
    if summary["mean_improvement_m"] > 5.0:
        lines.append("A global translation correction materially improves real-to-road alignment.")
    else:
        lines.append("A global translation correction does not materially fix the real-to-road mismatch.")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--coarse-range-m", type=float, default=120.0)
    parser.add_argument("--coarse-step-m", type=float, default=20.0)
    parser.add_argument("--fine-range-m", type=float, default=30.0)
    parser.add_argument("--fine-step-m", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

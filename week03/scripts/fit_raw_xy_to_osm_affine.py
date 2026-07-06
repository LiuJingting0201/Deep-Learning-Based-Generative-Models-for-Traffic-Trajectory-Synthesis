"""Fit an affine transform from raw local x/y coordinates to OSM CRS coordinates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY_PATH = Path("/home/jliu/gps_with_speed_224_UPDATED.xls")
DEFAULT_OSM_EDGES_PATH = PROJECT_ROOT / "week03" / "data" / "maps" / "osm_drive_edges.gpkg"
DEFAULT_TRANSFORM_PATH = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "week03" / "docs" / "raw_xy_to_osm_affine_report.md"
DEFAULT_DIAGNOSTIC_DIR = PROJECT_ROOT / "week03" / "results" / "map_alignment_diagnostics"


def main() -> None:
    args = parse_args()
    pd, np, gpd, plt = import_runtime_dependencies()

    trajectory_path = args.trajectory_path.resolve()
    osm_edges_path = args.osm_edges_path.resolve()
    transform_path = args.output_json.resolve()
    report_path = args.report_path.resolve()
    diagnostic_dir = args.diagnostic_dir.resolve()
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    rows = load_trajectory_frame(pd, trajectory_path)
    edges = gpd.read_file(osm_edges_path)
    if edges.crs is None:
        raise ValueError(f"OSM edge file has no CRS: {osm_edges_path}")
    osm_crs = edges.crs

    points = gpd.GeoDataFrame(
        rows.copy(),
        geometry=gpd.points_from_xy(rows["lon"].astype(float), rows["lat"].astype(float)),
        crs="EPSG:4326",
    ).to_crs(osm_crs)
    x_osm = points.geometry.x.to_numpy(dtype=float)
    y_osm = points.geometry.y.to_numpy(dtype=float)
    raw = rows[["x", "y"]].to_numpy(dtype=float)

    design = np.column_stack([raw, np.ones(len(raw), dtype=float)])
    target = np.column_stack([x_osm, y_osm])
    affine, residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    transformed = design @ affine
    errors = np.linalg.norm(transformed - target, axis=1)
    residual_stats = summarize_errors(np, errors)
    warning_messages = alignment_warnings(residual_stats, args.warn_mean_m, args.warn_p95_m)

    payload = {
        "description": "[x_raw, y_raw, 1] @ affine_matrix = [x_osm, y_osm]",
        "created_by": Path(__file__).name,
        "input_trajectory_path": str(trajectory_path),
        "osm_edges_path": str(osm_edges_path),
        "osm_crs": osm_crs.to_string(),
        "raw_columns": ["x", "y"],
        "target_columns": ["lon", "lat projected to OSM CRS"],
        "affine_matrix": affine.tolist(),
        "num_points": int(len(rows)),
        "num_vehicles": int(rows["vehicle_id"].nunique()),
        "least_squares_rank": int(rank),
        "least_squares_residuals": residuals.tolist(),
        "least_squares_singular_values": singular_values.tolist(),
        "residual_error_m": residual_stats,
        "warning_mean_threshold_m": float(args.warn_mean_m),
        "warning_p95_threshold_m": float(args.warn_p95_m),
        "warnings": warning_messages,
    }
    write_json(transform_path, payload)

    diagnostic_paths = make_diagnostic_plots(
        plt=plt,
        np=np,
        rows=rows,
        projected_xy=target,
        transformed_xy=transformed,
        output_dir=diagnostic_dir,
        num_trajectories=args.num_diagnostics,
        seed=args.seed,
    )
    write_report(report_path, payload, diagnostic_paths)

    print(f"Saved affine transform: {transform_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved {len(diagnostic_paths)} diagnostic plot(s): {diagnostic_dir}")
    print(format_stats("Affine residual errors (m)", residual_stats))
    if warning_messages:
        print("WARNING: affine alignment residuals are large; do not claim map validity yet.", file=sys.stderr)
        for message in warning_messages:
            print(f"WARNING: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-path", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_TRANSFORM_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--num-diagnostics", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warn-mean-m", type=float, default=10.0)
    parser.add_argument("--warn-p95-m", type=float, default=25.0)
    return parser.parse_args()


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    missing = []
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        pd = None
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import geopandas as gpd
    except ImportError:
        missing.append("geopandas")
        gpd = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        missing.append("matplotlib")
        plt = None

    if missing:
        raise SystemExit(
            "Missing runtime package(s): "
            + ", ".join(sorted(set(missing)))
            + ". Run this script in the thesis geospatial Python environment."
        )
    return pd, np, gpd, plt


def load_trajectory_frame(pd: Any, path: Path) -> Any:
    required = ["vehicle_id", "time", "x", "y", "lon", "lat"]
    try:
        frame = pd.read_csv(path, usecols=required)
    except Exception:
        frame = pd.read_excel(path, usecols=required)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=required).copy()
    frame = frame.sort_values(["vehicle_id", "time"]).reset_index(drop=True)
    for column in ["time", "x", "y", "lon", "lat"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time", "x", "y", "lon", "lat"]).copy()
    if frame.empty:
        raise ValueError(f"No valid coordinate rows found in {path}")
    return frame


def summarize_errors(np: Any, errors: Any) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
    }


def alignment_warnings(stats: dict[str, float], mean_threshold: float, p95_threshold: float) -> list[str]:
    messages = []
    if stats["mean"] > mean_threshold:
        messages.append(f"mean residual {stats['mean']:.3f} m exceeds {mean_threshold:.3f} m")
    if stats["p95"] > p95_threshold:
        messages.append(f"p95 residual {stats['p95']:.3f} m exceeds {p95_threshold:.3f} m")
    return messages


def make_diagnostic_plots(
    plt: Any,
    np: Any,
    rows: Any,
    projected_xy: Any,
    transformed_xy: Any,
    output_dir: Path,
    num_trajectories: int,
    seed: int,
) -> list[Path]:
    vehicle_ids = sorted(rows["vehicle_id"].astype(str).unique().tolist())
    if not vehicle_ids:
        return []
    rng = np.random.default_rng(seed)
    selected = rng.choice(vehicle_ids, size=min(num_trajectories, len(vehicle_ids)), replace=False)
    paths = []
    vehicle_series = rows["vehicle_id"].astype(str).reset_index(drop=True)
    for vehicle_id in selected:
        mask = vehicle_series == str(vehicle_id)
        indices = np.flatnonzero(mask.to_numpy())
        if len(indices) < 2:
            continue
        path = output_dir / f"vehicle_{safe_name(str(vehicle_id))}_alignment.png"
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(projected_xy[indices, 0], projected_xy[indices, 1], label="lon/lat projected", linewidth=1.5)
        ax.plot(
            transformed_xy[indices, 0],
            transformed_xy[indices, 1],
            label="raw x/y affine",
            linewidth=1.2,
            linestyle="--",
        )
        ax.scatter(projected_xy[indices[0], 0], projected_xy[indices[0], 1], s=25, label="start")
        ax.set_title(f"Vehicle {vehicle_id}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(path: Path, payload: dict[str, Any], diagnostic_paths: list[Path]) -> None:
    stats = payload["residual_error_m"]
    warning_lines = (
        "\n".join(f"- WARNING: {message}" for message in payload["warnings"])
        if payload["warnings"]
        else "- No threshold warnings were triggered. This is not, by itself, proof of map validity."
    )
    diagnostics = "\n".join(f"- {plot}" for plot in diagnostic_paths) or "- None"
    matrix = payload["affine_matrix"]
    text = f"""# Raw x/y to OSM Affine Alignment

## Inputs

- Raw trajectory file: `{payload["input_trajectory_path"]}`
- OSM edge file: `{payload["osm_edges_path"]}`
- OSM CRS: `{payload["osm_crs"]}`
- Points used: {payload["num_points"]}
- Vehicles used: {payload["num_vehicles"]}

## Affine Transform

Convention:

```text
[x_raw, y_raw, 1] @ A = [x_osm, y_osm]
```

Matrix:

```json
{json.dumps(matrix, indent=2)}
```

## Residual Error in Meters

{format_stats("", stats)}

## Warnings

{warning_lines}

Do not claim map validity unless these residuals are reasonable for the trajectory scale and downstream evaluation purpose.

## Diagnostic Plots

{diagnostics}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def format_stats(title: str, stats: dict[str, float]) -> str:
    prefix = f"{title}\n" if title else ""
    return (
        f"{prefix}"
        f"- mean: {stats['mean']:.6f}\n"
        f"- median: {stats['median']:.6f}\n"
        f"- p90: {stats['p90']:.6f}\n"
        f"- p95: {stats['p95']:.6f}\n"
        f"- p99: {stats['p99']:.6f}\n"
        f"- max: {stats['max']:.6f}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    main()

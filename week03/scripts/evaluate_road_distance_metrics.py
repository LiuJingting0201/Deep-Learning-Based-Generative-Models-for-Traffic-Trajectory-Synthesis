"""Evaluate pointwise distance from predicted and GT trajectories to OSM roads."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_TRANSFORM_PATH = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OSM_EDGES_PATH = PROJECT_ROOT / "week03" / "data" / "maps" / "osm_drive_edges.gpkg"


def main() -> None:
    args = parse_args()
    np, pd, gpd, shapely_module, Point, STRtree = import_runtime_dependencies()

    output_dir = args.output_dir.resolve()
    predictions_dir = resolve_predictions_dir(output_dir, args.predictions_dir_name)
    metric_output_dir = output_dir if output_dir.name != args.predictions_dir_name else output_dir.parent.parent
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    transform = load_transform(args.transform_path.resolve())
    check_alignment_reasonable(transform, args.max_affine_mean_m, args.max_affine_p95_m, args.allow_large_affine_error)

    metadata = load_metadata(pd, args.data_root.resolve(), output_dir)
    pred_paths = sorted(predictions_dir.glob(f"*{args.prediction_suffix}.npy"))
    if args.max_samples is not None:
        pred_paths = pred_paths[: args.max_samples]
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    edges = gpd.read_file(args.osm_edges_path.resolve())
    if edges.crs is None:
        raise ValueError(f"OSM edge file has no CRS: {args.osm_edges_path}")
    road_index = RoadDistanceIndex(edges.geometry.dropna().tolist(), STRtree)

    rows = []
    pred_all_distances = []
    gt_all_distances = []
    missing = []
    for pred_path in pred_paths:
        sample_id = sample_id_from_prediction_path(pred_path, args.prediction_suffix)
        meta = metadata.get(sample_id)
        if meta is None:
            missing.append(sample_id)
            continue
        gt_path = args.data_root.resolve() / str(meta["label_absolute_path"])
        if not gt_path.exists():
            raise FileNotFoundError(f"GT absolute trajectory not found for {sample_id}: {gt_path}")
        pred_raw = load_xy_array(np, pred_path)
        gt_raw = load_xy_array(np, gt_path)
        pred_osm = apply_affine(np, pred_raw, transform["affine_matrix"])
        gt_osm = apply_affine(np, gt_raw, transform["affine_matrix"])
        pred_distances = road_index.distances_for_xy(np, shapely_module, Point, pred_osm)
        gt_distances = road_index.distances_for_xy(np, shapely_module, Point, gt_osm)
        pred_all_distances.append(pred_distances)
        gt_all_distances.append(gt_distances)

        row = {
            "sample_id": sample_id,
            "vehicle_id": meta.get("vehicle_id", ""),
            "split": meta.get("split", ""),
            "num_pred_points": int(len(pred_distances)),
            "num_gt_points": int(len(gt_distances)),
        }
        row.update(metric_row(np, pred_distances, "pred"))
        row.update(metric_row(np, gt_distances, "gt"))
        rows.append(row)

    if missing:
        preview = ", ".join(missing[:10])
        warnings.warn(f"Skipped {len(missing)} prediction(s) missing metadata. First missing: {preview}")
    if not rows:
        raise ValueError("No samples could be evaluated after pairing predictions with metadata.")

    per_sample_csv = metric_output_dir / "road_distance_per_sample.csv"
    write_csv(per_sample_csv, rows)
    summary = {
        "output_dir": str(output_dir),
        "predictions_dir": str(predictions_dir),
        "data_root": str(args.data_root.resolve()),
        "osm_edges_path": str(args.osm_edges_path.resolve()),
        "transform_path": str(args.transform_path.resolve()),
        "num_prediction_files_found": int(len(pred_paths)),
        "num_samples_evaluated": int(len(rows)),
        "num_missing_metadata": int(len(missing)),
        "road_index_mode": road_index.mode,
        "affine_residual_error_m": transform.get("residual_error_m", {}),
        "pred": summarize_distances(np, concat(np, pred_all_distances)),
        "gt": summarize_distances(np, concat(np, gt_all_distances)),
        "per_sample_mean": summarize_per_sample(np, rows),
    }
    write_json(metric_output_dir / "road_distance_summary.json", summary)
    write_markdown(metric_output_dir / "road_distance_summary.md", summary)

    print(f"Saved per-sample road metrics: {per_sample_csv}")
    print(f"Saved road-distance summary: {metric_output_dir / 'road_distance_summary.json'}")
    print(f"Road index mode: {road_index.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Experiment output dir, evaluation dir, or predictions dir.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM_PATH)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES_PATH)
    parser.add_argument("--predictions-dir-name", default="predictions_absolute_integrated")
    parser.add_argument("--prediction-suffix", default="_pred_absolute_integrated")
    parser.add_argument("--max-affine-mean-m", type=float, default=10.0)
    parser.add_argument("--max-affine-p95-m", type=float, default=25.0)
    parser.add_argument("--allow-large-affine-error", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        pd = None
    try:
        import geopandas as gpd
    except ImportError:
        missing.append("geopandas")
        gpd = None
    try:
        import shapely as shapely_module
    except ImportError:
        shapely_module = None
    try:
        from shapely.geometry import Point
    except ImportError:
        missing.append("shapely")
        Point = None
    try:
        from shapely.strtree import STRtree
    except ImportError:
        STRtree = None
    if missing:
        raise SystemExit(
            "Missing runtime package(s): "
            + ", ".join(sorted(set(missing)))
            + ". Run this script in the thesis geospatial Python environment."
        )
    return np, pd, gpd, shapely_module, Point, STRtree


class RoadDistanceIndex:
    def __init__(self, geometries: list[Any], strtree_class: Any) -> None:
        self.geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
        if not self.geometries:
            raise ValueError("No valid OSM road geometries were loaded.")
        self.tree = None
        self.mode = "brute_force"
        if strtree_class is not None:
            try:
                self.tree = strtree_class(self.geometries)
                self.mode = "shapely_strtree"
            except Exception as exc:
                warnings.warn(f"Could not build Shapely STRtree spatial index; falling back to brute force: {exc}")
        if self.tree is None:
            warnings.warn("No Shapely spatial index is available; nearest-road distance may be slow.")

    def distances_for_xy(self, np: Any, shapely_module: Any, point_class: Any, xy: Any) -> Any:
        if self.tree is not None and hasattr(self.tree, "query_nearest") and hasattr(shapely_module, "points"):
            try:
                points = shapely_module.points(xy[:, 0].astype(float), xy[:, 1].astype(float))
                indices, distances = self.tree.query_nearest(points, return_distance=True, all_matches=False)
                ordered = np.empty(len(points), dtype=float)
                ordered[indices[0]] = distances
                return ordered
            except Exception as exc:
                warnings.warn(f"Vectorized STRtree distance query failed; using per-point nearest queries: {exc}")
        distances = []
        for x_coord, y_coord in xy:
            point = point_class(float(x_coord), float(y_coord))
            road = self.nearest_geometry(point)
            distances.append(float(point.distance(road)))
        return np.asarray(distances, dtype=float)

    def nearest_geometry(self, point: Any) -> Any:
        if self.tree is None:
            return min(self.geometries, key=point.distance)
        nearest = self.tree.nearest(point)
        if isinstance(nearest, int):
            return self.geometries[nearest]
        try:
            import numpy as np

            if isinstance(nearest, np.integer):
                return self.geometries[int(nearest)]
        except ImportError:
            pass
        return nearest


def resolve_predictions_dir(output_dir: Path, predictions_dir_name: str) -> Path:
    candidates = [
        output_dir,
        output_dir / predictions_dir_name,
        output_dir / "evaluation" / predictions_dir_name,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.npy")):
            return candidate
    raise FileNotFoundError(
        f"Could not find prediction .npy files under {output_dir}, "
        f"{output_dir / predictions_dir_name}, or {output_dir / 'evaluation' / predictions_dir_name}"
    )


def load_transform(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Affine transform not found: {path}")
    payload = json.loads(path.read_text())
    if "affine_matrix" not in payload:
        raise ValueError(f"{path} does not contain affine_matrix")
    return payload


def check_alignment_reasonable(
    transform: dict[str, Any],
    max_mean_m: float,
    max_p95_m: float,
    allow_large: bool,
) -> None:
    stats = transform.get("residual_error_m", {})
    mean = float(stats.get("mean", float("inf")))
    p95 = float(stats.get("p95", float("inf")))
    messages = []
    if mean > max_mean_m:
        messages.append(f"mean residual {mean:.3f} m exceeds {max_mean_m:.3f} m")
    if p95 > max_p95_m:
        messages.append(f"p95 residual {p95:.3f} m exceeds {max_p95_m:.3f} m")
    if messages and not allow_large:
        raise SystemExit(
            "Affine alignment error is large; road-distance metrics are blocked by default. "
            + "; ".join(messages)
            + ". Re-run with --allow-large-affine-error only for diagnostics."
        )
    if messages:
        warnings.warn(
            "Affine alignment error is large; road-distance metrics should be treated as diagnostic only. "
            + "; ".join(messages)
        )


def load_metadata(pd: Any, data_root: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = [
        output_dir / "split_copy.csv",
        output_dir.parent / "split_copy.csv",
        output_dir.parent.parent / "split_copy.csv",
        data_root / "splits" / "split_metadata.csv",
        data_root / "metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"No metadata CSV found for data root {data_root} or output {output_dir}")
    required = {"sample_id", "label_absolute_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata {path} is missing columns: {sorted(missing)}")
    if "valid_or_skipped" in frame.columns:
        frame = frame[frame["valid_or_skipped"].fillna("valid") == "valid"].copy()
    return {str(row.sample_id): row._asdict() for row in frame.itertuples(index=False)}


def sample_id_from_prediction_path(path: Path, suffix: str) -> str:
    stem = path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def load_xy_array(np: Any, path: Path) -> Any:
    arr = np.load(path).astype(float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected [T,2+] x/y array in {path}, got shape {arr.shape}")
    return arr[:, :2]


def apply_affine(np: Any, xy: Any, affine_matrix: Any) -> Any:
    matrix = np.asarray(affine_matrix, dtype=float)
    if matrix.shape != (3, 2):
        raise ValueError(f"Expected affine_matrix shape [3,2], got {matrix.shape}")
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy), dtype=float)])
    return design @ matrix


def metric_row(np: Any, distances: Any, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_road_distance": float(np.mean(distances)),
        f"{prefix}_median_road_distance": float(np.median(distances)),
        f"{prefix}_p90_road_distance": float(np.percentile(distances, 90)),
        f"{prefix}_offroad_ratio_5m": float(np.mean(distances > 5.0)),
        f"{prefix}_offroad_ratio_10m": float(np.mean(distances > 10.0)),
        f"{prefix}_offroad_ratio_15m": float(np.mean(distances > 15.0)),
    }


def summarize_distances(np: Any, distances: Any) -> dict[str, float]:
    return {
        "mean_road_distance": float(np.mean(distances)),
        "median_road_distance": float(np.median(distances)),
        "p90_road_distance": float(np.percentile(distances, 90)),
        "p95_road_distance": float(np.percentile(distances, 95)),
        "p99_road_distance": float(np.percentile(distances, 99)),
        "max_road_distance": float(np.max(distances)),
        "offroad_ratio_5m": float(np.mean(distances > 5.0)),
        "offroad_ratio_10m": float(np.mean(distances > 10.0)),
        "offroad_ratio_15m": float(np.mean(distances > 15.0)),
    }


def summarize_per_sample(np: Any, rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    columns = [
        "pred_mean_road_distance",
        "pred_median_road_distance",
        "pred_p90_road_distance",
        "pred_offroad_ratio_5m",
        "pred_offroad_ratio_10m",
        "pred_offroad_ratio_15m",
        "gt_mean_road_distance",
        "gt_median_road_distance",
        "gt_p90_road_distance",
        "gt_offroad_ratio_5m",
        "gt_offroad_ratio_10m",
        "gt_offroad_ratio_15m",
    ]
    summary = {}
    for column in columns:
        values = np.asarray([row[column] for row in rows], dtype=float)
        summary[column] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return summary


def concat(np: Any, arrays: list[Any]) -> Any:
    if not arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(arrays)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    pred = summary["pred"]
    gt = summary["gt"]
    text = f"""# Road-Distance Metrics

## Inputs

- Output directory: `{summary["output_dir"]}`
- Predictions: `{summary["predictions_dir"]}`
- Data root: `{summary["data_root"]}`
- OSM edges: `{summary["osm_edges_path"]}`
- Affine transform: `{summary["transform_path"]}`
- Road index mode: `{summary["road_index_mode"]}`
- Samples evaluated: {summary["num_samples_evaluated"]}

## Pointwise Summary

| Metric | Prediction | GT |
| --- | ---: | ---: |
| mean_road_distance | {pred["mean_road_distance"]:.6f} | {gt["mean_road_distance"]:.6f} |
| median_road_distance | {pred["median_road_distance"]:.6f} | {gt["median_road_distance"]:.6f} |
| p90_road_distance | {pred["p90_road_distance"]:.6f} | {gt["p90_road_distance"]:.6f} |
| offroad_ratio_5m | {pred["offroad_ratio_5m"]:.6f} | {gt["offroad_ratio_5m"]:.6f} |
| offroad_ratio_10m | {pred["offroad_ratio_10m"]:.6f} | {gt["offroad_ratio_10m"]:.6f} |
| offroad_ratio_15m | {pred["offroad_ratio_15m"]:.6f} | {gt["offroad_ratio_15m"]:.6f} |

## Affine Residuals

```json
{json.dumps(summary["affine_residual_error_m"], indent=2)}
```

Treat road-distance results as map-validity evidence only if the affine residuals are reasonable.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    main()

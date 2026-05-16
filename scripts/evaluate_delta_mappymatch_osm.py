"""Evaluate delta decoder traces with real OSM/mappymatch post-processing.

This is a whole-trace map-matching baseline using OSM through mappymatch. It is
not the nearest-GT-polyline oracle experiment and does not use ground-truth
trajectories as a map.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point


SEQUENCE_LENGTH = 224


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    delta_eval_dir = args.delta_eval_dir.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dirs(output_dir)

    transform = load_transform(args.xy_to_lonlat_json)
    metadata = load_split_metadata(data_root, args.split, args.max_samples)
    validate_inputs(data_root, delta_eval_dir, metadata)
    write_json(output_dir / "config.json", make_config(args, metadata))

    per_sample = evaluate_samples(data_root, delta_eval_dir, output_dir, metadata, transform, args.padding_meters)
    per_sample.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    summary = summarize_results(per_sample, args)
    write_json(output_dir / "metrics_summary.json", summary)
    write_chinese_report(output_dir / "delta_mappymatch_osm_report_zh.txt", summary, args)
    print_summary(summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_delta_displacement_paired"))
    parser.add_argument(
        "--delta-eval-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_delta_displacement_resnet18_ablation/evaluation"),
    )
    parser.add_argument("--xy-to-lonlat-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/delta_mappymatch_osm"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--padding-meters", type=float, default=1000.0)
    return parser.parse_args()


def prepare_output_dirs(output_dir: Path) -> None:
    for directory in [
        output_dir,
        output_dir / "trace_csvs",
        output_dir / "matched_outputs",
        output_dir / "predictions_matched_xy",
        output_dir / "plots",
        output_dir / "plots" / "per_sample",
        output_dir / "plots" / "per_sample_with_roads",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    for directory in [
        output_dir / "trace_csvs",
        output_dir / "matched_outputs",
        output_dir / "predictions_matched_xy",
        output_dir / "plots",
        output_dir / "plots" / "per_sample",
        output_dir / "plots" / "per_sample_with_roads",
    ]:
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()


def load_transform(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    transform = json.loads(path.read_text(encoding="utf-8"))
    required = {"lon_coefficients", "lat_coefficients", "coordinate_type"}
    missing = required.difference(transform)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    if transform["coordinate_type"] != "affine_xy_to_lonlat":
        raise ValueError(f"Unsupported coordinate_type: {transform['coordinate_type']}")
    return transform


def load_split_metadata(data_root: Path, split: str, max_samples: int | None) -> pd.DataFrame:
    metadata_path = data_root / "splits" / "split_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path)
    required = {"sample_id", "vehicle_id", "split", "label_absolute_path", "start_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} missing required columns: {sorted(missing)}")
    split_frame = metadata.loc[metadata["split"] == split].copy().reset_index(drop=True)
    if split_frame.empty:
        raise ValueError(f"No samples found for split={split!r} in {metadata_path}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided.")
        split_frame = split_frame.head(max_samples).copy()
    return split_frame


def validate_inputs(data_root: Path, delta_eval_dir: Path, metadata: pd.DataFrame) -> None:
    pred_delta_dir = delta_eval_dir / "predictions_delta_displacement"
    if not pred_delta_dir.exists():
        raise FileNotFoundError(pred_delta_dir)
    missing: list[str] = []
    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        paths = [
            data_root / row.label_absolute_path,
            data_root / row.start_path,
            pred_delta_dir / f"{sample_id}_pred_delta.npy",
        ]
        missing.extend(str(path) for path in paths if not path.exists())
    if missing:
        preview = "\n".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Required input files are missing:\n{preview}{suffix}")


def make_config(args: argparse.Namespace, metadata: pd.DataFrame) -> dict[str, Any]:
    return {
        "experiment": "delta_mappymatch_osm",
        "num_samples": int(len(metadata)),
        "data_root": str(args.data_root),
        "delta_eval_dir": str(args.delta_eval_dir),
        "xy_to_lonlat_json": str(args.xy_to_lonlat_json),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "padding_meters": args.padding_meters,
        "map_matching": "mappymatch whole-trace LCSSMatcher with OSM NxMap",
        "not_oracle": "Does not use ground-truth trajectory as a projection map.",
    }


def evaluate_samples(
    data_root: Path,
    delta_eval_dir: Path,
    output_dir: Path,
    metadata: pd.DataFrame,
    transform: dict[str, Any],
    padding_meters: float,
) -> pd.DataFrame:
    pred_delta_dir = delta_eval_dir / "predictions_delta_displacement"
    rows: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, np.ndarray | None]] = {}
    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        true_abs = load_xy_array(data_root / row.label_absolute_path, sample_id, "ground-truth absolute")
        start = load_start(data_root / row.start_path, sample_id)
        pred_delta = load_xy_array(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_id, "predicted delta")
        validate_shapes(sample_id, true_abs, start, pred_delta)
        pred_xy = integrate_delta(start, pred_delta)
        plain_metrics, _ = compute_metrics(pred_xy, true_abs)

        pred_lonlat = xy_to_lonlat(pred_xy, transform)
        trace_path = write_trace_csv(output_dir / "trace_csvs" / f"{sample_id}_pred_trace.csv", pred_lonlat)
        match_metadata, match_debug, matched_lonlat, road_lines_3857, path_lines_3857 = run_mappymatch(
            trace_path,
            padding_meters,
            expected_length=len(pred_xy),
        )
        matched_metrics_available = matched_lonlat is not None and len(matched_lonlat) == len(true_abs)
        matched_metrics = empty_metrics()
        matched_xy: np.ndarray | None = None
        road_lines_xy = epsg3857_lines_to_xy(road_lines_3857, transform)
        path_lines_xy = epsg3857_lines_to_xy(path_lines_3857, transform)
        if matched_metrics_available:
            matched_xy = lonlat_to_xy(matched_lonlat, transform)
            matched_metrics, _ = compute_metrics(matched_xy, true_abs)
            np.save(output_dir / "predictions_matched_xy" / f"{sample_id}_matched_points_xy.npy", matched_xy)
            match_metadata["matched_metrics_available"] = True
            match_metadata["matched_lonlat_preview"] = matched_lonlat[:5].tolist()
        else:
            match_metadata["matched_metrics_available"] = False
        write_json(
            output_dir / "matched_outputs" / f"{sample_id}_road_geometries_xy.json",
            {"road_geometries_xy": [line.tolist() for line in road_lines_xy]},
        )
        if path_lines_xy:
            np.save(
                output_dir / "matched_outputs" / f"{sample_id}_matched_path_xy.npy",
                lines_to_nan_separated_array(path_lines_xy),
            )
        write_json(output_dir / "matched_outputs" / f"{sample_id}_match.json", match_metadata)
        write_json(output_dir / "matched_outputs" / f"{sample_id}_match_debug.json", match_debug)

        rows.append(
            {
                "sample_id": sample_id,
                "vehicle_id": str(row.vehicle_id),
                "split": str(row.split),
                **prefix_metrics("plain", plain_metrics),
                "match_success": match_metadata["match_success"],
                "match_failure_reason": match_metadata.get("failure_reason", ""),
                "matched_path_length": match_metadata.get("matched_path_length", np.nan),
                "number_of_matches": match_metadata.get("number_of_matches", np.nan),
                "mean_trace_to_match_distance": match_metadata.get("mean_trace_to_match_distance", np.nan),
                **prefix_metrics("matched", matched_metrics),
                "matched_metrics_available": matched_metrics_available,
                "trace_csv": str(trace_path.relative_to(output_dir)),
                "match_json": f"matched_outputs/{sample_id}_match.json",
                "match_debug_json": f"matched_outputs/{sample_id}_match_debug.json",
            }
        )
        trajectories[sample_id] = {
            "ground_truth": true_abs,
            "plain": pred_xy,
            "matched": matched_xy,
            "road_lines": road_lines_xy,
            "matched_path_lines": path_lines_xy,
        }
    make_plots(pd.DataFrame(rows), trajectories, output_dir / "plots")
    return pd.DataFrame(rows)


def load_xy_array(path: Path, sample_id: str, label: str) -> np.ndarray:
    array = np.load(path).astype(np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{sample_id}: {label} must have shape [T, 2], got {array.shape}")
    return array


def load_start(path: Path, sample_id: str) -> np.ndarray:
    start = np.load(path).astype(np.float64)
    if start.shape != (2,):
        raise ValueError(f"{sample_id}: start must have shape (2,), got {start.shape}")
    return start


def validate_shapes(sample_id: str, true_abs: np.ndarray, start: np.ndarray, pred_delta: np.ndarray) -> None:
    if true_abs.shape != pred_delta.shape:
        raise ValueError(f"{sample_id}: true_abs shape {true_abs.shape} != pred_delta shape {pred_delta.shape}")
    if true_abs.shape[0] != SEQUENCE_LENGTH:
        raise ValueError(f"{sample_id}: expected {SEQUENCE_LENGTH} timesteps, got {true_abs.shape[0]}")
    if start.shape != (2,):
        raise ValueError(f"{sample_id}: start must have shape (2,), got {start.shape}")


def integrate_delta(start: np.ndarray, pred_delta: np.ndarray) -> np.ndarray:
    pred_xy = np.empty_like(pred_delta, dtype=np.float64)
    pred_xy[0] = start
    pred_xy[1:] = start + np.cumsum(pred_delta[1:], axis=0)
    return pred_xy


def xy_to_lonlat(xy: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    design = np.column_stack([np.ones(len(xy), dtype=np.float64), xy[:, 0], xy[:, 1]])
    lon = design @ np.asarray(transform["lon_coefficients"], dtype=np.float64)
    lat = design @ np.asarray(transform["lat_coefficients"], dtype=np.float64)
    return np.column_stack([lon, lat])


def lonlat_to_xy(lonlat: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    lon_coefficients = np.asarray(transform["lon_coefficients"], dtype=np.float64)
    lat_coefficients = np.asarray(transform["lat_coefficients"], dtype=np.float64)
    linear = np.array(
        [
            [lon_coefficients[1], lon_coefficients[2]],
            [lat_coefficients[1], lat_coefficients[2]],
        ],
        dtype=np.float64,
    )
    shifted = np.column_stack(
        [
            lonlat[:, 0] - lon_coefficients[0],
            lonlat[:, 1] - lat_coefficients[0],
        ]
    )
    return np.linalg.solve(linear, shifted.T).T


def epsg3857_lines_to_xy(lines: list[np.ndarray], transform: dict[str, Any]) -> list[np.ndarray]:
    if not lines:
        return []
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    converted = []
    for line in lines:
        if len(line) == 0:
            continue
        lon, lat = transformer.transform(line[:, 0], line[:, 1])
        converted.append(lonlat_to_xy(np.column_stack([lon, lat]), transform))
    return converted


def lines_to_nan_separated_array(lines: list[np.ndarray]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for line in lines:
        if len(line):
            rows.append(line)
            rows.append(np.full((1, 2), np.nan, dtype=np.float64))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(rows[:-1])


def write_trace_csv(path: Path, lonlat: np.ndarray) -> Path:
    frame = pd.DataFrame({"latitude": lonlat[:, 1], "longitude": lonlat[:, 0]})
    frame.to_csv(path, index=False)
    return path


def run_mappymatch(
    trace_path: Path,
    padding_meters: float,
    expected_length: int,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray | None, list[np.ndarray], list[np.ndarray]]:
    metadata: dict[str, Any] = {
        "trace_csv": str(trace_path),
        "padding_meters": padding_meters,
        "match_success": False,
        "matched_metrics_available": False,
    }
    debug: dict[str, Any] = {
        "trace_csv": str(trace_path),
        "padding_meters": padding_meters,
        "match_success": False,
    }
    matched_lonlat: np.ndarray | None = None
    road_lines_3857: list[np.ndarray] = []
    path_lines_3857: list[np.ndarray] = []
    try:
        from mappymatch.constructs.geofence import Geofence
        from mappymatch.constructs.trace import Trace
        from mappymatch.maps.nx.nx_map import NetworkType, NxMap
        from mappymatch.matchers.lcss.lcss import LCSSMatcher

        trace = Trace.from_csv(
            str(trace_path),
            lat_column="latitude",
            lon_column="longitude",
            xy=True,
        )
        geofence = Geofence.from_trace(trace, padding=padding_meters)
        nx_map = NxMap.from_geofence(geofence, network_type=NetworkType.DRIVE)
        road_lines_3857 = extract_road_lines_3857(getattr(nx_map, "roads", []))
        matcher = LCSSMatcher(nx_map)
        result = matcher.match_trace(trace)
        debug = build_match_debug(result, trace_path, padding_meters)
        metadata.update(extract_match_metadata(result))
        path_lines_3857 = extract_road_lines_3857(getattr(result, "path", []))
        matched_lonlat = extract_timestep_matched_lonlat(result, expected_length)
        if matched_lonlat is not None:
            metadata["matched_coordinate_source"] = "match.coordinate projected to match.road.geom"
            metadata["matched_coordinate_count"] = int(len(matched_lonlat))
        else:
            metadata["matched_coordinate_source"] = ""
            metadata["matched_coordinate_count"] = 0
        metadata["match_success"] = True
        debug["match_success"] = True
    except Exception as exc:  # noqa: BLE001 - preserve batch progress and report reason.
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
        debug["failure_reason"] = metadata["failure_reason"]
        debug["traceback_tail"] = metadata["traceback_tail"]
    return metadata, debug, matched_lonlat, road_lines_3857, path_lines_3857


def extract_road_lines_3857(roads: list[Any]) -> list[np.ndarray]:
    lines = []
    for road in roads:
        geom = getattr(road, "geom", None)
        if geom is None or not hasattr(geom, "coords"):
            continue
        coords = np.asarray(list(geom.coords), dtype=np.float64)
        if coords.ndim == 2 and coords.shape[1] >= 2 and len(coords) >= 2:
            lines.append(coords[:, :2])
    return lines


def extract_match_metadata(result: Any) -> dict[str, Any]:
    path = getattr(result, "path", None)
    matches = getattr(result, "matches", None)
    distances = collect_distance_like_values(result)
    metadata: dict[str, Any] = {
        "result_type": type(result).__name__,
        "matched_path_length": len(path) if path is not None else np.nan,
        "number_of_matches": len(matches) if matches is not None else np.nan,
        "mean_trace_to_match_distance": float(np.mean(distances)) if distances else np.nan,
    }
    if path is not None:
        metadata["path_preview"] = [safe_string(item) for item in list(path)[:20]]
        metadata["road_path_coordinates_preview"] = road_path_coordinates(path, max_roads=10)
    return metadata


def extract_timestep_matched_lonlat(result: Any, expected_length: int) -> np.ndarray | None:
    matches = getattr(result, "matches", None)
    if matches is None or len(matches) != expected_length:
        return None
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lonlat_rows: list[tuple[float, float]] = []
    for match in matches:
        coordinate = getattr(match, "coordinate", None)
        road = getattr(match, "road", None)
        geom = getattr(road, "geom", None)
        if coordinate is None or geom is None:
            return None
        try:
            point = Point(float(coordinate.x), float(coordinate.y))
            matched_point = geom.interpolate(geom.project(point))
            lon, lat = transformer.transform(float(matched_point.x), float(matched_point.y))
        except Exception:
            return None
        lonlat_rows.append((float(lon), float(lat)))
    return np.asarray(lonlat_rows, dtype=np.float64)


def build_match_debug(result: Any, trace_path: Path, padding_meters: float) -> dict[str, Any]:
    matches = getattr(result, "matches", None)
    path = getattr(result, "path", None)
    return {
        "trace_csv": str(trace_path),
        "padding_meters": padding_meters,
        "match_result_type": full_type_name(result),
        "match_result_dir": public_dir(result),
        "match_result_attributes": summarize_public_attributes(result),
        "number_of_matches": len(matches) if matches is not None else None,
        "number_of_path_elements": len(path) if path is not None else None,
        "first_matches": [summarize_object(match) for match in list(matches or [])[:5]],
        "first_path_elements": [summarize_object(element) for element in list(path or [])[:5]],
    }


def summarize_public_attributes(value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for attr in public_dir(value):
        try:
            attr_value = getattr(value, attr)
        except Exception as exc:
            output[attr] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if callable(attr_value):
            continue
        output[attr] = summarize_value(attr_value)
    return output


def summarize_object(value: Any) -> dict[str, Any]:
    return {
        "type": full_type_name(value),
        "dir": public_dir(value),
        "attributes": summarize_public_attributes(value),
        "repr": safe_string(value),
    }


def summarize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): summarize_value(item) for key, item in list(value.items())[:20]}
    if hasattr(value, "geom_type") and hasattr(value, "coords"):
        coords = list(value.coords)
        return {
            "type": full_type_name(value),
            "geom_type": getattr(value, "geom_type", ""),
            "num_coords": len(coords),
            "coords_preview": [[float(x), float(y)] for x, y, *rest in coords[:5]],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": full_type_name(value),
            "length": len(value),
            "preview": [summarize_value(item) for item in list(value)[:5]],
        }
    if hasattr(value, "_asdict"):
        return {str(key): summarize_value(item) for key, item in value._asdict().items()}
    if hasattr(value, "__dict__"):
        return {
            "type": full_type_name(value),
            "attributes": {
                str(key): summarize_value(item)
                for key, item in list(vars(value).items())[:20]
                if not str(key).startswith("_")
            },
        }
    return {"type": full_type_name(value), "repr": safe_string(value)}


def public_dir(value: Any) -> list[str]:
    return [attr for attr in dir(value) if not attr.startswith("_")]


def full_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def road_path_coordinates(path: list[Any], max_roads: int) -> list[dict[str, Any]]:
    rows = []
    for road in list(path)[:max_roads]:
        geom = getattr(road, "geom", None)
        coords = list(geom.coords) if geom is not None and hasattr(geom, "coords") else []
        rows.append(
            {
                "road": safe_string(road),
                "num_coords": len(coords),
                "coords_preview_epsg3857": [[float(x), float(y)] for x, y, *rest in coords[:10]],
            }
        )
    return rows


def collect_distance_like_values(result: Any) -> list[float]:
    values: list[float] = []
    for attr in ["distances", "distance", "match_distances"]:
        if hasattr(result, attr):
            values.extend(numeric_values(getattr(result, attr)))
    matches = getattr(result, "matches", None)
    if matches is not None:
        for match in matches:
            for attr in ["distance", "dist"]:
                if hasattr(match, attr):
                    values.extend(numeric_values(getattr(match, attr)))
    return values


def numeric_values(value: Any) -> list[float]:
    array = np.asarray(value, dtype=object).reshape(-1)
    output = []
    for item in array:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return output


def safe_string(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 300 else text[:297] + "..."


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    diff = pred - true
    point_errors = np.linalg.norm(diff, axis=1)
    return (
        {
            "ADE": float(point_errors.mean()),
            "FDE": float(point_errors[-1]),
            "RMSE": float(np.sqrt(np.mean(diff**2))),
            "MAE": float(np.mean(np.abs(diff))),
        },
        point_errors,
    )


def empty_metrics() -> dict[str, float]:
    return {"ADE": np.nan, "FDE": np.nan, "RMSE": np.nan, "MAE": np.nan}


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def summarize_results(per_sample: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "split": args.split,
        "num_samples": int(len(per_sample)),
        "padding_meters": args.padding_meters,
        "plain": {
            metric: summarize_values(per_sample[f"plain_{metric}"].to_numpy(dtype=float))
            for metric in ["ADE", "FDE", "RMSE", "MAE"]
        },
        "map_matching": {
            "success_count": int(per_sample["match_success"].sum()),
            "failure_count": int((~per_sample["match_success"]).sum()),
            "success_rate": float(per_sample["match_success"].mean()),
            "matched_metrics_available_count": int(per_sample["matched_metrics_available"].sum()),
        },
    }
    available = per_sample.loc[per_sample["matched_metrics_available"]].copy()
    if len(available):
        summary["matched"] = {
            metric: summarize_values(available[f"matched_{metric}"].to_numpy(dtype=float))
            for metric in ["ADE", "FDE", "RMSE", "MAE"]
        }
    else:
        summary["matched"] = {
            metric: summarize_values(np.asarray([], dtype=float))
            for metric in ["ADE", "FDE", "RMSE", "MAE"]
        }
    if "matched_path_length" in per_sample:
        summary["map_matching"]["matched_path_length"] = summarize_values(
            per_sample["matched_path_length"].to_numpy(dtype=float)
        )
    failures = per_sample.loc[~per_sample["match_success"], "match_failure_reason"].value_counts().head(10)
    summary["map_matching"]["top_failure_reasons"] = failures.to_dict()
    return summary


def summarize_values(values: np.ndarray) -> dict[str, float]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {key: float("nan") for key in ["mean", "median", "std", "min", "p95", "max"]}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


def make_plots(
    per_sample: pd.DataFrame,
    trajectories: dict[str, dict[str, np.ndarray | None]],
    plots_dir: Path,
) -> None:
    per_sample_dir = plots_dir / "per_sample"
    per_sample_with_roads_dir = plots_dir / "per_sample_with_roads"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    per_sample_with_roads_dir.mkdir(parents=True, exist_ok=True)
    rows_with_paths = []
    for row in per_sample.itertuples(index=False):
        sample_id = str(row.sample_id)
        path = per_sample_dir / f"{sample_id}_trajectory_comparison.png"
        road_path = per_sample_with_roads_dir / f"{sample_id}_trajectory_with_roads.png"
        plot_sample_comparison(row, trajectories[sample_id], path, include_roads=False)
        plot_sample_comparison(row, trajectories[sample_id], road_path, include_roads=True)
        row_dict = row._asdict()
        row_dict["plot_path"] = path
        row_dict["plot_with_roads_path"] = road_path
        rows_with_paths.append(row_dict)
    plot_frame = pd.DataFrame(rows_with_paths)
    make_group_contact_sheets(plot_frame, plots_dir, path_column="plot_path", suffix="")
    make_group_contact_sheets(plot_frame, plots_dir, path_column="plot_with_roads_path", suffix="_with_roads")
    plot_metric_distributions(per_sample, plots_dir)
    plot_plain_vs_matched_scatter(per_sample, "ADE", plots_dir / "plain_vs_matched_ADE_scatter.png")
    plot_plain_vs_matched_scatter(per_sample, "FDE", plots_dir / "plain_vs_matched_FDE_scatter.png")
    plot_road_distance_distribution(per_sample, plots_dir / "plain_to_road_distance_distribution.png")


def plot_sample_comparison(
    row: Any,
    trajectory_set: dict[str, np.ndarray | None],
    path: Path,
    include_roads: bool,
) -> None:
    true_abs = trajectory_set["ground_truth"]
    plain = trajectory_set["plain"]
    matched = trajectory_set["matched"]
    road_lines = trajectory_set.get("road_lines") or []
    matched_path_lines = trajectory_set.get("matched_path_lines") or []
    fig, axis = plt.subplots(figsize=(6, 6))
    legend_labels = set()
    if include_roads:
        for line in road_lines:
            label = "OSM road network" if "OSM road network" not in legend_labels else None
            axis.plot(line[:, 0], line[:, 1], color="0.82", linewidth=0.55, alpha=0.75, label=label, zorder=1)
            if label:
                legend_labels.add(label)
    axis.plot(true_abs[:, 0], true_abs[:, 1], label="ground truth", linewidth=1.8)
    axis.plot(plain[:, 0], plain[:, 1], label="plain prediction", linewidth=1.3)
    if include_roads and matched_path_lines:
        for line in matched_path_lines:
            label = "matched path" if "matched path" not in legend_labels else None
            axis.plot(line[:, 0], line[:, 1], color="tab:purple", linewidth=1.8, alpha=0.85, label=label, zorder=3)
            if label:
                legend_labels.add(label)
    if matched is not None:
        axis.plot(matched[:, 0], matched[:, 1], label="matched points", linewidth=1.3, zorder=4)
    axis.scatter(true_abs[0, 0], true_abs[0, 1], s=28, marker="o", label="start")
    axis.scatter(true_abs[-1, 0], true_abs[-1, 1], s=28, marker="s", label="gt end")
    axis.scatter(plain[-1, 0], plain[-1, 1], s=30, marker="^", label="plain end")
    if matched is not None:
        axis.scatter(matched[-1, 0], matched[-1, 1], s=34, marker="x", label="matched end")
        all_points = np.vstack([true_abs, plain, matched])
        matched_text = f"{row.matched_ADE:.2f}/{row.matched_FDE:.2f}"
    else:
        all_points = np.vstack([true_abs, plain])
        matched_text = "unavailable"
    set_equal_xy_limits(axis, all_points)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{row.sample_id}\n"
        f"plain ADE/FDE={row.plain_ADE:.2f}/{row.plain_FDE:.2f}; "
        f"matched={matched_text}\n"
        f"success={row.match_success} matches={row.number_of_matches}"
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_group_contact_sheets(plot_frame: pd.DataFrame, plots_dir: Path, path_column: str, suffix: str) -> None:
    sort_column = "matched_ADE" if plot_frame["matched_metrics_available"].any() else "plain_ADE"
    sorted_rows = plot_frame.sort_values(sort_column, na_position="last").reset_index(drop=True)
    groups = {
        "best": sorted_rows.head(10),
        "median": median_window(sorted_rows, 10),
        "worst": sorted_rows.tail(10).sort_values(sort_column, ascending=False, na_position="last"),
    }
    for group_name, group in groups.items():
        make_contact_sheet(
            [Path(path) for path in group[path_column]],
            plots_dir / f"{group_name}_10_grid{suffix}.png",
        )


def median_window(sorted_rows: pd.DataFrame, size: int) -> pd.DataFrame:
    if len(sorted_rows) <= size:
        return sorted_rows
    center = len(sorted_rows) // 2
    start = max(0, center - size // 2)
    end = min(len(sorted_rows), start + size)
    start = max(0, end - size)
    return sorted_rows.iloc[start:end]


def make_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    images = [plt.imread(path) for path in image_paths]
    cols = 5
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, image, image_path in zip(axes_array, images, image_paths):
        axis.imshow(image)
        axis.set_title(image_path.stem.replace("_trajectory_comparison", ""), fontsize=7)
        axis.axis("off")
    for axis in axes_array[len(images) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_distributions(per_sample: pd.DataFrame, plots_dir: Path) -> None:
    for metric in ["ADE", "FDE"]:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.hist(per_sample[f"plain_{metric}"].dropna(), bins=30, alpha=0.55, label="plain")
        matched_values = per_sample.loc[per_sample["matched_metrics_available"], f"matched_{metric}"].dropna()
        if len(matched_values):
            axis.hist(matched_values, bins=30, alpha=0.55, label="mappymatched")
        axis.set_xlabel(metric)
        axis.set_ylabel("count")
        axis.legend()
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_distribution_comparison.png", dpi=150)
        plt.close(fig)


def plot_plain_vs_matched_scatter(per_sample: pd.DataFrame, metric: str, path: Path) -> None:
    available = per_sample.loc[per_sample["matched_metrics_available"]].copy()
    fig, axis = plt.subplots(figsize=(5, 5))
    if len(available):
        x = available[f"plain_{metric}"]
        y = available[f"matched_{metric}"]
        axis.scatter(x, y, s=18, alpha=0.75)
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
    axis.set_xlabel(f"plain {metric}")
    axis.set_ylabel(f"mappymatched {metric}")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_road_distance_distribution(per_sample: pd.DataFrame, path: Path) -> None:
    if "mean_trace_to_match_distance" not in per_sample.columns:
        return
    values = per_sample["mean_trace_to_match_distance"].dropna()
    if values.empty:
        return
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=30)
    axis.set_xlabel("mean trace-to-road distance")
    axis.set_ylabel("count")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def set_equal_xy_limits(axis: plt.Axes, points: np.ndarray) -> None:
    pad_x = max(1.0, float(np.ptp(points[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(points[:, 1])) * 0.05)
    axis.set_xlim(float(points[:, 0].min() - pad_x), float(points[:, 0].max() + pad_x))
    axis.set_ylim(float(points[:, 1].min() - pad_y), float(points[:, 1].max() + pad_y))


def write_chinese_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    plain = summary["plain"]
    matched = summary["matched"]
    matching = summary["map_matching"]
    matched_line = (
        f"- matched ADE mean = {matched['ADE']['mean']:.4f}\n"
        f"- matched FDE mean = {matched['FDE']['mean']:.4f}\n"
        if matching["matched_metrics_available_count"]
        else "- matched ADE/FDE 不可用：mappymatch 结果没有可确认的一一 timestep 对齐 matched points。\n"
    )
    text = f"""Delta Mappymatch OSM 实验报告

实验说明
- 本实验使用 mappymatch 和 OSM，因此道路网络独立于 ground truth。
- 这不是之前的 nearest_gt_polyline oracle projection。
- 本实验不把 ground-truth trajectory 当作道路或投影线。
- 本实验是 whole-trace map matching，不是 step-wise delta projection。
- 它可以作为真实道路网络 post-processing baseline。
- plots/ 中的图在同一个 XY 坐标系中比较 ground truth、plain integrated prediction 和 OSM/mappymatched trajectory。
- per_sample_with_roads/ 额外显示 OSM road network 背景和 match_result.path 对应的 matched road path。
- 图中的 mapmatched line 当前表示逐 timestep 对齐的 snapped points；matched road path 和 OSM road background 也需要一起目视检查，因为 snapped points 可能在不同 road segments 之间跳动。
- ADE/FDE 的改善可能比较有限，即使 road adherence 变好；因为道路吸附不一定等价于逐 timestep 对齐的真实轨迹误差下降。

输入与设置
- split = {summary['split']}
- 样本数 = {summary['num_samples']}
- padding_meters = {summary['padding_meters']}
- xy_to_lonlat_json = {args.xy_to_lonlat_json}

Plain delta reconstruction 结果
- plain ADE mean = {plain['ADE']['mean']:.4f}
- plain FDE mean = {plain['FDE']['mean']:.4f}
- plain RMSE mean = {plain['RMSE']['mean']:.4f}
- plain MAE mean = {plain['MAE']['mean']:.4f}

Mappymatch/OSM 诊断
- match success count = {matching['success_count']}
- match failure count = {matching['failure_count']}
- match success rate = {matching['success_rate']:.4f}
- matched metrics available count = {matching['matched_metrics_available_count']}
{matched_line}

解释建议
- 如果 mappymatch 失败，常见原因包括 XY-to-lonlat transform 误差、地理位置不正确、OSM coverage 问题，或 predicted trace 离真实道路太远。
- 如果匹配成功但无法得到逐 timestep matched coordinates，本脚本报告 matched_path_length、number_of_matches 等诊断，而不伪造 matched ADE/FDE。
- 若后续需要 matched ADE/FDE，需要明确从 mappymatch result 中提取与原始 224 个点一一对应的 matched lon/lat，再转换回 XY 或在 lon/lat 空间评估。
"""
    path.write_text(text, encoding="utf-8")


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    plain = summary["plain"]
    matched = summary["matched"]
    matching = summary["map_matching"]
    print("Delta mappymatch OSM summary")
    print(f"number of samples: {summary['num_samples']}")
    print(f"plain ADE/FDE: {plain['ADE']['mean']:.4f} / {plain['FDE']['mean']:.4f}")
    print(f"plain RMSE/MAE: {plain['RMSE']['mean']:.4f} / {plain['MAE']['mean']:.4f}")
    if matching["matched_metrics_available_count"]:
        print(f"matched ADE/FDE: {matched['ADE']['mean']:.4f} / {matched['FDE']['mean']:.4f}")
    else:
        print("matched ADE/FDE: unavailable (no timestep-aligned matched points extracted)")
    print(f"matched metrics availability count: {matching['matched_metrics_available_count']}")
    print(
        "match success/failure/rate: "
        f"{matching['success_count']} / {matching['failure_count']} / {matching['success_rate']:.4f}"
    )
    print(f"output directory: {output_dir}")
    print(f"plots directory: {output_dir / 'plots'}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

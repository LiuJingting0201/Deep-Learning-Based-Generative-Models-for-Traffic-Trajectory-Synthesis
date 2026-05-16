"""Fit an affine transform from local XY coordinates to longitude/latitude."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    raw = read_raw_table(args.raw_csv)
    frame = clean_frame(raw)
    transform = fit_transform(frame)
    diagnostics = evaluate_transform(frame, transform)

    output = {
        "coordinate_type": "affine_xy_to_lonlat",
        "lon_coefficients": transform["lon_coefficients"],
        "lat_coefficients": transform["lat_coefficients"],
    }
    write_json(args.output_json, output)
    write_json(args.output_report, {"transform": output, "diagnostics": diagnostics})
    print_summary(diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()
    if args.output_report is None:
        args.output_report = args.output_json.with_name(args.output_json.stem + "_report.json")
    return args


def read_raw_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_excel(path)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["x", "y", "lon", "lat"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Raw file missing required columns: {sorted(missing)}")
    cleaned = frame[required].copy()
    for column in required:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna().reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid x/y/lon/lat rows after dropping missing values.")
    return cleaned


def fit_transform(frame: pd.DataFrame) -> dict[str, list[float]]:
    design = np.column_stack(
        [
            np.ones(len(frame), dtype=np.float64),
            frame["x"].to_numpy(dtype=np.float64),
            frame["y"].to_numpy(dtype=np.float64),
        ]
    )
    lon_coefficients, *_ = np.linalg.lstsq(design, frame["lon"].to_numpy(dtype=np.float64), rcond=None)
    lat_coefficients, *_ = np.linalg.lstsq(design, frame["lat"].to_numpy(dtype=np.float64), rcond=None)
    return {
        "lon_coefficients": lon_coefficients.astype(float).tolist(),
        "lat_coefficients": lat_coefficients.astype(float).tolist(),
    }


def evaluate_transform(frame: pd.DataFrame, transform: dict[str, list[float]]) -> dict[str, Any]:
    xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
    predicted = xy_to_lonlat(xy, transform)
    true_lon = frame["lon"].to_numpy(dtype=np.float64)
    true_lat = frame["lat"].to_numpy(dtype=np.float64)
    lon_error = predicted[:, 0] - true_lon
    lat_error = predicted[:, 1] - true_lat
    meter_error = approximate_meter_error(lon_error, lat_error, true_lat)
    return {
        "num_points": int(len(frame)),
        "lon_abs_error": summarize_abs(lon_error),
        "lat_abs_error": summarize_abs(lat_error),
        "lon_RMSE": float(np.sqrt(np.mean(lon_error**2))),
        "lat_RMSE": float(np.sqrt(np.mean(lat_error**2))),
        "approx_meter_error": summarize_abs(meter_error),
        "approx_meter_RMSE": float(np.sqrt(np.mean(meter_error**2))),
    }


def xy_to_lonlat(xy: np.ndarray, transform: dict[str, list[float]]) -> np.ndarray:
    design = np.column_stack([np.ones(len(xy), dtype=np.float64), xy[:, 0], xy[:, 1]])
    lon = design @ np.asarray(transform["lon_coefficients"], dtype=np.float64)
    lat = design @ np.asarray(transform["lat_coefficients"], dtype=np.float64)
    return np.column_stack([lon, lat])


def approximate_meter_error(lon_error: np.ndarray, lat_error: np.ndarray, lat: np.ndarray) -> np.ndarray:
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.deg2rad(float(np.nanmean(lat))))
    dx = lon_error * meters_per_degree_lon
    dy = lat_error * meters_per_degree_lat
    return np.sqrt(dx**2 + dy**2)


def summarize_abs(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values.astype(np.float64))
    return {
        "mean": float(np.mean(abs_values)),
        "median": float(np.median(abs_values)),
        "p95": float(np.percentile(abs_values, 95)),
        "max": float(np.max(abs_values)),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(diagnostics: dict[str, Any]) -> None:
    print("XY -> lon/lat affine transform summary")
    print(f"number of points: {diagnostics['num_points']}")
    print(
        "lon MAE/RMSE: "
        f"{diagnostics['lon_abs_error']['mean']:.10f} / {diagnostics['lon_RMSE']:.10f}"
    )
    print(
        "lat MAE/RMSE: "
        f"{diagnostics['lat_abs_error']['mean']:.10f} / {diagnostics['lat_RMSE']:.10f}"
    )
    print(
        "approx meter error mean/p95: "
        f"{diagnostics['approx_meter_error']['mean']:.4f} / {diagnostics['approx_meter_error']['p95']:.4f}"
    )


if __name__ == "__main__":
    main()

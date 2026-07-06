"""Compute train-split normalization diagnostics for 224-step absolute positions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "absolute_position_normalization_diagnostics.json"
DIAGNOSTIC_PERCENTILES = [0, 0.5, 1, 2, 5, 25, 50, 75, 95, 98, 99, 99.5, 100]
SIGMOID_K_VALUES = [1.0, 1.5, 2.0]
NORMALIZED_RATIO_INTERVALS = [
    (0.0, 0.01),
    (0.0, 0.02),
    (0.0, 0.05),
    (0.45, 0.55),
    (0.4, 0.6),
    (0.25, 0.75),
    (0.95, 1.0),
    (0.98, 1.0),
    (0.99, 1.0),
]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sample_ids = read_sample_ids(data_root / "splits" / f"{args.split}_ids.csv")
    values = collect_absolute_values(data_root / "labels_absolute", sample_ids)
    diagnostics = build_diagnostics(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"data_root: {data_root}")
    print(f"split: {args.split}")
    print(f"num_samples: {len(sample_ids)}")
    print(f"num_values: {int(values.shape[0])}")
    print(f"output: {output}")
    print(f"x_mean: {diagnostics['x_mean']:.10g}")
    print(f"x_std: {diagnostics['x_std']:.10g}")
    print(f"y_mean: {diagnostics['y_mean']:.10g}")
    print(f"y_std: {diagnostics['y_std']:.10g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    return parser.parse_args()


def read_sample_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a sample_id column.")
        return [row["sample_id"] for row in reader if row.get("sample_id")]


def collect_absolute_values(labels_dir: Path, sample_ids: list[str]) -> np.ndarray:
    arrays = []
    for sample_id in sample_ids:
        path = labels_dir / f"{sample_id}.npy"
        xy = np.load(path).astype(np.float64)
        if xy.shape != (224, 2):
            raise ValueError(f"Expected shape (224, 2), got {xy.shape} in {path}")
        arrays.append(xy)
    if not arrays:
        raise ValueError("No absolute-position arrays found.")
    return np.concatenate(arrays, axis=0)


def build_diagnostics(values: np.ndarray) -> dict[str, object]:
    x = values[:, 0]
    y = values[:, 1]
    payload: dict[str, object] = {
        "normalization_source": "train_split_absolute_positions",
        "num_train_values": int(values.shape[0]),
        "percentiles": DIAGNOSTIC_PERCENTILES,
        "sigmoid_k_values": SIGMOID_K_VALUES,
        "sigmoid_definition": "sigmoid(k * z), where z = (value - train_mean) / train_std",
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "x_mean": float(x.mean()),
        "x_std": float(x.std()),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "x_percentiles": percentiles(x),
        "y_percentiles": percentiles(y),
    }
    for k_value in SIGMOID_K_VALUES:
        x_norm = sigmoid_normalize(x, payload["x_mean"], payload["x_std"], k_value)
        y_norm = sigmoid_normalize(y, payload["y_mean"], payload["y_std"], k_value)
        payload[f"sigmoid_k_{k_value:g}"] = {
            "x_norm_percentiles": percentiles(x_norm),
            "y_norm_percentiles": percentiles(y_norm),
            "x_norm_interval_ratios": interval_ratios(x_norm),
            "y_norm_interval_ratios": interval_ratios(y_norm),
        }
    return payload


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {f"{percentile:g}": float(np.percentile(values, percentile)) for percentile in DIAGNOSTIC_PERCENTILES}


def sigmoid_normalize(values: np.ndarray, mean: float, std: float, k_value: float) -> np.ndarray:
    if std == 0.0:
        raise ValueError("Cannot apply sigmoid normalization with zero std.")
    logits = np.clip(float(k_value) * ((values - float(mean)) / float(std)), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def interval_ratios(values: np.ndarray) -> dict[str, float]:
    ratios = {}
    for lower, upper in NORMALIZED_RATIO_INTERVALS:
        ratios[f"[{lower:g},{upper:g}]"] = float(np.mean((values >= lower) & (values <= upper)))
    return ratios


if __name__ == "__main__":
    main()

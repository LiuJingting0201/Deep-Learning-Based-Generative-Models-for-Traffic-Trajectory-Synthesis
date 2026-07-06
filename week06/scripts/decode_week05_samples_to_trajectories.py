"""Decode Week05 generated GASF float arrays back to raw XY trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / "week05"
    / "results_npy_samples"
    / "week05_sig15_mse_diag"
    / "week05_sig15_mse_diag_step_040000_samples_1000.npy"
)
DEFAULT_CONFIG = PROJECT_ROOT / "week05" / "results" / "week05_sig15_mse_diag" / "startup_config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    normalization = config["normalization_stats"]
    position_stats = config["position_stats"]
    sigmoid_k = float(args.sigmoid_k if args.sigmoid_k is not None else config["sigmoid_k"])

    output_dir = args.output_dir.resolve()
    trajectories_dir = output_dir / "trajectories_raw_xy"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    samples = np.load(args.samples, mmap_mode="r")
    validate_samples(samples)
    num_samples = min(int(samples.shape[0]), args.max_samples) if args.max_samples is not None else int(samples.shape[0])
    sequence_length = int(samples.shape[1])

    starts = np.zeros((num_samples, 2), dtype=np.float32)
    deltas = np.zeros((num_samples, sequence_length, 2), dtype=np.float32)
    metrics_rows = []

    for index in range(num_samples):
        image = np.asarray(samples[index], dtype=np.float32)
        decoded = decode_one_image(
            image=image,
            normalization=normalization,
            position_stats=position_stats,
            sigmoid_k=sigmoid_k,
            start_mode=args.start_mode,
            softargmax_temperature=args.softargmax_temperature,
            eps=args.eps,
        )
        trajectory = integrate_delta(decoded["start_xy"], decoded["delta_xy"])
        sample_name = f"generated_{index:06d}"
        np.save(trajectories_dir / f"{sample_name}.npy", trajectory.astype(np.float32))
        starts[index] = decoded["start_xy"]
        deltas[index] = decoded["delta_xy"]
        metrics_rows.append(metric_row(sample_name, decoded, trajectory, position_stats))

    np.save(output_dir / "starts_raw_xy.npy", starts)
    np.save(output_dir / "delta_raw_xy.npy", deltas)
    write_csv(output_dir / "decode_metrics.csv", metrics_rows)

    summary = {
        "samples_path": str(args.samples.resolve()),
        "config_path": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "num_samples": num_samples,
        "sequence_length": sequence_length,
        "sigmoid_k": sigmoid_k,
        "start_mode": args.start_mode,
        "softargmax_temperature": args.softargmax_temperature,
        "normalization_stats": normalization,
        "position_stats": position_stats,
        "decode_metrics": summarize_decode_metrics(metrics_rows),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def decode_one_image(
    image: np.ndarray,
    normalization: dict[str, float],
    position_stats: dict[str, float],
    sigmoid_k: float,
    start_mode: str,
    softargmax_temperature: float,
    eps: float,
) -> dict[str, np.ndarray | float]:
    dx = decode_gasf_channel_01(image[:, :, 0], normalization["dx_mean"], normalization["dx_std"], sigmoid_k, eps)
    dy = decode_gasf_channel_01(image[:, :, 1], normalization["dy_mean"], normalization["dy_std"], sigmoid_k, eps)
    delta_xy = np.stack([dx, dy], axis=-1).astype(np.float32)
    delta_xy[0] = 0.0
    start_xy, heatmap_peak, heatmap_mass = decode_start_heatmap(
        image[:, :, 2],
        position_stats=position_stats,
        mode=start_mode,
        softargmax_temperature=softargmax_temperature,
        eps=eps,
    )
    return {
        "delta_xy": delta_xy,
        "start_xy": start_xy.astype(np.float32),
        "heatmap_peak": float(heatmap_peak),
        "heatmap_mass": float(heatmap_mass),
    }


def decode_gasf_channel_01(channel_01: np.ndarray, mean: float, std: float, sigmoid_k: float, eps: float) -> np.ndarray:
    diag = np.diagonal(np.clip(channel_01, 0.0, 1.0)).astype(np.float64)
    u = np.sqrt(np.clip(diag, eps, 1.0 - eps))
    logits = np.log(u / (1.0 - u))
    values = float(mean) + float(std) * logits / float(sigmoid_k)
    return values.astype(np.float32)


def decode_start_heatmap(
    heatmap: np.ndarray,
    position_stats: dict[str, float],
    mode: str,
    softargmax_temperature: float,
    eps: float,
) -> tuple[np.ndarray, float, float]:
    heatmap = np.clip(heatmap.astype(np.float64), 0.0, None)
    height, width = heatmap.shape
    peak = float(np.max(heatmap))
    mass = float(np.sum(heatmap))
    if mode == "argmax" or mass <= eps:
        y_index, x_index = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        px = float(x_index)
        py = float(y_index)
    else:
        weights = np.exp((heatmap - peak) / max(float(softargmax_temperature), eps))
        weights_sum = float(weights.sum())
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        px = float((weights * grid_x).sum() / weights_sum)
        py = float((weights * grid_y).sum() / weights_sum)
    x_norm = px / max(width - 1, 1)
    y_norm = py / max(height - 1, 1)
    start_xy = np.array(
        [
            denormalize_position(x_norm, position_stats["x_min"], position_stats["x_max"]),
            denormalize_position(y_norm, position_stats["y_min"], position_stats["y_max"]),
        ],
        dtype=np.float32,
    )
    return start_xy, peak, mass


def denormalize_position(value: float, minimum: float, maximum: float) -> float:
    return float(minimum) + float(np.clip(value, 0.0, 1.0)) * (float(maximum) - float(minimum))


def integrate_delta(start_xy: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    trajectory = np.zeros_like(delta_xy, dtype=np.float32)
    trajectory[0] = start_xy.astype(np.float32)
    trajectory[1:] = start_xy.astype(np.float32) + np.cumsum(delta_xy[1:], axis=0)
    return trajectory


def metric_row(
    sample_name: str,
    decoded: dict[str, np.ndarray | float],
    trajectory: np.ndarray,
    position_stats: dict[str, float],
) -> dict[str, Any]:
    delta_xy = decoded["delta_xy"]
    step_lengths = np.linalg.norm(delta_xy[1:], axis=1)
    range_mask = (
        (trajectory[:, 0] < position_stats["x_min"])
        | (trajectory[:, 0] > position_stats["x_max"])
        | (trajectory[:, 1] < position_stats["y_min"])
        | (trajectory[:, 1] > position_stats["y_max"])
    )
    return {
        "sample_id": sample_name,
        "start_x": float(trajectory[0, 0]),
        "start_y": float(trajectory[0, 1]),
        "end_x": float(trajectory[-1, 0]),
        "end_y": float(trajectory[-1, 1]),
        "heatmap_peak": decoded["heatmap_peak"],
        "heatmap_mass": decoded["heatmap_mass"],
        "mean_step_length": float(np.mean(step_lengths)),
        "p95_step_length": float(np.percentile(step_lengths, 95)),
        "max_step_length": float(np.max(step_lengths)),
        "path_length": float(np.sum(step_lengths)),
        "displacement": float(np.linalg.norm(trajectory[-1] - trajectory[0])),
        "bbox_width": float(np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])),
        "bbox_height": float(np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])),
        "out_of_position_range_ratio": float(np.mean(range_mask)),
    }


def summarize_decode_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "heatmap_peak",
        "heatmap_mass",
        "mean_step_length",
        "p95_step_length",
        "max_step_length",
        "path_length",
        "displacement",
        "bbox_width",
        "bbox_height",
        "out_of_position_range_ratio",
    ]
    return {key: summarize(np.asarray([row[key] for row in rows], dtype=float)) for key in keys}


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def validate_samples(samples: np.ndarray) -> None:
    if samples.ndim != 4:
        raise ValueError(f"Expected samples with shape N,H,W,C, got {samples.shape}")
    if samples.shape[1] != samples.shape[2]:
        raise ValueError(f"Expected square GASF samples, got {samples.shape}")
    if samples.shape[3] < 3:
        raise ValueError(f"Expected at least 3 channels, got {samples.shape}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sigmoid-k", type=float, default=None)
    parser.add_argument("--start-mode", choices=("softargmax", "argmax"), default="softargmax")
    parser.add_argument("--softargmax-temperature", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

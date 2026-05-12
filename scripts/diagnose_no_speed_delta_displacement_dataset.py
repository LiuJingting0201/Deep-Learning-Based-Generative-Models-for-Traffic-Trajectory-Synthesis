"""Run diagnostics for the no-speed delta-displacement paired dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    diagnostics_dir = data_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_root / "metadata.csv")
    valid_metadata = metadata[metadata["valid_or_skipped"] == "valid"].copy()
    if valid_metadata.empty:
        raise ValueError("No valid samples found in metadata.csv")

    deltas = load_delta_labels(data_root, valid_metadata)
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    magnitudes = np.linalg.norm(deltas, axis=1)

    normalization = read_normalization(data_root, metadata, dx, dy)
    normalized_dx = minmax_normalize(dx, normalization["delta_x_global_min"], normalization["delta_x_global_max"])
    normalized_dy = minmax_normalize(dy, normalization["delta_y_global_min"], normalization["delta_y_global_max"])

    symmetric_scale = float(max(np.max(np.abs(dx)), np.max(np.abs(dy))))
    symmetric_dx = symmetric_maxabs_normalize(dx, symmetric_scale)
    symmetric_dy = symmetric_maxabs_normalize(dy, symmetric_scale)

    summary = {
        "data_root": str(data_root),
        "num_samples": int(len(valid_metadata)),
        "num_delta_values": int(len(dx)),
        "zero_ratios": {
            "dx_exact_zero_ratio": ratio(dx == 0.0),
            "dy_exact_zero_ratio": ratio(dy == 0.0),
            "either_axis_exact_zero_ratio": ratio((dx == 0.0) | (dy == 0.0)),
            "both_axes_exact_zero_ratio": ratio((dx == 0.0) & (dy == 0.0)),
            "magnitude_exact_zero_ratio": ratio(magnitudes == 0.0),
            "dx_near_zero_ratio_eps_1e-6": ratio(np.abs(dx) <= 1e-6),
            "dy_near_zero_ratio_eps_1e-6": ratio(np.abs(dy) <= 1e-6),
            "magnitude_near_zero_ratio_eps_1e-6": ratio(magnitudes <= 1e-6),
            "dx_near_zero_ratio_eps_1e-3": ratio(np.abs(dx) <= 1e-3),
            "dy_near_zero_ratio_eps_1e-3": ratio(np.abs(dy) <= 1e-3),
            "magnitude_near_zero_ratio_eps_1e-3": ratio(magnitudes <= 1e-3),
        },
        "delta_magnitude_stats": stats(magnitudes),
        "dx_stats": stats(dx),
        "dy_stats": stats(dy),
        "normalization": {
            **normalization,
            "global_minmax_zero_maps_to": {
                "dx": zero_minmax(normalization["delta_x_global_min"], normalization["delta_x_global_max"]),
                "dy": zero_minmax(normalization["delta_y_global_min"], normalization["delta_y_global_max"]),
            },
            "symmetric_maxabs": {
                "scale": symmetric_scale,
                "zero_maps_to_dx": 0.5,
                "zero_maps_to_dy": 0.5,
                "method": "(value / max_abs_global + 1) / 2, using one global max_abs over dx and dy",
            },
        },
        "saturation": {
            "thresholds": {
                "near_0": args.saturation_eps,
                "near_1": 1.0 - args.saturation_eps,
            },
            "global_minmax": saturation_summary(normalized_dx, normalized_dy, args.saturation_eps),
            "symmetric_maxabs": saturation_summary(symmetric_dx, symmetric_dy, args.saturation_eps),
        },
    }

    write_histogram_csv(diagnostics_dir / "dx_histogram.csv", dx, args.bins, "dx")
    write_histogram_csv(diagnostics_dir / "dy_histogram.csv", dy, args.bins, "dy")
    write_histogram_csv(
        diagnostics_dir / "normalized_dx_histogram.csv",
        normalized_dx,
        args.bins,
        "normalized_dx_global_minmax",
        value_range=(0.0, 1.0),
    )
    write_histogram_csv(
        diagnostics_dir / "normalized_dy_histogram.csv",
        normalized_dy,
        args.bins,
        "normalized_dy_global_minmax",
        value_range=(0.0, 1.0),
    )

    plot_histograms(diagnostics_dir, dx, dy, normalized_dx, normalized_dy, args.bins)
    with (diagnostics_dir / "diagnostics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print_summary(summary, diagnostics_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_no_speed_delta_displacement_paired"),
    )
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument(
        "--saturation-eps",
        type=float,
        default=1e-3,
        help="Count normalized values <= eps or >= 1 - eps as near saturation.",
    )
    return parser.parse_args()


def load_delta_labels(data_root: Path, metadata: pd.DataFrame) -> np.ndarray:
    arrays = []
    for row in metadata.itertuples(index=False):
        path = data_root / row.label_delta_displacement_path
        array = np.load(path)
        if array.shape != (224, 2):
            raise ValueError(f"Unexpected delta label shape for {row.sample_id}: {array.shape}")
        arrays.append(array.astype(np.float64, copy=False))
    return np.concatenate(arrays, axis=0)


def read_normalization(
    data_root: Path, metadata: pd.DataFrame, dx: np.ndarray, dy: np.ndarray
) -> dict[str, float | str]:
    path = data_root / "encoding_normalization.json"
    if path.exists():
        payload = json.load(path.open())
        return {
            "delta_x_global_min": float(payload["delta_x_global_min"]),
            "delta_x_global_max": float(payload["delta_x_global_max"]),
            "delta_y_global_min": float(payload["delta_y_global_min"]),
            "delta_y_global_max": float(payload["delta_y_global_max"]),
            "normalization_method": str(payload.get("normalization_method", "unknown")),
        }

    return {
        "delta_x_global_min": float(dx.min()),
        "delta_x_global_max": float(dx.max()),
        "delta_y_global_min": float(dy.min()),
        "delta_y_global_max": float(dy.max()),
        "normalization_method": "recomputed_dataset_level_min_max_to_unit_interval",
    }


def minmax_normalize(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum == minimum:
        raise ValueError("Cannot min/max normalize a zero-width range.")
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def symmetric_maxabs_normalize(values: np.ndarray, scale: float) -> np.ndarray:
    if scale == 0.0:
        raise ValueError("Cannot max-abs normalize with zero scale.")
    return np.clip((values / scale + 1.0) / 2.0, 0.0, 1.0)


def ratio(mask: np.ndarray) -> float:
    return float(np.mean(mask))


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def zero_minmax(minimum: float, maximum: float) -> float:
    return float((0.0 - minimum) / (maximum - minimum))


def saturation_summary(
    normalized_dx: np.ndarray, normalized_dy: np.ndarray, eps: float
) -> dict[str, float | int]:
    dx_low = normalized_dx <= eps
    dx_high = normalized_dx >= 1.0 - eps
    dy_low = normalized_dy <= eps
    dy_high = normalized_dy >= 1.0 - eps
    any_axis = dx_low | dx_high | dy_low | dy_high
    return {
        "dx_near_0_count": int(dx_low.sum()),
        "dx_near_0_ratio": ratio(dx_low),
        "dx_near_1_count": int(dx_high.sum()),
        "dx_near_1_ratio": ratio(dx_high),
        "dy_near_0_count": int(dy_low.sum()),
        "dy_near_0_ratio": ratio(dy_low),
        "dy_near_1_count": int(dy_high.sum()),
        "dy_near_1_ratio": ratio(dy_high),
        "any_axis_near_0_or_1_count": int(any_axis.sum()),
        "any_axis_near_0_or_1_ratio": ratio(any_axis),
    }


def write_histogram_csv(
    path: Path,
    values: np.ndarray,
    bins: int,
    value_name: str,
    value_range: tuple[float, float] | None = None,
) -> None:
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    frame = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "ratio": counts / counts.sum(),
            "value": value_name,
        }
    )
    frame.to_csv(path, index=False)


def plot_histograms(
    diagnostics_dir: Path,
    dx: np.ndarray,
    dy: np.ndarray,
    normalized_dx: np.ndarray,
    normalized_dy: np.ndarray,
    bins: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dx, bins=bins, alpha=0.7, label="dx")
    axes[0].hist(dy, bins=bins, alpha=0.7, label="dy")
    axes[0].set_title("delta displacement histogram")
    axes[0].legend()
    axes[1].hist(normalized_dx, bins=bins, alpha=0.7, label="normalized dx")
    axes[1].hist(normalized_dy, bins=bins, alpha=0.7, label="normalized dy")
    axes[1].set_title("global min/max normalized histogram")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "delta_and_normalized_histograms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(dx, bins=bins)
    axes[0, 0].set_title("dx")
    axes[0, 1].hist(dy, bins=bins)
    axes[0, 1].set_title("dy")
    axes[1, 0].hist(normalized_dx, bins=bins, range=(0.0, 1.0))
    axes[1, 0].set_title("normalized dx")
    axes[1, 1].hist(normalized_dy, bins=bins, range=(0.0, 1.0))
    axes[1, 1].set_title("normalized dy")
    fig.tight_layout()
    fig.savefig(diagnostics_dir / "histograms_grid.png", dpi=150)
    plt.close(fig)


def print_summary(summary: dict[str, object], diagnostics_dir: Path) -> None:
    zero = summary["zero_ratios"]
    mag = summary["delta_magnitude_stats"]
    norm = summary["normalization"]
    sat = summary["saturation"]["global_minmax"]

    print(f"Saved diagnostics to {diagnostics_dir}")
    print(f"Samples: {summary['num_samples']}; delta timesteps: {summary['num_delta_values']}")
    print(
        "Zero ratios: "
        f"both exact={zero['both_axes_exact_zero_ratio']:.6f}, "
        f"magnitude<=1e-6={zero['magnitude_near_zero_ratio_eps_1e-6']:.6f}, "
        f"magnitude<=1e-3={zero['magnitude_near_zero_ratio_eps_1e-3']:.6f}"
    )
    print(
        "Delta magnitude: "
        f"mean={mag['mean']:.4f}, median={mag['median']:.4f}, "
        f"p95={mag['p95']:.4f}, p99={mag['p99']:.4f}, max={mag['max']:.4f}"
    )
    print(
        "Zero maps under current global min/max: "
        f"dx={norm['global_minmax_zero_maps_to']['dx']:.6f}, "
        f"dy={norm['global_minmax_zero_maps_to']['dy']:.6f}; "
        "under symmetric max-abs: dx=0.500000, dy=0.500000"
    )
    print(
        "Global min/max saturation at eps=1e-3: "
        f"dx near 0={sat['dx_near_0_count']}, dx near 1={sat['dx_near_1_count']}, "
        f"dy near 0={sat['dy_near_0_count']}, dy near 1={sat['dy_near_1_count']}, "
        f"any-axis ratio={sat['any_axis_near_0_or_1_ratio']:.6f}"
    )

    looks_safe = (
        zero["magnitude_near_zero_ratio_eps_1e-3"] < 0.02
        and sat["any_axis_near_0_or_1_ratio"] < 0.01
        and mag["max"] < 100.0
    )
    if looks_safe:
        print("Training safety read: looks safe to train; no obvious zero-collapse or saturation issue.")
    else:
        print("Training safety read: review diagnostics before training; one or more heuristic checks flagged.")


if __name__ == "__main__":
    main()

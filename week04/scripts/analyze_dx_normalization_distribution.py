"""Analyze train-split dx and globally normalized dx distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path("/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "sanity_checks" / "dx_normalization_distribution"


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = read_train_ids(data_root)
    stats = read_normalization(data_root)
    dx_values = load_train_dx(data_root, train_ids)
    dx_norm = normalize_to_unit(dx_values, stats["dx_min"], stats["dx_max"])

    summary = summarize(dx_values, dx_norm, args.epsilon)
    write_summary(output_dir / "dx_normalization_summary.json", data_root, train_ids, stats, summary)
    write_csv(output_dir / "dx_normalization_summary.csv", summary)
    plot_histograms(output_dir, dx_values, dx_norm, args.epsilon)

    print(f"Train samples: {len(train_ids)}")
    print(f"dx count: {len(dx_values)}")
    print(f"dx_min={stats['dx_min']:.12g}, dx_max={stats['dx_max']:.12g}")
    print(f"(dx_norm < 0.5).mean()={summary['frac_dx_norm_lt_0p5']:.12g}")
    print(f"(dx_norm > 0.5).mean()={summary['frac_dx_norm_gt_0p5']:.12g}")
    print(f"(abs(dx) < epsilon).mean()={summary['frac_abs_dx_lt_epsilon']:.12g}")
    print(f"epsilon={args.epsilon:.12g}")
    print(f"Output dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    return parser.parse_args()


def read_train_ids(data_root: Path) -> list[str]:
    split_path = data_root / "splits" / "train_ids.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{split_path} must contain a sample_id column")
        return [row["sample_id"] for row in reader if row.get("sample_id")]


def read_normalization(data_root: Path) -> dict[str, float]:
    path = data_root / "constrained_gasf_normalization.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    return {"dx_min": float(payload["dx_min"]), "dx_max": float(payload["dx_max"])}


def load_train_dx(data_root: Path, sample_ids: list[str]) -> np.ndarray:
    dx_chunks: list[np.ndarray] = []
    labels_dir = data_root / "labels_delta_displacement"
    for sample_id in sample_ids:
        path = labels_dir / f"{sample_id}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        delta_xy = np.load(path).astype(np.float64)
        if delta_xy.ndim != 2 or delta_xy.shape[1] != 2:
            raise ValueError(f"{path} has unexpected shape {delta_xy.shape}")
        dx_chunks.append(delta_xy[:, 0])
    if not dx_chunks:
        raise ValueError("No train dx values found.")
    return np.concatenate(dx_chunks)


def normalize_to_unit(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def summarize(dx_values: np.ndarray, dx_norm: np.ndarray, epsilon: float) -> dict[str, Any]:
    near_half = np.abs(dx_norm - 0.5)
    return {
        "num_dx": int(len(dx_values)),
        "dx_min": float(np.min(dx_values)),
        "dx_max": float(np.max(dx_values)),
        "dx_mean": float(np.mean(dx_values)),
        "dx_std": float(np.std(dx_values)),
        "dx_median": float(np.median(dx_values)),
        "dx_norm_min": float(np.min(dx_norm)),
        "dx_norm_max": float(np.max(dx_norm)),
        "dx_norm_mean": float(np.mean(dx_norm)),
        "dx_norm_std": float(np.std(dx_norm)),
        "dx_norm_median": float(np.median(dx_norm)),
        "frac_dx_norm_lt_0p5": float(np.mean(dx_norm < 0.5)),
        "frac_dx_norm_gt_0p5": float(np.mean(dx_norm > 0.5)),
        "frac_dx_norm_eq_0p5": float(np.mean(dx_norm == 0.5)),
        "frac_abs_dx_lt_epsilon": float(np.mean(np.abs(dx_values) < epsilon)),
        "frac_dx_norm_within_0p01_of_0p5": float(np.mean(near_half <= 0.01)),
        "frac_dx_norm_within_0p02_of_0p5": float(np.mean(near_half <= 0.02)),
        "frac_dx_norm_within_0p05_of_0p5": float(np.mean(near_half <= 0.05)),
        "dx_quantiles": quantiles(dx_values),
        "dx_norm_quantiles": quantiles(dx_norm),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    return {f"q{q:g}": float(np.quantile(values, q)) for q in qs}


def write_summary(
    path: Path,
    data_root: Path,
    train_ids: list[str],
    stats: dict[str, float],
    summary: dict[str, Any],
) -> None:
    payload = {
        "data_root": str(data_root),
        "train_sample_count": len(train_ids),
        "normalization": stats,
        "summary": summary,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey}"] = subvalue
        else:
            flat[key] = value
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def plot_histograms(output_dir: Path, dx_values: np.ndarray, dx_norm: np.ndarray, epsilon: float) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(dx_values, bins=160, color="#2f6f9f", alpha=0.88)
    axis.axvline(0.0, color="#d1495b", linewidth=1.5, label="dx = 0")
    axis.axvspan(-epsilon, epsilon, color="#edae49", alpha=0.24, label=f"|dx| < {epsilon:g}")
    axis.set_title("Train dx distribution")
    axis.set_xlabel("dx")
    axis.set_ylabel("count")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_dx.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(dx_norm, bins=160, color="#287c71", alpha=0.88)
    axis.axvline(0.5, color="#d1495b", linewidth=1.5, label="dx_norm = 0.5")
    axis.axvspan(0.48, 0.52, color="#edae49", alpha=0.24, label="0.48 <= dx_norm <= 0.52")
    axis.set_title("Train dx_norm distribution")
    axis.set_xlabel("dx_norm")
    axis.set_ylabel("count")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_dx_norm.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

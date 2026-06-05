"""Check diagonal invertibility of the Week 4 [-1, 1] GASF ablation dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path("/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start_neg11")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    stats = read_normalization(data_root)
    sample_ids = sorted(path.stem for path in (data_root / "labels_delta_displacement").glob("*.npy"))
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]
    if not sample_ids:
        raise ValueError(f"No samples found under {data_root}")

    rows = [check_sample(data_root, sample_id, stats, args.sign_epsilon) for sample_id in sample_ids]
    output_csv = data_root / "sanity_checks" / "neg11_invertibility_metrics.csv"
    output_json = data_root / "sanity_checks" / "neg11_invertibility_summary.json"
    write_rows(output_csv, rows)
    summary = summarize(rows, data_root, args.sign_epsilon)
    write_summary(output_json, summary)

    print(f"Checked samples: {len(rows)}")
    print(f"Metrics CSV: {output_csv}")
    print(f"Summary JSON: {output_json}")
    print(f"mean_abs_diag_mae_dx={summary['mean_abs_diag_mae_dx']:.12g}")
    print(f"mean_abs_diag_mae_dy={summary['mean_abs_diag_mae_dy']:.12g}")
    print(f"mean_positive_branch_mae_dx={summary['mean_positive_branch_mae_dx']:.12g}")
    print(f"mean_positive_branch_mae_dy={summary['mean_positive_branch_mae_dy']:.12g}")
    print(f"mean_oracle_signed_mae_dx={summary['mean_oracle_signed_mae_dx']:.12g}")
    print(f"mean_oracle_signed_mae_dy={summary['mean_oracle_signed_mae_dy']:.12g}")
    print(f"fraction_negative_dx={summary['mean_fraction_negative_dx']:.12g}")
    print(f"fraction_negative_dy={summary['mean_fraction_negative_dy']:.12g}")
    print("Strict diagonal signed invertible: false")
    print("Reason: diagonal recovers abs(x), so sign is lost for nonzero negative normalized values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sign-epsilon", type=float, default=1e-8)
    return parser.parse_args()


def read_normalization(data_root: Path) -> dict[str, float]:
    path = data_root / "constrained_gasf_normalization.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    return {
        key: float(payload[key])
        for key in ["dx_min", "dx_max", "dy_min", "dy_max"]
    }


def check_sample(
    data_root: Path,
    sample_id: str,
    stats: dict[str, float],
    sign_epsilon: float,
) -> dict[str, Any]:
    delta_xy = np.load(data_root / "labels_delta_displacement" / f"{sample_id}.npy").astype(np.float64)
    encoded = np.load(data_root / "arrays" / f"{sample_id}.npy").astype(np.float64)

    dx_norm = normalize_to_neg11(delta_xy[:, 0], stats["dx_min"], stats["dx_max"])
    dy_norm = normalize_to_neg11(delta_xy[:, 1], stats["dy_min"], stats["dy_max"])
    abs_dx_rec = decode_abs_from_saved_channel(encoded[:, :, 0])
    abs_dy_rec = decode_abs_from_saved_channel(encoded[:, :, 1])

    dx_positive_branch = abs_dx_rec
    dy_positive_branch = abs_dy_rec
    dx_oracle_signed = np.sign(dx_norm) * abs_dx_rec
    dy_oracle_signed = np.sign(dy_norm) * abs_dy_rec

    return {
        "sample_id": sample_id,
        "abs_diag_mae_dx": mae(abs_dx_rec, np.abs(dx_norm)),
        "abs_diag_max_error_dx": max_abs_error(abs_dx_rec, np.abs(dx_norm)),
        "abs_diag_mae_dy": mae(abs_dy_rec, np.abs(dy_norm)),
        "abs_diag_max_error_dy": max_abs_error(abs_dy_rec, np.abs(dy_norm)),
        "positive_branch_mae_dx": mae(dx_positive_branch, dx_norm),
        "positive_branch_max_error_dx": max_abs_error(dx_positive_branch, dx_norm),
        "positive_branch_mae_dy": mae(dy_positive_branch, dy_norm),
        "positive_branch_max_error_dy": max_abs_error(dy_positive_branch, dy_norm),
        "oracle_signed_mae_dx": mae(dx_oracle_signed, dx_norm),
        "oracle_signed_max_error_dx": max_abs_error(dx_oracle_signed, dx_norm),
        "oracle_signed_mae_dy": mae(dy_oracle_signed, dy_norm),
        "oracle_signed_max_error_dy": max_abs_error(dy_oracle_signed, dy_norm),
        "fraction_negative_dx": float(np.mean(dx_norm < -sign_epsilon)),
        "fraction_negative_dy": float(np.mean(dy_norm < -sign_epsilon)),
        "fraction_positive_dx": float(np.mean(dx_norm > sign_epsilon)),
        "fraction_positive_dy": float(np.mean(dy_norm > sign_epsilon)),
        "sign_lost": True,
        "strict_signed_diagonal_invertible": False,
        "status": "not_strictly_invertible_from_diagonal",
    }


def normalize_to_neg11(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scaled = 2.0 * (values - minimum) / (maximum - minimum) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def decode_abs_from_saved_channel(channel_01: np.ndarray) -> np.ndarray:
    gasf = channel_01 * 2.0 - 1.0
    diag = np.diag(gasf)
    return np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, 1.0))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], data_root: Path, sign_epsilon: float) -> dict[str, Any]:
    numeric_keys = [
        "abs_diag_mae_dx",
        "abs_diag_max_error_dx",
        "abs_diag_mae_dy",
        "abs_diag_max_error_dy",
        "positive_branch_mae_dx",
        "positive_branch_max_error_dx",
        "positive_branch_mae_dy",
        "positive_branch_max_error_dy",
        "oracle_signed_mae_dx",
        "oracle_signed_max_error_dx",
        "oracle_signed_mae_dy",
        "oracle_signed_max_error_dy",
        "fraction_negative_dx",
        "fraction_negative_dy",
        "fraction_positive_dx",
        "fraction_positive_dy",
    ]
    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "num_samples": len(rows),
        "sign_epsilon": sign_epsilon,
        "strict_signed_diagonal_invertible": False,
        "sign_lost": True,
        "reason": "For GASF with x in [-1,1], diag = 2*x^2 - 1, so diagonal recovers abs(x) only.",
    }
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(np.mean(values))
        summary[f"max_{key}"] = float(np.max(values))
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

"""Check whether stored delta labels equal first differences of absolute labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATA_ROOT = Path("/home/jliu/Thesis/week04/data_constrained_gasf_dxdy_start")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_csv = args.output_csv.resolve()
    output_summary = args.output_summary.resolve()

    rows = check_dataset(data_root, args.max_samples)
    write_metrics(output_csv, rows)
    write_summary(output_summary, data_root, rows)

    print(f"Checked samples: {len(rows)}")
    print(f"Metrics CSV: {output_csv}")
    print(f"Summary JSON: {output_summary}")
    if rows:
        max_error = max(float(row["delta_consistency_max_error"]) for row in rows)
        mean_mse = float(np.mean([float(row["delta_consistency_mse"]) for row in rows]))
        close_count = sum(float(row["delta_consistency_max_error"]) <= args.ok_threshold for row in rows)
        print(f"mean_mse={mean_mse:.12g}")
        print(f"global_max_error={max_error:.12g}")
        print(f"close_count_at_{args.ok_threshold:g}={close_count}/{len(rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ok-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sanity_checks" / "delta_consistency_metrics.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sanity_checks" / "delta_consistency_summary.json",
    )
    return parser.parse_args()


def check_dataset(data_root: Path, max_samples: int | None) -> list[dict[str, Any]]:
    absolute_dir = data_root / "labels_absolute"
    delta_dir = data_root / "labels_delta_displacement"
    if not absolute_dir.exists():
        raise FileNotFoundError(absolute_dir)
    if not delta_dir.exists():
        raise FileNotFoundError(delta_dir)

    sample_ids = sorted(path.stem for path in absolute_dir.glob("*.npy"))
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        abs_path = absolute_dir / f"{sample_id}.npy"
        delta_path = delta_dir / f"{sample_id}.npy"
        if not delta_path.exists():
            rows.append(error_row(sample_id, f"missing_delta:{delta_path}"))
            continue

        abs_xy = np.load(abs_path).astype(np.float64)
        delta_xy = np.load(delta_path).astype(np.float64)
        if abs_xy.ndim != 2 or abs_xy.shape[1] != 2:
            rows.append(error_row(sample_id, f"bad_absolute_shape:{abs_xy.shape}"))
            continue
        if delta_xy.ndim != 2 or delta_xy.shape[1] != 2:
            rows.append(error_row(sample_id, f"bad_delta_shape:{delta_xy.shape}"))
            continue
        if delta_xy.shape[0] != abs_xy.shape[0]:
            rows.append(error_row(sample_id, f"length_mismatch:absolute={abs_xy.shape},delta={delta_xy.shape}"))
            continue

        delta_from_abs = abs_xy[1:] - abs_xy[:-1]
        stored_delta = delta_xy[1:]
        diff = delta_from_abs - stored_delta
        mse = float(np.mean(diff**2))
        max_error = float(np.max(np.abs(diff)))
        rows.append(
            {
                "sample_id": sample_id,
                "delta_consistency_mse": mse,
                "delta_consistency_max_error": max_error,
                "delta_consistency_error": max_error,
                "absolute_shape": str(tuple(abs_xy.shape)),
                "delta_shape": str(tuple(delta_xy.shape)),
                "status": "ok" if max_error <= 1e-6 else "mismatch",
            }
        )
    return rows


def error_row(sample_id: str, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "delta_consistency_mse": "",
        "delta_consistency_max_error": "",
        "delta_consistency_error": "",
        "absolute_shape": "",
        "delta_shape": "",
        "status": reason,
    }


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "delta_consistency_mse",
        "delta_consistency_max_error",
        "delta_consistency_error",
        "absolute_shape",
        "delta_shape",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, data_root: Path, rows: list[dict[str, Any]]) -> None:
    numeric_rows = [row for row in rows if row["delta_consistency_max_error"] != ""]
    max_errors = np.asarray([float(row["delta_consistency_max_error"]) for row in numeric_rows], dtype=np.float64)
    mses = np.asarray([float(row["delta_consistency_mse"]) for row in numeric_rows], dtype=np.float64)
    payload: dict[str, Any] = {
        "data_root": str(data_root),
        "num_rows": len(rows),
        "num_numeric_rows": len(numeric_rows),
        "num_status_ok_at_1e-6": int(np.sum(max_errors <= 1e-6)) if len(max_errors) else 0,
        "num_status_mismatch_at_1e-6": int(np.sum(max_errors > 1e-6)) if len(max_errors) else 0,
        "mean_mse": float(np.mean(mses)) if len(mses) else None,
        "max_mse": float(np.max(mses)) if len(mses) else None,
        "mean_max_error": float(np.mean(max_errors)) if len(max_errors) else None,
        "global_max_error": float(np.max(max_errors)) if len(max_errors) else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

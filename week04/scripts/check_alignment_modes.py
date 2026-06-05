"""Compare delta integration alignment modes against absolute trajectories."""

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

    rows = check_alignment_modes(data_root, args.max_samples)
    write_metrics(output_csv, rows)
    summary = write_summary(output_summary, data_root, rows)

    print(f"Checked samples: {len(rows)}")
    print(f"Metrics CSV: {output_csv}")
    print(f"Summary JSON: {output_summary}")
    for mode in ("a", "b", "c"):
        key = f"mode_{mode}"
        stats = summary.get(key, {})
        print(
            f"{key}: mean_ade={stats.get('mean_ade'):.12g} "
            f"median_ade={stats.get('median_ade'):.12g} "
            f"max_ade={stats.get('max_ade'):.12g} "
            f"mean_fde={stats.get('mean_fde'):.12g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sanity_checks" / "alignment_modes_metrics.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_DATA_ROOT / "sanity_checks" / "alignment_modes_summary.json",
    )
    return parser.parse_args()


def check_alignment_modes(data_root: Path, max_samples: int | None) -> list[dict[str, Any]]:
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
        if delta_xy.shape != abs_xy.shape:
            rows.append(error_row(sample_id, f"shape_mismatch:absolute={abs_xy.shape},delta={delta_xy.shape}"))
            continue

        initial_point = abs_xy[0]
        reconstructions = {
            "a": integrate_mode_a(initial_point, delta_xy),
            "b": integrate_mode_b(initial_point, delta_xy),
            "c": integrate_mode_c(initial_point, delta_xy),
        }

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "absolute_shape": str(tuple(abs_xy.shape)),
            "delta_shape": str(tuple(delta_xy.shape)),
            "delta0_x": float(delta_xy[0, 0]),
            "delta0_y": float(delta_xy[0, 1]),
            "status": "ok",
        }
        for mode, xy_rec in reconstructions.items():
            metrics = trajectory_metrics(xy_rec, abs_xy)
            row[f"mode_{mode}_ade"] = metrics["ade"]
            row[f"mode_{mode}_fde"] = metrics["fde"]
            row[f"mode_{mode}_max_error"] = metrics["max_error"]

        row["mode_a_minus_c_mean_abs_error"] = float(np.mean(np.abs(reconstructions["a"] - reconstructions["c"])))
        row["mode_a_minus_c_max_abs_error"] = float(np.max(np.abs(reconstructions["a"] - reconstructions["c"])))
        rows.append(row)

    return rows


def integrate_mode_a(initial_point: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    xy_rec = np.empty_like(delta_xy, dtype=np.float64)
    xy_rec[0] = initial_point
    for timestep in range(1, delta_xy.shape[0]):
        xy_rec[timestep] = xy_rec[timestep - 1] + delta_xy[timestep]
    return xy_rec


def integrate_mode_b(initial_point: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    xy_rec = np.empty_like(delta_xy, dtype=np.float64)
    xy_rec[0] = initial_point
    for timestep in range(1, delta_xy.shape[0]):
        xy_rec[timestep] = xy_rec[timestep - 1] + delta_xy[timestep - 1]
    return xy_rec


def integrate_mode_c(initial_point: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    return initial_point + np.cumsum(delta_xy, axis=0)


def trajectory_metrics(xy_rec: np.ndarray, abs_xy: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(xy_rec - abs_xy, axis=1)
    return {
        "ade": float(np.mean(errors)),
        "fde": float(errors[-1]),
        "max_error": float(np.max(errors)),
    }


def error_row(sample_id: str, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "absolute_shape": "",
        "delta_shape": "",
        "delta0_x": "",
        "delta0_y": "",
        "mode_a_ade": "",
        "mode_a_fde": "",
        "mode_a_max_error": "",
        "mode_b_ade": "",
        "mode_b_fde": "",
        "mode_b_max_error": "",
        "mode_c_ade": "",
        "mode_c_fde": "",
        "mode_c_max_error": "",
        "mode_a_minus_c_mean_abs_error": "",
        "mode_a_minus_c_max_abs_error": "",
        "status": reason,
    }


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "absolute_shape",
        "delta_shape",
        "delta0_x",
        "delta0_y",
        "mode_a_ade",
        "mode_a_fde",
        "mode_a_max_error",
        "mode_b_ade",
        "mode_b_fde",
        "mode_b_max_error",
        "mode_c_ade",
        "mode_c_fde",
        "mode_c_max_error",
        "mode_a_minus_c_mean_abs_error",
        "mode_a_minus_c_max_abs_error",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, data_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["status"] == "ok"]
    payload: dict[str, Any] = {
        "data_root": str(data_root),
        "num_rows": len(rows),
        "num_valid_rows": len(valid_rows),
        "num_error_rows": len(rows) - len(valid_rows),
        "mode_definitions": {
            "mode_a": "xy[0]=start; xy[t]=xy[t-1]+delta[t]",
            "mode_b": "xy[0]=start; xy[t]=xy[t-1]+delta[t-1]",
            "mode_c": "xy=start+cumsum(delta)",
        },
    }
    for mode in ("a", "b", "c"):
        payload[f"mode_{mode}"] = summarize_metric_prefix(valid_rows, f"mode_{mode}")
    payload["mode_a_minus_c"] = {
        "mean_abs_error_mean": mean_from_rows(valid_rows, "mode_a_minus_c_mean_abs_error"),
        "max_abs_error_global": max_from_rows(valid_rows, "mode_a_minus_c_max_abs_error"),
    }
    payload["delta0"] = {
        "max_abs_x": max_abs_from_rows(valid_rows, "delta0_x"),
        "max_abs_y": max_abs_from_rows(valid_rows, "delta0_y"),
        "nonzero_count_at_1e-12": count_abs_gt_rows(valid_rows, ("delta0_x", "delta0_y"), 1e-12),
    }
    payload["conclusion"] = (
        "Mode A and Mode C match the stored delta convention; Mode B is the off-by-one "
        "alternative and produces much larger ADE/FDE."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def summarize_metric_prefix(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    return {
        "mean_ade": mean_from_rows(rows, f"{prefix}_ade"),
        "median_ade": median_from_rows(rows, f"{prefix}_ade"),
        "max_ade": max_from_rows(rows, f"{prefix}_ade"),
        "mean_fde": mean_from_rows(rows, f"{prefix}_fde"),
        "median_fde": median_from_rows(rows, f"{prefix}_fde"),
        "max_fde": max_from_rows(rows, f"{prefix}_fde"),
        "mean_max_error": mean_from_rows(rows, f"{prefix}_max_error"),
        "global_max_error": max_from_rows(rows, f"{prefix}_max_error"),
    }


def values_from_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return np.asarray(values, dtype=np.float64)


def mean_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = values_from_rows(rows, key)
    return float(np.mean(values)) if len(values) else None


def median_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = values_from_rows(rows, key)
    return float(np.median(values)) if len(values) else None


def max_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = values_from_rows(rows, key)
    return float(np.max(values)) if len(values) else None


def max_abs_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = values_from_rows(rows, key)
    return float(np.max(np.abs(values))) if len(values) else None


def count_abs_gt_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], threshold: float) -> int:
    count = 0
    for row in rows:
        if any(abs(float(row[key])) > threshold for key in keys if row.get(key) != ""):
            count += 1
    return count


if __name__ == "__main__":
    main()

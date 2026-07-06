"""Filter decoded generated trajectories using real-baseline trajectory/map thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_map_eval_week05_osm_full_real"
DEFAULT_DECODE_METRICS = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000" / "decode_metrics.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_rejection_filter"


DEFAULT_CRITERIA = [
    "road_mean_m",
    "road_p95_m",
    "displacement",
    "bbox_width",
    "bbox_height",
    "max_step_length",
    "large_jump_ratio",
]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_rows = read_csv(args.eval_dir.resolve() / "generated_per_sample_metrics.csv")
    real_rows = read_csv(args.eval_dir.resolve() / "real_per_sample_metrics.csv")
    decode_rows = read_csv(args.decode_metrics.resolve()) if args.decode_metrics.exists() else []
    decode_by_id = {row["sample_id"]: row for row in decode_rows}

    generated_rows = [merge_decode(row, decode_by_id.get(row["sample_id"])) for row in generated_rows]
    criteria = list(args.criteria)
    if args.max_out_of_range_ratio is not None:
        criteria.append("out_of_position_range_ratio")

    thresholds = build_thresholds(
        real_rows=real_rows,
        generated_rows=generated_rows,
        criteria=criteria,
        quantile=args.real_quantile,
        max_out_of_range_ratio=args.max_out_of_range_ratio,
        overrides=parse_threshold_overrides(args.threshold),
    )
    filtered_rows = [apply_filter(row, thresholds) for row in generated_rows]
    accepted = [row for row in filtered_rows if row["accepted"]]
    rejected = [row for row in filtered_rows if not row["accepted"]]

    write_csv(output_dir / "generated_filter_all.csv", filtered_rows)
    write_csv(output_dir / "accepted_generated.csv", accepted)
    write_csv(output_dir / "rejected_generated.csv", rejected)

    summary = build_summary(
        generated_rows=filtered_rows,
        accepted=accepted,
        rejected=rejected,
        real_rows=real_rows,
        thresholds=thresholds,
        real_quantile=args.real_quantile,
    )
    write_json(output_dir / "filter_summary.json", summary)
    write_markdown(output_dir / "filter_summary.md", summary)
    plot_filter_summary(output_dir / "filter_metric_comparison.png", filtered_rows, real_rows, thresholds)
    print(json.dumps(summary, indent=2))


def build_thresholds(
    real_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    criteria: list[str],
    quantile: float,
    max_out_of_range_ratio: float | None,
    overrides: dict[str, float],
) -> dict[str, float]:
    thresholds = {}
    for key in criteria:
        if key in overrides:
            thresholds[key] = overrides[key]
        elif key == "out_of_position_range_ratio":
            thresholds[key] = 0.0 if max_out_of_range_ratio is None else max_out_of_range_ratio
        elif key in real_rows[0]:
            thresholds[key] = float(np.quantile(column_values(real_rows, key), quantile))
        else:
            thresholds[key] = float(np.quantile(column_values(generated_rows, key), quantile))
    return thresholds


def apply_filter(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    failures = []
    for key, threshold in thresholds.items():
        value = as_float(row.get(key, np.nan))
        if not np.isfinite(value) or value > threshold:
            failures.append(f"{key}>{threshold:.6g}")
    output = dict(row)
    output["accepted"] = not failures
    output["num_failed_criteria"] = len(failures)
    output["failed_criteria"] = ";".join(failures)
    return output


def merge_decode(metric_row: dict[str, str], decode_row: dict[str, str] | None) -> dict[str, Any]:
    output: dict[str, Any] = dict(metric_row)
    if decode_row is not None:
        for key in ("heatmap_peak", "heatmap_mass", "out_of_position_range_ratio"):
            if key in decode_row:
                output[key] = decode_row[key]
    return output


def build_summary(
    generated_rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    real_quantile: float,
) -> dict[str, Any]:
    return {
        "num_generated": len(generated_rows),
        "num_real_baseline": len(real_rows),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "acceptance_rate": len(accepted) / len(generated_rows) if generated_rows else 0.0,
        "real_quantile_for_thresholds": real_quantile,
        "thresholds": thresholds,
        "accepted_summary": summarize_rows(accepted),
        "rejected_summary": summarize_rows(rejected),
        "generated_summary": summarize_rows(generated_rows),
        "real_summary": summarize_rows(real_rows),
        "failure_counts": failure_counts(generated_rows, thresholds),
    }


def failure_counts(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, int]:
    counts = {key: 0 for key in thresholds}
    for row in rows:
        for key, threshold in thresholds.items():
            value = as_float(row.get(key, np.nan))
            if not np.isfinite(value) or value > threshold:
                counts[key] += 1
    return counts


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "road_mean_m",
        "road_p95_m",
        "support_mean_m",
        "support_p95_m",
        "displacement",
        "bbox_width",
        "bbox_height",
        "max_step_length",
        "large_jump_ratio",
        "path_length",
        "out_of_position_range_ratio",
    ]
    summary = {}
    for key in keys:
        values = [as_float(row[key]) for row in rows if key in row and is_number(row[key])]
        if values:
            summary[key] = summarize(np.asarray(values, dtype=float))
    return summary


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def plot_filter_summary(path: Path, generated: list[dict[str, Any]], real: list[dict[str, Any]], thresholds: dict[str, float]) -> None:
    candidates = ["support_mean_m", "support_p95_m", "road_mean_m", "displacement", "bbox_width", "bbox_height", "max_step_length"]
    plot_keys = [key for key in candidates if any(key in row for row in generated + real)]
    if not plot_keys:
        return
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(4.2 * len(plot_keys), 4.0))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, plot_keys):
        real_values = column_values(real, key)
        gen_values = column_values(generated, key)
        accepted_values = np.asarray([as_float(row[key]) for row in generated if row.get("accepted") and key in row], dtype=float)
        ax.boxplot(
            [real_values, gen_values, accepted_values if len(accepted_values) else np.asarray([np.nan])],
            labels=["real", "gen", "accepted"],
            showfliers=False,
        )
        if key in thresholds:
            ax.axhline(thresholds[key], color="#dc2626", linestyle="--", linewidth=1.2)
        ax.set_title(key)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def column_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([as_float(row[key]) for row in rows if key in row and is_number(row[key])], dtype=float)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week06 Generated Trajectory Rejection Filter",
        "",
        f"- Generated samples: {summary['num_generated']}",
        f"- Real baseline samples: {summary['num_real_baseline']}",
        f"- Accepted: {summary['num_accepted']}",
        f"- Rejected: {summary['num_rejected']}",
        f"- Acceptance rate: {summary['acceptance_rate']:.4f}",
        f"- Real quantile for thresholds: {summary['real_quantile_for_thresholds']}",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in summary["thresholds"].items():
        lines.append(f"- `{key}` <= {value:.6g}")
    lines.extend(["", "## Failure Counts", ""])
    for key, value in summary["failure_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Accepted Summary", ""])
    for key, stats in summary["accepted_summary"].items():
        lines.append(
            f"- `{key}`: mean={stats['mean']:.6g}, median={stats['median']:.6g}, "
            f"p95={stats['p95']:.6g}, max={stats['max']:.6g}"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_threshold_overrides(values: list[str]) -> dict[str, float]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Threshold override must be KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        overrides[key] = float(raw)
    return overrides


def is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number)


def as_float(value: Any) -> float:
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--decode-metrics", type=Path, default=DEFAULT_DECODE_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--real-quantile", type=float, default=0.95)
    parser.add_argument("--criteria", nargs="+", default=DEFAULT_CRITERIA)
    parser.add_argument("--max-out-of-range-ratio", type=float, default=0.0)
    parser.add_argument("--threshold", action="append", default=[], help="Override threshold as KEY=VALUE")
    return parser.parse_args()


if __name__ == "__main__":
    main()

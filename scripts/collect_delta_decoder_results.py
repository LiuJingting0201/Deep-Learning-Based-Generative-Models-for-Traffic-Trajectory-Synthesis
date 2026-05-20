"""Collect delta-displacement decoder ablation metrics into CSV/Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RESULT_DIRS = [
    "smoke_multibranch_delta",
    "decoder_multibranch_delta_beta0",
    "decoder_multibranch_delta_beta005",
    "decoder_multibranch_delta_beta02",
    "decoder_multibranch_delta_norm_beta0",
    "decoder_multibranch_delta_norm_beta005",
    "decoder_multibranch_delta_norm_beta02",
    "decoder_channel_r_beta005",
    "decoder_channel_g_beta005",
    "decoder_channel_b_beta005",
    "decoder_simplecnn_rgb_beta005",
    "decoder_simplecnn_rgb_norm_beta005",
]

METRICS = [
    "delta_ADE",
    "delta_FDE",
    "integrated_ADE",
    "integrated_FDE",
    "start_aligned_ADE",
    "centroid_aligned_ADE",
    "scale_normalized_shape_ADE",
]


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_prefix = args.output_prefix.resolve()
    rows = [collect_one(path) for path in discover_result_dirs(results_root)]
    rows = [row for row in rows if row is not None]

    table = pd.DataFrame(rows)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".md").write_text(to_markdown(table) + "\n")
    print(f"Collected {len(table)} result directories")
    print(f"CSV: {output_prefix.with_suffix('.csv')}")
    print(f"Markdown: {output_prefix.with_suffix('.md')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results_hpc"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results_hpc/delta_decoder_ablation_summary"),
    )
    return parser.parse_args()


def discover_result_dirs(results_root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in DEFAULT_RESULT_DIRS:
        path = results_root / name
        if path.exists():
            paths.append(path)

    for path in sorted(results_root.glob("decoder_*")):
        if path.is_dir() and path not in paths:
            paths.append(path)
    return paths


def collect_one(output_dir: Path) -> dict[str, Any] | None:
    if not output_dir.is_dir():
        return None

    row: dict[str, Any] = {"output_dir": output_dir.as_posix()}
    config = read_json(output_dir / "config.json")
    best = read_json(output_dir / "best_summary.json")
    val_metrics = read_json(output_dir / "metrics_val.json")
    test_metrics = read_json(output_dir / "metrics_test.json")

    if config is None:
        row["config_status"] = "missing"
    else:
        row["config_status"] = "ok"
    config = config or {}
    best = best or {}

    row.update(
        {
            "model": config.get("model"),
            "channels": config.get("channels"),
            "encoder_type": config.get("encoder_type"),
            "beta_integrated_loss": config.get("beta_integrated_loss"),
            "normalize_target_delta": config.get("normalize_target_delta"),
            "best_epoch": best.get("best_epoch"),
            "best_val_total_loss": best.get("best_val_total_loss"),
        }
    )

    add_metric_means(row, val_metrics, "val")
    add_metric_means(row, test_metrics, "test")
    return row


def add_metric_means(row: dict[str, Any], metrics: dict[str, Any] | None, split: str) -> None:
    metrics = metrics or {}
    for metric in METRICS:
        row[f"{split}_{metric}_mean"] = metric_mean(metrics, metric)


def metric_mean(metrics: dict[str, Any], metric: str) -> Any:
    value = metrics.get(metric)
    if isinstance(value, dict):
        return value.get("mean")
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"_json_error": f"Could not parse {path}"}


def to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "No delta decoder results found."
    columns = [
        "output_dir",
        "model",
        "channels",
        "encoder_type",
        "beta_integrated_loss",
        "normalize_target_delta",
        "best_epoch",
        "best_val_total_loss",
        "val_delta_ADE_mean",
        "val_integrated_ADE_mean",
        "test_delta_ADE_mean",
        "test_integrated_ADE_mean",
    ]
    columns = [column for column in columns if column in table.columns]
    display = table[columns].fillna("")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in display.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

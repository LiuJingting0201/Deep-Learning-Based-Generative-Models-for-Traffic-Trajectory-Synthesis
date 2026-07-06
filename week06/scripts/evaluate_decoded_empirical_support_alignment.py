"""Evaluate decoded trajectories against the empirical Week05 real-trajectory support."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from evaluate_decoded_map_alignment import (
    distance_metric_row,
    load_xy,
    summarize_rows,
    trajectory_metric_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECODED_DIR = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_empirical_support_eval"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_paths = sorted((args.real_data_root.resolve() / "labels_absolute").glob("*.npy"))
    if not real_paths:
        raise FileNotFoundError(args.real_data_root / "labels_absolute")
    generated_paths = sorted((args.decoded_dir.resolve() / "trajectories_raw_xy").glob("*.npy"))
    if args.max_generated is not None:
        generated_paths = generated_paths[: args.max_generated]
    if args.max_real is not None:
        eval_real_paths = real_paths[: args.max_real]
    else:
        eval_real_paths = real_paths
    if not generated_paths:
        raise FileNotFoundError(args.decoded_dir / "trajectories_raw_xy")

    support_index = EmpiricalSupportIndex.from_paths(real_paths)
    generated_rows = evaluate_paths_batched(generated_paths, "generated", support_index, args.large_jump_threshold)
    real_rows = evaluate_paths_batched(eval_real_paths, "real_bounce_regularized", support_index, args.large_jump_threshold)

    write_csv(output_dir / "generated_per_sample_metrics.csv", generated_rows)
    write_csv(output_dir / "real_per_sample_metrics.csv", real_rows)
    summary = {
        "decoded_dir": str(args.decoded_dir.resolve()),
        "real_data_root": str(args.real_data_root.resolve()),
        "support_source": "all Week05 real bounce-regularized raw XY points",
        "num_support_trajectories": len(real_paths),
        "num_support_points": int(len(support_index.points)),
        "num_generated": len(generated_rows),
        "num_real": len(real_rows),
        "large_jump_threshold": args.large_jump_threshold,
        "generated": summarize_rows(generated_rows),
        "real_bounce_regularized": summarize_rows(real_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary)
    print(json.dumps(summary, indent=2))


def evaluate_paths_batched(
    paths: list[Path],
    group: str,
    support_index: "EmpiricalSupportIndex",
    large_jump_threshold: float,
) -> list[dict[str, Any]]:
    trajectories = [load_xy(path) for path in paths]
    if trajectories:
        lengths = [len(trajectory) for trajectory in trajectories]
        all_points = np.concatenate(trajectories, axis=0)
        all_distances = support_index.distances_for_xy(all_points)
        distance_chunks = np.split(all_distances, np.cumsum(lengths)[:-1])
    else:
        distance_chunks = []
    rows = []
    for path, trajectory, distances in zip(paths, trajectories, distance_chunks):
        row = trajectory_metric_row(path.stem, group, trajectory, large_jump_threshold)
        row.update(distance_metric_row(distances, "support"))
        rows.append(row)
    return rows


class EmpiricalSupportIndex:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.tree = cKDTree(points)

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "EmpiricalSupportIndex":
        point_arrays = []
        for path in paths:
            xy = load_xy(path)
            if len(xy):
                point_arrays.append(xy)
        if not point_arrays:
            raise ValueError("No valid empirical support points were loaded.")
        return cls(np.concatenate(point_arrays, axis=0).astype(np.float64))

    def distances_for_xy(self, xy: np.ndarray) -> np.ndarray:
        distances, _ = self.tree.query(xy[:, :2].astype(np.float64), k=1, workers=-1)
        return np.asarray(distances, dtype=float)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Week06 Empirical Support Evaluation",
        "",
        f"- Support source: {summary['support_source']}",
        f"- Support trajectories: {summary['num_support_trajectories']}",
        f"- Support points: {summary['num_support_points']}",
        f"- Generated samples evaluated: {summary['num_generated']}",
        f"- Real baseline samples evaluated: {summary['num_real']}",
        "",
    ]
    for group_key, title in (("generated", "Generated"), ("real_bounce_regularized", "Real Bounce-Regularized")):
        lines.extend([f"## {title}", ""])
        group = summary[group_key]
        for key in sorted(group):
            values = group[key]
            lines.append(
                f"- `{key}`: mean={values['mean']:.6g}, median={values['median']:.6g}, "
                f"p95={values['p95']:.6g}, max={values['max']:.6g}"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--large-jump-threshold", type=float, default=50.0)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--max-real", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

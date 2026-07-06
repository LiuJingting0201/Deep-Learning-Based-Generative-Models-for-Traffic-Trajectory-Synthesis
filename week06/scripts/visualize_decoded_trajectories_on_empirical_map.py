"""Visualize generated trajectories on the empirical Week05 real-trajectory support."""

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
DEFAULT_DECODED_DIR = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_empirical_support_eval"
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "trajectory_visualizations_empirical_real_map"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_paths = sorted((args.real_data_root.resolve() / "labels_absolute").glob("*.npy"))
    if not real_paths:
        raise FileNotFoundError(args.real_data_root / "labels_absolute")
    background_paths = choose_background_paths(real_paths, args.max_background_trajectories, args.seed)

    generated_rows = read_csv(args.eval_dir.resolve() / "generated_per_sample_metrics.csv")
    real_rows = read_csv(args.eval_dir.resolve() / "real_per_sample_metrics.csv")
    selected_generated = select_by_metric(
        generated_rows, metric=args.selection_metric, random_count=args.num_random, seed=args.seed
    )
    selected_real = select_by_metric(real_rows, metric=args.selection_metric, random_count=args.num_real, seed=args.seed + 17)

    write_manifest(output_dir / "selection_manifest.json", background_paths, selected_generated, selected_real)
    plot_overlay(
        output_dir / "generated_on_empirical_real_map.png",
        background_paths,
        selected_generated,
        args.decoded_dir.resolve(),
        generated=True,
    )
    plot_overlay(
        output_dir / "real_on_empirical_real_map.png",
        background_paths,
        selected_real,
        args.real_data_root.resolve(),
        generated=False,
    )
    plot_side_by_side(
        output_dir / "generated_vs_real_on_empirical_real_map.png",
        background_paths,
        selected_generated,
        selected_real,
        args.decoded_dir.resolve(),
        args.real_data_root.resolve(),
    )
    plot_contact_sheet(
        output_dir / "generated_contact_sheet_empirical_real_map.png",
        background_paths,
        selected_generated,
        args.decoded_dir.resolve(),
        generated=True,
    )
    plot_contact_sheet(
        output_dir / "real_contact_sheet_empirical_real_map.png",
        background_paths,
        selected_real,
        args.real_data_root.resolve(),
        generated=False,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "background_trajectories": len(background_paths),
                "generated": selected_generated,
                "real": selected_real,
            },
            indent=2,
        )
    )


def choose_background_paths(paths: list[Path], max_count: int, seed: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(paths), size=max_count, replace=False))
    return [paths[int(index)] for index in indices]


def select_by_metric(rows: list[dict[str, str]], metric: str, random_count: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[metric]))
    selected = [
        annotate(ordered[0], f"best_{metric}"),
        annotate(ordered[len(ordered) // 2], f"median_{metric}"),
        annotate(ordered[-1], f"worst_{metric}"),
    ]
    rng = np.random.default_rng(seed)
    for index in rng.choice(len(rows), size=min(random_count, len(rows)), replace=False):
        selected.append(annotate(rows[int(index)], "random"))
    return dedupe(selected)


def annotate(row: dict[str, str], reason: str) -> dict[str, Any]:
    output: dict[str, Any] = {"reason": reason}
    output.update(row)
    return output


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        output.append(row)
    return output


def plot_background(ax: Any, background_paths: list[Path], linewidth: float = 0.18, alpha: float = 0.12) -> None:
    for path in background_paths:
        xy = np.load(path)
        ax.plot(xy[:, 0], xy[:, 1], color="#111827", linewidth=linewidth, alpha=alpha, zorder=1)


def plot_overlay(
    path: Path,
    background_paths: list[Path],
    rows: list[dict[str, Any]],
    data_root: Path,
    generated: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    plot_background(ax, background_paths)
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    for row, color in zip(rows, colors):
        xy = load_trajectory(row["sample_id"], data_root, generated=generated)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.8, color=color, alpha=0.85, label=f"{row['sample_id']} {row['reason']}", zorder=3)
        ax.scatter(xy[0, 0], xy[0, 1], s=18, marker="o", color=color, zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=24, marker="x", color=color, zorder=4)
    ax.set_title("Generated on empirical Week05 real support" if generated else "Real on empirical Week05 real support")
    style_axes(ax)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_side_by_side(
    path: Path,
    background_paths: list[Path],
    generated_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    decoded_dir: Path,
    real_root: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, title in zip(axes, ["Generated selected", "Real selected"]):
        plot_background(ax, background_paths)
        ax.set_title(title)
        style_axes(ax)
    for row in generated_rows:
        xy = load_trajectory(row["sample_id"], decoded_dir, generated=True)
        axes[0].plot(xy[:, 0], xy[:, 1], linewidth=1.5, alpha=0.85, zorder=3)
    for row in real_rows:
        xy = load_trajectory(row["sample_id"], real_root, generated=False)
        axes[1].plot(xy[:, 0], xy[:, 1], linewidth=1.5, alpha=0.85, zorder=3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_contact_sheet(
    path: Path,
    background_paths: list[Path],
    rows: list[dict[str, Any]],
    data_root: Path,
    generated: bool,
) -> None:
    cols = 3
    row_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(15, 5 * row_count), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        xy = load_trajectory(row["sample_id"], data_root, generated=generated)
        plot_background_near(ax, background_paths, xy)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, color="#dc2626", zorder=3)
        ax.scatter(xy[0, 0], xy[0, 1], s=30, marker="o", color="#16a34a", zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=34, marker="x", color="#2563eb", zorder=4)
        metric_value = float(row.get("support_mean_m", row.get("road_mean_m", np.nan)))
        ax.set_title(f"{row['sample_id']} ({row['reason']})\nsupport_mean={metric_value:.1f}m", fontsize=9)
        pad = 180.0
        ax.set_xlim(float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad))
        ax.set_ylim(float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad))
        style_axes(ax)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_background_near(ax: Any, background_paths: list[Path], xy: np.ndarray) -> None:
    pad = 220.0
    x_min, x_max = float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad)
    y_min, y_max = float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad)
    for path in background_paths:
        bg = np.load(path)
        if bg[:, 0].max() < x_min or bg[:, 0].min() > x_max or bg[:, 1].max() < y_min or bg[:, 1].min() > y_max:
            continue
        ax.plot(bg[:, 0], bg[:, 1], color="#111827", linewidth=0.25, alpha=0.18, zorder=1)


def style_axes(ax: Any) -> None:
    ax.set_xlabel("Week05 raw x")
    ax.set_ylabel("Week05 raw y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)


def load_trajectory(sample_id: str, data_root: Path, generated: bool) -> np.ndarray:
    if generated:
        path = data_root / "trajectories_raw_xy" / f"{sample_id}.npy"
    else:
        path = data_root / "labels_absolute" / f"{sample_id}.npy"
    array = np.load(path).astype(float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected trajectory shape N,2 in {path}, got {array.shape}")
    return array


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(
    path: Path,
    background_paths: list[Path],
    generated_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "background_real_trajectories": [str(path) for path in background_paths],
        "generated": generated_rows,
        "real": real_rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-background-trajectories", type=int, default=2500)
    parser.add_argument("--num-random", type=int, default=3)
    parser.add_argument("--num-real", type=int, default=3)
    parser.add_argument("--selection-metric", default="support_mean_m")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Create a compact model-comparison sheet for Week06 raw diffusion samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "raw_diffusion_visual_comparison"
DEFAULT_RUNS = (
    "raw_absolute_unet1d",
    "raw_absolute_temporal_resnet_dilated",
    "raw_absolute_temporal_resnet",
    "raw_delta_unet1d",
    "raw_delta_temporal_resnet_dilated",
    "raw_delta_temporal_resnet",
)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    background_paths = choose_background_paths(
        sorted((args.real_data_root / "labels_absolute").glob("*.npy")),
        args.max_background_trajectories,
        args.seed,
    )
    fig, axes = plt.subplots(
        len(args.runs),
        3,
        figsize=(13, 3.4 * len(args.runs)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, run_name in enumerate(args.runs):
        eval_dir = args.results_root / f"{run_name}_empirical_support_eval"
        decoded_dir = args.decoded_root / f"{run_name}_step_050000"
        rows = read_csv(eval_dir / "generated_per_sample_metrics.csv")
        selected = select_best_median_worst(rows, "support_mean_m")
        for col_index, row in enumerate(selected):
            ax = axes[row_index, col_index]
            trajectory = np.load(decoded_dir / "trajectories_raw_xy" / f"{row['sample_id']}.npy").astype(float)
            plot_background_near(ax, background_paths, trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="#dc2626", linewidth=1.7, zorder=3)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color="#16a34a", marker="o", s=16, zorder=4)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#2563eb", marker="x", s=20, zorder=4)
            if row_index == 0:
                ax.set_title(["Best", "Median", "Worst"][col_index], fontsize=11)
            label = run_name.replace("raw_", "").replace("_temporal_resnet", "_resnet")
            ax.text(
                0.02,
                0.98,
                f"{label}\n{row['sample_id']}\nsupport={float(row['support_mean_m']):.1f}m",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.82, "pad": 3},
            )
            pad = 180.0
            ax.set_xlim(float(trajectory[:, 0].min() - pad), float(trajectory[:, 0].max() + pad))
            ax.set_ylim(float(trajectory[:, 1].min() - pad), float(trajectory[:, 1].max() + pad))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.12)
            ax.tick_params(labelsize=7)
    output_path = args.output_dir / "raw_diffusion_best_median_worst_support.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


def choose_background_paths(paths: list[Path], max_count: int, seed: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(paths), size=max_count, replace=False))
    return [paths[int(index)] for index in indices]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_best_median_worst(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: float(row[metric]))
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def plot_background_near(ax: plt.Axes, background_paths: list[Path], trajectory: np.ndarray) -> None:
    pad = 220.0
    x_min, x_max = float(trajectory[:, 0].min() - pad), float(trajectory[:, 0].max() + pad)
    y_min, y_max = float(trajectory[:, 1].min() - pad), float(trajectory[:, 1].max() + pad)
    for path in background_paths:
        xy = np.load(path)
        if xy[:, 0].max() < x_min or xy[:, 0].min() > x_max or xy[:, 1].max() < y_min or xy[:, 1].min() > y_max:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color="#111827", linewidth=0.2, alpha=0.14, zorder=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-root", type=Path, default=PROJECT_ROOT / "week06" / "decoded")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "week06" / "results")
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--max-background-trajectories", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

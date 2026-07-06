"""Visualize decoded generated trajectories and real baselines on the Week05 OSM map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECODED_DIR = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_step_040000_map_eval_week05_osm_full_real"
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OSM_EDGES = PROJECT_ROOT / "week06" / "data" / "maps_week05" / "osm_drive_week05_edges.gpkg"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "trajectory_visualizations"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = gpd.read_file(args.osm_edges_path.resolve())
    transform = json.loads(args.transform_path.resolve().read_text())["affine_matrix"]
    generated_metrics_path = (
        args.generated_metrics_path.resolve()
        if args.generated_metrics_path is not None
        else args.eval_dir.resolve() / "generated_per_sample_metrics.csv"
    )
    generated_rows = read_csv(generated_metrics_path)
    real_rows = read_csv(args.eval_dir.resolve() / "real_per_sample_metrics.csv")

    selected_generated = select_generated(generated_rows, args.num_random, args.seed)
    selected_real = select_real(real_rows, args.num_real, args.seed)

    write_selection_manifest(output_dir / "selection_manifest.json", selected_generated, selected_real)
    offset_xy = np.asarray([args.offset_x, args.offset_y], dtype=float)
    plot_contact_sheet(output_dir / "generated_contact_sheet.png", edges, selected_generated, args.decoded_dir, transform, offset_xy)
    plot_contact_sheet(
        output_dir / "real_contact_sheet.png",
        edges,
        selected_real,
        args.real_data_root,
        transform,
        offset_xy,
        real=True,
    )
    plot_overlay(output_dir / "generated_selected_overlay.png", edges, selected_generated, args.decoded_dir, transform, offset_xy)
    if args.plot_all_generated:
        plot_overlay(
            output_dir / "generated_all_overlay.png",
            edges,
            generated_rows[: args.max_all_generated],
            args.decoded_dir,
            transform,
            offset_xy,
            title=f"Generated accepted trajectories (n={min(len(generated_rows), args.max_all_generated)}/{len(generated_rows)})",
            linewidth=0.8,
            alpha=0.35,
            legend=False,
        )
    plot_overlay(output_dir / "real_selected_overlay.png", edges, selected_real, args.real_data_root, transform, offset_xy, real=True)
    plot_side_by_side(
        output_dir / "generated_vs_real_overlay.png",
        edges,
        selected_generated,
        selected_real,
        args.decoded_dir,
        args.real_data_root,
        transform,
        offset_xy,
    )
    print(json.dumps({"output_dir": str(output_dir), "generated": selected_generated, "real": selected_real}, indent=2))


def select_generated(rows: list[dict[str, str]], num_random: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["road_mean_m"]))
    selected = [
        annotate(ordered[0], "best_road_mean"),
        annotate(ordered[len(ordered) // 2], "median_road_mean"),
        annotate(ordered[-1], "worst_road_mean"),
    ]
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(rows), size=min(num_random, len(rows)), replace=False)
    for index in random_indices:
        selected.append(annotate(rows[int(index)], "random"))
    return dedupe(selected)


def select_real(rows: list[dict[str, str]], num_real: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["road_mean_m"]))
    selected = [
        annotate(ordered[0], "best_real_road_mean"),
        annotate(ordered[len(ordered) // 2], "median_real_road_mean"),
        annotate(ordered[-1], "worst_real_road_mean"),
    ]
    rng = np.random.default_rng(seed + 17)
    random_indices = rng.choice(len(rows), size=min(num_real, len(rows)), replace=False)
    for index in random_indices:
        selected.append(annotate(rows[int(index)], "random_real"))
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


def plot_contact_sheet(
    path: Path,
    edges: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
    data_root: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
    real: bool = False,
) -> None:
    cols = 3
    rows_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(15, 5 * rows_count), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        trajectory = load_trajectory(row["sample_id"], data_root, real=real)
        trajectory_osm = apply_affine(trajectory, transform) + offset_xy
        plot_single(ax, edges, trajectory_osm, row)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_overlay(
    path: Path,
    edges: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
    data_root: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
    real: bool = False,
    title: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 0.75,
    legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    edges.plot(ax=ax, linewidth=0.35, color="#111827", alpha=0.35)
    for row in rows:
        trajectory = load_trajectory(row["sample_id"], data_root, real=real)
        trajectory_osm = apply_affine(trajectory, transform) + offset_xy
        ax.plot(trajectory_osm[:, 0], trajectory_osm[:, 1], linewidth=linewidth, alpha=alpha, label=row["sample_id"])
        ax.scatter(trajectory_osm[0, 0], trajectory_osm[0, 1], s=18, marker="o", alpha=min(1.0, alpha + 0.25))
        ax.scatter(trajectory_osm[-1, 0], trajectory_osm[-1, 1], s=22, marker="x", alpha=min(1.0, alpha + 0.25))
    if title is None:
        title = "Selected real trajectories" if real else "Selected generated trajectories"
    ax.set_title(title)
    ax.set_xlabel("OSM x")
    ax.set_ylabel("OSM y")
    ax.set_aspect("equal", adjustable="box")
    if legend:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_side_by_side(
    path: Path,
    edges: gpd.GeoDataFrame,
    generated_rows: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
    decoded_dir: Path,
    real_root: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    for ax, title in zip(axes, ["Generated selected", "Real selected"]):
        edges.plot(ax=ax, linewidth=0.35, color="#111827", alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("OSM x")
        ax.set_ylabel("OSM y")
        ax.set_aspect("equal", adjustable="box")
    for row in generated_rows:
        trajectory = apply_affine(load_trajectory(row["sample_id"], decoded_dir), transform) + offset_xy
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.4, alpha=0.75)
    for row in real_rows:
        trajectory = apply_affine(load_trajectory(row["sample_id"], real_root, real=True), transform) + offset_xy
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.4, alpha=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_single(ax: Any, edges: gpd.GeoDataFrame, trajectory_osm: np.ndarray, row: dict[str, Any]) -> None:
    edges.plot(ax=ax, linewidth=0.3, color="#111827", alpha=0.35)
    ax.plot(trajectory_osm[:, 0], trajectory_osm[:, 1], linewidth=2.0, color="#dc2626")
    ax.scatter(trajectory_osm[0, 0], trajectory_osm[0, 1], s=30, marker="o", color="#16a34a", label="start")
    ax.scatter(trajectory_osm[-1, 0], trajectory_osm[-1, 1], s=34, marker="x", color="#2563eb", label="end")
    pad = 180.0
    ax.set_xlim(float(trajectory_osm[:, 0].min() - pad), float(trajectory_osm[:, 0].max() + pad))
    ax.set_ylim(float(trajectory_osm[:, 1].min() - pad), float(trajectory_osm[:, 1].max() + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{row['sample_id']} ({row['reason']})\n"
        f"road_mean={float(row['road_mean_m']):.1f}m disp={float(row['displacement']):.0f}",
        fontsize=9,
    )
    ax.axis("on")


def load_trajectory(sample_id: str, data_root: Path, real: bool = False) -> np.ndarray:
    if real:
        path = data_root / "labels_absolute" / f"{sample_id}.npy"
    else:
        path = data_root / "trajectories_raw_xy" / f"{sample_id}.npy"
    array = np.load(path).astype(float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected trajectory shape N,2 in {path}, got {array.shape}")
    return array


def apply_affine(xy: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    affine = np.asarray(matrix, dtype=float)
    design = np.column_stack([xy[:, :2], np.ones(len(xy), dtype=float)])
    return design @ affine


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_selection_manifest(path: Path, generated: list[dict[str, Any]], real: list[dict[str, Any]]) -> None:
    payload = {"generated": generated, "real": real}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-metrics-path", type=Path, default=None)
    parser.add_argument("--plot-all-generated", action="store_true")
    parser.add_argument("--max-all-generated", type=int, default=200)
    parser.add_argument("--num-random", type=int, default=3)
    parser.add_argument("--num-real", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

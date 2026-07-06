"""Visualize accepted Week06 generated trajectories on the offset-corrected OSM road map."""

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
from matplotlib.collections import LineCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACCEPTED_CSV = (
    PROJECT_ROOT
    / "week06"
    / "results"
    / "week05_sig15_mse_diag_step_040000_rejection_filter_p95_strict_offset_dx220_dy-720"
    / "accepted_generated.csv"
)
DEFAULT_DECODED_DIR = PROJECT_ROOT / "week06" / "decoded" / "week05_sig15_mse_diag_step_040000"
DEFAULT_OSM_EDGES = PROJECT_ROOT / "week06" / "data" / "maps_week05" / "osm_drive_week05_edges.gpkg"
DEFAULT_TRANSFORM = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "accepted_generated_on_osm_offset_dx220_dy-720"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows = read_csv(args.accepted_csv.resolve())
    if not accepted_rows:
        raise ValueError(f"No accepted rows found in {args.accepted_csv}")

    edges = gpd.read_file(args.osm_edges_path.resolve())
    transform = json.loads(args.transform_path.resolve().read_text())["affine_matrix"]
    offset_xy = np.asarray([args.offset_x, args.offset_y], dtype=float)

    rows_for_sheet = select_representatives(accepted_rows, args.num_contact, args.seed)
    manifest = {
        "accepted_csv": str(args.accepted_csv.resolve()),
        "num_accepted": len(accepted_rows),
        "offset_x": args.offset_x,
        "offset_y": args.offset_y,
        "contact_sheet_samples": rows_for_sheet,
    }
    (output_dir / "accepted_visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plot_all_overlay(
        output_dir / "accepted_all_overlay_osm.png",
        edges,
        accepted_rows,
        args.decoded_dir.resolve(),
        transform,
        offset_xy,
    )
    plot_contact_sheet(
        output_dir / "accepted_contact_sheet_osm.png",
        edges,
        rows_for_sheet,
        args.decoded_dir.resolve(),
        transform,
        offset_xy,
    )
    plot_representative_overlay(
        output_dir / "accepted_representative_overlay_osm.png",
        edges,
        rows_for_sheet,
        args.decoded_dir.resolve(),
        transform,
        offset_xy,
    )
    print(json.dumps({"output_dir": str(output_dir), "num_accepted": len(accepted_rows)}, indent=2))


def select_representatives(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["road_mean_m"]))
    selected: list[dict[str, Any]] = []
    if count <= 0:
        return selected

    quantile_slots = min(count, 9)
    for slot in range(quantile_slots):
        q = 0.0 if quantile_slots == 1 else slot / (quantile_slots - 1)
        index = int(round(q * (len(ordered) - 1)))
        row = dict(ordered[index])
        row["reason"] = f"road_mean_q{q:.2f}"
        selected.append(row)

    if len(selected) < count:
        rng = np.random.default_rng(seed)
        chosen_ids = {row["sample_id"] for row in selected}
        candidates = [row for row in rows if row["sample_id"] not in chosen_ids]
        random_count = min(count - len(selected), len(candidates))
        for index in rng.choice(len(candidates), size=random_count, replace=False):
            row = dict(candidates[int(index)])
            row["reason"] = "random_accepted"
            selected.append(row)
    return dedupe(selected)[:count]


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


def plot_all_overlay(
    path: Path,
    edges: gpd.GeoDataFrame,
    rows: list[dict[str, str]],
    decoded_dir: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    plot_edges(ax, edges, linewidth=0.35, alpha=0.32)

    trajectories = [load_osm_trajectory(row["sample_id"], decoded_dir, transform, offset_xy) for row in rows]
    road_means = np.asarray([float(row["road_mean_m"]) for row in rows], dtype=float)
    collection = LineCollection(trajectories, cmap="viridis_r", linewidths=1.1, alpha=0.7)
    collection.set_array(road_means)
    ax.add_collection(collection)
    cbar = fig.colorbar(collection, ax=ax, shrink=0.78)
    cbar.set_label("road_mean_m")

    starts = np.asarray([xy[0] for xy in trajectories])
    ends = np.asarray([xy[-1] for xy in trajectories])
    ax.scatter(starts[:, 0], starts[:, 1], s=9, color="#16a34a", alpha=0.65, label="start")
    ax.scatter(ends[:, 0], ends[:, 1], s=12, color="#dc2626", alpha=0.65, marker="x", label="end")
    set_bounds_from_trajectories(ax, trajectories, pad=420.0)
    ax.set_title(f"Accepted generated trajectories on Week05 OSM roads (n={len(rows)})")
    style_axes(ax)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_representative_overlay(
    path: Path,
    edges: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
    decoded_dir: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    plot_edges(ax, edges, linewidth=0.35, alpha=0.32)
    colors = plt.cm.tab20(np.linspace(0, 1, len(rows)))
    trajectories = []
    for row, color in zip(rows, colors):
        xy = load_osm_trajectory(row["sample_id"], decoded_dir, transform, offset_xy)
        trajectories.append(xy)
        label = f"{row['sample_id']} mean={float(row['road_mean_m']):.1f}m"
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.8, color=color, alpha=0.9, label=label)
        ax.scatter(xy[0, 0], xy[0, 1], s=22, color=color, marker="o")
        ax.scatter(xy[-1, 0], xy[-1, 1], s=28, color=color, marker="x")
    set_bounds_from_trajectories(ax, trajectories, pad=420.0)
    ax.set_title("Representative accepted generated trajectories")
    style_axes(ax)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_contact_sheet(
    path: Path,
    edges: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
    decoded_dir: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
) -> None:
    cols = 3
    row_count = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(row_count, cols, figsize=(15, 5 * row_count), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, row in zip(axes.ravel(), rows):
        xy = load_osm_trajectory(row["sample_id"], decoded_dir, transform, offset_xy)
        plot_edges_near(ax, edges, xy, pad=260.0)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.0, color="#dc2626", zorder=3)
        ax.scatter(xy[0, 0], xy[0, 1], s=30, marker="o", color="#16a34a", zorder=4)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=34, marker="x", color="#2563eb", zorder=4)
        ax.set_title(
            f"{row['sample_id']} ({row['reason']})\n"
            f"road_mean={float(row['road_mean_m']):.1f}m road_p95={float(row['road_p95_m']):.1f}m",
            fontsize=9,
        )
        set_bounds_from_trajectories(ax, [xy], pad=180.0)
        style_axes(ax)
    fig.savefig(path, dpi=175)
    plt.close(fig)


def plot_edges(ax: Any, edges: gpd.GeoDataFrame, linewidth: float, alpha: float) -> None:
    edges.plot(ax=ax, linewidth=linewidth, color="#111827", alpha=alpha, zorder=1)


def plot_edges_near(ax: Any, edges: gpd.GeoDataFrame, xy: np.ndarray, pad: float) -> None:
    x_min, x_max = float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad)
    y_min, y_max = float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad)
    nearby = edges.cx[x_min:x_max, y_min:y_max]
    plot_edges(ax, nearby if len(nearby) else edges, linewidth=0.35, alpha=0.35)


def set_bounds_from_trajectories(ax: Any, trajectories: list[np.ndarray], pad: float) -> None:
    points = np.vstack(trajectories)
    ax.set_xlim(float(points[:, 0].min() - pad), float(points[:, 0].max() + pad))
    ax.set_ylim(float(points[:, 1].min() - pad), float(points[:, 1].max() + pad))


def style_axes(ax: Any) -> None:
    ax.set_xlabel("OSM x")
    ax.set_ylabel("OSM y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.14)


def load_osm_trajectory(
    sample_id: str,
    decoded_dir: Path,
    transform: list[list[float]],
    offset_xy: np.ndarray,
) -> np.ndarray:
    raw_xy = np.load(decoded_dir / "trajectories_raw_xy" / f"{sample_id}.npy").astype(float)
    return apply_affine(raw_xy, transform) + offset_xy


def apply_affine(xy: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    affine = np.asarray(matrix, dtype=float)
    design = np.column_stack([xy[:, :2], np.ones(len(xy), dtype=float)])
    return design @ affine


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-csv", type=Path, default=DEFAULT_ACCEPTED_CSV)
    parser.add_argument("--decoded-dir", type=Path, default=DEFAULT_DECODED_DIR)
    parser.add_argument("--osm-edges-path", type=Path, default=DEFAULT_OSM_EDGES)
    parser.add_argument("--transform-path", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-contact", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset-x", type=float, default=220.0)
    parser.add_argument("--offset-y", type=float, default=-720.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

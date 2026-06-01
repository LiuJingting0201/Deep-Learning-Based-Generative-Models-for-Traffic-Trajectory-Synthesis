"""Map-aware trajectory refinement for already reconstructed delta-decoder outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "week03" / "results" / "baseline_raw_resnet18_bs16" / "evaluation"
DEFAULT_DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_AFFINE_JSON = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
IMAGE_SIZE = 224


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    for subdir in ["refined_absolute", "refined_delta", "debug_plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if args.plot_distance_background:
        (output_dir / "debug_plots_map_background").mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.evaluation_dir / "per_sample_metrics.csv")
    split_metadata = pd.read_csv(args.data_root / "splits" / "split_metadata.csv")
    metadata_lookup = {str(row.sample_id): row._asdict() for row in split_metadata.itertuples(index=False)}
    extent_lookup = load_extent_lookup(args.distance_field_metadata.resolve())
    affine = load_affine(args.raw_to_osm_affine_json.resolve())
    offsets = make_search_offsets(args.search_radius_m, args.search_step_m)

    pred_paths = sorted((args.evaluation_dir / "predictions_absolute_integrated").glob("*_pred_absolute_integrated.npy"))
    if args.max_samples is not None:
        pred_paths = pred_paths[: args.max_samples]
    rows: list[dict[str, Any]] = []
    arrays_for_best10: dict[str, dict[str, np.ndarray]] = {}

    for index, pred_path in enumerate(pred_paths):
        sample_id = pred_path.stem.replace("_pred_absolute_integrated", "")
        if sample_id not in metadata_lookup:
            continue
        pred = np.load(pred_path).astype(np.float32)
        pred_delta_path = args.evaluation_dir / "predictions_delta_displacement" / f"{sample_id}_pred_delta.npy"
        pred_delta = np.load(pred_delta_path).astype(np.float32) if pred_delta_path.exists() else absolute_to_delta(pred)
        gt = np.load(args.data_root / metadata_lookup[sample_id]["label_absolute_path"]).astype(np.float32)
        distance_field = np.load(args.distance_field_dir / f"{sample_id}_distance.npy").astype(np.float32)
        if not np.isfinite(distance_field).all():
            raise ValueError(f"Distance field contains NaN/Inf for {sample_id}")
        extent = extent_array(extent_lookup, sample_id)
        context = RefinementContext(args=args, affine=affine, extent=extent, distance_field=distance_field, offsets=offsets)

        if args.method == "pointwise_grid":
            refined = refine_pointwise_grid(pred, context)
        elif args.method == "sliding_window":
            refined = refine_sliding_window(pred, context)
        else:
            raise ValueError(f"Unsupported method: {args.method}")
        refined_delta = absolute_to_delta(refined)

        np.save(output_dir / "refined_absolute" / f"{sample_id}_refined_absolute.npy", refined.astype(np.float32))
        np.save(output_dir / "refined_delta" / f"{sample_id}_refined_delta.npy", refined_delta.astype(np.float32))

        row = make_metric_row(
            sample_id=sample_id,
            vehicle_id=str(metadata_lookup[sample_id].get("vehicle_id", "")),
            split=str(metadata_lookup[sample_id].get("split", "")),
            raw=pred,
            refined=refined,
            gt=gt,
            context=context,
        )
        rows.append(row)
        arrays_for_best10[sample_id] = {
            "raw": pred,
            "refined": refined,
            "gt": gt,
            "pred_delta": pred_delta,
            "extent": extent,
        }

        if index < args.debug_samples:
            plot_debug(
                output_dir / "debug_plots" / f"{sample_id}_debug.png",
                sample_id,
                pred,
                refined,
                gt,
                row,
            )
            if args.plot_distance_background:
                plot_debug_map_background(
                    output_dir / "debug_plots_map_background" / f"{sample_id}_debug_map_background.png",
                    sample_id,
                    pred,
                    refined,
                    gt,
                    context,
                    row,
                )
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(pred_paths)} samples")

    if not rows:
        raise ValueError("No samples were processed.")
    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(output_dir / "refinement_per_sample_metrics.csv", index=False)
    summary = summarize_refinement(args, per_sample)
    write_json(output_dir / "refinement_summary.json", summary)
    write_summary_md(output_dir / "refinement_summary.md", summary)
    write_best10_plots(output_dir, baseline, per_sample, arrays_for_best10, args, affine)
    print(f"Wrote refinement outputs to {output_dir}")
    print((output_dir / "refinement_summary.md").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--distance-field-dir", type=Path, required=True)
    parser.add_argument("--distance-field-metadata", type=Path, required=True)
    parser.add_argument("--raw-to-osm-affine-json", type=Path, default=DEFAULT_AFFINE_JSON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-mode", choices=["oracle_bbox", "start_center"], required=True)
    parser.add_argument("--method", choices=["pointwise_grid", "sliding_window"], default="pointwise_grid")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--search-radius-m", type=float, default=15.0)
    parser.add_argument("--search-step-m", type=float, default=2.0)
    parser.add_argument("--w-road", type=float, default=1.0)
    parser.add_argument("--w-ref", type=float, default=0.05)
    parser.add_argument("--w-curvature", type=float, default=0.1)
    parser.add_argument("--w-step", type=float, default=0.1)
    parser.add_argument("--w-heading", type=float, default=0.05)
    parser.add_argument("--w-terminal", type=float, default=0.2)
    parser.add_argument("--terminal-power", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--debug-samples", type=int, default=20)
    parser.add_argument("--plot-distance-background", action="store_true")
    parser.add_argument("--background-clip-m", type=float, default=50.0)
    return parser.parse_args()


class RefinementContext:
    def __init__(
        self,
        args: argparse.Namespace,
        affine: np.ndarray,
        extent: np.ndarray,
        distance_field: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        self.args = args
        self.affine = affine
        self.extent = extent
        self.distance_field = distance_field
        self.offsets = offsets
        self.oob_penalty = max(float(np.nanmax(distance_field)) + 50.0, 100.0)


def load_extent_lookup(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"sample_id", "min_x", "max_x", "min_y", "max_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return {str(row.sample_id): row._asdict() for row in frame.itertuples(index=False)}


def load_affine(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    affine = np.asarray(payload["affine_matrix"], dtype=np.float64)
    if affine.shape != (3, 2):
        raise ValueError(f"Expected affine matrix [3,2], got {affine.shape}")
    return affine


def extent_array(extent_lookup: dict[str, dict[str, Any]], sample_id: str) -> np.ndarray:
    if sample_id not in extent_lookup:
        raise KeyError(f"Missing distance-field metadata for {sample_id}")
    row = extent_lookup[sample_id]
    return np.asarray([row["min_x"], row["max_x"], row["min_y"], row["max_y"]], dtype=np.float64)


def make_search_offsets(radius_m: float, step_m: float) -> np.ndarray:
    if radius_m < 0 or step_m <= 0:
        raise ValueError("search radius must be >= 0 and search step must be > 0")
    values = np.arange(-radius_m, radius_m + 0.5 * step_m, step_m, dtype=np.float32)
    dx, dy = np.meshgrid(values, values)
    offsets = np.column_stack([dx.ravel(), dy.ravel()])
    return offsets[np.linalg.norm(offsets, axis=1) <= radius_m + 1e-6]


def raw_to_osm(points_raw: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points_raw, dtype=np.float64).reshape(-1, 2)
    design = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return design @ affine


def sample_distance_raw(points_raw: np.ndarray, context: RefinementContext) -> np.ndarray:
    points_osm = raw_to_osm(points_raw, context.affine)
    min_x, max_x, min_y, max_y = context.extent
    px = (points_osm[:, 0] - min_x) / (max_x - min_x) * (IMAGE_SIZE - 1)
    py = (max_y - points_osm[:, 1]) / (max_y - min_y) * (IMAGE_SIZE - 1)
    oob = (px < 0) | (px > IMAGE_SIZE - 1) | (py < 0) | (py > IMAGE_SIZE - 1)
    px_clip = np.clip(px, 0, IMAGE_SIZE - 1)
    py_clip = np.clip(py, 0, IMAGE_SIZE - 1)
    x0 = np.floor(px_clip).astype(np.int32)
    y0 = np.floor(py_clip).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, IMAGE_SIZE - 1)
    y1 = np.clip(y0 + 1, 0, IMAGE_SIZE - 1)
    wx = px_clip - x0
    wy = py_clip - y0
    field = context.distance_field
    sampled = (
        field[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + field[y0, x1] * wx * (1.0 - wy)
        + field[y1, x0] * (1.0 - wx) * wy
        + field[y1, x1] * wx * wy
    )
    sampled = sampled.astype(np.float64)
    sampled[oob] += context.oob_penalty
    return sampled


def refine_pointwise_grid(pred: np.ndarray, context: RefinementContext) -> np.ndarray:
    refined = np.empty_like(pred, dtype=np.float32)
    refined[0] = pred[0]
    for t in range(1, len(pred)):
        candidates = pred[t] + context.offsets
        costs = candidate_costs(candidates, pred, refined, t, context)
        refined[t] = candidates[int(np.argmin(costs))]
    return refined


def refine_sliding_window(pred: np.ndarray, context: RefinementContext) -> np.ndarray:
    refined = np.empty_like(pred, dtype=np.float32)
    refined[0] = pred[0]
    horizon = max(1, int(context.args.window_size))
    for t in range(1, len(pred)):
        candidates = pred[t] + context.offsets
        costs = candidate_costs(candidates, pred, refined, t, context)
        future_count = min(horizon - 1, len(pred) - t - 1)
        if future_count > 0:
            future_offsets = pred[t + 1 : t + 1 + future_count] - pred[t]
            rollout = candidates[:, None, :] + future_offsets[None, :, :]
            road = sample_distance_raw(rollout.reshape(-1, 2), context).reshape(len(candidates), future_count)
            ref = np.sum((rollout - pred[t + 1 : t + 1 + future_count][None, :, :]) ** 2, axis=2)
            costs += context.args.w_road * road.mean(axis=1) + context.args.w_ref * ref.mean(axis=1)
        refined[t] = candidates[int(np.argmin(costs))]
    return refined


def candidate_costs(
    candidates: np.ndarray,
    pred: np.ndarray,
    refined: np.ndarray,
    t: int,
    context: RefinementContext,
) -> np.ndarray:
    args = context.args
    road = sample_distance_raw(candidates, context)
    ref = np.sum((candidates - pred[t]) ** 2, axis=1)
    progress = t / max(1, len(pred) - 1)
    terminal_weight = args.w_terminal * (progress**args.terminal_power)
    terminal = terminal_weight * ref
    prev_step_len = np.linalg.norm(pred[t] - pred[t - 1])
    refined_vectors = candidates - refined[t - 1]
    step = (np.linalg.norm(refined_vectors, axis=1) - prev_step_len) ** 2
    costs = args.w_road * road + args.w_ref * ref + terminal + args.w_step * step
    raw_dir = pred[t] - pred[t - 1]
    costs += args.w_heading * heading_cost(refined_vectors, raw_dir)
    if t >= 2:
        refined_acc = candidates - 2.0 * refined[t - 1] + refined[t - 2]
        raw_acc = pred[t] - 2.0 * pred[t - 1] + pred[t - 2]
        curvature = np.sum((refined_acc - raw_acc) ** 2, axis=1)
        costs += args.w_curvature * curvature
    return costs


def heading_cost(candidate_vectors: np.ndarray, previous_vector: np.ndarray) -> np.ndarray:
    prev_norm = float(np.linalg.norm(previous_vector))
    if prev_norm < 1e-6:
        return np.zeros(len(candidate_vectors), dtype=np.float64)
    cand_norm = np.linalg.norm(candidate_vectors, axis=1)
    valid = cand_norm >= 1e-6
    costs = np.zeros(len(candidate_vectors), dtype=np.float64)
    denom = cand_norm[valid] * prev_norm
    if not np.any(valid):
        return costs
    cos = np.clip((candidate_vectors[valid] @ previous_vector) / denom, -1.0, 1.0)
    costs[valid] = 1.0 - cos
    return costs


def absolute_to_delta(absolute: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(absolute, dtype=np.float32)
    delta[1:] = absolute[1:] - absolute[:-1]
    return delta


def integrated_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(pred - gt, axis=1)
    return {
        "integrated_ADE": float(errors.mean()),
        "integrated_FDE": float(errors[-1]),
        "trajectory_length_ratio": float(trajectory_length(pred) / trajectory_length(gt)) if trajectory_length(gt) else np.nan,
    }


def road_metrics(points: np.ndarray, context: RefinementContext) -> dict[str, float]:
    distances = sample_distance_raw(points, context)
    return {
        "mean_road_distance": float(np.mean(distances)),
        "median_road_distance": float(np.median(distances)),
        "p90_road_distance": float(np.percentile(distances, 90)),
        "offroad_ratio_5m": float(np.mean(distances > 5.0)),
        "offroad_ratio_10m": float(np.mean(distances > 10.0)),
        "offroad_ratio_15m": float(np.mean(distances > 15.0)),
    }


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def make_metric_row(
    sample_id: str,
    vehicle_id: str,
    split: str,
    raw: np.ndarray,
    refined: np.ndarray,
    gt: np.ndarray,
    context: RefinementContext,
) -> dict[str, Any]:
    row: dict[str, Any] = {"sample_id": sample_id, "vehicle_id": vehicle_id, "split": split}
    for prefix, points in [("raw", raw), ("refined", refined), ("gt", gt)]:
        if prefix == "gt":
            row.update({f"{prefix}_{key}": value for key, value in integrated_metrics(gt, gt).items()})
        else:
            row.update({f"{prefix}_{key}": value for key, value in integrated_metrics(points, gt).items()})
        row.update({f"{prefix}_{key}": value for key, value in road_metrics(points, context).items()})
    row["delta_ADE_change_refined_minus_raw"] = row["refined_integrated_ADE"] - row["raw_integrated_ADE"]
    row["delta_FDE_change_refined_minus_raw"] = row["refined_integrated_FDE"] - row["raw_integrated_FDE"]
    row["road_mean_improvement_raw_minus_refined"] = row["raw_mean_road_distance"] - row["refined_mean_road_distance"]
    row["offroad10_improvement_raw_minus_refined"] = row["raw_offroad_ratio_10m"] - row["refined_offroad_ratio_10m"]
    return row


def summarize_refinement(args: argparse.Namespace, per_sample: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "description": "map-aware trajectory refinement post-processing, not decoder training",
        "crop_mode": args.crop_mode,
        "crop_mode_note": (
            "oracle_bbox uses GT trajectory extent and is an upper-bound diagnostic"
            if args.crop_mode == "oracle_bbox"
            else "start_center is more practical but may have limited coverage"
        ),
        "method": args.method,
        "num_samples": int(len(per_sample)),
        "parameters": {
            "search_radius_m": args.search_radius_m,
            "search_step_m": args.search_step_m,
            "w_road": args.w_road,
            "w_ref": args.w_ref,
            "w_curvature": args.w_curvature,
            "w_step": args.w_step,
            "w_heading": args.w_heading,
            "w_terminal": args.w_terminal,
            "terminal_power": args.terminal_power,
        },
        "means": {},
        "comparisons": {},
    }
    columns = [
        "integrated_ADE",
        "integrated_FDE",
        "trajectory_length_ratio",
        "mean_road_distance",
        "median_road_distance",
        "p90_road_distance",
        "offroad_ratio_5m",
        "offroad_ratio_10m",
        "offroad_ratio_15m",
    ]
    for prefix in ["raw", "refined", "gt"]:
        summary["means"][prefix] = {column: float(per_sample[f"{prefix}_{column}"].mean()) for column in columns}
    summary["comparisons"] = {
        "road_distance_improvement_raw_minus_refined": float(
            per_sample["road_mean_improvement_raw_minus_refined"].mean()
        ),
        "offroad_ratio_10m_improvement_raw_minus_refined": float(
            per_sample["offroad10_improvement_raw_minus_refined"].mean()
        ),
        "ADE_change_refined_minus_raw": float(per_sample["delta_ADE_change_refined_minus_raw"].mean()),
        "FDE_change_refined_minus_raw": float(per_sample["delta_FDE_change_refined_minus_raw"].mean()),
        "trajectory_length_ratio_change_refined_minus_raw": float(
            (per_sample["refined_trajectory_length_ratio"] - per_sample["raw_trajectory_length_ratio"]).mean()
        ),
        "refined_vs_gt_mean_road_distance_gap": float(
            (per_sample["refined_mean_road_distance"] - per_sample["gt_mean_road_distance"]).mean()
        ),
        "refined_vs_gt_offroad10_gap": float(
            (per_sample["refined_offroad_ratio_10m"] - per_sample["gt_offroad_ratio_10m"]).mean()
        ),
    }
    return summary


def plot_debug(path: Path, sample_id: str, raw: np.ndarray, refined: np.ndarray, gt: np.ndarray, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_trajectories(ax, gt=gt, raw=raw, refined=refined)
    ax.set_title(
        f"{sample_id}\n"
        f"raw ADE/FDE={row['raw_integrated_ADE']:.2f}/{row['raw_integrated_FDE']:.2f}, "
        f"ref={row['refined_integrated_ADE']:.2f}/{row['refined_integrated_FDE']:.2f}\n"
        f"road mean raw/ref={row['raw_mean_road_distance']:.2f}/{row['refined_mean_road_distance']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_debug_map_background(
    path: Path,
    sample_id: str,
    raw: np.ndarray,
    refined: np.ndarray,
    gt: np.ndarray,
    context: RefinementContext,
    row: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_distance_background(ax, context.distance_field, context.extent, context.args.background_clip_m)
    plot_projected_trajectories(ax, context.affine, gt=gt, raw=raw, refined=refined)
    ax.set_title(
        f"{sample_id} projected OSM frame\n"
        f"raw ADE/FDE={row['raw_integrated_ADE']:.2f}/{row['raw_integrated_FDE']:.2f}, "
        f"ref={row['refined_integrated_ADE']:.2f}/{row['refined_integrated_FDE']:.2f}\n"
        f"road mean raw/ref={row['raw_mean_road_distance']:.2f}/{row['refined_mean_road_distance']:.2f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trajectories(
    ax: Any,
    gt: np.ndarray,
    raw: np.ndarray | None = None,
    refined: np.ndarray | None = None,
) -> None:
    ax.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=1.8, color="#1b9e77")
    ax.scatter(gt[0, 0], gt[0, 1], marker="o", s=25, color="#1b9e77", label="GT start")
    ax.scatter(gt[-1, 0], gt[-1, 1], marker="s", s=25, color="#1b9e77", label="GT end")
    if raw is not None:
        ax.plot(raw[:, 0], raw[:, 1], label="raw pred", linewidth=1.5, color="#d95f02")
        ax.scatter(raw[-1, 0], raw[-1, 1], marker="^", s=28, color="#d95f02", label="raw end")
    if refined is not None:
        ax.plot(refined[:, 0], refined[:, 1], label="refined", linewidth=1.5, color="#7570b3")
        ax.scatter(refined[-1, 0], refined[-1, 1], marker="D", s=25, color="#7570b3", label="refined end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def plot_distance_background(ax: Any, distance_field: np.ndarray, extent: np.ndarray, clip_m: float) -> None:
    min_x, max_x, min_y, max_y = extent
    image = np.clip(distance_field, 0.0, clip_m)
    ax.imshow(
        image,
        cmap="gray_r",
        origin="upper",
        extent=(min_x, max_x, min_y, max_y),
        vmin=0.0,
        vmax=clip_m,
        alpha=0.72,
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


def plot_projected_trajectories(
    ax: Any,
    affine: np.ndarray,
    gt: np.ndarray,
    raw: np.ndarray | None = None,
    refined: np.ndarray | None = None,
) -> None:
    gt_osm = raw_to_osm(gt, affine)
    raw_osm = raw_to_osm(raw, affine) if raw is not None else None
    refined_osm = raw_to_osm(refined, affine) if refined is not None else None
    plot_trajectories(ax, gt=gt_osm, raw=raw_osm, refined=refined_osm)
    ax.set_xlabel("OSM projected x")
    ax.set_ylabel("OSM projected y")


def write_best10_plots(
    output_dir: Path,
    baseline: pd.DataFrame,
    per_sample: pd.DataFrame,
    arrays: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    affine: np.ndarray,
) -> None:
    root = output_dir / "best10_plots"
    for subdir in ["raw_vs_gt", "refined_vs_gt", "raw_refined_gt"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    map_root = output_dir / "best10_plots_map_background"
    if args.plot_distance_background:
        for subdir in ["raw_vs_gt", "refined_vs_gt", "raw_refined_gt"]:
            (map_root / subdir).mkdir(parents=True, exist_ok=True)
    processed = set(per_sample["sample_id"].astype(str))
    best = baseline[(baseline["split"] == "test") & (baseline["sample_id"].astype(str).isin(processed))]
    best = best.sort_values("integrated_ADE").head(10).reset_index(drop=True)
    metric_lookup = {str(row.sample_id): row._asdict() for row in per_sample.itertuples(index=False)}
    rows = []
    for rank, row in enumerate(best.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        data = arrays[sample_id]
        metrics = metric_lookup[sample_id]
        stem = f"{rank:02d}_{sample_id}"
        save_best_plot(root / "raw_vs_gt" / f"{stem}_raw_vs_gt.png", sample_id, data, metrics, mode="raw")
        save_best_plot(root / "refined_vs_gt" / f"{stem}_refined_vs_gt.png", sample_id, data, metrics, mode="refined")
        save_best_plot(root / "raw_refined_gt" / f"{stem}_raw_refined_gt.png", sample_id, data, metrics, mode="all")
        if args.plot_distance_background:
            distance_field = np.load(args.distance_field_dir / f"{sample_id}_distance.npy").astype(np.float32)
            save_best_map_background_plot(
                map_root / "raw_vs_gt" / f"{stem}_raw_vs_gt.png",
                sample_id,
                data,
                metrics,
                mode="raw",
                distance_field=distance_field,
                extent=data["extent"],
                affine=affine,
                clip_m=args.background_clip_m,
            )
            save_best_map_background_plot(
                map_root / "refined_vs_gt" / f"{stem}_refined_vs_gt.png",
                sample_id,
                data,
                metrics,
                mode="refined",
                distance_field=distance_field,
                extent=data["extent"],
                affine=affine,
                clip_m=args.background_clip_m,
            )
            save_best_map_background_plot(
                map_root / "raw_refined_gt" / f"{stem}_raw_refined_gt.png",
                sample_id,
                data,
                metrics,
                mode="all",
                distance_field=distance_field,
                extent=data["extent"],
                affine=affine,
                clip_m=args.background_clip_m,
            )
        rows.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "vehicle_id": str(row.vehicle_id),
                "baseline_integrated_ADE": float(row.integrated_ADE),
                "baseline_integrated_FDE": float(row.integrated_FDE),
                "baseline_delta_ADE": float(row.delta_ADE),
                "baseline_delta_FDE": float(row.delta_FDE),
                "raw_mean_road_distance": metrics["raw_mean_road_distance"],
                "refined_mean_road_distance": metrics["refined_mean_road_distance"],
                "gt_mean_road_distance": metrics["gt_mean_road_distance"],
                "raw_offroad_ratio_10m": metrics["raw_offroad_ratio_10m"],
                "refined_offroad_ratio_10m": metrics["refined_offroad_ratio_10m"],
                "gt_offroad_ratio_10m": metrics["gt_offroad_ratio_10m"],
            }
        )
    write_csv(root / "best10_selected_samples.csv", rows)


def save_best_plot(path: Path, sample_id: str, data: dict[str, np.ndarray], metrics: dict[str, Any], mode: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    raw = data["raw"] if mode in {"raw", "all"} else None
    refined = data["refined"] if mode in {"refined", "all"} else None
    plot_trajectories(ax, gt=data["gt"], raw=raw, refined=refined)
    ax.set_title(best_title(sample_id, metrics, mode))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_best_map_background_plot(
    path: Path,
    sample_id: str,
    data: dict[str, np.ndarray],
    metrics: dict[str, Any],
    mode: str,
    distance_field: np.ndarray,
    extent: np.ndarray,
    affine: np.ndarray,
    clip_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_distance_background(ax, distance_field, extent, clip_m)
    raw = data["raw"] if mode in {"raw", "all"} else None
    refined = data["refined"] if mode in {"refined", "all"} else None
    plot_projected_trajectories(ax, affine, gt=data["gt"], raw=raw, refined=refined)
    ax.set_title(best_title(sample_id, metrics, mode) + "\nprojected OSM frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def best_title(sample_id: str, metrics: dict[str, Any], mode: str) -> str:
    if mode == "raw":
        return (
            f"{sample_id} raw vs GT\n"
            f"ADE/FDE={metrics['raw_integrated_ADE']:.2f}/{metrics['raw_integrated_FDE']:.2f}, "
            f"road={metrics['raw_mean_road_distance']:.2f}"
        )
    if mode == "refined":
        return (
            f"{sample_id} refined vs GT\n"
            f"ADE/FDE={metrics['refined_integrated_ADE']:.2f}/{metrics['refined_integrated_FDE']:.2f}, "
            f"road={metrics['refined_mean_road_distance']:.2f}"
        )
    return (
        f"{sample_id} raw/refined/GT\n"
        f"raw ADE/FDE={metrics['raw_integrated_ADE']:.2f}/{metrics['raw_integrated_FDE']:.2f}, "
        f"ref={metrics['refined_integrated_ADE']:.2f}/{metrics['refined_integrated_FDE']:.2f}\n"
        f"road raw/ref={metrics['raw_mean_road_distance']:.2f}/{metrics['refined_mean_road_distance']:.2f}"
    )


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    raw = summary["means"]["raw"]
    refined = summary["means"]["refined"]
    gt = summary["means"]["gt"]
    comp = summary["comparisons"]
    params = summary["parameters"]
    text = f"""# Map-Aware Trajectory Refinement

This is post-processing, not decoder training.

- crop_mode: {summary["crop_mode"]}
- crop note: {summary["crop_mode_note"]}
- method: {summary["method"]}
- samples: {summary["num_samples"]}
- w_terminal: {params["w_terminal"]}
- terminal_power: {params["terminal_power"]}

| Metric | Raw prediction | Refined prediction | GT |
| --- | ---: | ---: | ---: |
| integrated_ADE | {raw["integrated_ADE"]:.6f} | {refined["integrated_ADE"]:.6f} | {gt["integrated_ADE"]:.6f} |
| integrated_FDE | {raw["integrated_FDE"]:.6f} | {refined["integrated_FDE"]:.6f} | {gt["integrated_FDE"]:.6f} |
| trajectory_length_ratio | {raw["trajectory_length_ratio"]:.6f} | {refined["trajectory_length_ratio"]:.6f} | {gt["trajectory_length_ratio"]:.6f} |
| mean_road_distance | {raw["mean_road_distance"]:.6f} | {refined["mean_road_distance"]:.6f} | {gt["mean_road_distance"]:.6f} |
| median_road_distance | {raw["median_road_distance"]:.6f} | {refined["median_road_distance"]:.6f} | {gt["median_road_distance"]:.6f} |
| p90_road_distance | {raw["p90_road_distance"]:.6f} | {refined["p90_road_distance"]:.6f} | {gt["p90_road_distance"]:.6f} |
| offroad_ratio_5m | {raw["offroad_ratio_5m"]:.6f} | {refined["offroad_ratio_5m"]:.6f} | {gt["offroad_ratio_5m"]:.6f} |
| offroad_ratio_10m | {raw["offroad_ratio_10m"]:.6f} | {refined["offroad_ratio_10m"]:.6f} | {gt["offroad_ratio_10m"]:.6f} |
| offroad_ratio_15m | {raw["offroad_ratio_15m"]:.6f} | {refined["offroad_ratio_15m"]:.6f} | {gt["offroad_ratio_15m"]:.6f} |

## Changes

- road-distance improvement raw minus refined: {comp["road_distance_improvement_raw_minus_refined"]:.6f}
- offroad ratio >10m improvement raw minus refined: {comp["offroad_ratio_10m_improvement_raw_minus_refined"]:.6f}
- ADE change refined minus raw: {comp["ADE_change_refined_minus_raw"]:.6f}
- FDE change refined minus raw: {comp["FDE_change_refined_minus_raw"]:.6f}
- trajectory length ratio change refined minus raw: {comp["trajectory_length_ratio_change_refined_minus_raw"]:.6f}
- refined vs GT mean road-distance gap: {comp["refined_vs_gt_mean_road_distance_gap"]:.6f}
- refined vs GT offroad >10m gap: {comp["refined_vs_gt_offroad10_gap"]:.6f}
"""
    path.write_text(text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

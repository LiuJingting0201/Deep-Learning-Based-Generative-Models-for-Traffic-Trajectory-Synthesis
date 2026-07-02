"""Evaluate affine upper-bound alignment for delta-displacement decoder outputs.

This diagnostic compares raw integrated predictions against oracle translation
and oracle similarity alignment. It does not train or modify any model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_no_speed_delta_displacement_paired"


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = experiment_dir / "evaluation" / "affine_upper_bound"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    split_copy = load_split_copy(experiment_dir)
    split_rows = split_copy[split_copy["split"] == args.split].copy()
    if split_rows.empty:
        raise ValueError(f"No rows found for split={args.split!r} in {experiment_dir / 'split_copy.csv'}")

    data_root = resolve_data_root(args.data_root, experiment_dir)
    prediction_dir = experiment_dir / "evaluation" / "predictions_absolute_integrated"
    if not prediction_dir.exists():
        raise FileNotFoundError(prediction_dir)

    rows: list[dict[str, object]] = []
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for row in split_rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        gt_path = data_root / str(row.label_absolute_path)
        pred_path = prediction_dir / f"{sample_id}_pred_absolute_integrated.npy"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)

        gt = load_xy(gt_path)
        pred_raw = load_xy(pred_path)
        pred_translation, translation_offset = apply_centroid_translation(pred_raw, gt)
        pred_affine, affine_params = apply_similarity_alignment(pred_raw, gt)

        sample_metrics = {
            "sample_id": sample_id,
            "vehicle_id": str(getattr(row, "vehicle_id", "")),
            "split": str(row.split),
            "rotation_deg": affine_params["rotation_deg"],
            "estimated_scale": affine_params["scale"],
            "translation_norm": affine_params["translation_norm"],
            "centroid_translation_norm": float(np.linalg.norm(translation_offset)),
            "raw_rotation_deg": 0.0,
            "raw_estimated_scale": 1.0,
            "raw_translation_norm": 0.0,
            "translation_rotation_deg": 0.0,
            "translation_estimated_scale": 1.0,
            "translation_translation_norm": float(np.linalg.norm(translation_offset)),
            "affine_rotation_deg": affine_params["rotation_deg"],
            "affine_estimated_scale": affine_params["scale"],
            "affine_translation_norm": affine_params["translation_norm"],
        }

        method_metrics = {
            "raw": compute_metrics(pred_raw, gt),
            "translation": compute_metrics(pred_translation, gt),
            "affine": compute_metrics(pred_affine, gt),
        }
        for method, metrics in method_metrics.items():
            for key, value in metrics.items():
                sample_metrics[f"{method}_{key}"] = value

        raw_ade = float(method_metrics["raw"]["ADE"])
        raw_fde = float(method_metrics["raw"]["FDE"])
        sample_metrics["raw_ADE_improvement_vs_raw"] = 0.0
        sample_metrics["raw_FDE_improvement_vs_raw"] = 0.0
        sample_metrics["translation_ADE_improvement_vs_raw"] = raw_ade - float(method_metrics["translation"]["ADE"])
        sample_metrics["translation_FDE_improvement_vs_raw"] = raw_fde - float(method_metrics["translation"]["FDE"])
        sample_metrics["affine_ADE_improvement_vs_raw"] = raw_ade - float(method_metrics["affine"]["ADE"])
        sample_metrics["affine_FDE_improvement_vs_raw"] = raw_fde - float(method_metrics["affine"]["FDE"])

        rows.append(sample_metrics)
        trajectories[sample_id] = {
            "gt": gt,
            "raw": pred_raw,
            "translation": pred_translation,
            "affine": pred_affine,
        }

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(output_dir / "per_sample_affine_metrics.csv", index=False)

    summary = make_summary(per_sample, args.split)
    write_json(output_dir / "affine_summary.json", summary)
    write_report(output_dir / "affine_report_zh.txt", summary[args.split])

    make_plots(per_sample, plots_dir)
    make_qualitative_plots(per_sample, trajectories, plots_dir)

    print(f"Saved affine upper-bound diagnostics to {output_dir}")
    print(
        f"{args.split}: raw ADE={summary[args.split]['raw_ADE_mean']:.4f}, "
        f"translation ADE={summary[args.split]['translation_ADE_mean']:.4f}, "
        f"affine ADE={summary[args.split]['affine_ADE_mean']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("results_hpc/delta_displacement_resnet18_decoder_ablation"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root for label_absolute_path. Defaults to config.json data_root, then local paired dataset.",
    )
    return parser.parse_args()


def load_split_copy(experiment_dir: Path) -> pd.DataFrame:
    split_path = experiment_dir / "split_copy.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    frame = pd.read_csv(split_path)
    required = {"sample_id", "vehicle_id", "label_absolute_path", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split_path} missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{split_path} contains duplicate sample_id values.")
    return frame


def resolve_data_root(cli_data_root: Path | None, experiment_dir: Path) -> Path:
    if cli_data_root is not None:
        return cli_data_root.resolve()

    config_path = experiment_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        data_root = Path(str(config.get("data_root", ""))).expanduser()
        if data_root.exists():
            return data_root.resolve()
        if not data_root.is_absolute() and (PROJECT_ROOT / data_root).exists():
            return (PROJECT_ROOT / data_root).resolve()

    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT.resolve()
    raise FileNotFoundError(
        "Could not resolve dataset root. Pass --data-root or provide config.json with a valid data_root."
    )


def load_xy(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected [N,2] trajectory at {path}, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {path}")
    return array


def apply_centroid_translation(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offset = gt.mean(axis=0) - pred.mean(axis=0)
    return pred + offset, offset


def apply_similarity_alignment(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and GT shape mismatch: {pred.shape} vs {gt.shape}")

    pred_centroid = pred.mean(axis=0)
    gt_centroid = gt.mean(axis=0)
    pred_centered = pred - pred_centroid
    gt_centered = gt - gt_centroid
    denom = float(np.sum(pred_centered**2))

    if denom <= 0.0:
        rotation = np.eye(2, dtype=np.float64)
        scale = 1.0
    else:
        covariance = pred_centered.T @ gt_centered
        u, singular_values, vt = np.linalg.svd(covariance)
        correction = np.eye(2, dtype=np.float64)
        if np.linalg.det(u @ vt) < 0.0:
            correction[-1, -1] = -1.0
        rotation = u @ correction @ vt
        scale = float(np.sum(singular_values * np.diag(correction)) / denom)

    translation = gt_centroid - scale * (pred_centroid @ rotation)
    aligned = scale * (pred @ rotation) + translation
    rotation_deg = math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
    return aligned, {
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "translation_norm": float(np.linalg.norm(translation)),
    }


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and GT shape mismatch: {pred.shape} vs {gt.shape}")
    diff = pred - gt
    point_errors = np.linalg.norm(diff, axis=1)
    gt_length = trajectory_length(gt)
    pred_length = trajectory_length(pred)
    pred_turn = mean_abs_turn_deg(pred)
    gt_turn = mean_abs_turn_deg(gt)
    return {
        "ADE": float(np.mean(point_errors)),
        "FDE": float(point_errors[-1]),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "MAE": float(np.mean(np.abs(diff))),
        "centroid_error": float(np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))),
        "trajectory_length_error": float(abs(pred_length - gt_length)),
        "trajectory_length_ratio": safe_ratio(pred_length, gt_length),
        "turn_sharpness_ratio": safe_ratio(pred_turn, gt_turn),
        "mean_abs_turn_deg": float(pred_turn),
        "gt_mean_abs_turn_deg": float(gt_turn),
    }


def trajectory_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def mean_abs_turn_deg(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    vectors = np.diff(xy, axis=0)
    prev = vectors[:-1]
    nxt = vectors[1:]
    prev_norm = np.linalg.norm(prev, axis=1)
    next_norm = np.linalg.norm(nxt, axis=1)
    valid = (prev_norm > 1e-12) & (next_norm > 1e-12)
    if not np.any(valid):
        return 0.0
    prev = prev[valid]
    nxt = nxt[valid]
    cross = prev[:, 0] * nxt[:, 1] - prev[:, 1] * nxt[:, 0]
    dot = np.sum(prev * nxt, axis=1)
    turns = np.degrees(np.arctan2(cross, dot))
    return float(np.mean(np.abs(turns)))


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def make_summary(per_sample: pd.DataFrame, split: str) -> dict[str, dict[str, float]]:
    summary = {
        split: {
            "raw_ADE_mean": mean(per_sample["raw_ADE"]),
            "translation_ADE_mean": mean(per_sample["translation_ADE"]),
            "affine_ADE_mean": mean(per_sample["affine_ADE"]),
            "raw_FDE_mean": mean(per_sample["raw_FDE"]),
            "translation_FDE_mean": mean(per_sample["translation_FDE"]),
            "affine_FDE_mean": mean(per_sample["affine_FDE"]),
            "raw_centroid_error_mean": mean(per_sample["raw_centroid_error"]),
            "translation_centroid_error_mean": mean(per_sample["translation_centroid_error"]),
            "affine_centroid_error_mean": mean(per_sample["affine_centroid_error"]),
            "raw_turn_sharpness_ratio": mean(per_sample["raw_turn_sharpness_ratio"]),
            "affine_turn_sharpness_ratio": mean(per_sample["affine_turn_sharpness_ratio"]),
            "mean_rotation_deg": mean(np.abs(per_sample["rotation_deg"].to_numpy(dtype=float))),
            "mean_scale": mean(per_sample["estimated_scale"]),
            "mean_translation_norm": mean(per_sample["translation_norm"]),
            "translation_improvement_mean": mean(per_sample["translation_ADE_improvement_vs_raw"]),
            "affine_improvement_mean": mean(per_sample["affine_ADE_improvement_vs_raw"]),
            "count_affine_better_than_translation": int(np.sum(per_sample["affine_ADE"] < per_sample["translation_ADE"])),
            "count_affine_better_than_raw": int(np.sum(per_sample["affine_ADE"] < per_sample["raw_ADE"])),
            "num_samples": int(len(per_sample)),
        }
    }
    return summary


def mean(values: pd.Series | np.ndarray) -> float:
    return float(np.nanmean(np.asarray(values, dtype=float)))


def make_plots(per_sample: pd.DataFrame, plots_dir: Path) -> None:
    hist_specs = [
        ("raw_ADE", "hist_raw_ADE.png", "Raw ADE"),
        ("translation_ADE", "hist_translation_ADE.png", "Translation ADE"),
        ("affine_ADE", "hist_affine_ADE.png", "Similarity-aligned ADE"),
    ]
    for column, filename, title in hist_specs:
        draw_histogram(per_sample[column].to_numpy(dtype=float), plots_dir / filename, title, "ADE", "count")

    scatter(
        per_sample["raw_ADE"],
        per_sample["affine_ADE"],
        plots_dir / "scatter_raw_vs_affine_ADE.png",
        "raw ADE",
        "affine ADE",
    )
    scatter(
        per_sample["estimated_scale"],
        per_sample["affine_ADE_improvement_vs_raw"],
        plots_dir / "scatter_scale_vs_ADE_improvement.png",
        "estimated scale",
        "ADE improvement vs raw",
    )
    scatter(
        np.abs(per_sample["rotation_deg"]),
        per_sample["affine_ADE_improvement_vs_raw"],
        plots_dir / "scatter_rotation_vs_ADE_improvement.png",
        "|rotation| (deg)",
        "ADE improvement vs raw",
    )
    scatter(
        per_sample["translation_norm"],
        per_sample["affine_ADE_improvement_vs_raw"],
        plots_dir / "scatter_translation_norm_vs_ADE_improvement.png",
        "similarity translation norm",
        "ADE improvement vs raw",
    )


def scatter(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, path: Path, xlabel: str, ylabel: str) -> None:
    draw_scatter(np.asarray(x, dtype=float), np.asarray(y, dtype=float), path, xlabel, ylabel)


def make_qualitative_plots(
    per_sample: pd.DataFrame,
    trajectories: dict[str, dict[str, np.ndarray]],
    plots_dir: Path,
) -> None:
    groups = {
        "top_affine_improvement": per_sample.sort_values("affine_ADE_improvement_vs_raw", ascending=False).head(10),
        "top_translation_improvement": per_sample.sort_values(
            "translation_ADE_improvement_vs_raw", ascending=False
        ).head(10),
        "smallest_affine_improvement": per_sample.sort_values("affine_ADE_improvement_vs_raw", ascending=True).head(10),
    }
    for group_name, frame in groups.items():
        group_dir = plots_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            path = group_dir / f"{sample_id}_trajectory.png"
            plot_four_way(row, trajectories[sample_id], path)
            image_paths.append(path)
        make_contact_sheet(image_paths, plots_dir / f"{group_name}_10_grid.png")


def plot_four_way(row: object, curves: dict[str, np.ndarray], path: Path) -> None:
    draw_trajectory_plot(
        curves,
        path,
        f"{row.sample_id} {row.vehicle_id}",
        f"raw={row.raw_ADE:.2f} trans={row.translation_ADE:.2f} affine={row.affine_ADE:.2f}",
    )


def draw_histogram(values: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    values = values[np.isfinite(values)]
    image, draw, box = make_canvas(title, xlabel, ylabel, size=(900, 600))
    left, top, right, bottom = box
    if values.size == 0:
        image.save(path)
        return
    counts, edges = np.histogram(values, bins=30)
    max_count = max(1, int(counts.max()))
    draw_grid(draw, box)
    for index, count in enumerate(counts):
        x0 = left + (right - left) * index / len(counts)
        x1 = left + (right - left) * (index + 1) / len(counts)
        y0 = bottom - (bottom - top) * count / max_count
        draw.rectangle([x0 + 1, y0, x1 - 1, bottom], fill=(82, 130, 190), outline=(55, 90, 140))
    draw.text((left, bottom + 8), f"{edges[0]:.2f}", fill=(40, 40, 40))
    draw.text((right - 70, bottom + 8), f"{edges[-1]:.2f}", fill=(40, 40, 40))
    image.save(path)


def draw_scatter(x: np.ndarray, y: np.ndarray, path: Path, xlabel: str, ylabel: str) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    image, draw, box = make_canvas("", xlabel, ylabel, size=(760, 700))
    draw_grid(draw, box)
    if x.size == 0:
        image.save(path)
        return
    mapper = make_mapper(np.column_stack([x, y]), box, equal_aspect=False)
    for xi, yi in zip(x, y):
        px, py = mapper(np.array([xi, yi]))
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(47, 120, 173))
    left, top, right, bottom = box
    draw.text((left, bottom + 8), f"{float(np.min(x)):.2f}", fill=(40, 40, 40))
    draw.text((right - 70, bottom + 8), f"{float(np.max(x)):.2f}", fill=(40, 40, 40))
    draw.text((left - 58, bottom - 8), f"{float(np.min(y)):.2f}", fill=(40, 40, 40))
    draw.text((left - 58, top), f"{float(np.max(y)):.2f}", fill=(40, 40, 40))
    image.save(path)


def draw_trajectory_plot(curves: dict[str, np.ndarray], path: Path, title: str, subtitle: str) -> None:
    image = Image.new("RGB", (840, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 18), title, fill=(20, 20, 20))
    draw.text((30, 40), subtitle, fill=(70, 70, 70))
    box = (80, 80, 800, 690)
    draw_grid(draw, box)
    mapper = make_mapper(np.vstack(list(curves.values())), box, equal_aspect=True)
    colors = {
        "gt": (20, 20, 20),
        "raw": (213, 94, 0),
        "translation": (0, 114, 178),
        "affine": (0, 158, 115),
    }
    labels = [("GT", "gt"), ("raw", "raw"), ("translation", "translation"), ("affine", "affine")]
    for label, name in labels:
        curve = curves[name]
        points = [mapper(point) for point in curve]
        if len(points) > 1:
            draw.line(points, fill=colors[name], width=3 if name == "gt" else 2)
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=colors[name])
        draw.rectangle([ex - 4, ey - 4, ex + 4, ey + 4], fill=colors[name])
    legend_x = 90
    legend_y = 705
    for offset, (label, name) in enumerate(labels):
        x = legend_x + offset * 160
        draw.line([(x, legend_y + 8), (x + 30, legend_y + 8)], fill=colors[name], width=3)
        draw.text((x + 36, legend_y), label, fill=(20, 20, 20))
    image.save(path)


def make_canvas(
    title: str,
    xlabel: str,
    ylabel: str,
    size: tuple[int, int],
) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int, int]]:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    box = (90, 60, width - 40, height - 90)
    if title:
        draw.text((30, 20), title, fill=(20, 20, 20))
    draw.text(((box[0] + box[2]) // 2 - 40, height - 35), xlabel, fill=(40, 40, 40))
    draw.text((18, (box[1] + box[3]) // 2), ylabel, fill=(40, 40, 40))
    return image, draw, box


def draw_grid(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(40, 40, 40))
    for index in range(1, 5):
        x = left + (right - left) * index / 5
        y = top + (bottom - top) * index / 5
        draw.line([(x, top), (x, bottom)], fill=(225, 225, 225))
        draw.line([(left, y), (right, y)], fill=(225, 225, 225))


def make_mapper(
    xy: np.ndarray,
    box: tuple[int, int, int, int],
    equal_aspect: bool,
) -> callable:
    left, top, right, bottom = box
    xmin = float(np.min(xy[:, 0]))
    xmax = float(np.max(xy[:, 0]))
    ymin = float(np.min(xy[:, 1]))
    ymax = float(np.max(xy[:, 1]))
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    xmin -= max(1.0, xspan * 0.05)
    xmax += max(1.0, xspan * 0.05)
    ymin -= max(1.0, yspan * 0.05)
    ymax += max(1.0, yspan * 0.05)
    if equal_aspect:
        data_w = xmax - xmin
        data_h = ymax - ymin
        pixel_w = right - left
        pixel_h = bottom - top
        target_ratio = pixel_w / pixel_h
        data_ratio = data_w / data_h
        if data_ratio > target_ratio:
            new_h = data_w / target_ratio
            center = (ymin + ymax) / 2.0
            ymin = center - new_h / 2.0
            ymax = center + new_h / 2.0
        else:
            new_w = data_h * target_ratio
            center = (xmin + xmax) / 2.0
            xmin = center - new_w / 2.0
            xmax = center + new_w / 2.0

    def mapper(point: np.ndarray) -> tuple[float, float]:
        px = left + (float(point[0]) - xmin) / (xmax - xmin) * (right - left)
        py = bottom - (float(point[1]) - ymin) / (ymax - ymin) * (bottom - top)
        return px, py

    return mapper


def make_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        with Image.open(path) as image:
            thumbs.append(image.convert("RGB").resize((320, 320)))
    cols = 5
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows * 320), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * 320, (index // cols) * 320))
    sheet.save(output_path)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: dict[str, float]) -> None:
    raw = summary["raw_ADE_mean"]
    translation = summary["translation_ADE_mean"]
    affine = summary["affine_ADE_mean"]
    translation_gain = summary["translation_improvement_mean"]
    affine_gain = summary["affine_improvement_mean"]
    extra_affine_gain = affine_gain - translation_gain
    turn_delta = abs(summary["affine_turn_sharpness_ratio"] - summary["raw_turn_sharpness_ratio"])

    if extra_affine_gain > max(0.05 * max(raw, 1e-12), 0.1 * max(abs(translation_gain), 1e-12)):
        interpretation = "除平移漂移外，还存在尺度或方向失配。"
    elif translation_gain > 0.75 * max(affine_gain, 1e-12) and translation_gain > 0.0:
        interpretation = "误差主要来自整体平移。"
    elif affine_gain <= 0.05 * max(raw, 1e-12):
        interpretation = "局部形状误差占主导。"
    else:
        interpretation = "平移、尺度/方向与局部形状误差可能共同存在。"

    turn_sentence = ""
    if turn_delta <= 0.05:
        turn_sentence = "affine 对齐后的 turn_sharpness_ratio 与 raw 很接近，说明局部转弯几何本身已存在，只是全局对齐存在问题。"
    else:
        turn_sentence = "turn_sharpness_ratio 在 affine 对齐后有明显变化，需要谨慎检查局部转弯几何是否也有误差。"

    text = f"""目的：
验证 delta baseline 是否主要受全局 affine 误差影响。

方法：
比较 raw prediction、仅 centroid translation、以及 similarity transform upper bound。similarity 只包含平移、uniform scale 和 rotation，不包含 non-uniform scale 或 shear，因此保持局部轨迹拓扑。

结果：
raw_ADE_mean = {raw:.6f}
translation_ADE_mean = {translation:.6f}
affine_ADE_mean = {affine:.6f}
raw_FDE_mean = {summary["raw_FDE_mean"]:.6f}
translation_FDE_mean = {summary["translation_FDE_mean"]:.6f}
affine_FDE_mean = {summary["affine_FDE_mean"]:.6f}
mean_rotation_deg = {summary["mean_rotation_deg"]:.6f}
mean_scale = {summary["mean_scale"]:.6f}
mean_translation_norm = {summary["mean_translation_norm"]:.6f}
translation_improvement_mean = {translation_gain:.6f}
affine_improvement_mean = {affine_gain:.6f}
raw_turn_sharpness_ratio = {summary["raw_turn_sharpness_ratio"]:.6f}
affine_turn_sharpness_ratio = {summary["affine_turn_sharpness_ratio"]:.6f}

解释：
{interpretation}
{turn_sentence}

This experiment acts as an upper bound before designing learned affine correction.
"""
    path.write_text(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

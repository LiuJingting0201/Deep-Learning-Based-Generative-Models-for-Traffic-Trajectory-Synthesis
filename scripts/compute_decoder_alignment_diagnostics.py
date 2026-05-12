"""Compute alignment diagnostics for absolute and delta-integrated decoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "raw_ADE",
    "raw_FDE",
    "start_aligned_ADE",
    "start_aligned_FDE",
    "centroid_aligned_ADE",
    "centroid_aligned_FDE",
    "scale_normalized_shape_ADE",
    "scale_normalized_shape_FDE",
    "pred_centroid_x",
    "pred_centroid_y",
    "gt_centroid_x",
    "gt_centroid_y",
    "pred_shape_scale",
    "gt_shape_scale",
    "shape_scale_ratio",
]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    delta_experiment = args.delta_experiment.resolve()
    absolute_experiment = args.absolute_experiment.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = pd.read_csv(data_root / "splits" / "split_metadata.csv")
    rows: list[dict[str, object]] = []

    for row in split_metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        gt = np.load(data_root / row.label_absolute_path).astype(np.float64)
        model_specs = [
            (
                "absolute_resnet18",
                absolute_experiment / "evaluation" / "predictions" / f"{sample_id}_pred.npy",
            ),
            (
                "delta_integrated_resnet18",
                delta_experiment
                / "evaluation"
                / "predictions_absolute_integrated"
                / f"{sample_id}_pred_absolute_integrated.npy",
            ),
        ]
        for model_name, pred_path in model_specs:
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = np.load(pred_path).astype(np.float64)
            metrics = compute_alignment_metrics(pred, gt)
            rows.append(
                {
                    "sample_id": sample_id,
                    "vehicle_id": str(row.vehicle_id),
                    "split": str(row.split),
                    "model": model_name,
                    **metrics,
                }
            )

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(output_dir / "alignment_per_sample_metrics.csv", index=False)

    summary = summarize(per_sample)
    comparison = make_test_comparison(per_sample)
    write_json(output_dir / "alignment_summary.json", summary)
    write_json(output_dir / "alignment_test_comparison.json", comparison)
    make_comparison_table(comparison).to_csv(output_dir / "alignment_test_comparison_table.csv", index=False)

    print(f"Saved alignment diagnostics to {output_dir}")
    print("Test split mean metrics:")
    table = make_comparison_table(comparison)
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_no_speed_delta_displacement_paired"),
    )
    parser.add_argument(
        "--delta-experiment",
        type=Path,
        default=Path("experiments/decoder_no_speed_delta_displacement_resnet18_ablation"),
    )
    parser.add_argument(
        "--absolute-experiment",
        type=Path,
        default=Path("experiments/decoder_no_speed_resnet18_ablation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/decoder_no_speed_delta_displacement_resnet18_ablation/"
            "comparison_with_absolute_resnet18/alignment_diagnostics"
        ),
    )
    return parser.parse_args()


def compute_alignment_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and GT shape mismatch: {pred.shape} vs {gt.shape}")
    raw = point_errors(pred, gt)

    start_aligned = pred + (gt[0] - pred[0])
    start_errors = point_errors(start_aligned, gt)

    pred_centroid = pred.mean(axis=0)
    gt_centroid = gt.mean(axis=0)
    centroid_aligned = pred + (gt_centroid - pred_centroid)
    centroid_errors = point_errors(centroid_aligned, gt)

    pred_centered = pred - pred_centroid
    gt_centered = gt - gt_centroid
    pred_scale = rms_radius(pred_centered)
    gt_scale = rms_radius(gt_centered)
    if pred_scale > 0.0 and gt_scale > 0.0:
        scale_normalized = pred_centered * (gt_scale / pred_scale) + gt_centroid
        shape_scale_ratio = pred_scale / gt_scale
    else:
        scale_normalized = centroid_aligned
        shape_scale_ratio = float("nan")
    scale_errors = point_errors(scale_normalized, gt)

    return {
        "raw_ADE": float(raw.mean()),
        "raw_FDE": float(raw[-1]),
        "start_aligned_ADE": float(start_errors.mean()),
        "start_aligned_FDE": float(start_errors[-1]),
        "centroid_aligned_ADE": float(centroid_errors.mean()),
        "centroid_aligned_FDE": float(centroid_errors[-1]),
        "scale_normalized_shape_ADE": float(scale_errors.mean()),
        "scale_normalized_shape_FDE": float(scale_errors[-1]),
        "pred_centroid_x": float(pred_centroid[0]),
        "pred_centroid_y": float(pred_centroid[1]),
        "gt_centroid_x": float(gt_centroid[0]),
        "gt_centroid_y": float(gt_centroid[1]),
        "pred_shape_scale": float(pred_scale),
        "gt_shape_scale": float(gt_scale),
        "shape_scale_ratio": float(shape_scale_ratio),
    }


def point_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=1)


def rms_radius(centered: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def summarize(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for split, split_frame in per_sample.groupby("split", sort=False):
        output[str(split)] = {}
        for model, model_frame in split_frame.groupby("model", sort=False):
            output[str(split)][str(model)] = {}
            for metric in METRIC_COLUMNS:
                values = model_frame[metric].to_numpy(dtype=float)
                output[str(split)][str(model)][metric] = metric_summary(values)
    return output


def metric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
        "p75": float(np.nanpercentile(values, 75)),
        "p90": float(np.nanpercentile(values, 90)),
        "p95": float(np.nanpercentile(values, 95)),
        "max": float(np.nanmax(values)),
    }


def make_test_comparison(per_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
    test = per_sample[per_sample["split"] == "test"].copy()
    comparison: dict[str, dict[str, float]] = {}
    by_model = {model: frame for model, frame in test.groupby("model")}
    absolute = by_model["absolute_resnet18"].set_index("sample_id")
    delta = by_model["delta_integrated_resnet18"].set_index("sample_id")
    common_ids = sorted(set(absolute.index) & set(delta.index))
    if len(common_ids) != len(absolute) or len(common_ids) != len(delta):
        raise RuntimeError("Absolute and delta test sample_id sets do not match.")

    for metric in [
        "raw_ADE",
        "raw_FDE",
        "start_aligned_ADE",
        "start_aligned_FDE",
        "centroid_aligned_ADE",
        "centroid_aligned_FDE",
        "scale_normalized_shape_ADE",
        "scale_normalized_shape_FDE",
        "shape_scale_ratio",
    ]:
        abs_values = absolute.loc[common_ids, metric].to_numpy(dtype=float)
        delta_values = delta.loc[common_ids, metric].to_numpy(dtype=float)
        abs_mean = float(np.nanmean(abs_values))
        delta_mean = float(np.nanmean(delta_values))
        comparison[metric] = {
            "absolute_resnet18_mean": abs_mean,
            "delta_integrated_resnet18_mean": delta_mean,
            "delta_minus_absolute": float(delta_mean - abs_mean),
            "percent_change_vs_absolute": float((delta_mean - abs_mean) / abs_mean * 100.0)
            if abs_mean
            else float("nan"),
            "delta_better_count": int(np.nansum(delta_values < abs_values)),
            "absolute_better_count": int(np.nansum(abs_values < delta_values)),
        }
    return comparison


def make_comparison_table(comparison: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for metric, values in comparison.items():
        rows.append({"metric": metric, **values})
    return pd.DataFrame(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

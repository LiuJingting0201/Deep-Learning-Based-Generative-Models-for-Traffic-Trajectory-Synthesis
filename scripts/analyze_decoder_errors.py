"""Analyze failure modes of the no-speed decoder experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = experiment_dir / "error_analysis"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(experiment_dir / "evaluation" / "per_sample_metrics.csv")
    split_metadata = pd.read_csv(data_root / "splits" / "split_metadata.csv")
    merged = metrics.merge(
        split_metadata[["sample_id", "label_path"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if merged["label_path"].isna().any():
        raise RuntimeError("Some per-sample metric rows do not match split_metadata.csv.")

    rows = []
    for row in merged.itertuples(index=False):
        true = np.load(data_root / row.label_path).astype(np.float32)
        pred_path = experiment_dir / "evaluation" / "predictions" / f"{row.sample_id}_pred.npy"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred = np.load(pred_path).astype(np.float32)
        traj_features = compute_trajectory_features(true)
        error_features = compute_error_features(pred, true)
        rows.append(
            {
                "sample_id": row.sample_id,
                "vehicle_id": row.vehicle_id,
                "split": row.split,
                "ADE": row.ADE,
                "FDE": row.FDE,
                "RMSE": row.RMSE,
                "MAE": row.MAE,
                **traj_features,
                **error_features,
            }
        )

    table = pd.DataFrame(rows)
    table = assign_ade_groups(table)
    table.to_csv(output_dir / "error_feature_table.csv", index=False)

    correlation_summary = compute_correlations(table)
    (output_dir / "correlation_summary.json").write_text(json.dumps(correlation_summary, indent=2))

    group_summary = (
        table.groupby(["split", "ade_group"])
        .mean(numeric_only=True)
        .reset_index()
    )
    group_summary.to_csv(output_dir / "ade_group_feature_summary.csv", index=False)

    plot_outputs(table, plots_dir)
    write_summary(table, correlation_summary, output_dir)
    print((output_dir / "summary_zh.txt").read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_original_config"),
    )
    return parser.parse_args()


def compute_trajectory_features(points: np.ndarray) -> dict[str, float]:
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    trajectory_length = float(segment_lengths.sum())
    displacement = float(np.linalg.norm(points[-1] - points[0]))
    tortuosity = trajectory_length / max(displacement, 1e-6)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    width = float(maxs[0] - mins[0])
    height = float(maxs[1] - mins[1])
    angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
    angle_delta = np.abs(np.diff(np.unwrap(angles)))
    sharp_turn_count = int((angle_delta > np.deg2rad(45.0)).sum())
    return {
        "trajectory_length": trajectory_length,
        "start_end_displacement": displacement,
        "tortuosity": float(tortuosity),
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "sharp_turn_count": sharp_turn_count,
        "start_x": float(points[0, 0]),
        "start_y": float(points[0, 1]),
        "end_x": float(points[-1, 0]),
        "end_y": float(points[-1, 1]),
    }


def compute_error_features(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred_center = pred.mean(axis=0)
    true_center = true.mean(axis=0)
    pred_start_aligned = pred + (true[0] - pred[0])
    pred_centroid_aligned = pred + (true_center - pred_center)
    length_error = abs(trajectory_length(pred) - trajectory_length(true))
    return {
        "start_point_error": float(np.linalg.norm(pred[0] - true[0])),
        "end_point_error": float(np.linalg.norm(pred[-1] - true[-1])),
        "centroid_error": float(np.linalg.norm(pred_center - true_center)),
        "shape_error_start_aligned": float(
            np.linalg.norm(pred_start_aligned - true, axis=1).mean()
        ),
        "shape_error_centroid_aligned": float(
            np.linalg.norm(pred_centroid_aligned - true, axis=1).mean()
        ),
        "trajectory_length_error_recomputed": float(length_error),
    }


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def assign_ade_groups(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["ade_group"] = "middle"
    for split, idx in table.groupby("split").groups.items():
        split_values = table.loc[idx, "ADE"]
        q10 = split_values.quantile(0.10)
        q45 = split_values.quantile(0.45)
        q55 = split_values.quantile(0.55)
        q90 = split_values.quantile(0.90)
        table.loc[idx.intersection(table.index[table["ADE"] <= q10]), "ade_group"] = "good"
        median_mask = table.index[
            (table["split"] == split) & (table["ADE"] >= q45) & (table["ADE"] <= q55)
        ]
        table.loc[median_mask, "ade_group"] = "median"
        table.loc[idx.intersection(table.index[table["ADE"] >= q90]), "ade_group"] = "bad"
    return table


def compute_correlations(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    features = [
        "trajectory_length",
        "tortuosity",
        "bbox_area",
        "start_end_displacement",
        "sharp_turn_count",
        "start_point_error",
        "end_point_error",
        "centroid_error",
        "shape_error_start_aligned",
        "shape_error_centroid_aligned",
    ]
    summary: dict[str, dict[str, float]] = {}
    for split, frame in table.groupby("split"):
        summary[split] = {}
        for feature in features:
            summary[split][f"ADE_vs_{feature}"] = float(frame["ADE"].corr(frame[feature]))
            summary[split][f"FDE_vs_{feature}"] = float(frame["FDE"].corr(frame[feature]))
    return summary


def plot_outputs(table: pd.DataFrame, plots_dir: Path) -> None:
    test = table[table["split"] == "test"].copy()
    plot_hist(test["ADE"], plots_dir / "hist_ADE.png", "Test ADE")
    plot_hist(test["FDE"], plots_dir / "hist_FDE.png", "Test FDE")
    for feature, label in [
        ("trajectory_length", "trajectory length"),
        ("tortuosity", "tortuosity"),
        ("bbox_area", "bbox area"),
        ("start_point_error", "start error"),
        ("end_point_error", "end error"),
        ("centroid_error", "centroid error"),
    ]:
        plot_scatter(test, feature, plots_dir / f"ADE_vs_{feature}.png", label)


def plot_hist(values: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=30)
    ax.set_title(title)
    ax.set_xlabel(title)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(table: pd.DataFrame, feature: str, path: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(table[feature], table["ADE"], s=16, alpha=0.75)
    ax.set_xlabel(label)
    ax.set_ylabel("ADE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(table: pd.DataFrame, correlations: dict[str, dict[str, float]], output_dir: Path) -> None:
    test = table[table["split"] == "test"].copy()
    grouped = test.groupby("ade_group").mean(numeric_only=True)
    good = grouped.loc["good"] if "good" in grouped.index else None
    bad = grouped.loc["bad"] if "bad" in grouped.index else None
    corr = correlations["test"]

    if good is not None and bad is not None:
        length_note = "更长" if bad["trajectory_length"] > good["trajectory_length"] else "不更长"
        tort_note = "更曲折" if bad["tortuosity"] > good["tortuosity"] else "不更曲折"
        endpoint_note = (
            "端点误差很重要"
            if bad["end_point_error"] > bad["shape_error_centroid_aligned"]
            else "形状/拓扑误差不小于端点误差"
        )
        shift_note = (
            "空间平移/定位误差占比较大"
            if bad["centroid_error"] > bad["shape_error_centroid_aligned"]
            else "对齐质心后仍有较大形状误差"
        )
    else:
        length_note = tort_note = endpoint_note = shift_note = "样本分组不足，无法判断"

    summary = f"""
Decoder 错误诊断摘要

1. worst cases 是否更长：{length_note}。
2. worst cases 是否更曲折：{tort_note}。
3. test ADE 与轨迹长度相关系数：{corr['ADE_vs_trajectory_length']:.4f}。
4. test ADE 与 tortuosity 相关系数：{corr['ADE_vs_tortuosity']:.4f}。
5. test ADE 与 bbox area 相关系数：{corr['ADE_vs_bbox_area']:.4f}。
6. test ADE 与 sharp turn count 相关系数：{corr['ADE_vs_sharp_turn_count']:.4f}。
7. 端点误差判断：{endpoint_note}。
8. 空间 shift vs shape 判断：{shift_note}。
9. 如果 centroid/start 对齐后的 shape error 仍高，说明失败不是单纯绝对位置偏移；如果对齐后明显降低，则说明模型恢复了部分形状但定位较弱。
"""
    (output_dir / "summary_zh.txt").write_text(summary.strip() + "\n")


if __name__ == "__main__":
    main()

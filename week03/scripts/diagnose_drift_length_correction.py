"""Post-hoc drift and trajectory-length diagnostics for Week3 experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/home/jliu/Thesis/week03/logs/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DEFAULT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
OUTPUT_DIR_DEFAULT = PROJECT_ROOT / "week03" / "results" / "drift_length_diagnostics"
DEFAULT_EXPERIMENTS = {
    "raw_resnet18": PROJECT_ROOT / "week03" / "results" / "baseline_raw_resnet18_bs16",
    "hard_resnet18": PROJECT_ROOT / "week03" / "results" / "experiment_b_hard_resnet18_bs16",
    "decomp_resnet18": PROJECT_ROOT / "week03" / "results" / "experiment_c_decomp_resnet18_bs16",
    "decomp_mtf_local_resnet18": PROJECT_ROOT
    / "week03"
    / "results"
    / "experiment_d_decomp_mtf_local_resnet18_bs16",
}
STEPS = 224
SCALE_MIN = 0.5
SCALE_MAX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--experiment", action="append", default=[], help="Repeatable: name=/path/to/experiment_dir")
    parser.add_argument("--splits", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    experiments = parse_experiments(args.experiment)
    splits = ["train", "val", "test"] if args.splits == "all" else [args.splits]
    rng = np.random.default_rng(args.seed)

    summary_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    analyzed: list[dict[str, str]] = []
    for experiment_name, experiment_dir in experiments.items():
        experiment_dir = experiment_dir.resolve()
        load_result = load_experiment_tables(experiment_name, experiment_dir, args.data_root.resolve())
        if load_result["skip_reason"]:
            skipped.append({"experiment_name": experiment_name, "reason": load_result["skip_reason"]})
            print(f"WARNING: skipping {experiment_name}: {load_result['skip_reason']}", file=sys.stderr)
            continue

        per_sample = load_result["per_sample"]
        metadata = load_result["metadata"]
        scale_info = estimate_length_scale(experiment_dir, metadata, per_sample, args.data_root.resolve())

        for split in splits:
            split_metrics = per_sample[per_sample["split"] == split].copy()
            if split_metrics.empty:
                skipped.append({"experiment_name": experiment_name, "reason": f"no rows for split={split}"})
                print(f"WARNING: skipping {experiment_name}/{split}: no per-sample rows", file=sys.stderr)
                continue

            rows, skipped_count = diagnose_split(
                experiment_name,
                experiment_dir,
                split,
                split_metrics,
                metadata,
                args.data_root.resolve(),
                scale_info,
            )
            if not rows:
                skipped.append(
                    {
                        "experiment_name": experiment_name,
                        "reason": f"no loadable prediction/label pairs for split={split}",
                    }
                )
                print(f"WARNING: skipping {experiment_name}/{split}: no loadable samples", file=sys.stderr)
                continue

            diagnostics = pd.DataFrame(rows)
            csv_path = output_dir / f"{experiment_name}_{split}_per_sample_drift_diagnostics.csv"
            diagnostics.to_csv(csv_path, index=False)
            summary_rows.append(summarize_split(experiment_name, split, diagnostics, scale_info))
            analyzed.append(
                {
                    "experiment_name": experiment_name,
                    "split": split,
                    "num_samples": str(len(diagnostics)),
                    "skipped_samples": str(skipped_count),
                }
            )
            make_plots(
                output_dir,
                experiment_name,
                split,
                experiment_dir,
                metadata,
                args.data_root.resolve(),
                diagnostics,
                args.num_plots,
                rng,
                float(scale_info["val_estimated_scale_clamped"]),
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "drift_length_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_markdown_report(output_dir, summary, analyzed, skipped)
    write_run_manifest(output_dir, args, analyzed, skipped)

    print(f"summary_csv={summary_path}")
    print(f"summary_md={output_dir / 'drift_length_summary.md'}")
    if analyzed:
        print("analyzed=" + ", ".join(f"{row['experiment_name']}:{row['split']}" for row in analyzed))
    if skipped:
        print("skipped=" + json.dumps(skipped, indent=2))


def parse_experiments(raw_experiments: list[str]) -> dict[str, Path]:
    if not raw_experiments:
        return dict(DEFAULT_EXPERIMENTS)
    experiments: dict[str, Path] = {}
    for raw in raw_experiments:
        if "=" not in raw:
            raise ValueError(f"Invalid --experiment {raw!r}; expected name=/path")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --experiment {raw!r}; empty name")
        experiments[name] = Path(path).expanduser()
    return experiments


def load_experiment_tables(experiment_name: str, experiment_dir: Path, data_root: Path) -> dict[str, Any]:
    evaluation_dir = experiment_dir / "evaluation"
    pred_delta_dir = evaluation_dir / "predictions_delta_displacement"
    pred_abs_dir = evaluation_dir / "predictions_absolute_integrated"
    per_sample_path = evaluation_dir / "per_sample_metrics.csv"
    if not experiment_dir.exists():
        return {"skip_reason": f"missing experiment directory: {experiment_dir}"}
    if not per_sample_path.exists():
        return {"skip_reason": f"missing per_sample_metrics.csv: {per_sample_path}"}
    if not pred_delta_dir.is_dir() or not pred_abs_dir.is_dir():
        return {"skip_reason": f"missing prediction directories under {evaluation_dir}"}

    metadata_path = experiment_dir / "split_copy.csv"
    if not metadata_path.exists():
        split_path = data_root / "splits" / "split_metadata.csv"
        metadata_path = split_path if split_path.exists() else data_root / "metadata.csv"
    if not metadata_path.exists():
        return {"skip_reason": f"missing metadata fallback for {experiment_name}"}

    per_sample = pd.read_csv(per_sample_path)
    metadata = pd.read_csv(metadata_path)
    if "sample_id" not in per_sample or "split" not in per_sample:
        return {"skip_reason": f"per_sample_metrics.csv lacks sample_id/split columns: {per_sample_path}"}
    if "sample_id" not in metadata:
        return {"skip_reason": f"metadata lacks sample_id column: {metadata_path}"}
    return {"skip_reason": "", "per_sample": per_sample, "metadata": metadata, "metadata_path": metadata_path}


def estimate_length_scale(
    experiment_dir: Path, metadata: pd.DataFrame, per_sample: pd.DataFrame, data_root: Path
) -> dict[str, Any]:
    scale_split = "val" if (per_sample["split"] == "val").any() else str(per_sample["split"].iloc[0])
    source = "val_estimated_scale" if scale_split == "val" else "same_split_oracle_scale"
    ratios: list[float] = []
    for sample_id in per_sample.loc[per_sample["split"] == scale_split, "sample_id"].astype(str):
        try:
            pred_delta, _, true_abs, start = load_sample_arrays(experiment_dir, metadata, data_root, sample_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        pred_abs = integrate_delta(start, pred_delta)
        pred_length = trajectory_length(pred_abs)
        true_length = trajectory_length(true_abs)
        if pred_length > 1e-8 and math.isfinite(true_length):
            ratios.append(true_length / pred_length)
    if ratios:
        raw_scale = float(np.mean(ratios))
    else:
        raw_scale = 1.0
        source = "fallback_scale_1_missing_val_predictions"
    return {
        "scale_split": scale_split,
        "scale_source": source,
        "val_estimated_scale": raw_scale,
        "val_estimated_scale_clamped": clamp_scale(raw_scale),
        "scale_num_samples": len(ratios),
    }


def diagnose_split(
    experiment_name: str,
    experiment_dir: Path,
    split: str,
    split_metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    data_root: Path,
    scale_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_count = 0
    scale = float(scale_info["val_estimated_scale_clamped"])
    for sample_id in split_metrics["sample_id"].astype(str):
        try:
            pred_delta, pred_abs, true_abs, start = load_sample_arrays(experiment_dir, metadata, data_root, sample_id)
            true_delta = load_true_delta(metadata, data_root, sample_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped_count += 1
            print(f"WARNING: skipping {experiment_name}/{split}/{sample_id}: {exc}", file=sys.stderr)
            continue

        row = compute_sample_diagnostics(sample_id, split, pred_delta, pred_abs, true_delta, true_abs, start, scale)
        rows.append(row)
    return rows, skipped_count


def load_sample_arrays(
    experiment_dir: Path, metadata: pd.DataFrame, data_root: Path, sample_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_delta_path = experiment_dir / "evaluation" / "predictions_delta_displacement" / f"{sample_id}_pred_delta.npy"
    pred_abs_path = (
        experiment_dir
        / "evaluation"
        / "predictions_absolute_integrated"
        / f"{sample_id}_pred_absolute_integrated.npy"
    )
    pred_delta = load_npy(pred_delta_path, (STEPS, 2))
    pred_abs = load_npy(pred_abs_path, (STEPS, 2))
    true_abs = load_true_abs(metadata, data_root, sample_id)
    start = load_start(metadata, data_root, sample_id)
    return pred_delta, pred_abs, true_abs, start


def metadata_row(metadata: pd.DataFrame, sample_id: str) -> pd.Series:
    matches = metadata[metadata["sample_id"].astype(str) == sample_id]
    if matches.empty:
        raise KeyError(f"sample_id {sample_id} not found in metadata")
    return matches.iloc[0]


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_root / path


def load_true_abs(metadata: pd.DataFrame, data_root: Path, sample_id: str) -> np.ndarray:
    row = metadata_row(metadata, sample_id)
    return load_npy(resolve_data_path(data_root, row["label_absolute_path"]), (STEPS, 2))


def load_true_delta(metadata: pd.DataFrame, data_root: Path, sample_id: str) -> np.ndarray:
    row = metadata_row(metadata, sample_id)
    return load_npy(resolve_data_path(data_root, row["label_delta_displacement_path"]), (STEPS, 2))


def load_start(metadata: pd.DataFrame, data_root: Path, sample_id: str) -> np.ndarray:
    row = metadata_row(metadata, sample_id)
    return load_npy(resolve_data_path(data_root, row["start_path"]), (2,))


def load_npy(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    array = np.load(path).astype(np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{path} expected shape {expected_shape}, got {array.shape}")
    return array


def compute_sample_diagnostics(
    sample_id: str,
    split: str,
    pred_delta: np.ndarray,
    pred_abs: np.ndarray,
    true_delta: np.ndarray,
    true_abs: np.ndarray,
    start: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    original_delta = basic_metrics(pred_delta, true_delta)
    original_integrated = basic_metrics(pred_abs, true_abs)
    pred_length = trajectory_length(pred_abs)
    true_length = trajectory_length(true_abs)
    original_ratio = safe_divide(pred_length, true_length)

    pred_delta_scaled = pred_delta * scale
    pred_abs_scaled = integrate_delta(start, pred_delta_scaled)
    scaled_integrated = basic_metrics(pred_abs_scaled, true_abs)
    scaled_ratio = safe_divide(trajectory_length(pred_abs_scaled), true_length)

    oracle_scale_raw = safe_divide(true_length, pred_length)
    oracle_scale = clamp_scale(oracle_scale_raw)
    pred_abs_oracle = integrate_delta(start, pred_delta * oracle_scale)
    oracle_integrated = basic_metrics(pred_abs_oracle, true_abs)
    oracle_ratio = safe_divide(trajectory_length(pred_abs_oracle), true_length)

    endpoint_error = true_abs[-1] - pred_abs[-1]
    alpha = np.linspace(0.0, 1.0, STEPS, dtype=np.float64).reshape(STEPS, 1)
    pred_abs_endpoint = pred_abs + alpha * endpoint_error
    endpoint_integrated = basic_metrics(pred_abs_endpoint, true_abs)
    endpoint_ratio = safe_divide(trajectory_length(pred_abs_endpoint), true_length)

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "original_delta_ADE": original_delta["ADE"],
        "original_delta_FDE": original_delta["FDE"],
        "original_integrated_ADE": original_integrated["ADE"],
        "original_integrated_FDE": original_integrated["FDE"],
        "original_trajectory_length_ratio": original_ratio,
        "original_pred_length": pred_length,
        "true_length": true_length,
        "original_end_error": original_integrated["FDE"],
        "global_length_scale": scale,
        "scaled_integrated_ADE": scaled_integrated["ADE"],
        "scaled_integrated_FDE": scaled_integrated["FDE"],
        "scaled_trajectory_length_ratio": scaled_ratio,
        "scaled_end_error": scaled_integrated["FDE"],
        "oracle_length_scale_raw": oracle_scale_raw,
        "oracle_length_scale_clamped": oracle_scale,
        "oracle_length_scaled_integrated_ADE": oracle_integrated["ADE"],
        "oracle_length_scaled_integrated_FDE": oracle_integrated["FDE"],
        "oracle_length_scaled_trajectory_length_ratio": oracle_ratio,
        "oracle_length_scaled_end_error": oracle_integrated["FDE"],
        "endpoint_corrected_ADE": endpoint_integrated["ADE"],
        "endpoint_corrected_FDE": endpoint_integrated["FDE"],
        "endpoint_corrected_trajectory_length_ratio": endpoint_ratio,
        "endpoint_corrected_end_error": endpoint_integrated["FDE"],
    }
    for anchor_count in [4, 8, 16]:
        corrected = piecewise_correct(pred_abs, true_abs, anchor_count)
        metrics = basic_metrics(corrected, true_abs)
        row[f"piecewise{anchor_count}_corrected_ADE"] = metrics["ADE"]
        row[f"piecewise{anchor_count}_corrected_FDE"] = metrics["FDE"]
        row[f"piecewise{anchor_count}_corrected_end_error"] = metrics["FDE"]
    row["improvement_from_endpoint_correction"] = row["original_integrated_ADE"] - row["endpoint_corrected_ADE"]
    row["improvement_from_length_scaling"] = row["original_integrated_ADE"] - row["scaled_integrated_ADE"]
    row["abs_length_ratio_error"] = abs(1.0 - row["original_trajectory_length_ratio"])
    return row


def basic_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(pred - true, axis=1)
    return {"ADE": float(errors.mean()), "FDE": float(errors[-1])}


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def integrate_delta(start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    absolute = np.empty_like(delta, dtype=np.float64)
    absolute[0] = start
    absolute[1:] = start.reshape(1, 2) + np.cumsum(delta[1:], axis=0)
    return absolute


def clamp_scale(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return float(np.clip(value, SCALE_MIN, SCALE_MAX))


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return float("nan")
    return float(numerator / denominator)


def piecewise_correct(pred_abs: np.ndarray, true_abs: np.ndarray, anchor_count: int) -> np.ndarray:
    anchors = np.linspace(0, STEPS - 1, anchor_count).round().astype(int)
    anchors = np.unique(anchors)
    corrections = true_abs[anchors] - pred_abs[anchors]
    timesteps = np.arange(STEPS)
    interp_x = np.interp(timesteps, anchors, corrections[:, 0])
    interp_y = np.interp(timesteps, anchors, corrections[:, 1])
    return pred_abs + np.stack([interp_x, interp_y], axis=1)


def summarize_split(
    experiment_name: str, split: str, diagnostics: pd.DataFrame, scale_info: dict[str, Any]
) -> dict[str, Any]:
    original = float(diagnostics["original_integrated_ADE"].mean())
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "split": split,
        "num_samples": int(len(diagnostics)),
        "val_estimated_scale": scale_info["val_estimated_scale"],
        "val_estimated_scale_clamped": scale_info["val_estimated_scale_clamped"],
        "scale_source": scale_info["scale_source"],
        "scale_num_samples": scale_info["scale_num_samples"],
    }
    mean_columns = [
        "original_integrated_ADE",
        "original_integrated_FDE",
        "original_trajectory_length_ratio",
        "scaled_integrated_ADE",
        "scaled_integrated_FDE",
        "scaled_trajectory_length_ratio",
        "oracle_length_scaled_integrated_ADE",
        "oracle_length_scaled_integrated_FDE",
        "endpoint_corrected_ADE",
        "endpoint_corrected_FDE",
        "piecewise4_corrected_ADE",
        "piecewise8_corrected_ADE",
        "piecewise16_corrected_ADE",
    ]
    for column in mean_columns:
        row[f"{column}_mean"] = float(diagnostics[column].mean())
    for label, column in [
        ("endpoint_ADE_improvement_percent", "endpoint_corrected_ADE"),
        ("scaled_ADE_improvement_percent", "scaled_integrated_ADE"),
        ("oracle_length_scaled_ADE_improvement_percent", "oracle_length_scaled_integrated_ADE"),
        ("piecewise4_ADE_improvement_percent", "piecewise4_corrected_ADE"),
        ("piecewise8_ADE_improvement_percent", "piecewise8_corrected_ADE"),
        ("piecewise16_ADE_improvement_percent", "piecewise16_corrected_ADE"),
    ]:
        row[label] = percent_improvement(original, float(diagnostics[column].mean()))
    row.update(correlation_summary(diagnostics))
    return row


def percent_improvement(original: float, corrected: float) -> float:
    if abs(original) <= 1e-12:
        return float("nan")
    return float(100.0 * (original - corrected) / original)


def correlation_summary(diagnostics: pd.DataFrame) -> dict[str, float]:
    pairs = {
        "corr_ade_vs_length_ratio": ("original_integrated_ADE", "original_trajectory_length_ratio"),
        "corr_ade_vs_abs_length_ratio_error": ("original_integrated_ADE", "abs_length_ratio_error"),
        "corr_fde_vs_end_error": ("original_integrated_FDE", "original_end_error"),
        "corr_endpoint_improvement_vs_original_ade": (
            "improvement_from_endpoint_correction",
            "original_integrated_ADE",
        ),
        "corr_length_scaling_improvement_vs_abs_length_ratio_error": (
            "improvement_from_length_scaling",
            "abs_length_ratio_error",
        ),
    }
    out: dict[str, float] = {}
    scipy_pearson = None
    try:
        from scipy.stats import pearsonr  # type: ignore

        scipy_pearson = pearsonr
    except Exception:
        scipy_pearson = None
    for name, (left, right) in pairs.items():
        x = diagnostics[left].to_numpy(dtype=np.float64)
        y = diagnostics[right].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
            out[name] = float("nan")
            out[f"{name}_pvalue"] = float("nan")
        elif scipy_pearson is not None:
            r, p = scipy_pearson(x[valid], y[valid])
            out[name] = float(r)
            out[f"{name}_pvalue"] = float(p)
        else:
            out[name] = float(np.corrcoef(x[valid], y[valid])[0, 1])
            out[f"{name}_pvalue"] = float("nan")
    return out


def make_plots(
    output_dir: Path,
    experiment_name: str,
    split: str,
    experiment_dir: Path,
    metadata: pd.DataFrame,
    data_root: Path,
    diagnostics: pd.DataFrame,
    num_plots: int,
    rng: np.random.Generator,
    scale: float,
) -> None:
    plot_dir = output_dir / "plots" / experiment_name / split
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_histograms(plot_dir, diagnostics)
    if num_plots <= 0:
        return
    selected = select_plot_samples(diagnostics, num_plots, rng)
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        plot_sample(plot_dir, rank, row, experiment_dir, metadata, data_root, scale)


def make_histograms(plot_dir: Path, diagnostics: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    bins = 40
    plt.hist(diagnostics["original_integrated_ADE"], bins=bins, alpha=0.45, label="original")
    plt.hist(diagnostics["scaled_integrated_ADE"], bins=bins, alpha=0.45, label="scaled")
    plt.hist(diagnostics["endpoint_corrected_ADE"], bins=bins, alpha=0.45, label="endpoint-corrected")
    plt.xlabel("Integrated ADE")
    plt.ylabel("Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_integrated_ade_original_scaled_endpoint.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(diagnostics["original_trajectory_length_ratio"], bins=40, color="#4c78a8", alpha=0.8)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Original trajectory length ratio")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_original_trajectory_length_ratio.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    improvement = diagnostics["original_integrated_ADE"] - diagnostics["endpoint_corrected_ADE"]
    plt.hist(improvement, bins=40, color="#59a14f", alpha=0.8)
    plt.xlabel("ADE improvement from endpoint correction")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_endpoint_correction_improvement.png", dpi=150)
    plt.close()


def select_plot_samples(diagnostics: pd.DataFrame, num_plots: int, rng: np.random.Generator) -> pd.DataFrame:
    sorted_diag = diagnostics.sort_values("original_integrated_ADE").reset_index(drop=True)
    indices = {0, len(sorted_diag) // 2, len(sorted_diag) - 1}
    remaining = [idx for idx in range(len(sorted_diag)) if idx not in indices]
    extra_count = max(0, min(num_plots, len(sorted_diag)) - len(indices))
    if extra_count and remaining:
        indices.update(rng.choice(remaining, size=min(extra_count, len(remaining)), replace=False).tolist())
    return sorted_diag.iloc[sorted(indices)].copy()


def plot_sample(
    plot_dir: Path,
    rank: int,
    row: pd.Series,
    experiment_dir: Path,
    metadata: pd.DataFrame,
    data_root: Path,
    scale: float,
) -> None:
    sample_id = str(row["sample_id"])
    pred_delta, pred_abs, true_abs, start = load_sample_arrays(experiment_dir, metadata, data_root, sample_id)
    scaled_abs = integrate_delta(start, pred_delta * scale)
    endpoint_error = true_abs[-1] - pred_abs[-1]
    alpha = np.linspace(0.0, 1.0, STEPS, dtype=np.float64).reshape(STEPS, 1)
    endpoint_abs = pred_abs + alpha * endpoint_error
    piecewise16_abs = piecewise_correct(pred_abs, true_abs, 16)

    plt.figure(figsize=(7, 7))
    plt.plot(true_abs[:, 0], true_abs[:, 1], color="black", linewidth=2.0, label="true absolute")
    plt.plot(pred_abs[:, 0], pred_abs[:, 1], color="#4c78a8", linewidth=1.4, label="original integrated")
    plt.plot(scaled_abs[:, 0], scaled_abs[:, 1], color="#f58518", linewidth=1.4, label="validation-scaled")
    plt.plot(endpoint_abs[:, 0], endpoint_abs[:, 1], color="#54a24b", linewidth=1.4, label="endpoint-corrected")
    plt.plot(piecewise16_abs[:, 0], piecewise16_abs[:, 1], color="#b279a2", linewidth=1.4, label="piecewise16")
    plt.scatter(true_abs[0, 0], true_abs[0, 1], color="black", s=25, marker="o")
    plt.scatter(true_abs[-1, 0], true_abs[-1, 1], color="black", s=30, marker="x")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=8)
    plt.title(
        f"{sample_id} | original ADE/FDE {row['original_integrated_ADE']:.2f}/{row['original_integrated_FDE']:.2f} | "
        f"scaled {row['scaled_integrated_ADE']:.2f}/{row['scaled_integrated_FDE']:.2f} | "
        f"endpoint ADE {row['endpoint_corrected_ADE']:.2f} | "
        f"piecewise16 ADE {row['piecewise16_corrected_ADE']:.2f} | "
        f"len ratio {row['original_trajectory_length_ratio']:.3f}",
        fontsize=8,
    )
    plt.tight_layout()
    safe_sample = sample_id.replace("/", "_")
    plt.savefig(plot_dir / f"{rank:02d}_{safe_sample}_drift_corrections.png", dpi=150)
    plt.close()


def write_markdown_report(
    output_dir: Path, summary: pd.DataFrame, analyzed: list[dict[str, str]], skipped: list[dict[str, str]]
) -> None:
    lines = [
        "# Drift And Length Diagnostics",
        "",
        "## Purpose",
        "This is a post-hoc diagnostic to decide whether accumulated drift or trajectory length mismatch is the main bottleneck behind high integrated ADE/FDE. It reads existing model predictions only and does not train or modify any model.",
        "",
        "## Methods",
        "- Validation-estimated length scaling applies one global delta scale estimated from validation true/predicted trajectory length ratios.",
        "- Per-sample oracle length scaling uses the true length for each sample as an upper bound on length-bias correction.",
        "- Oracle endpoint linear correction spreads final endpoint error linearly across the trajectory.",
        "- Piecewise oracle correction uses linearly interpolated corrections from K=4/8/16 anchor points.",
        "",
        "## Summary Table",
    ]
    if summary.empty:
        lines.append("No experiments with complete predictions were found.")
    else:
        display_columns = [
            "experiment_name",
            "split",
            "num_samples",
            "original_integrated_ADE_mean",
            "original_integrated_FDE_mean",
            "scaled_integrated_ADE_mean",
            "scaled_integrated_FDE_mean",
            "oracle_length_scaled_integrated_ADE_mean",
            "oracle_length_scaled_integrated_FDE_mean",
            "endpoint_corrected_ADE_mean",
            "piecewise16_corrected_ADE_mean",
        ]
        lines.extend(markdown_table(summary[display_columns]))
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "- If validation-estimated scaling improves ADE/FDE, systematic delta magnitude bias exists.",
            "- If oracle length scaling helps but validation scale does not, length mismatch is sample-specific.",
            "- If endpoint correction drastically reduces ADE, low-frequency drift dominates.",
            "- If piecewise correction helps much more than endpoint correction, drift changes over time.",
            "- If none helps, representation or local shape mismatch dominates.",
            "",
            "## Per-Experiment Interpretation",
        ]
    )
    if summary.empty:
        lines.append("- No completed experiment predictions were available.")
    else:
        for _, row in summary.iterrows():
            lines.append(f"- {row['experiment_name']} / {row['split']}: {automatic_interpretation(row)}")
    lines.extend(["", "## Analyzed"])
    lines.extend([f"- {row['experiment_name']} / {row['split']}: {row['num_samples']} samples" for row in analyzed] or ["- None"])
    lines.extend(["", "## Skipped"])
    lines.extend([f"- {row['experiment_name']}: {row['reason']}" for row in skipped] or ["- None"])
    (output_dir / "drift_length_summary.md").write_text("\n".join(lines) + "\n")


def markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = frame.columns.tolist()
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for header in headers:
            value = row[header]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def automatic_interpretation(row: pd.Series) -> str:
    scaled = float(row["scaled_ADE_improvement_percent"])
    oracle = float(row["oracle_length_scaled_ADE_improvement_percent"])
    endpoint = float(row["endpoint_ADE_improvement_percent"])
    piecewise16 = float(row["piecewise16_ADE_improvement_percent"])
    parts = [
        f"global length scaling improves ADE by {scaled:.1f}%",
        f"oracle length scaling by {oracle:.1f}%",
        f"endpoint correction by {endpoint:.1f}%",
        f"piecewise16 by {piecewise16:.1f}%",
    ]
    if endpoint >= 50:
        parts.append("low-frequency endpoint drift appears dominant")
    elif piecewise16 >= endpoint + 20:
        parts.append("drift varies over time rather than only at the endpoint")
    elif oracle >= 20 and scaled < 10:
        parts.append("length error is mostly sample-specific")
    elif scaled >= 20:
        parts.append("systematic delta magnitude bias is substantial")
    else:
        parts.append("local path or representation mismatch likely remains important")
    return "; ".join(parts) + "."


def write_run_manifest(
    output_dir: Path, args: argparse.Namespace, analyzed: list[dict[str, str]], skipped: list[dict[str, str]]
) -> None:
    manifest = {
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(output_dir),
        "splits": args.splits,
        "num_plots": args.num_plots,
        "seed": args.seed,
        "analyzed": analyzed,
        "skipped": skipped,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Train and evaluate a ResNet18 decoder for delta-displacement images.

This experiment keeps the existing delta-displacement dataset unchanged. The
model predicts delta_xy, and evaluation also integrates predicted deltas with
the true start point to measure absolute trajectory reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_resnet18_decoder_ablation import (  # noqa: E402
    LabelScaler,
    ResNet18TrajectoryDecoder,
    format_metric,
    plot_training_curves,
    resolve_device,
    run_epoch,
    set_seed,
    write_json,
)
from run_original_config_decoder_experiment import make_contact_sheet  # noqa: E402


DELTA_METRICS = ["delta_ADE", "delta_FDE", "delta_RMSE", "delta_MAE"]
INTEGRATED_METRICS = [
    "integrated_ADE",
    "integrated_FDE",
    "integrated_RMSE",
    "integrated_MAE",
    "start_error",
    "end_error",
    "trajectory_length_error",
    "trajectory_length_ratio",
    "cumulative_drift_final",
]


class DeltaDisplacementDataset(Dataset):
    def __init__(self, data_root: Path, metadata: pd.DataFrame, scaler: LabelScaler) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        delta = np.load(self.data_root / row.label_delta_displacement_path).astype(np.float32)
        absolute = np.load(self.data_root / row.label_absolute_path).astype(np.float32)
        start = np.load(self.data_root / row.start_path).astype(np.float32)
        return {
            "image": image_tensor.contiguous(),
            "label": torch.from_numpy(self.scaler.transform(delta).astype(np.float32)),
            "label_raw": torch.from_numpy(delta),
            "absolute": torch.from_numpy(absolute),
            "start": torch.from_numpy(start),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["checkpoints", "plots", "evaluation", "evaluation/plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(data_root)
    if not args.skip_training:
        train(args, split_metadata, device)

    summary = evaluate(args, split_metadata, device)
    compare_with_absolute_resnet18(args, summary)
    write_chinese_report(args, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_no_speed_delta_displacement_paired"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_delta_displacement_resnet18_ablation"),
    )
    parser.add_argument(
        "--absolute-resnet18-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_resnet18_ablation"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--near-zero-eps", type=float, default=1e-3)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def verify_delta_data_and_split(data_root: Path) -> pd.DataFrame:
    metadata_path = data_root / "metadata.csv"
    split_path = data_root / "splits" / "split_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    metadata = pd.read_csv(metadata_path)
    split_metadata = pd.read_csv(split_path)
    required = {
        "sample_id",
        "vehicle_id",
        "image_path",
        "label_delta_displacement_path",
        "label_absolute_path",
        "start_path",
        "split",
    }
    missing = required.difference(split_metadata.columns)
    if missing:
        raise ValueError(f"split_metadata.csv missing columns: {sorted(missing)}")
    if set(metadata["sample_id"]) != set(split_metadata["sample_id"]):
        raise RuntimeError("metadata.csv and split_metadata.csv sample_id sets do not match.")
    if split_metadata["sample_id"].duplicated().any():
        raise RuntimeError("split_metadata.csv contains duplicate sample_id values.")

    split_sets = {
        split: set(split_metadata.loc[split_metadata["split"] == split, "sample_id"])
        for split in ["train", "val", "test"]
    }
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise RuntimeError("Train/val/test split overlap detected.")
    if set.union(*split_sets.values()) != set(split_metadata["sample_id"]):
        raise RuntimeError("Some samples are not assigned to exactly one split.")

    image_min = 255
    image_max = 0
    delta_mins = []
    delta_maxs = []
    bad: list[dict[str, object]] = []
    for row in split_metadata.itertuples(index=False):
        image_path = data_root / row.image_path
        delta_path = data_root / row.label_delta_displacement_path
        absolute_path = data_root / row.label_absolute_path
        start_path = data_root / row.start_path
        for name, path in [
            ("image", image_path),
            ("delta", delta_path),
            ("absolute", absolute_path),
            ("start", start_path),
        ]:
            if not path.exists():
                bad.append({"sample_id": row.sample_id, "reason": f"missing_{name}"})
        if bad and str(bad[-1].get("sample_id")) == str(row.sample_id):
            continue
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                bad.append({"sample_id": row.sample_id, "reason": f"bad_image:{image.mode}:{image.size}"})
            pixels = np.asarray(image)
            image_min = min(image_min, int(pixels.min()))
            image_max = max(image_max, int(pixels.max()))
        delta = np.load(delta_path)
        absolute = np.load(absolute_path)
        start = np.load(start_path)
        if delta.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_delta:{delta.shape}"})
        if absolute.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_absolute:{absolute.shape}"})
        if start.shape != (2,):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_start:{start.shape}"})
        delta_mins.append(delta.min(axis=0))
        delta_maxs.append(delta.max(axis=0))

    if bad:
        report_path = data_root / "splits" / "delta_training_consistency_report.csv"
        pd.DataFrame(bad).to_csv(report_path, index=False)
        raise RuntimeError(f"Delta dataset consistency check failed. Report: {report_path}")

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    print("Delta data and split validation passed")
    print(f"total samples: {len(split_metadata)}")
    print(f"train count: {int(counts['train'])}")
    print(f"val count: {int(counts['val'])}")
    print(f"test count: {int(counts['test'])}")
    print(f"delta min: {np.vstack(delta_mins).min(axis=0).tolist()}")
    print(f"delta max: {np.vstack(delta_maxs).max(axis=0).tolist()}")
    print(f"image channel min/max: {image_min}/{image_max}")
    return split_metadata


def fit_delta_label_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [
        np.load(data_root / row.label_delta_displacement_path).astype(np.float32)
        for row in metadata.itertuples(index=False)
    ]
    stacked = np.concatenate(labels, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return LabelScaler(mean=mean.tolist(), std=std.tolist())


def train(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    scaler = fit_delta_label_scaler(data_root, train_metadata)

    write_json(output_dir / "label_scaler.json", asdict(scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = DataLoader(
        DeltaDisplacementDataset(data_root, train_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DeltaDisplacementDataset(data_root, val_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNet18TrajectoryDecoder(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    log_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        compute_diag = epoch == 1 or epoch % args.metrics_every == 0
        train_diag = evaluate_delta_loader(model, train_loader, criterion, scaler, device, compute_diag)
        val_diag = evaluate_delta_loader(model, val_loader, criterion, scaler, device, compute_diag)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_diag["loss"],
            "val_loss": val_diag["loss"],
            "train_ADE": train_diag["ade"],
            "train_FDE": train_diag["fde"],
            "val_ADE": val_diag["ade"],
            "val_FDE": val_diag["fde"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time": time.perf_counter() - start_time,
        }
        log_rows.append(row)
        write_train_log(output_dir / "train_log.csv", log_rows)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_diag["loss"],
            "train_loss": train_loss,
            "config": config,
            "label_scaler": asdict(scaler),
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last_model.pt")
        if val_diag["loss"] < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_diag["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "checkpoints" / "best_model.pt")
        else:
            epochs_without_improvement += 1

        plot_training_curves(log_rows, output_dir / "plots")
        print(
            f"epoch={epoch:04d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_diag['loss']:.6f} "
            f"train_delta_ADE={format_metric(train_diag['ade'])} "
            f"val_delta_ADE={format_metric(val_diag['ade'])} "
            f"best_val_loss={best_val_loss:.6f} best_epoch={best_epoch}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopping_triggered = True
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epochs without validation improvement."
            )
            break

    write_json(
        output_dir / "best_summary.json",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_epoch": int(log_rows[-1]["epoch"]) if log_rows else 0,
            "early_stopping_triggered": early_stopping_triggered,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
    )


def write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_eval_loss",
                "val_loss",
                "train_ADE",
                "train_FDE",
                "val_ADE",
                "val_FDE",
                "learning_rate",
                "elapsed_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_delta_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scaler: LabelScaler,
    device: torch.device,
    compute_metrics: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels_norm = batch["label"].to(device, non_blocking=True)
            pred_norm = model(images)
            loss = criterion(pred_norm, labels_norm)
            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            if compute_metrics:
                pred = scaler.inverse_transform_tensor(pred_norm)
                true = scaler.inverse_transform_tensor(labels_norm)
                point_errors = torch.linalg.norm(pred - true, dim=2)
                ade_sum += float(point_errors.mean(dim=1).sum().detach().cpu())
                fde_sum += float(point_errors[:, -1].sum().detach().cpu())
            total_samples += batch_size
    return {
        "loss": total_loss / max(1, total_samples),
        "ade": ade_sum / max(1, total_samples) if compute_metrics else float("nan"),
        "fde": fde_sum / max(1, total_samples) if compute_metrics else float("nan"),
    }


def make_config(
    args: argparse.Namespace,
    device: torch.device,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config.update(
        {
            "model": "ResNet18TrajectoryDecoder",
            "backbone": "torchvision.models.resnet18(weights=None)",
            "pretrained": False,
            "head": "feature_dim -> 1024 -> 448 -> reshape[224,2]",
            "optimizer": "Adam",
            "scheduler": "none",
            "image_normalization": "[0, 1]",
            "label_normalization": "train_mean_std_delta_displacement_xy",
            "loss": "MSELoss in normalized delta-displacement space",
            "coordinate_space": "delta_displacement_xy",
            "integrated_absolute_evaluation": "oracle true start point",
            "device_resolved": str(device),
            "train_samples": len(train_metadata),
            "val_samples": len(val_metadata),
        }
    )
    return config


def evaluate(
    args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device
) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    pred_delta_dir = evaluation_dir / "predictions_delta_displacement"
    pred_absolute_dir = evaluation_dir / "predictions_absolute_integrated"
    plots_dir = evaluation_dir / "plots"
    for directory in [pred_delta_dir, pred_absolute_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    scaler_data = json.loads((output_dir / "label_scaler.json").read_text())
    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    model = ResNet18TrajectoryDecoder(dropout=args.dropout).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    delta_timestep_rows: list[dict[str, object]] = []
    integrated_timestep_rows: list[dict[str, object]] = []
    predictions_abs: dict[str, dict[str, np.ndarray]] = {}
    truths_abs: dict[str, dict[str, np.ndarray]] = {}
    predictions_delta: dict[str, dict[str, np.ndarray]] = {}
    truths_delta: dict[str, dict[str, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = DataLoader(
            DeltaDisplacementDataset(data_root, split_frame, scaler),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        (
            split_pred_delta,
            split_true_delta,
            split_pred_abs,
            split_true_abs,
            split_rows,
            delta_timestep_ade,
            integrated_timestep_ade,
        ) = predict_split(model, loader, scaler, device, pred_delta_dir, pred_absolute_dir, split, args.near_zero_eps)
        predictions_delta[split] = split_pred_delta
        truths_delta[split] = split_true_delta
        predictions_abs[split] = split_pred_abs
        truths_abs[split] = split_true_abs
        rows.extend(split_rows)
        delta_timestep_rows.extend(
            {"split": split, "timestep": index, "delta_ade": float(value)}
            for index, value in enumerate(delta_timestep_ade)
        )
        integrated_timestep_rows.extend(
            {"split": split, "timestep": index, "integrated_ade": float(value)}
            for index, value in enumerate(integrated_timestep_ade)
        )

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.merge(
        pd.DataFrame(delta_timestep_rows),
        pd.DataFrame(integrated_timestep_rows),
        on=["split", "timestep"],
    )
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_delta_experiment_metrics(per_sample)
    distribution_summary = summarize_delta_distribution(per_sample)
    variance_summary = compute_variance_diagnostics(predictions_delta, truths_delta, predictions_abs, truths_abs)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    write_json(evaluation_dir / "delta_distribution_diagnostics.json", distribution_summary)
    write_json(evaluation_dir / "prediction_variance_diagnostics.json", variance_summary)
    plot_evaluation_outputs(per_sample, per_timestep, predictions_abs["test"], truths_abs["test"], plots_dir)
    plot_delta_distribution_diagnostics(per_sample, plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "distribution_summary": distribution_summary,
        "variance_summary": variance_summary,
    }


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    scaler: LabelScaler,
    device: torch.device,
    pred_delta_dir: Path,
    pred_absolute_dir: Path,
    split: str,
    near_zero_eps: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    pred_delta_by_id: dict[str, np.ndarray] = {}
    true_delta_by_id: dict[str, np.ndarray] = {}
    pred_abs_by_id: dict[str, np.ndarray] = {}
    true_abs_by_id: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    delta_timestep_error_sum = np.zeros(224, dtype=np.float64)
    integrated_timestep_error_sum = np.zeros(224, dtype=np.float64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            pred_norm = model(images)
            pred_delta = scaler.inverse_transform_tensor(pred_norm).cpu().numpy().astype(np.float32)
            true_delta = batch["label_raw"].cpu().numpy().astype(np.float32)
            true_abs = batch["absolute"].cpu().numpy().astype(np.float32)
            starts = batch["start"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred_delta = pred_delta[index]
                sample_true_delta = true_delta[index]
                sample_true_abs = true_abs[index]
                sample_start = starts[index]
                sample_pred_abs = integrate_delta(sample_start, sample_pred_delta)

                np.save(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_pred_delta)
                np.save(pred_absolute_dir / f"{sample_id}_pred_absolute_integrated.npy", sample_pred_abs)

                delta_metrics, delta_errors = compute_basic_metrics(sample_pred_delta, sample_true_delta, "delta")
                integrated_metrics, integrated_errors = compute_basic_metrics(
                    sample_pred_abs, sample_true_abs, "integrated"
                )
                pred_length = trajectory_length(sample_pred_abs)
                true_length = trajectory_length(sample_true_abs)
                pred_delta_mag = np.linalg.norm(sample_pred_delta, axis=1)
                true_delta_mag = np.linalg.norm(sample_true_delta, axis=1)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **delta_metrics,
                        **integrated_metrics,
                        "start_error": float(integrated_errors[0]),
                        "end_error": float(integrated_errors[-1]),
                        "trajectory_length_error": float(abs(pred_length - true_length)),
                        "trajectory_length_ratio": float(pred_length / true_length) if true_length else np.nan,
                        "cumulative_drift_final": float(integrated_errors[-1]),
                        "pred_delta_magnitude_mean": float(pred_delta_mag.mean()),
                        "true_delta_magnitude_mean": float(true_delta_mag.mean()),
                        "pred_delta_magnitude_median": float(np.median(pred_delta_mag)),
                        "true_delta_magnitude_median": float(np.median(true_delta_mag)),
                        "pred_near_zero_delta_ratio": float(np.mean(pred_delta_mag <= near_zero_eps)),
                        "true_near_zero_delta_ratio": float(np.mean(true_delta_mag <= near_zero_eps)),
                    }
                )
                pred_delta_by_id[sample_id] = sample_pred_delta
                true_delta_by_id[sample_id] = sample_true_delta
                pred_abs_by_id[sample_id] = sample_pred_abs
                true_abs_by_id[sample_id] = sample_true_abs
                delta_timestep_error_sum += delta_errors
                integrated_timestep_error_sum += integrated_errors
                total += 1

    return (
        pred_delta_by_id,
        true_delta_by_id,
        pred_abs_by_id,
        true_abs_by_id,
        rows,
        delta_timestep_error_sum / max(1, total),
        integrated_timestep_error_sum / max(1, total),
    )


def integrate_delta(start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    absolute = np.empty_like(delta, dtype=np.float32)
    absolute[0] = start.astype(np.float32)
    absolute[1:] = start.astype(np.float32) + np.cumsum(delta[1:], axis=0)
    return absolute


def compute_basic_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> tuple[dict[str, float], np.ndarray]:
    diff = pred - true
    point_errors = np.linalg.norm(diff, axis=1)
    return (
        {
            f"{prefix}_ADE": float(point_errors.mean()),
            f"{prefix}_FDE": float(point_errors[-1]),
            f"{prefix}_RMSE": float(np.sqrt(np.mean(diff**2))),
            f"{prefix}_MAE": float(np.mean(np.abs(diff))),
        },
        point_errors,
    )


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def summarize_delta_experiment_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in DELTA_METRICS + INTEGRATED_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            summary[split][metric] = {
                "mean": float(np.nanmean(values)),
                "median": float(np.nanmedian(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "p50": float(np.nanpercentile(values, 50)),
                "p75": float(np.nanpercentile(values, 75)),
                "p90": float(np.nanpercentile(values, 90)),
                "p95": float(np.nanpercentile(values, 95)),
                "max": float(np.nanmax(values)),
            }
    return summary


def summarize_delta_distribution(per_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in [
            "pred_delta_magnitude_mean",
            "true_delta_magnitude_mean",
            "pred_delta_magnitude_median",
            "true_delta_magnitude_median",
            "pred_near_zero_delta_ratio",
            "true_near_zero_delta_ratio",
            "trajectory_length_ratio",
            "cumulative_drift_final",
        ]:
            values = frame[metric].to_numpy(dtype=float)
            summary[split][metric] = float(np.nanmean(values))
    return summary


def compute_variance_diagnostics(
    predictions_delta: dict[str, dict[str, np.ndarray]],
    truths_delta: dict[str, dict[str, np.ndarray]],
    predictions_abs: dict[str, dict[str, np.ndarray]],
    truths_abs: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        pred_delta = np.stack(list(predictions_delta[split].values()))
        true_delta = np.stack(list(truths_delta[split].values()))
        pred_abs = np.stack(list(predictions_abs[split].values()))
        true_abs = np.stack(list(truths_abs[split].values()))
        delta_true_var = float(np.var(true_delta, axis=0).mean())
        abs_true_var = float(np.var(true_abs, axis=0).mean())
        diagnostics[split] = {
            "delta_ground_truth_variance": delta_true_var,
            "delta_predicted_variance": float(np.var(pred_delta, axis=0).mean()),
            "delta_predicted_to_ground_truth_variance_ratio": float(np.var(pred_delta, axis=0).mean() / delta_true_var)
            if delta_true_var
            else float("nan"),
            "integrated_ground_truth_variance": abs_true_var,
            "integrated_predicted_variance": float(np.var(pred_abs, axis=0).mean()),
            "integrated_predicted_to_ground_truth_variance_ratio": float(np.var(pred_abs, axis=0).mean() / abs_true_var)
            if abs_true_var
            else float("nan"),
        }
    return diagnostics


def plot_evaluation_outputs(
    per_sample: pd.DataFrame,
    per_timestep: pd.DataFrame,
    test_predictions_abs: dict[str, np.ndarray],
    test_truths_abs: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    test_rows = per_sample[per_sample["split"] == "test"].copy()
    plot_hist(test_rows["integrated_ADE"], plots_dir / "hist_test_integrated_ADE.png", "Test integrated ADE")
    plot_hist(test_rows["integrated_FDE"], plots_dir / "hist_test_integrated_FDE.png", "Test integrated FDE")
    plot_hist(test_rows["delta_ADE"], plots_dir / "hist_test_delta_ADE.png", "Test delta ADE")
    plot_scatter(test_rows, plots_dir / "scatter_integrated_ADE_vs_FDE.png")
    plot_per_timestep(per_timestep, plots_dir / "per_timestep_ADE_curve.png")
    plot_groups(test_rows, test_predictions_abs, test_truths_abs, plots_dir)


def plot_hist(values: pd.Series, path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=30)
    axis.set_title(title)
    axis.set_xlabel(title)
    axis.set_ylabel("count")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(test_rows: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(test_rows["integrated_ADE"], test_rows["integrated_FDE"], s=16, alpha=0.8)
    axis.set_xlabel("integrated ADE")
    axis.set_ylabel("integrated FDE")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_timestep(per_timestep: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    for split, frame in per_timestep.groupby("split", sort=False):
        axis.plot(frame["timestep"], frame["integrated_ade"], label=f"{split} integrated")
    axis.set_xlabel("timestep")
    axis.set_ylabel("integrated ADE")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for split, frame in per_timestep.groupby("split", sort=False):
        axis.plot(frame["timestep"], frame["delta_ade"], label=f"{split} delta")
    axis.set_xlabel("timestep")
    axis.set_ylabel("delta ADE")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_name("per_timestep_delta_ADE_curve.png"), dpi=150)
    plt.close(fig)


def plot_groups(
    test_rows: pd.DataFrame,
    predictions_abs: dict[str, np.ndarray],
    truths_abs: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    sorted_rows = test_rows.sort_values("integrated_ADE").reset_index(drop=True)
    groups = {
        "best": sorted_rows.head(10),
        "median": sorted_rows.iloc[
            max(0, len(sorted_rows) // 2 - 5) : min(len(sorted_rows), len(sorted_rows) // 2 + 5)
        ],
        "worst": sorted_rows.tail(10).sort_values("integrated_ADE", ascending=False),
    }
    for group_name, frame in groups.items():
        group_dir = plots_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            path = group_dir / f"{sample_id}_trajectory.png"
            plot_prediction(
                sample_id,
                str(row.vehicle_id),
                truths_abs[sample_id],
                predictions_abs[sample_id],
                float(row.delta_ADE),
                float(row.integrated_ADE),
                float(row.integrated_FDE),
                path,
            )
            paths.append(path)
        make_contact_sheet(paths, plots_dir / f"{group_name}_10_grid.png")


def plot_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    pred: np.ndarray,
    delta_ade: float,
    integrated_ade: float,
    integrated_fde: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
    axis.plot(pred[:, 0], pred[:, 1], label="pred integrated", linewidth=1.5)
    axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", label="gt/start")
    axis.scatter(true[-1, 0], true[-1, 1], s=22, marker="s", label="gt end")
    axis.scatter(pred[-1, 0], pred[-1, 1], s=26, marker="^", label="pred end")
    all_xy = np.vstack([true, pred])
    pad_x = max(1.0, float(np.ptp(all_xy[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(all_xy[:, 1])) * 0.05)
    axis.set_xlim(float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
    axis.set_ylim(float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{sample_id} {vehicle_id}\n"
        f"delta ADE={delta_ade:.2f} int ADE={integrated_ade:.2f} int FDE={integrated_fde:.2f}"
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_distribution_diagnostics(per_sample: pd.DataFrame, plots_dir: Path) -> None:
    test = per_sample[per_sample["split"] == "test"]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(test["true_delta_magnitude_mean"], bins=30, alpha=0.55, label="GT mean |delta|")
    axis.hist(test["pred_delta_magnitude_mean"], bins=30, alpha=0.55, label="Pred mean |delta|")
    axis.set_xlabel("per-sample mean delta magnitude")
    axis.set_ylabel("count")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_vs_gt_delta_magnitude_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(test["trajectory_length_ratio"], bins=30)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("predicted / ground-truth trajectory length")
    axis.set_ylabel("count")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "trajectory_length_ratio_distribution.png", dpi=150)
    plt.close(fig)


def compare_with_absolute_resnet18(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    absolute_dir = args.absolute_resnet18_dir.resolve()
    comparison_dir = output_dir / "comparison_with_absolute_resnet18"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    absolute_metrics = json.loads((absolute_dir / "evaluation" / "metrics_summary.json").read_text())
    delta_metrics = summary["metrics_summary"]
    rows = []
    for metric_abs, metric_delta in [
        ("ADE", "integrated_ADE"),
        ("FDE", "integrated_FDE"),
        ("RMSE", "integrated_RMSE"),
        ("MAE", "integrated_MAE"),
    ]:
        absolute_mean = absolute_metrics["test"][metric_abs]["mean"]
        delta_mean = delta_metrics["test"][metric_delta]["mean"]
        rows.append(
            {
                "split": "test",
                "metric": metric_abs,
                "absolute_resnet18_mean": absolute_mean,
                "delta_integrated_resnet18_mean": delta_mean,
                "delta_minus_absolute": delta_mean - absolute_mean,
                "percent_change_vs_absolute": (delta_mean - absolute_mean) / absolute_mean * 100.0
                if absolute_mean
                else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(comparison_dir / "comparison_table.csv", index=False)

    absolute_samples = pd.read_csv(absolute_dir / "evaluation" / "per_sample_metrics.csv")
    delta_samples = pd.read_csv(output_dir / "evaluation" / "per_sample_metrics.csv")
    comparison_summary = {
        "absolute_resnet18_dir": str(absolute_dir),
        "delta_displacement_resnet18_dir": str(output_dir),
        "mean_metric_table": table.to_dict(orient="records"),
        "test": make_absolute_delta_test_comparison(absolute_samples, delta_samples),
    }
    write_json(comparison_dir / "comparison_summary.json", comparison_summary)
    plot_absolute_delta_distribution_comparisons(absolute_samples, delta_samples, comparison_dir)
    plot_per_timestep_comparison(absolute_dir, output_dir, comparison_dir)
    plot_side_by_side_examples(absolute_dir, output_dir, comparison_dir, absolute_samples, delta_samples)


def make_absolute_delta_test_comparison(
    absolute_samples: pd.DataFrame, delta_samples: pd.DataFrame
) -> dict[str, object]:
    absolute_test = absolute_samples[absolute_samples["split"] == "test"].copy()
    delta_test = delta_samples[delta_samples["split"] == "test"].copy()
    merged = absolute_test.merge(
        delta_test,
        on=["sample_id", "vehicle_id", "split"],
        suffixes=("_absolute", "_delta"),
    )
    result: dict[str, object] = {}
    for abs_metric, delta_metric in [("ADE", "integrated_ADE"), ("FDE", "integrated_FDE")]:
        before_column = f"{abs_metric}_absolute" if f"{abs_metric}_absolute" in merged.columns else abs_metric
        before = merged[before_column].to_numpy(dtype=float)
        after = merged[delta_metric].to_numpy(dtype=float)
        result[abs_metric] = {
            "absolute_resnet18_mean": float(before.mean()),
            "delta_integrated_mean": float(after.mean()),
            "delta_minus_absolute": float(after.mean() - before.mean()),
            "percent_change_vs_absolute": float((after.mean() - before.mean()) / before.mean() * 100.0),
            "samples_improved_count": int((after < before).sum()),
            "samples_worse_count": int((after > before).sum()),
            "absolute_p50": float(np.percentile(before, 50)),
            "delta_p50": float(np.percentile(after, 50)),
            "absolute_p75": float(np.percentile(before, 75)),
            "delta_p75": float(np.percentile(after, 75)),
            "absolute_p90": float(np.percentile(before, 90)),
            "delta_p90": float(np.percentile(after, 90)),
            "absolute_p95": float(np.percentile(before, 95)),
            "delta_p95": float(np.percentile(after, 95)),
            "absolute_max": float(before.max()),
            "delta_max": float(after.max()),
        }
    return result


def plot_absolute_delta_distribution_comparisons(
    absolute_samples: pd.DataFrame, delta_samples: pd.DataFrame, comparison_dir: Path
) -> None:
    absolute_test = absolute_samples[absolute_samples["split"] == "test"]
    delta_test = delta_samples[delta_samples["split"] == "test"]
    for metric, delta_metric in [("ADE", "integrated_ADE"), ("FDE", "integrated_FDE")]:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(absolute_test[metric], bins=30, alpha=0.55, label="Absolute x/y ResNet18")
        axis.hist(delta_test[delta_metric], bins=30, alpha=0.55, label="Delta integrated ResNet18")
        axis.set_xlabel(f"Test {metric}")
        axis.set_ylabel("count")
        axis.legend()
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(comparison_dir / f"{metric}_distribution_comparison.png", dpi=150)
        plt.close(fig)


def plot_per_timestep_comparison(absolute_dir: Path, output_dir: Path, comparison_dir: Path) -> None:
    absolute_curve = pd.read_csv(absolute_dir / "evaluation" / "per_timestep_ade.csv")
    delta_curve = pd.read_csv(output_dir / "evaluation" / "per_timestep_ade.csv")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        absolute_curve[absolute_curve["split"] == "test"]["timestep"],
        absolute_curve[absolute_curve["split"] == "test"]["ade"],
        label="Absolute x/y ResNet18",
    )
    axis.plot(
        delta_curve[delta_curve["split"] == "test"]["timestep"],
        delta_curve[delta_curve["split"] == "test"]["integrated_ade"],
        label="Delta integrated ResNet18",
    )
    axis.set_xlabel("timestep")
    axis.set_ylabel("test ADE")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(comparison_dir / "per_timestep_ADE_comparison.png", dpi=150)
    plt.close(fig)


def plot_side_by_side_examples(
    absolute_dir: Path,
    output_dir: Path,
    comparison_dir: Path,
    absolute_samples: pd.DataFrame,
    delta_samples: pd.DataFrame,
) -> None:
    test_rows = delta_samples[delta_samples["split"] == "test"].sort_values("integrated_ADE").reset_index(drop=True)
    groups = {
        "best": test_rows.head(3),
        "median": test_rows.iloc[
            max(0, len(test_rows) // 2 - 1) : min(len(test_rows), len(test_rows) // 2 + 2)
        ],
        "worst": test_rows.tail(3).sort_values("integrated_ADE", ascending=False),
    }
    absolute_by_id = absolute_samples.set_index("sample_id")
    split_copy = pd.read_csv(output_dir / "split_copy.csv").set_index("sample_id")
    paths = []
    for group, frame in groups.items():
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            true = np.load(PROJECT_ROOT / "data_no_speed_delta_displacement_paired" / split_copy.loc[sample_id, "label_absolute_path"])
            absolute_pred = np.load(absolute_dir / "evaluation" / "predictions" / f"{sample_id}_pred.npy")
            delta_pred = np.load(
                output_dir / "evaluation" / "predictions_absolute_integrated" / f"{sample_id}_pred_absolute_integrated.npy"
            )
            path = comparison_dir / f"{group}_{sample_id}_side_by_side.png"
            plot_comparison_prediction(
                sample_id,
                str(row.vehicle_id),
                true,
                absolute_pred,
                delta_pred,
                float(absolute_by_id.loc[sample_id, "ADE"]),
                float(absolute_by_id.loc[sample_id, "FDE"]),
                float(row.integrated_ADE),
                float(row.integrated_FDE),
                path,
            )
            paths.append(path)
    make_contact_sheet(paths, comparison_dir / "side_by_side_best_median_worst_grid.png")


def plot_comparison_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    absolute_pred: np.ndarray,
    delta_pred: np.ndarray,
    absolute_ade: float,
    absolute_fde: float,
    delta_ade: float,
    delta_fde: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    all_xy = np.vstack([true, absolute_pred, delta_pred])
    pad_x = max(1.0, float(np.ptp(all_xy[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(all_xy[:, 1])) * 0.05)
    panels = [
        ("Absolute x/y ResNet18", absolute_pred, absolute_ade, absolute_fde),
        ("Delta integrated ResNet18", delta_pred, delta_ade, delta_fde),
    ]
    for axis, (title, pred, ade, fde) in zip(axes, panels):
        axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
        axis.plot(pred[:, 0], pred[:, 1], label="prediction", linewidth=1.5)
        axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", label="gt/start")
        axis.scatter(true[-1, 0], true[-1, 1], s=22, marker="s", label="gt end")
        axis.scatter(pred[-1, 0], pred[-1, 1], s=26, marker="^", label="pred end")
        axis.set_title(f"{title}\nADE={ade:.2f} FDE={fde:.2f}")
        axis.set_xlim(float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
        axis.set_ylim(float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{sample_id} {vehicle_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_chinese_report(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    metrics = summary["metrics_summary"]
    distribution = summary["distribution_summary"]
    variance = summary["variance_summary"]
    best = json.loads((output_dir / "best_summary.json").read_text())
    comparison = json.loads(
        (output_dir / "comparison_with_absolute_resnet18" / "comparison_summary.json").read_text()
    )
    table = pd.read_csv(output_dir / "comparison_with_absolute_resnet18" / "comparison_table.csv")

    test_delta_ade = metrics["test"]["delta_ADE"]["mean"]
    test_delta_fde = metrics["test"]["delta_FDE"]["mean"]
    test_int_ade = metrics["test"]["integrated_ADE"]["mean"]
    test_int_fde = metrics["test"]["integrated_FDE"]["mean"]
    abs_ade = float(table.query("metric == 'ADE'")["absolute_resnet18_mean"].iloc[0])
    abs_fde = float(table.query("metric == 'FDE'")["absolute_resnet18_mean"].iloc[0])
    ade_change = float(table.query("metric == 'ADE'")["percent_change_vs_absolute"].iloc[0])
    fde_change = float(table.query("metric == 'FDE'")["percent_change_vs_absolute"].iloc[0])
    ade_cmp = comparison["test"]["ADE"]
    fde_cmp = comparison["test"]["FDE"]

    if test_int_ade < abs_ade and test_int_fde < abs_fde:
        verdict = "delta-displacement 表示在本次 ResNet18 设置下同时降低了 integrated ADE 和 FDE。"
    elif test_int_ade < abs_ade and test_int_fde >= abs_fde:
        verdict = "delta-displacement 表示降低了平均 integrated ADE，但 FDE 没有同步改善，可能存在积分漂移。"
    else:
        verdict = "delta-displacement 表示没有在主要 integrated 指标上超过 absolute x/y ResNet18 baseline。"

    report = f"""
Delta-Displacement Representation Ablation 中文报告

实验设置：
- 数据集：data_no_speed_delta_displacement_paired/，训练前未重新生成或修改。
- Split：data_no_speed_delta_displacement_paired/splits/split_metadata.csv，与 absolute no-speed split 的 sample_id 对齐。
- 输入：delta-displacement Hilbert/GAF/MTF RGB 224x224 图像。
- 目标：labels_delta_displacement，shape [224,2]。
- 模型：scratch ResNet18 backbone + MLP head 512 -> 1024 -> 448，dropout=0.3。
- 训练：Adam lr=1e-4, batch_size=8, weight_decay=0, scheduler=none, seed=42。
- 标签归一化：train-only mean/std，在 normalized delta-displacement 空间使用 MSELoss。
- 绝对轨迹评估：使用 true start point 的 oracle-start integration。

训练结果：
- best epoch={int(best['best_epoch'])}, final epoch={int(best['final_epoch'])}, early stopping={'触发' if best['early_stopping_triggered'] else '未触发'}。
- best validation loss={float(best['best_val_loss']):.6f}。

1. Delta-space 解码误差：
- test delta ADE={test_delta_ade:.4f}
- test delta FDE={test_delta_fde:.4f}
- test delta RMSE={metrics['test']['delta_RMSE']['mean']:.4f}
- test delta MAE={metrics['test']['delta_MAE']['mean']:.4f}

2. Integrated absolute 轨迹误差：
- test integrated ADE={test_int_ade:.4f}
- test integrated FDE={test_int_fde:.4f}
- test integrated RMSE={metrics['test']['integrated_RMSE']['mean']:.4f}
- test integrated MAE={metrics['test']['integrated_MAE']['mean']:.4f}
- start error={metrics['test']['start_error']['mean']:.6f}，由于使用 true start，理论上应接近 0。
- trajectory length error={metrics['test']['trajectory_length_error']['mean']:.4f}

3. 与 absolute x/y ResNet18 baseline 对比：
- absolute ResNet18 test ADE={abs_ade:.4f}, FDE={abs_fde:.4f}
- delta integrated test ADE={test_int_ade:.4f}, FDE={test_int_fde:.4f}
- ADE change vs absolute={ade_change:.2f}%, FDE change vs absolute={fde_change:.2f}%。
- ADE improved samples={ade_cmp['samples_improved_count']}, worse samples={ade_cmp['samples_worse_count']}。
- FDE improved samples={fde_cmp['samples_improved_count']}, worse samples={fde_cmp['samples_worse_count']}。

4. Long-tail 与分布：
- absolute ADE p95={ade_cmp['absolute_p95']:.4f}, max={ade_cmp['absolute_max']:.4f}
- delta integrated ADE p95={ade_cmp['delta_p95']:.4f}, max={ade_cmp['delta_max']:.4f}
- predicted near-zero delta ratio={distribution['test']['pred_near_zero_delta_ratio']:.4f}
- ground-truth near-zero delta ratio={distribution['test']['true_near_zero_delta_ratio']:.4f}
- predicted/GT trajectory length ratio={distribution['test']['trajectory_length_ratio']:.4f}
- delta predicted/GT variance ratio={variance['test']['delta_predicted_to_ground_truth_variance_ratio']:.4f}

5. 谨慎解释：
{verdict}
这个结果不说明 absolute 表示无效；它只是在相同 split、相同 ResNet18 capacity、相同 no-speed/no-map 条件下比较两种图像表示的可解码性。
如果 ADE 有改善但 FDE 或 end error 较差，应优先解释为局部位移误差在积分后累积造成 drift。
如果 delta 表示整体更差，则说明 Hilbert/GAF/MTF 可能更适合编码 absolute spatial coordinates，而不是局部 displacement 序列。
若 delta 表示更好，它提示 local-motion representation 对轨迹生成可能有价值，但后续仍需要 start-point、地图或条件信息来约束绝对位置与可行路径。

输出文件：
- checkpoints/best_model.pt, checkpoints/last_model.pt
- train_log.csv, config.json, label_scaler.json
- evaluation/metrics_summary.json
- evaluation/per_sample_metrics.csv
- evaluation/per_timestep_ade.csv
- evaluation/delta_distribution_diagnostics.json
- evaluation/predictions_delta_displacement/
- evaluation/predictions_absolute_integrated/
- comparison_with_absolute_resnet18/
"""
    path = output_dir / "delta_displacement_ablation_report_zh.txt"
    path.write_text(report.strip() + "\n")
    print(report.strip())


if __name__ == "__main__":
    main()

"""Train Week3 E1 raw ResNet18 delta decoder with low-frequency drift correction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/home/jliu/Thesis/week03/logs/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
for path in [SCRIPTS_DIR, WEEK03_SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drift_aware_resnet18_models import DriftAwareResNet18DeltaDecoder  # noqa: E402
from run_delta_displacement_resnet18_decoder_ablation import (  # noqa: E402
    DeltaDisplacementDataset,
    compute_basic_metrics,
    fit_delta_label_scaler,
    trajectory_length,
    verify_delta_data_and_split,
)
from run_resnet18_decoder_ablation import LabelScaler, format_metric, resolve_device, set_seed, write_json  # noqa: E402


TRAJECTORY_STEPS = 224
TRAJECTORY_DIMS = 2


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.resolve()
    for subdir in [
        "checkpoints",
        "plots",
        "evaluation",
        "evaluation/plots",
        "evaluation/predictions_delta_displacement",
        "evaluation/predictions_absolute_integrated_raw",
        "evaluation/predictions_absolute_corrected",
        "evaluation/corrections",
    ]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(args.data_root.resolve())
    if not args.skip_training:
        train(args, split_metadata, device)
    summary = evaluate(args, split_metadata, device)
    write_week3_e1_report(args, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jliu/data_no_speed_delta_displacement_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "week03" / "results" / "experiment_e1_drift_aware_raw_resnet18_bs16",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--correction-anchors", type=int, default=16)
    parser.add_argument("--lambda-abs", type=float, default=0.05)
    parser.add_argument("--lambda-smooth", type=float, default=0.001)
    parser.add_argument("--lambda-mag", type=float, default=0.0001)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def fit_absolute_label_scaler(data_root: Path, train_metadata: pd.DataFrame) -> LabelScaler:
    labels = [
        np.load(data_root / row.label_absolute_path).astype(np.float32)
        for row in train_metadata.itertuples(index=False)
    ]
    stacked = np.concatenate(labels, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return LabelScaler(mean=mean.tolist(), std=std.tolist())


def scaler_transform_tensor(scaler: LabelScaler, values: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(scaler.mean, dtype=values.dtype, device=values.device).view(1, 1, TRAJECTORY_DIMS)
    std = torch.tensor(scaler.std, dtype=values.dtype, device=values.device).view(1, 1, TRAJECTORY_DIMS)
    return (values - mean) / std


def integrate_delta_torch(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    absolute = torch.empty_like(delta)
    absolute[:, 0, :] = start
    absolute[:, 1:, :] = start[:, None, :] + torch.cumsum(delta[:, 1:, :], dim=1)
    return absolute


def train(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    delta_scaler = fit_delta_label_scaler(data_root, train_metadata)
    absolute_scaler = fit_absolute_label_scaler(data_root, train_metadata)

    write_json(output_dir / "delta_scaler.json", asdict(delta_scaler))
    write_json(output_dir / "absolute_scaler.json", asdict(absolute_scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = DataLoader(
        DeltaDisplacementDataset(data_root, train_metadata, delta_scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_metadata) % args.batch_size == 1,
    )
    val_loader = DataLoader(
        DeltaDisplacementDataset(data_root, val_metadata, delta_scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DriftAwareResNet18DeltaDecoder(
        dropout=args.dropout,
        correction_anchors=args.correction_anchors,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    log_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(model, train_loader, criterion, delta_scaler, absolute_scaler, device, optimizer, args)
        val_losses = evaluate_loss_loader(model, val_loader, criterion, delta_scaler, absolute_scaler, device, args)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in val_losses.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time": time.perf_counter() - start_time,
        }
        log_rows.append(row)
        write_train_log(output_dir / "train_log.csv", log_rows)
        plot_training_curves(log_rows, output_dir / "plots")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses["total_loss"],
            "train_loss": train_losses["total_loss"],
            "config": config,
            "delta_scaler": asdict(delta_scaler),
            "absolute_scaler": asdict(absolute_scaler),
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last_model.pt")
        if val_losses["total_loss"] < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_losses["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "checkpoints" / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:04d}/{args.epochs} train_total={train_losses['total_loss']:.6f} "
            f"val_total={val_losses['total_loss']:.6f} train_delta={train_losses['delta_loss']:.6f} "
            f"val_delta={val_losses['delta_loss']:.6f} train_abs={train_losses['abs_loss']:.6f} "
            f"val_abs={val_losses['abs_loss']:.6f} best_val={best_val_loss:.6f} best_epoch={best_epoch}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopping_triggered = True
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
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


def make_config(
    args: argparse.Namespace,
    device: torch.device,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
) -> dict[str, object]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "experiment": "E1 raw ResNet18 low-frequency drift correction",
            "model": "DriftAwareResNet18DeltaDecoder",
            "backbone": "torchvision.models.resnet18(weights=None)",
            "pretrained": False,
            "conv1_input_channels": 3,
            "input_representation": "raw RGB [GASF,GADF,MTF] in [0,1]",
            "delta_head": "feature_dim -> 1024 -> 448 -> reshape[224,2]",
            "correction_head": "feature_dim -> 512 -> correction_anchors*2 -> interpolate[224,2]",
            "correction_space": "normalized absolute-coordinate residual added after delta integration",
            "delta_label_normalization": "train_mean_std_delta_displacement_xy",
            "absolute_label_normalization": "train_mean_std_absolute_xy",
            "integrated_absolute_evaluation": "oracle true start point",
            "optimizer": "Adam",
            "scheduler": "none",
            "device_resolved": str(device),
            "train_samples": len(train_metadata),
            "val_samples": len(val_metadata),
        }
    )
    return config


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    delta_scaler: LabelScaler,
    absolute_scaler: LabelScaler,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = empty_loss_totals()
    sample_count = 0
    for batch in loader:
        images, true_delta_norm, true_abs_raw, start = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = compute_losses(outputs, true_delta_norm, true_abs_raw, start, criterion, delta_scaler, absolute_scaler, args)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = images.shape[0]
        add_loss_totals(totals, losses, batch_size)
        sample_count += batch_size
    return finalize_loss_totals(totals, sample_count)


def evaluate_loss_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    delta_scaler: LabelScaler,
    absolute_scaler: LabelScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = empty_loss_totals()
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            images, true_delta_norm, true_abs_raw, start = batch_to_device(batch, device)
            outputs = model(images)
            losses = compute_losses(outputs, true_delta_norm, true_abs_raw, start, criterion, delta_scaler, absolute_scaler, args)
            batch_size = images.shape[0]
            add_loss_totals(totals, losses, batch_size)
            sample_count += batch_size
    return finalize_loss_totals(totals, sample_count)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
        batch["absolute"].to(device, non_blocking=True),
        batch["start"].to(device, non_blocking=True),
    )


def compute_losses(
    outputs: dict[str, torch.Tensor],
    true_delta_norm: torch.Tensor,
    true_abs_raw: torch.Tensor,
    start: torch.Tensor,
    criterion: nn.Module,
    delta_scaler: LabelScaler,
    absolute_scaler: LabelScaler,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pred_delta_norm = outputs["pred_delta_norm"]
    corr_full_norm = outputs["corr_full_norm"]
    delta_loss = criterion(pred_delta_norm, true_delta_norm)
    pred_delta_raw = delta_scaler.inverse_transform_tensor(pred_delta_norm)
    pred_abs_raw = integrate_delta_torch(start, pred_delta_raw)
    pred_abs_raw_norm = scaler_transform_tensor(absolute_scaler, pred_abs_raw)
    true_abs_norm = scaler_transform_tensor(absolute_scaler, true_abs_raw)
    pred_abs_corrected_norm = pred_abs_raw_norm + corr_full_norm
    abs_loss = criterion(pred_abs_corrected_norm, true_abs_norm)
    smooth_loss = torch.mean((corr_full_norm[:, 1:, :] - corr_full_norm[:, :-1, :]) ** 2)
    mag_loss = torch.mean(corr_full_norm**2)
    total_loss = (
        delta_loss
        + args.lambda_abs * abs_loss
        + args.lambda_smooth * smooth_loss
        + args.lambda_mag * mag_loss
    )
    return {
        "total_loss": total_loss,
        "delta_loss": delta_loss,
        "abs_loss": abs_loss,
        "smooth_loss": smooth_loss,
        "mag_loss": mag_loss,
    }


def empty_loss_totals() -> dict[str, float]:
    return {key: 0.0 for key in ["total_loss", "delta_loss", "abs_loss", "smooth_loss", "mag_loss"]}


def add_loss_totals(totals: dict[str, float], losses: dict[str, torch.Tensor], batch_size: int) -> None:
    for key in totals:
        totals[key] += float(losses[key].detach().cpu()) * batch_size


def finalize_loss_totals(totals: dict[str, float], sample_count: int) -> dict[str, float]:
    return {key: value / max(1, sample_count) for key, value in totals.items()}


def write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "train_total_loss",
        "train_delta_loss",
        "train_abs_loss",
        "train_smooth_loss",
        "train_mag_loss",
        "val_total_loss",
        "val_delta_loss",
        "val_abs_loss",
        "val_smooth_loss",
        "val_mag_loss",
        "learning_rate",
        "elapsed_time",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(rows: list[dict[str, object]], plots_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    for name, columns in {
        "total_loss_curves.png": ["train_total_loss", "val_total_loss"],
        "component_train_loss_curves.png": ["train_delta_loss", "train_abs_loss", "train_smooth_loss", "train_mag_loss"],
        "component_val_loss_curves.png": ["val_delta_loss", "val_abs_loss", "val_smooth_loss", "val_mag_loss"],
    }.items():
        fig, axis = plt.subplots(figsize=(7, 4))
        for column in columns:
            axis.plot(frame["epoch"], frame[column], label=column)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=150)
        plt.close(fig)


def evaluate(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    pred_delta_dir = evaluation_dir / "predictions_delta_displacement"
    pred_abs_raw_dir = evaluation_dir / "predictions_absolute_integrated_raw"
    pred_abs_corrected_dir = evaluation_dir / "predictions_absolute_corrected"
    corrections_dir = evaluation_dir / "corrections"
    plots_dir = evaluation_dir / "plots"
    for directory in [pred_delta_dir, pred_abs_raw_dir, pred_abs_corrected_dir, corrections_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    delta_scaler = load_scaler(output_dir / "delta_scaler.json")
    absolute_scaler = load_scaler(output_dir / "absolute_scaler.json")
    model = DriftAwareResNet18DeltaDecoder(
        dropout=args.dropout,
        correction_anchors=args.correction_anchors,
    ).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    test_predictions: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = DataLoader(
            DeltaDisplacementDataset(data_root, split_frame, delta_scaler),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        split_rows, timestep_ade, split_predictions = predict_split(
            model,
            loader,
            delta_scaler,
            absolute_scaler,
            device,
            pred_delta_dir,
            pred_abs_raw_dir,
            pred_abs_corrected_dir,
            corrections_dir,
            split,
        )
        rows.extend(split_rows)
        timestep_rows.extend(
            {
                "split": split,
                "timestep": index,
                "raw_integrated_ade": float(timestep_ade["raw"][index]),
                "corrected_integrated_ade": float(timestep_ade["corrected"][index]),
                "delta_ade": float(timestep_ade["delta"][index]),
            }
            for index in range(TRAJECTORY_STEPS)
        )
        if split == "test":
            test_predictions = split_predictions

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.DataFrame(timestep_rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)
    metrics_summary = summarize_metrics(per_sample)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    plot_evaluation_outputs(per_sample, per_timestep, test_predictions, plots_dir)
    return {"checkpoint": checkpoint, "metrics_summary": metrics_summary}


def load_scaler(path: Path) -> LabelScaler:
    data = json.loads(path.read_text())
    return LabelScaler(mean=data["mean"], std=data["std"])


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    delta_scaler: LabelScaler,
    absolute_scaler: LabelScaler,
    device: torch.device,
    pred_delta_dir: Path,
    pred_abs_raw_dir: Path,
    pred_abs_corrected_dir: Path,
    corrections_dir: Path,
    split: str,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    timestep_sums = {
        "delta": np.zeros(TRAJECTORY_STEPS, dtype=np.float64),
        "raw": np.zeros(TRAJECTORY_STEPS, dtype=np.float64),
        "corrected": np.zeros(TRAJECTORY_STEPS, dtype=np.float64),
    }
    predictions = {"true": {}, "raw": {}, "corrected": {}}
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            start = batch["start"].to(device, non_blocking=True)
            outputs = model(images)
            pred_delta_norm = outputs["pred_delta_norm"]
            corr_full_norm = outputs["corr_full_norm"]
            pred_delta_raw_t = delta_scaler.inverse_transform_tensor(pred_delta_norm)
            pred_abs_raw_t = integrate_delta_torch(start, pred_delta_raw_t)
            pred_abs_raw_norm_t = scaler_transform_tensor(absolute_scaler, pred_abs_raw_t)
            pred_abs_corrected_norm_t = pred_abs_raw_norm_t + corr_full_norm
            pred_abs_corrected_t = absolute_scaler.inverse_transform_tensor(pred_abs_corrected_norm_t)

            pred_delta_raw = pred_delta_raw_t.cpu().numpy().astype(np.float32)
            pred_abs_raw = pred_abs_raw_t.cpu().numpy().astype(np.float32)
            pred_abs_corrected = pred_abs_corrected_t.cpu().numpy().astype(np.float32)
            corr_full_norm_np = corr_full_norm.cpu().numpy().astype(np.float32)
            true_delta = batch["label_raw"].cpu().numpy().astype(np.float32)
            true_abs = batch["absolute"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred_delta = pred_delta_raw[index]
                sample_pred_abs_raw = pred_abs_raw[index]
                sample_pred_abs_corrected = pred_abs_corrected[index]
                sample_corr = corr_full_norm_np[index]
                sample_true_delta = true_delta[index]
                sample_true_abs = true_abs[index]

                np.save(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_pred_delta)
                np.save(pred_abs_raw_dir / f"{sample_id}_pred_absolute_raw.npy", sample_pred_abs_raw)
                np.save(pred_abs_corrected_dir / f"{sample_id}_pred_absolute_corrected.npy", sample_pred_abs_corrected)
                np.save(corrections_dir / f"{sample_id}_corr_full_norm.npy", sample_corr)

                delta_metrics, delta_errors = compute_basic_metrics(sample_pred_delta, sample_true_delta, "delta")
                raw_metrics, raw_errors = compute_basic_metrics(sample_pred_abs_raw, sample_true_abs, "raw_integrated")
                corrected_metrics, corrected_errors = compute_basic_metrics(
                    sample_pred_abs_corrected,
                    sample_true_abs,
                    "corrected_integrated",
                )
                raw_length = trajectory_length(sample_pred_abs_raw)
                corrected_length = trajectory_length(sample_pred_abs_corrected)
                true_length = trajectory_length(sample_true_abs)
                corr_norm = np.linalg.norm(sample_corr, axis=1)
                corr_smooth = float(np.mean(np.diff(sample_corr, axis=0) ** 2))
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **delta_metrics,
                        **raw_metrics,
                        **corrected_metrics,
                        "raw_end_error": float(raw_errors[-1]),
                        "corrected_end_error": float(corrected_errors[-1]),
                        "raw_trajectory_length_ratio": float(raw_length / true_length) if true_length else np.nan,
                        "corrected_trajectory_length_ratio": float(corrected_length / true_length)
                        if true_length
                        else np.nan,
                        "correction_norm_mean": float(corr_norm.mean()),
                        "correction_norm_max": float(corr_norm.max()),
                        "correction_smoothness": corr_smooth,
                    }
                )
                timestep_sums["delta"] += delta_errors
                timestep_sums["raw"] += raw_errors
                timestep_sums["corrected"] += corrected_errors
                total += 1
                if split == "test":
                    predictions["true"][sample_id] = sample_true_abs
                    predictions["raw"][sample_id] = sample_pred_abs_raw
                    predictions["corrected"][sample_id] = sample_pred_abs_corrected

    timestep_ade = {key: value / max(1, total) for key, value in timestep_sums.items()}
    return rows, timestep_ade, predictions


def summarize_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "delta_ADE",
        "delta_FDE",
        "delta_RMSE",
        "delta_MAE",
        "raw_integrated_ADE",
        "raw_integrated_FDE",
        "raw_integrated_RMSE",
        "raw_integrated_MAE",
        "corrected_integrated_ADE",
        "corrected_integrated_FDE",
        "corrected_integrated_RMSE",
        "corrected_integrated_MAE",
        "raw_end_error",
        "corrected_end_error",
        "raw_trajectory_length_ratio",
        "corrected_trajectory_length_ratio",
        "correction_norm_mean",
        "correction_norm_max",
        "correction_smoothness",
    ]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in metrics:
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


def plot_evaluation_outputs(
    per_sample: pd.DataFrame,
    per_timestep: pd.DataFrame,
    test_predictions: dict[str, dict[str, np.ndarray]],
    plots_dir: Path,
) -> None:
    plot_per_timestep(per_timestep, plots_dir / "per_timestep_raw_vs_corrected_ADE.png")
    test_rows = per_sample[per_sample["split"] == "test"].copy()
    if not test_rows.empty:
        plot_histograms(test_rows, plots_dir)
        plot_qualitative_groups(test_rows, test_predictions, plots_dir)


def plot_per_timestep(per_timestep: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    for split, frame in per_timestep.groupby("split", sort=False):
        axis.plot(frame["timestep"], frame["raw_integrated_ade"], label=f"{split} raw")
        axis.plot(frame["timestep"], frame["corrected_integrated_ade"], linestyle="--", label=f"{split} corrected")
    axis.set_xlabel("timestep")
    axis.set_ylabel("integrated ADE")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histograms(test_rows: pd.DataFrame, plots_dir: Path) -> None:
    for metric, title in [
        ("raw_integrated_ADE", "Test raw integrated ADE"),
        ("corrected_integrated_ADE", "Test corrected integrated ADE"),
        ("corrected_integrated_FDE", "Test corrected integrated FDE"),
    ]:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(test_rows[metric], bins=30)
        axis.set_xlabel(title)
        axis.set_ylabel("count")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"hist_{metric}.png", dpi=150)
        plt.close(fig)


def plot_qualitative_groups(
    test_rows: pd.DataFrame,
    test_predictions: dict[str, dict[str, np.ndarray]],
    plots_dir: Path,
) -> None:
    sorted_rows = test_rows.sort_values("raw_integrated_ADE").reset_index(drop=True)
    selected = [
        ("best", sorted_rows.iloc[0]),
        ("median", sorted_rows.iloc[len(sorted_rows) // 2]),
        ("worst", sorted_rows.iloc[-1]),
    ]
    for label, row in selected:
        sample_id = str(row["sample_id"])
        if sample_id not in test_predictions.get("true", {}):
            continue
        fig, axis = plt.subplots(figsize=(6, 6))
        true = test_predictions["true"][sample_id]
        raw = test_predictions["raw"][sample_id]
        corrected = test_predictions["corrected"][sample_id]
        axis.plot(true[:, 0], true[:, 1], color="black", linewidth=2, label="true")
        axis.plot(raw[:, 0], raw[:, 1], color="#4c78a8", label="raw integrated")
        axis.plot(corrected[:, 0], corrected[:, 1], color="#54a24b", label="corrected")
        axis.scatter(true[0, 0], true[0, 1], color="black", s=24)
        axis.scatter(true[-1, 0], true[-1, 1], color="black", s=30, marker="x")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
        axis.legend()
        axis.set_title(
            f"{label} {sample_id}: raw ADE/FDE {row['raw_integrated_ADE']:.2f}/{row['raw_integrated_FDE']:.2f}, "
            f"corrected {row['corrected_integrated_ADE']:.2f}/{row['corrected_integrated_FDE']:.2f}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(plots_dir / f"qual_{label}_{sample_id}_raw_vs_corrected.png", dpi=150)
        plt.close(fig)


def write_week3_e1_report(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    metrics = summary["metrics_summary"]
    best = json.loads((output_dir / "best_summary.json").read_text())
    report: dict[str, object] = {
        "model": "DriftAwareResNet18DeltaDecoder",
        "input_representation": "raw RGB [GASF,GADF,MTF] in [0,1]",
        "correction_anchors": args.correction_anchors,
        "loss_weights": {
            "lambda_abs": args.lambda_abs,
            "lambda_smooth": args.lambda_smooth,
            "lambda_mag": args.lambda_mag,
        },
        "best_epoch": best["best_epoch"],
        "best_val_loss": best["best_val_loss"],
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        split_metrics = metrics[split]
        raw_ade = split_metrics["raw_integrated_ADE"]["mean"]
        raw_fde = split_metrics["raw_integrated_FDE"]["mean"]
        corrected_ade = split_metrics["corrected_integrated_ADE"]["mean"]
        corrected_fde = split_metrics["corrected_integrated_FDE"]["mean"]
        report["splits"][split] = {
            "delta_ADE": split_metrics["delta_ADE"]["mean"],
            "delta_FDE": split_metrics["delta_FDE"]["mean"],
            "raw_integrated_ADE": raw_ade,
            "raw_integrated_FDE": raw_fde,
            "corrected_integrated_ADE": corrected_ade,
            "corrected_integrated_FDE": corrected_fde,
            "raw_trajectory_length_ratio": split_metrics["raw_trajectory_length_ratio"]["mean"],
            "corrected_trajectory_length_ratio": split_metrics["corrected_trajectory_length_ratio"]["mean"],
            "correction_magnitude_mean": split_metrics["correction_norm_mean"]["mean"],
            "correction_smoothness_mean": split_metrics["correction_smoothness"]["mean"],
            "corrected_ADE_improvement_percent": percent_improvement(raw_ade, corrected_ade),
            "corrected_FDE_improvement_percent": percent_improvement(raw_fde, corrected_fde),
        }
    write_json(output_dir / "week3_e1_report.json", report)
    write_markdown_report(output_dir / "week3_e1_report.md", report)
    print((output_dir / "week3_e1_report.md").read_text())


def percent_improvement(before: float, after: float) -> float:
    return float((before - after) / before * 100.0) if before else float("nan")


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Week3 E1 Drift-Aware Raw ResNet18 Report",
        "",
        f"- model: {report['model']}",
        f"- input_representation: {report['input_representation']}",
        f"- correction_anchors: {report['correction_anchors']}",
        f"- loss_weights: {report['loss_weights']}",
        f"- best_epoch: {report['best_epoch']}",
        f"- best_val_loss: {report['best_val_loss']}",
        "",
        "| split | delta ADE | delta FDE | raw ADE | raw FDE | corrected ADE | corrected FDE | ADE improvement % | FDE improvement % | raw length ratio | corrected length ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    splits: dict[str, dict[str, float]] = report["splits"]  # type: ignore[assignment]
    for split in ["train", "val", "test"]:
        row = splits[split]
        lines.append(
            f"| {split} | {row['delta_ADE']:.6f} | {row['delta_FDE']:.6f} | "
            f"{row['raw_integrated_ADE']:.6f} | {row['raw_integrated_FDE']:.6f} | "
            f"{row['corrected_integrated_ADE']:.6f} | {row['corrected_integrated_FDE']:.6f} | "
            f"{row['corrected_ADE_improvement_percent']:.3f} | {row['corrected_FDE_improvement_percent']:.3f} | "
            f"{row['raw_trajectory_length_ratio']:.6f} | {row['corrected_trajectory_length_ratio']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

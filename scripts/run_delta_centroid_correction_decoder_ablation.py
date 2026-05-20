"""Train and evaluate delta decoding with centroid correction diagnostics.

This standalone experiment keeps the delta-displacement GAF input and original
delta training behavior, then asks whether global centroid drift explains much
of the absolute trajectory error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_resnet18_decoder_ablation import LabelScaler, format_metric, plot_training_curves, write_json  # noqa: E402

try:
    from run_original_config_decoder_experiment import make_contact_sheet  # noqa: E402
except ImportError:  # pragma: no cover - optional plotting convenience
    make_contact_sheet = None


DELTA_METRICS = ["delta_ADE", "delta_FDE", "delta_RMSE", "delta_MAE"]
MODE_PREFIXES = [
    "oracle_integrated",
    "gt_centroid_corrected",
    "learned_centroid_corrected",
]
MODE_METRICS = [
    "ADE",
    "FDE",
    "RMSE",
    "MAE",
    "centroid_error",
    "end_error",
    "trajectory_length_error",
    "trajectory_length_ratio",
]
EXTRA_METRICS = [
    "learned_centroid_error",
    "centroid_offset_norm",
    "gt_centroid_correction_ADE_improvement",
    "learned_centroid_correction_ADE_improvement",
    "learned_vs_gt_centroid_corrected_ADE_gap",
    "gt_centroid_correction_FDE_improvement",
    "learned_centroid_correction_FDE_improvement",
    "learned_vs_gt_centroid_corrected_FDE_gap",
]


class DeltaCentroidCorrectionDataset(Dataset):
    def __init__(self, data_root: Path, metadata: pd.DataFrame, delta_scaler: LabelScaler, centroid_scaler: LabelScaler) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.delta_scaler = delta_scaler
        self.centroid_scaler = centroid_scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        delta = np.load(self.data_root / row.label_delta_displacement_path).astype(np.float32)
        absolute = np.load(self.data_root / row.label_absolute_path).astype(np.float32)
        start = np.load(self.data_root / row.start_path).astype(np.float32)
        centroid = absolute.mean(axis=0).astype(np.float32)
        return {
            "image": image_tensor.contiguous(),
            "label_delta": torch.from_numpy(self.delta_scaler.transform(delta).astype(np.float32)),
            "label_delta_raw": torch.from_numpy(delta),
            "label_centroid": torch.from_numpy(self.centroid_scaler.transform(centroid).astype(np.float32)),
            "label_centroid_raw": torch.from_numpy(centroid),
            "absolute": torch.from_numpy(absolute),
            "start": torch.from_numpy(start),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


class DeltaCentroidCorrectionResNet18(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.delta_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 448),
        )
        self.centroid_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature = self.backbone(images)
        return {
            "delta": self.delta_head(feature).view(-1, 224, 2),
            "centroid": self.centroid_head(feature),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.resolve()
    for subdir in ["checkpoints", "plots", "evaluation", "evaluation/plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(args.data_root.resolve())
    if not args.skip_training:
        train(args, split_metadata, device)
    summary = evaluate(args, split_metadata, device)
    write_chinese_report(args, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["SCRATCH"]) / "Thesis_data/data_no_speed_delta_displacement_paired",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results_hpc/delta_centroid_correction_decoder"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-centroid", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
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

    bad: list[dict[str, object]] = []
    for row in split_metadata.itertuples(index=False):
        paths = [
            ("image", data_root / row.image_path),
            ("delta", data_root / row.label_delta_displacement_path),
            ("absolute", data_root / row.label_absolute_path),
            ("start", data_root / row.start_path),
        ]
        for name, path in paths:
            if not path.exists():
                bad.append({"sample_id": row.sample_id, "reason": f"missing_{name}"})
        if bad and str(bad[-1].get("sample_id")) == str(row.sample_id):
            continue
        with Image.open(data_root / row.image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                bad.append({"sample_id": row.sample_id, "reason": f"bad_image:{image.mode}:{image.size}"})
        delta = np.load(data_root / row.label_delta_displacement_path)
        absolute = np.load(data_root / row.label_absolute_path)
        start = np.load(data_root / row.start_path)
        if delta.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_delta:{delta.shape}"})
        if absolute.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_absolute:{absolute.shape}"})
        if start.shape != (2,):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_start:{start.shape}"})

    if bad:
        report_path = data_root / "splits" / "delta_centroid_correction_consistency_report.csv"
        pd.DataFrame(bad).to_csv(report_path, index=False)
        raise RuntimeError(f"Dataset consistency check failed. Report: {report_path}")

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    print("Delta centroid correction data validation passed")
    print(f"total samples: {len(split_metadata)}")
    print(f"train count: {int(counts['train'])}")
    print(f"val count: {int(counts['val'])}")
    print(f"test count: {int(counts['test'])}")
    return split_metadata


def fit_delta_label_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [
        np.load(data_root / row.label_delta_displacement_path).astype(np.float32)
        for row in metadata.itertuples(index=False)
    ]
    stacked = np.concatenate(labels, axis=0)
    return make_scaler(stacked)


def fit_centroid_label_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    centroids = [
        np.load(data_root / row.label_absolute_path).astype(np.float32).mean(axis=0)
        for row in metadata.itertuples(index=False)
    ]
    return make_scaler(np.stack(centroids, axis=0))


def make_scaler(values: np.ndarray) -> LabelScaler:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return LabelScaler(mean=mean.tolist(), std=std.tolist())


def train(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    delta_scaler = fit_delta_label_scaler(data_root, train_metadata)
    centroid_scaler = fit_centroid_label_scaler(data_root, train_metadata)

    write_json(output_dir / "label_delta_scaler.json", asdict(delta_scaler))
    write_json(output_dir / "label_centroid_scaler.json", asdict(centroid_scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = make_loader(args, train_metadata, delta_scaler, centroid_scaler, True, device)
    val_loader = make_loader(args, val_metadata, delta_scaler, centroid_scaler, False, device)
    model = DeltaCentroidCorrectionResNet18(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    log_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer, args.lambda_centroid)
        compute_diag = epoch == 1 or epoch % args.metrics_every == 0
        train_diag = evaluate_loader(model, train_loader, device, args.lambda_centroid, compute_diag, delta_scaler, centroid_scaler)
        val_diag = evaluate_loader(model, val_loader, device, args.lambda_centroid, compute_diag, delta_scaler, centroid_scaler)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_diag["loss"],
            "val_loss": val_diag["loss"],
            "train_delta_loss": train_diag["delta_loss"],
            "train_centroid_loss": train_diag["centroid_loss"],
            "val_delta_loss": val_diag["delta_loss"],
            "val_centroid_loss": val_diag["centroid_loss"],
            "train_ADE": train_diag["delta_ade"],
            "train_FDE": train_diag["delta_fde"],
            "train_centroid_error": train_diag["centroid_error"],
            "val_ADE": val_diag["delta_ade"],
            "val_FDE": val_diag["delta_fde"],
            "val_centroid_error": val_diag["centroid_error"],
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
            "label_delta_scaler": asdict(delta_scaler),
            "label_centroid_scaler": asdict(centroid_scaler),
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
            f"epoch={epoch:04d}/{args.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_diag['loss']:.6f} train_delta_ADE={format_metric(train_diag['delta_ade'])} "
            f"val_delta_ADE={format_metric(val_diag['delta_ade'])} "
            f"val_centroid_error={format_metric(val_diag['centroid_error'])} "
            f"best_val_loss={best_val_loss:.6f} best_epoch={best_epoch}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopping_triggered = True
            print(f"Early stopping triggered after {epochs_without_improvement} epochs without validation improvement.")
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


def make_loader(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    delta_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        DeltaCentroidCorrectionDataset(args.data_root.resolve(), metadata, delta_scaler, centroid_scaler),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lambda_centroid: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.MSELoss()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        delta = batch["label_delta"].to(device, non_blocking=True)
        centroid = batch["label_centroid"].to(device, non_blocking=True)
        output = model(images)
        delta_loss = criterion(output["delta"], delta)
        centroid_loss = criterion(output["centroid"], centroid)
        loss = delta_loss + lambda_centroid * centroid_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_centroid: float,
    compute_metrics: bool,
    delta_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
) -> dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    totals = {"loss": 0.0, "delta_loss": 0.0, "centroid_loss": 0.0}
    total_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0
    centroid_error_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            delta = batch["label_delta"].to(device, non_blocking=True)
            centroid = batch["label_centroid"].to(device, non_blocking=True)
            output = model(images)
            delta_loss = criterion(output["delta"], delta)
            centroid_loss = criterion(output["centroid"], centroid)
            loss = delta_loss + lambda_centroid * centroid_loss
            batch_size = images.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["delta_loss"] += float(delta_loss.detach().cpu()) * batch_size
            totals["centroid_loss"] += float(centroid_loss.detach().cpu()) * batch_size
            if compute_metrics:
                pred_delta = inverse_sequence_tensor(delta_scaler, output["delta"])
                true_delta = inverse_sequence_tensor(delta_scaler, delta)
                pred_centroid = inverse_point_tensor(centroid_scaler, output["centroid"])
                true_centroid = inverse_point_tensor(centroid_scaler, centroid)
                point_errors = torch.linalg.norm(pred_delta - true_delta, dim=2)
                ade_sum += float(point_errors.mean(dim=1).sum().detach().cpu())
                fde_sum += float(point_errors[:, -1].sum().detach().cpu())
                centroid_error_sum += float(torch.linalg.norm(pred_centroid - true_centroid, dim=1).sum().detach().cpu())
            total_samples += batch_size
    return {
        "loss": totals["loss"] / max(1, total_samples),
        "delta_loss": totals["delta_loss"] / max(1, total_samples),
        "centroid_loss": totals["centroid_loss"] / max(1, total_samples),
        "delta_ade": ade_sum / max(1, total_samples) if compute_metrics else float("nan"),
        "delta_fde": fde_sum / max(1, total_samples) if compute_metrics else float("nan"),
        "centroid_error": centroid_error_sum / max(1, total_samples) if compute_metrics else float("nan"),
    }


def inverse_point_tensor(scaler: LabelScaler, values: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(scaler.mean, dtype=values.dtype, device=values.device).view(1, 2)
    std = torch.tensor(scaler.std, dtype=values.dtype, device=values.device).view(1, 2)
    return values * std + mean


def inverse_sequence_tensor(scaler: LabelScaler, values: torch.Tensor) -> torch.Tensor:
    if hasattr(scaler, "inverse_transform_tensor"):
        return scaler.inverse_transform_tensor(values)
    mean = torch.tensor(scaler.mean, dtype=values.dtype, device=values.device).view(1, 1, 2)
    std = torch.tensor(scaler.std, dtype=values.dtype, device=values.device).view(1, 1, 2)
    return values * std + mean


def write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_eval_loss",
        "val_loss",
        "train_delta_loss",
        "train_centroid_loss",
        "val_delta_loss",
        "val_centroid_loss",
        "train_ADE",
        "train_FDE",
        "train_centroid_error",
        "val_ADE",
        "val_FDE",
        "val_centroid_error",
        "learning_rate",
        "elapsed_time",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            "model": "DeltaCentroidCorrectionResNet18",
            "backbone": "torchvision.models.resnet18(weights=None)",
            "pretrained": False,
            "delta_head": "feature_dim -> 1024 -> dropout -> 448 -> reshape[224,2]",
            "centroid_head": "feature_dim -> 256 -> dropout -> 2",
            "input": "delta-displacement GAF RGB image / 255.0, shape [3,224,224]",
            "label_normalization": "train-only mean/std for delta displacement and absolute trajectory centroid",
            "loss": f"delta_MSE + {args.lambda_centroid} * centroid_MSE",
            "no_start_head": True,
            "no_absolute_ade_loss": True,
            "no_heading_or_turn_loss": True,
            "integration": "relative=cumsum(delta); relative[0]=0; pred_abs_oracle=true_start+relative",
            "centroid_correction": "trajectory translation by target_centroid - pred_abs_oracle.mean(axis=0)",
            "device_resolved": str(device),
            "train_samples": len(train_metadata),
            "val_samples": len(val_metadata),
        }
    )
    return config


def evaluate(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    pred_delta_dir = evaluation_dir / "predictions_delta"
    pred_oracle_dir = evaluation_dir / "predictions_absolute_oracle"
    pred_gt_corrected_dir = evaluation_dir / "predictions_absolute_gt_centroid_corrected"
    pred_learned_corrected_dir = evaluation_dir / "predictions_absolute_learned_centroid_corrected"
    pred_centroid_dir = evaluation_dir / "predictions_centroid"
    plots_dir = evaluation_dir / "plots"
    for directory in [
        pred_delta_dir,
        pred_oracle_dir,
        pred_gt_corrected_dir,
        pred_learned_corrected_dir,
        pred_centroid_dir,
        plots_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    delta_scaler = LabelScaler(**json.loads((output_dir / "label_delta_scaler.json").read_text()))
    centroid_scaler = LabelScaler(**json.loads((output_dir / "label_centroid_scaler.json").read_text()))
    model = DeltaCentroidCorrectionResNet18(dropout=args.dropout).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    test_predictions: dict[str, dict[str, np.ndarray]] = {}
    test_truths: dict[str, np.ndarray] = {}

    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = make_loader(args, split_frame, delta_scaler, centroid_scaler, False, device)
        split_predictions, split_truths, split_rows, timestep = predict_split(
            model,
            loader,
            delta_scaler,
            centroid_scaler,
            device,
            pred_delta_dir,
            pred_oracle_dir,
            pred_gt_corrected_dir,
            pred_learned_corrected_dir,
            pred_centroid_dir,
            split,
        )
        rows.extend(split_rows)
        timestep_rows.extend(
            {
                "split": split,
                "timestep": index,
                "delta_ade": float(timestep["delta"][index]),
                "oracle_integrated_ade": float(timestep["oracle_integrated"][index]),
                "gt_centroid_corrected_ade": float(timestep["gt_centroid_corrected"][index]),
                "learned_centroid_corrected_ade": float(timestep["learned_centroid_corrected"][index]),
            }
            for index in range(224)
        )
        if split == "test":
            test_predictions = split_predictions
            test_truths = split_truths

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.DataFrame(timestep_rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_metrics(per_sample)
    correction_summary = summarize_corrections(per_sample)
    correlation_summary = summarize_error_correlations(per_sample, plots_dir)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    write_json(evaluation_dir / "correction_summary.json", correction_summary)
    write_json(evaluation_dir / "error_correlation_summary.json", correlation_summary)
    plot_evaluation_outputs(per_sample, per_timestep, test_predictions, test_truths, plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "correction_summary": correction_summary,
        "error_correlation_summary": correlation_summary,
    }


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    delta_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    device: torch.device,
    pred_delta_dir: Path,
    pred_oracle_dir: Path,
    pred_gt_corrected_dir: Path,
    pred_learned_corrected_dir: Path,
    pred_centroid_dir: Path,
    split: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], list[dict[str, object]], dict[str, np.ndarray]]:
    predictions = {
        "oracle_integrated": {},
        "gt_centroid_corrected": {},
        "learned_centroid_corrected": {},
    }
    truths: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    timestep_error_sum = {
        "delta": np.zeros(224, dtype=np.float64),
        "oracle_integrated": np.zeros(224, dtype=np.float64),
        "gt_centroid_corrected": np.zeros(224, dtype=np.float64),
        "learned_centroid_corrected": np.zeros(224, dtype=np.float64),
    }
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            pred_delta = inverse_sequence_tensor(delta_scaler, output["delta"]).cpu().numpy().astype(np.float32)
            pred_centroid = inverse_point_tensor(centroid_scaler, output["centroid"]).cpu().numpy().astype(np.float32)
            true_delta = batch["label_delta_raw"].cpu().numpy().astype(np.float32)
            true_abs = batch["absolute"].cpu().numpy().astype(np.float32)
            true_start = batch["start"].cpu().numpy().astype(np.float32)
            true_centroid = batch["label_centroid_raw"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred_delta = pred_delta[index]
                sample_pred_centroid = pred_centroid[index]
                sample_true_delta = true_delta[index]
                sample_true_abs = true_abs[index]
                sample_true_start = true_start[index]
                sample_true_centroid = true_centroid[index]

                oracle_abs = integrate_delta(sample_true_start, sample_pred_delta)
                oracle_centroid = oracle_abs.mean(axis=0)
                gt_corrected = oracle_abs + (sample_true_centroid - oracle_centroid)
                learned_offset = sample_pred_centroid - oracle_centroid
                learned_corrected = oracle_abs + learned_offset

                np.save(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_pred_delta)
                np.save(pred_oracle_dir / f"{sample_id}_pred_absolute_oracle.npy", oracle_abs)
                np.save(pred_gt_corrected_dir / f"{sample_id}_pred_absolute_gt_centroid_corrected.npy", gt_corrected)
                np.save(pred_learned_corrected_dir / f"{sample_id}_pred_absolute_learned_centroid_corrected.npy", learned_corrected)
                np.save(pred_centroid_dir / f"{sample_id}_pred_centroid.npy", sample_pred_centroid)

                delta_metrics, delta_errors = compute_basic_metrics(sample_pred_delta, sample_true_delta, "delta")
                oracle_metrics, oracle_errors = compute_mode_metrics(oracle_abs, sample_true_abs, "oracle_integrated")
                gt_metrics, gt_errors = compute_mode_metrics(gt_corrected, sample_true_abs, "gt_centroid_corrected")
                learned_metrics, learned_errors = compute_mode_metrics(
                    learned_corrected, sample_true_abs, "learned_centroid_corrected"
                )
                row = {
                    "sample_id": sample_id,
                    "vehicle_id": vehicle_id,
                    "split": split,
                    **delta_metrics,
                    **oracle_metrics,
                    **gt_metrics,
                    **learned_metrics,
                    "learned_centroid_error": float(np.linalg.norm(sample_pred_centroid - sample_true_centroid)),
                    "centroid_offset_norm": float(np.linalg.norm(learned_offset)),
                    "gt_centroid_correction_ADE_improvement": float(
                        oracle_metrics["oracle_integrated_ADE"] - gt_metrics["gt_centroid_corrected_ADE"]
                    ),
                    "learned_centroid_correction_ADE_improvement": float(
                        oracle_metrics["oracle_integrated_ADE"] - learned_metrics["learned_centroid_corrected_ADE"]
                    ),
                    "learned_vs_gt_centroid_corrected_ADE_gap": float(
                        learned_metrics["learned_centroid_corrected_ADE"] - gt_metrics["gt_centroid_corrected_ADE"]
                    ),
                    "gt_centroid_correction_FDE_improvement": float(
                        oracle_metrics["oracle_integrated_FDE"] - gt_metrics["gt_centroid_corrected_FDE"]
                    ),
                    "learned_centroid_correction_FDE_improvement": float(
                        oracle_metrics["oracle_integrated_FDE"] - learned_metrics["learned_centroid_corrected_FDE"]
                    ),
                    "learned_vs_gt_centroid_corrected_FDE_gap": float(
                        learned_metrics["learned_centroid_corrected_FDE"] - gt_metrics["gt_centroid_corrected_FDE"]
                    ),
                }
                rows.append(row)
                predictions["oracle_integrated"][sample_id] = oracle_abs
                predictions["gt_centroid_corrected"][sample_id] = gt_corrected
                predictions["learned_centroid_corrected"][sample_id] = learned_corrected
                truths[sample_id] = sample_true_abs
                timestep_error_sum["delta"] += delta_errors
                timestep_error_sum["oracle_integrated"] += oracle_errors
                timestep_error_sum["gt_centroid_corrected"] += gt_errors
                timestep_error_sum["learned_centroid_corrected"] += learned_errors
                total += 1

    timestep = {key: value / max(1, total) for key, value in timestep_error_sum.items()}
    return predictions, truths, rows, timestep


def integrate_delta(start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    relative = np.cumsum(delta, axis=0).astype(np.float32)
    relative[0] = 0.0
    return start.astype(np.float32) + relative


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


def compute_mode_metrics(pred: np.ndarray, true: np.ndarray, prefix: str) -> tuple[dict[str, float], np.ndarray]:
    metrics, point_errors = compute_basic_metrics(pred, true, prefix)
    pred_length = trajectory_length(pred)
    true_length = trajectory_length(true)
    metrics.update(
        {
            f"{prefix}_centroid_error": float(np.linalg.norm(pred.mean(axis=0) - true.mean(axis=0))),
            f"{prefix}_end_error": float(point_errors[-1]),
            f"{prefix}_trajectory_length_error": float(abs(pred_length - true_length)),
            f"{prefix}_trajectory_length_ratio": float(pred_length / true_length) if true_length else np.nan,
        }
    )
    return metrics, point_errors


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def summarize_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics = DELTA_METRICS + [f"{prefix}_{metric}" for prefix in MODE_PREFIXES for metric in MODE_METRICS] + EXTRA_METRICS
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            summary[split][metric] = summarize_values(values)
    return summary


def summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
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


def summarize_corrections(per_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {
            "oracle_integrated_ADE_mean": float(np.nanmean(frame["oracle_integrated_ADE"])),
            "gt_centroid_corrected_ADE_mean": float(np.nanmean(frame["gt_centroid_corrected_ADE"])),
            "learned_centroid_corrected_ADE_mean": float(np.nanmean(frame["learned_centroid_corrected_ADE"])),
            "gt_centroid_ADE_improvement_mean": float(np.nanmean(frame["gt_centroid_correction_ADE_improvement"])),
            "learned_centroid_ADE_improvement_mean": float(np.nanmean(frame["learned_centroid_correction_ADE_improvement"])),
            "learned_vs_gt_ADE_gap_mean": float(np.nanmean(frame["learned_vs_gt_centroid_corrected_ADE_gap"])),
            "oracle_integrated_FDE_mean": float(np.nanmean(frame["oracle_integrated_FDE"])),
            "gt_centroid_corrected_FDE_mean": float(np.nanmean(frame["gt_centroid_corrected_FDE"])),
            "learned_centroid_corrected_FDE_mean": float(np.nanmean(frame["learned_centroid_corrected_FDE"])),
            "gt_centroid_FDE_improvement_mean": float(np.nanmean(frame["gt_centroid_correction_FDE_improvement"])),
            "learned_centroid_FDE_improvement_mean": float(np.nanmean(frame["learned_centroid_correction_FDE_improvement"])),
            "learned_vs_gt_FDE_gap_mean": float(np.nanmean(frame["learned_vs_gt_centroid_corrected_FDE_gap"])),
            "count_gt_centroid_correction_improved_ADE": int((frame["gt_centroid_correction_ADE_improvement"] > 0.0).sum()),
            "count_learned_centroid_correction_improved_ADE": int((frame["learned_centroid_correction_ADE_improvement"] > 0.0).sum()),
            "count_learned_close_to_gt_centroid_correction_ADE": int(
                (frame["learned_vs_gt_centroid_corrected_ADE_gap"].abs() < 10.0).sum()
            ),
        }
    return summary


def summarize_error_correlations(per_sample: pd.DataFrame, plots_dir: Path) -> dict[str, float]:
    test = per_sample[per_sample["split"] == "test"].copy()
    pairs = [
        (
            "learned_centroid_error",
            "learned_centroid_corrected_ADE",
            "learned_centroid_error_vs_learned_corrected_ADE.png",
        ),
        ("oracle_integrated_centroid_error", "oracle_integrated_ADE", "oracle_centroid_error_vs_oracle_ADE.png"),
        (
            "learned_centroid_corrected_centroid_error",
            "learned_centroid_corrected_ADE",
            "learned_corrected_centroid_error_vs_ADE.png",
        ),
        (
            "centroid_offset_norm",
            "learned_centroid_correction_ADE_improvement",
            "centroid_offset_norm_vs_learned_ADE_improvement.png",
        ),
    ]
    summary: dict[str, float] = {}
    for x_name, y_name, filename in pairs:
        x_values = test[x_name].to_numpy(dtype=float)
        y_values = test[y_name].to_numpy(dtype=float)
        pearson_r = pearson_corr(x_values, y_values)
        summary[f"{x_name}_corr_{y_name}"] = pearson_r
        plot_error_scatter(x_values, y_values, x_name, y_name, pearson_r, plots_dir / filename)
    return summary


def pearson_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x_values[mask]
    y = y_values[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_evaluation_outputs(
    per_sample: pd.DataFrame,
    per_timestep: pd.DataFrame,
    test_predictions: dict[str, dict[str, np.ndarray]],
    test_truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    test = per_sample[per_sample["split"] == "test"].copy()
    plot_hist(test["oracle_integrated_ADE"], plots_dir / "test_oracle_integrated_ADE.png", "Test oracle integrated ADE")
    plot_hist(
        test["gt_centroid_corrected_ADE"],
        plots_dir / "test_gt_centroid_corrected_ADE.png",
        "Test GT centroid corrected ADE",
    )
    plot_hist(
        test["learned_centroid_corrected_ADE"],
        plots_dir / "test_learned_centroid_corrected_ADE.png",
        "Test learned centroid corrected ADE",
    )
    plot_three_way_hist(test, plots_dir / "test_ADE_three_way_hist.png")
    plot_per_timestep(per_timestep, plots_dir / "test_per_timestep_ADE_three_way.png")
    plot_error_scatter(
        test["oracle_integrated_centroid_error"].to_numpy(dtype=float),
        test["oracle_integrated_ADE"].to_numpy(dtype=float),
        "oracle_integrated_centroid_error",
        "oracle_integrated_ADE",
        pearson_corr(test["oracle_integrated_centroid_error"].to_numpy(dtype=float), test["oracle_integrated_ADE"].to_numpy(dtype=float)),
        plots_dir / "scatter_oracle_centroid_error_vs_oracle_ADE.png",
    )
    plot_error_scatter(
        test["learned_centroid_error"].to_numpy(dtype=float),
        test["learned_centroid_corrected_ADE"].to_numpy(dtype=float),
        "learned_centroid_error",
        "learned_centroid_corrected_ADE",
        pearson_corr(test["learned_centroid_error"].to_numpy(dtype=float), test["learned_centroid_corrected_ADE"].to_numpy(dtype=float)),
        plots_dir / "scatter_learned_centroid_error_vs_learned_corrected_ADE.png",
    )
    plot_error_scatter(
        test["centroid_offset_norm"].to_numpy(dtype=float),
        test["learned_centroid_correction_ADE_improvement"].to_numpy(dtype=float),
        "centroid_offset_norm",
        "learned_centroid_correction_ADE_improvement",
        pearson_corr(
            test["centroid_offset_norm"].to_numpy(dtype=float),
            test["learned_centroid_correction_ADE_improvement"].to_numpy(dtype=float),
        ),
        plots_dir / "scatter_centroid_offset_norm_vs_learned_ADE_improvement.png",
    )
    plot_groups(test, test_predictions, test_truths, plots_dir)
    plot_improvement_groups(test, test_predictions, test_truths, plots_dir)


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


def plot_three_way_hist(test: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(test["oracle_integrated_ADE"], bins=30, alpha=0.5, label="oracle start")
    axis.hist(test["gt_centroid_corrected_ADE"], bins=30, alpha=0.5, label="GT centroid corrected")
    axis.hist(test["learned_centroid_corrected_ADE"], bins=30, alpha=0.5, label="learned centroid corrected")
    axis.set_xlabel("test ADE")
    axis.set_ylabel("count")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_timestep(per_timestep: pd.DataFrame, path: Path) -> None:
    test = per_timestep[per_timestep["split"] == "test"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(test["timestep"], test["oracle_integrated_ade"], label="oracle integrated")
    axis.plot(test["timestep"], test["gt_centroid_corrected_ade"], label="GT centroid corrected")
    axis.plot(test["timestep"], test["learned_centroid_corrected_ade"], label="learned centroid corrected")
    axis.set_xlabel("timestep")
    axis.set_ylabel("test ADE")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    pearson_r: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(x_values, y_values, s=16, alpha=0.8)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(f"Pearson r={pearson_r:.4f}")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_groups(
    test_rows: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    sorted_rows = test_rows.sort_values("learned_centroid_corrected_ADE").reset_index(drop=True)
    groups = {
        "best_learned_corrected": sorted_rows.head(10),
        "median_learned_corrected": sorted_rows.iloc[
            max(0, len(sorted_rows) // 2 - 5) : min(len(sorted_rows), len(sorted_rows) // 2 + 5)
        ],
        "worst_learned_corrected": sorted_rows.tail(10).sort_values("learned_centroid_corrected_ADE", ascending=False),
    }
    for group_name, frame in groups.items():
        plot_group(frame, group_name, predictions, truths, plots_dir)


def plot_improvement_groups(
    test_rows: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    groups = {
        "gt_centroid_correction_most_improved_ADE": test_rows.sort_values(
            "gt_centroid_correction_ADE_improvement", ascending=False
        ).head(10),
        "learned_centroid_correction_most_improved_ADE": test_rows.sort_values(
            "learned_centroid_correction_ADE_improvement", ascending=False
        ).head(10),
        "learned_centroid_correction_most_hurt_ADE": test_rows.sort_values(
            "learned_centroid_correction_ADE_improvement", ascending=True
        ).head(10),
    }
    for group_name, frame in groups.items():
        plot_group(frame, group_name, predictions, truths, plots_dir)


def plot_group(
    frame: pd.DataFrame,
    group_name: str,
    predictions: dict[str, dict[str, np.ndarray]],
    truths: dict[str, np.ndarray],
    plots_dir: Path,
) -> None:
    group_dir = plots_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in frame.itertuples(index=False):
        sample_id = str(row.sample_id)
        path = group_dir / f"{sample_id}_trajectory.png"
        plot_prediction(
            sample_id,
            str(row.vehicle_id),
            truths[sample_id],
            predictions["oracle_integrated"][sample_id],
            predictions["gt_centroid_corrected"][sample_id],
            predictions["learned_centroid_corrected"][sample_id],
            float(row.oracle_integrated_ADE),
            float(row.gt_centroid_corrected_ADE),
            float(row.learned_centroid_corrected_ADE),
            path,
        )
        paths.append(path)
    if make_contact_sheet is not None:
        make_contact_sheet(paths, plots_dir / f"{group_name}_10_grid.png")


def plot_prediction(
    sample_id: str,
    vehicle_id: str,
    true_abs: np.ndarray,
    oracle_abs: np.ndarray,
    gt_corrected: np.ndarray,
    learned_corrected: np.ndarray,
    oracle_ade: float,
    gt_ade: float,
    learned_ade: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(true_abs[:, 0], true_abs[:, 1], label="GT", linewidth=1.7)
    axis.plot(oracle_abs[:, 0], oracle_abs[:, 1], label="oracle start", linewidth=1.3)
    axis.plot(gt_corrected[:, 0], gt_corrected[:, 1], label="GT centroid corrected", linewidth=1.3)
    axis.plot(learned_corrected[:, 0], learned_corrected[:, 1], label="learned centroid corrected", linewidth=1.3)
    axis.scatter(true_abs[0, 0], true_abs[0, 1], s=20, marker="o", label="GT start")
    axis.scatter(true_abs[-1, 0], true_abs[-1, 1], s=20, marker="s", label="GT end")
    all_xy = np.vstack([true_abs, oracle_abs, gt_corrected, learned_corrected])
    pad_x = max(1.0, float(np.ptp(all_xy[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(all_xy[:, 1])) * 0.05)
    axis.set_xlim(float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
    axis.set_ylim(float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{sample_id} {vehicle_id}\n"
        f"oracle={oracle_ade:.2f} gt-corr={gt_ade:.2f} learned-corr={learned_ade:.2f}"
    )
    axis.legend(fontsize=6)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_chinese_report(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    metrics = summary["metrics_summary"]
    correction = summary["correction_summary"]
    correlations = summary["error_correlation_summary"]
    best_path = output_dir / "best_summary.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    test = metrics["test"]
    correction_test = correction["test"]

    gt_improves = correction_test["gt_centroid_ADE_improvement_mean"] > 0.0
    gt_large = correction_test["gt_centroid_ADE_improvement_mean"] > 10.0
    learned_gap = correction_test["learned_vs_gt_ADE_gap_mean"]
    learned_improves = correction_test["learned_centroid_ADE_improvement_mean"] > 0.0
    learned_close = abs(learned_gap) < 10.0
    learned_hurts = correction_test["learned_centroid_ADE_improvement_mean"] < 0.0

    report = f"""
Delta Centroid Correction Decoder 中文报告

动机：
- 本实验用于诊断 original delta-displacement decoder 的主要失败是否来自 global drift / centroid shift。
- 如果只用 true start 积分 delta 后形状仍较好，但整体中心偏移很大，centroid correction 应该显著降低 ADE/FDE。
- GT centroid correction 是 upper-bound diagnostic：它回答“如果 centroid 完美已知，可以消除多少 drift？”

实验设置：
- 数据集：{args.data_root.resolve()}
- 输入：delta-displacement GAF RGB image / 255.0，shape [3,224,224]。
- 模型：scratch ResNet18 shared backbone，delta head 512 -> 1024 -> dropout -> 448，centroid head 512 -> 256 -> dropout -> 2。
- 不包含 learned start head。
- 不使用 absolute ADE loss。
- 不使用 heading/turn losses，也不加入 geometry-aware supervision；这是为了保留 original delta baseline 的局部 shape 训练行为。
- 需要注意：本实验是 delta baseline + auxiliary centroid multitask supervision，不是完全等同于 original delta-only baseline。
- 归一化：delta scaler 与 centroid scaler 都只用 train split 拟合 mean/std。
- 训练目标：loss_delta + {args.lambda_centroid} * loss_centroid。
- 积分：relative=cumsum(delta)，relative[0]=0，pred_abs_oracle=true_start+relative，与 V2/V3 约定一致。

三种解码 / 修正模式：
- oracle-start delta baseline：true_start + relative，其中 relative=cumsum(pred_delta) 且 relative[0]=0。
- oracle-start + GT centroid correction：pred_abs_oracle + (gt_centroid - pred_abs_oracle.mean(axis=0))。
- oracle-start + learned centroid correction：pred_abs_oracle + (pred_centroid - pred_abs_oracle.mean(axis=0))。

训练结果：
- best epoch={best.get('best_epoch', 'NA')}, final epoch={best.get('final_epoch', 'NA')}, early stopping={best.get('early_stopping_triggered', 'NA')}。
- best validation loss={best.get('best_val_loss', 'NA')}。

测试集结果：
- delta ADE/FDE={test['delta_ADE']['mean']:.4f} / {test['delta_FDE']['mean']:.4f}
- oracle-start integrated ADE/FDE={test['oracle_integrated_ADE']['mean']:.4f} / {test['oracle_integrated_FDE']['mean']:.4f}
- GT-centroid-corrected ADE/FDE={test['gt_centroid_corrected_ADE']['mean']:.4f} / {test['gt_centroid_corrected_FDE']['mean']:.4f}
- learned-centroid-corrected ADE/FDE={test['learned_centroid_corrected_ADE']['mean']:.4f} / {test['learned_centroid_corrected_FDE']['mean']:.4f}
- learned centroid error={test['learned_centroid_error']['mean']:.4f}
- oracle integrated centroid error={test['oracle_integrated_centroid_error']['mean']:.4f}
- learned corrected centroid error={test['learned_centroid_corrected_centroid_error']['mean']:.4f}
- centroid offset norm={test['centroid_offset_norm']['mean']:.4f}
- GT centroid ADE improvement mean={correction_test['gt_centroid_ADE_improvement_mean']:.4f}
- learned centroid ADE improvement mean={correction_test['learned_centroid_ADE_improvement_mean']:.4f}
- learned vs GT ADE gap mean={correction_test['learned_vs_gt_ADE_gap_mean']:.4f}
- GT centroid FDE improvement mean={correction_test['gt_centroid_FDE_improvement_mean']:.4f}
- learned centroid FDE improvement mean={correction_test['learned_centroid_FDE_improvement_mean']:.4f}
- learned vs GT FDE gap mean={correction_test['learned_vs_gt_FDE_gap_mean']:.4f}
- GT centroid correction improves ADE samples={correction_test['count_gt_centroid_correction_improved_ADE']}
- learned centroid correction improves ADE samples={correction_test['count_learned_centroid_correction_improved_ADE']}
- learned close to GT centroid correction samples (abs gap < 10)={correction_test['count_learned_close_to_gt_centroid_correction_ADE']}

相关性分析（test split）：
- corr(learned_centroid_error, learned_centroid_corrected_ADE)={correlations['learned_centroid_error_corr_learned_centroid_corrected_ADE']:.4f}
- corr(oracle_integrated_centroid_error, oracle_integrated_ADE)={correlations['oracle_integrated_centroid_error_corr_oracle_integrated_ADE']:.4f}
- corr(learned_centroid_corrected_centroid_error, learned_centroid_corrected_ADE)={correlations['learned_centroid_corrected_centroid_error_corr_learned_centroid_corrected_ADE']:.4f}
- corr(centroid_offset_norm, learned_centroid_correction_ADE_improvement)={correlations['centroid_offset_norm_corr_learned_centroid_correction_ADE_improvement']:.4f}

解释：
- {'GT centroid correction greatly improves ADE: delta representation preserves useful shape, but suffers from global centroid drift.' if gt_large else 'GT centroid correction 没有大幅改善 ADE：当前失败不只是一阶整体 centroid shift，delta 局部形状或积分误差也可能重要。'}
- {'Learned centroid correction approaches GT correction: GAF contains learnable global centroid information.' if learned_close and learned_improves else 'Learned correction improves little while GT correction improves a lot: GAF/decoder struggles to infer global centroid, suggesting missing absolute localization information.' if gt_improves and not learned_improves else 'Learned centroid correction 有一定改善，但与 GT upper bound 仍有距离，需要看 learned centroid error 与 correction gap。'}
- {'Learned correction hurts: centroid head may be misaligned with delta prediction; correction should be treated carefully.' if learned_hurts else 'Learned correction 没有在均值上伤害 ADE。'}

输出文件：
- config.json
- label_delta_scaler.json, label_centroid_scaler.json
- split_copy.csv
- train_log.csv
- checkpoints/best_model.pt, checkpoints/last_model.pt
- best_summary.json
- evaluation/per_sample_metrics.csv
- evaluation/per_timestep_ade.csv
- evaluation/metrics_summary.json
- evaluation/correction_summary.json
- evaluation/error_correlation_summary.json
- evaluation/predictions_delta/
- evaluation/predictions_absolute_oracle/
- evaluation/predictions_absolute_gt_centroid_corrected/
- evaluation/predictions_absolute_learned_centroid_corrected/
- evaluation/predictions_centroid/
- evaluation/plots/
"""
    path = output_dir / "delta_centroid_correction_report_zh.txt"
    path.write_text(report.strip() + "\n")
    print(report.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


if __name__ == "__main__":
    main()

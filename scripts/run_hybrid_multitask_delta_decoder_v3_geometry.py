"""Train and evaluate a V3 geometry-aware hybrid delta-displacement decoder.

The model shares a scratch ResNet18 image encoder and predicts both normalized
delta displacement trajectories, normalized start points, and normalized
centroids. Absolute trajectories are evaluated by integrating predicted deltas
from the learned start, plus an oracle-start comparison using the true start
point. V3 adds output-side heading and turn-angle supervision to preserve local
maneuver geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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

from run_original_config_decoder_experiment import make_contact_sheet  # noqa: E402
from run_resnet18_decoder_ablation import (  # noqa: E402
    format_metric,
    plot_training_curves,
    resolve_device,
    set_seed,
    write_json,
)


DELTA_METRICS = ["delta_ADE", "delta_FDE", "delta_RMSE", "delta_MAE"]
INTEGRATED_METRICS = [
    "start_error",
    "integrated_ADE",
    "integrated_FDE",
    "integrated_RMSE",
    "integrated_MAE",
    "centroid_head_error",
    "integrated_centroid_error",
    "end_error",
    "trajectory_length_error",
    "trajectory_length_ratio",
    "cumulative_drift_final",
    "oracle_integrated_ADE",
    "oracle_integrated_FDE",
]
GEOMETRY_METRICS = [
    "heading_error_deg",
    "turn_error_deg",
    "mean_abs_turn_gt_deg",
    "mean_abs_turn_pred_deg",
    "turn_sharpness_ratio",
]


@dataclass(frozen=True)
class LabelScaler:
    mean: list[float]
    std: list[float]

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.tensor(self.std, dtype=values.dtype, device=values.device)
        if values.ndim == 3:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
        elif values.ndim == 2:
            mean = mean.view(1, -1)
            std = std.view(1, -1)
        return values * std + mean


class HybridDeltaDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        metadata: pd.DataFrame,
        delta_scaler: LabelScaler,
        start_scaler: LabelScaler,
        centroid_scaler: LabelScaler,
    ) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.delta_scaler = delta_scaler
        self.start_scaler = start_scaler
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
            "label_delta_norm": torch.from_numpy(self.delta_scaler.transform(delta)),
            "label_delta_raw": torch.from_numpy(delta),
            "label_absolute": torch.from_numpy(absolute),
            "start_norm": torch.from_numpy(self.start_scaler.transform(start)),
            "start_raw": torch.from_numpy(start),
            "centroid_norm": torch.from_numpy(self.centroid_scaler.transform(centroid)),
            "centroid_raw": torch.from_numpy(centroid),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


class HybridMultiHeadResNet18Decoder(nn.Module):
    """Scratch ResNet18 backbone with separate delta, start, and centroid regression heads."""

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
        self.start_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.centroid_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "delta": self.delta_head(features).view(-1, 224, 2),
            "start": self.start_head(features),
            "centroid": self.centroid_head(features),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.resolve()
    for subdir in ["checkpoints", "plots", "evaluation", "evaluation/plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.resolve()
    split_metadata = verify_delta_data_and_split(data_root)
    delta0_summary = compute_delta0_sanity(data_root, split_metadata)
    write_json(output_dir / "evaluation" / "delta0_sanity_summary.json", delta0_summary)
    print_delta0_sanity(delta0_summary)
    if not args.skip_training:
        train(args, split_metadata, device)

    summary = evaluate(args, split_metadata, device)
    write_chinese_report(args, summary)


def default_data_root() -> Path:
    return Path(os.environ["SCRATCH"]) / "Thesis_data/data_no_speed_delta_displacement_paired"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-dir", type=Path, default=Path("results_hpc/hybrid_multitask_delta_decoder_v3_geometry"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-start", type=float, default=1.0)
    parser.add_argument("--lambda-centroid", type=float, default=0.5)
    parser.add_argument("--lambda-abs", type=float, default=0.01)
    parser.add_argument("--lambda-heading", type=float, default=0.05)
    parser.add_argument("--lambda-turn", type=float, default=0.1)
    parser.add_argument("--turn-sharp-weight", type=float, default=2.0)
    parser.add_argument("--turn-weight-max", type=float, default=3.0)
    parser.add_argument("--speed-eps", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
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
    image_min = 255
    image_max = 0
    for row in split_metadata.itertuples(index=False):
        paths = {
            "image": data_root / row.image_path,
            "delta": data_root / row.label_delta_displacement_path,
            "absolute": data_root / row.label_absolute_path,
            "start": data_root / row.start_path,
        }
        missing_paths = [name for name, path in paths.items() if not path.exists()]
        if missing_paths:
            bad.append({"sample_id": row.sample_id, "reason": f"missing:{','.join(missing_paths)}"})
            continue
        with Image.open(paths["image"]) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                bad.append({"sample_id": row.sample_id, "reason": f"bad_image:{image.mode}:{image.size}"})
            pixels = np.asarray(image)
            image_min = min(image_min, int(pixels.min()))
            image_max = max(image_max, int(pixels.max()))
        delta = np.load(paths["delta"])
        absolute = np.load(paths["absolute"])
        start = np.load(paths["start"])
        if delta.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_delta:{delta.shape}"})
        if absolute.shape != (224, 2):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_absolute:{absolute.shape}"})
        if start.shape != (2,):
            bad.append({"sample_id": row.sample_id, "reason": f"bad_start:{start.shape}"})

    if bad:
        report_path = data_root / "splits" / "hybrid_multitask_consistency_report.csv"
        pd.DataFrame(bad).to_csv(report_path, index=False)
        raise RuntimeError(f"Dataset consistency check failed. Report: {report_path}")

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    print("Hybrid delta data and split validation passed")
    print(f"total samples: {len(split_metadata)}")
    print(f"train count: {int(counts['train'])}")
    print(f"val count: {int(counts['val'])}")
    print(f"test count: {int(counts['test'])}")
    print(f"image channel min/max: {image_min}/{image_max}")
    return split_metadata


def fit_delta_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [
        np.load(data_root / row.label_delta_displacement_path).astype(np.float32)
        for row in metadata.itertuples(index=False)
    ]
    stacked = np.concatenate(labels, axis=0)
    return make_scaler(stacked)


def fit_start_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [np.load(data_root / row.start_path).astype(np.float32) for row in metadata.itertuples(index=False)]
    stacked = np.stack(labels, axis=0)
    return make_scaler(stacked)


def fit_centroid_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [
        np.load(data_root / row.label_absolute_path).astype(np.float32).mean(axis=0)
        for row in metadata.itertuples(index=False)
    ]
    stacked = np.stack(labels, axis=0)
    return make_scaler(stacked)


def compute_delta0_sanity(data_root: Path, metadata: pd.DataFrame) -> dict[str, float]:
    delta0_values = np.stack(
        [
            np.load(data_root / row.label_delta_displacement_path).astype(np.float32)[0]
            for row in metadata.itertuples(index=False)
        ],
        axis=0,
    )
    delta0_norm = np.linalg.norm(delta0_values, axis=1)
    return {
        "num_samples": int(delta0_values.shape[0]),
        "delta0_norm_mean": float(np.mean(delta0_norm)),
        "delta0_norm_median": float(np.median(delta0_norm)),
        "delta0_norm_max": float(np.max(delta0_norm)),
        "delta0_abs_x_mean": float(np.mean(np.abs(delta0_values[:, 0]))),
        "delta0_abs_y_mean": float(np.mean(np.abs(delta0_values[:, 1]))),
    }


def print_delta0_sanity(summary: dict[str, float]) -> None:
    print("Delta[0] sanity diagnostics")
    print(f"num samples: {int(summary['num_samples'])}")
    print(f"delta0_norm_mean: {summary['delta0_norm_mean']:.8f}")
    print(f"delta0_norm_median: {summary['delta0_norm_median']:.8f}")
    print(f"delta0_norm_max: {summary['delta0_norm_max']:.8f}")
    print(f"delta0_abs_x_mean: {summary['delta0_abs_x_mean']:.8f}")
    print(f"delta0_abs_y_mean: {summary['delta0_abs_y_mean']:.8f}")


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
    delta_scaler = fit_delta_scaler(data_root, train_metadata)
    start_scaler = fit_start_scaler(data_root, train_metadata)
    centroid_scaler = fit_centroid_scaler(data_root, train_metadata)

    write_json(output_dir / "label_delta_scaler.json", asdict(delta_scaler))
    write_json(output_dir / "label_start_scaler.json", asdict(start_scaler))
    write_json(output_dir / "label_centroid_scaler.json", asdict(centroid_scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = make_loader(
        args, train_metadata, delta_scaler, start_scaler, centroid_scaler, shuffle=True, device=device
    )
    val_loader = make_loader(
        args, val_metadata, delta_scaler, start_scaler, centroid_scaler, shuffle=False, device=device
    )
    model = HybridMultiHeadResNet18Decoder(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    log_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, delta_scaler, start_scaler, centroid_scaler, device, args, optimizer)
        compute_diag = epoch == 1 or epoch % args.metrics_every == 0
        train_diag = evaluate_loader(
            model, train_loader, delta_scaler, start_scaler, centroid_scaler, device, args, compute_diag
        )
        val_diag = evaluate_loader(
            model, val_loader, delta_scaler, start_scaler, centroid_scaler, device, args, compute_diag
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_diag["loss"],
            "val_loss": val_diag["loss"],
            "train_ADE": train_diag["integrated_ade"],
            "train_FDE": train_diag["integrated_fde"],
            "val_ADE": val_diag["integrated_ade"],
            "val_FDE": val_diag["integrated_fde"],
            "train_delta_loss": train_diag["loss_delta"],
            "train_start_loss": train_diag["loss_start"],
            "train_centroid_loss": train_diag["loss_centroid"],
            "train_abs_loss": train_diag["loss_abs"],
            "train_heading_loss": train_diag["loss_heading"],
            "train_turn_loss": train_diag["loss_turn"],
            "val_delta_loss": val_diag["loss_delta"],
            "val_start_loss": val_diag["loss_start"],
            "val_centroid_loss": val_diag["loss_centroid"],
            "val_abs_loss": val_diag["loss_abs"],
            "val_heading_loss": val_diag["loss_heading"],
            "val_turn_loss": val_diag["loss_turn"],
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
            "label_start_scaler": asdict(start_scaler),
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
            f"epoch={epoch:04d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_diag['loss']:.6f} "
            f"train_integrated_ADE={format_metric(train_diag['integrated_ade'])} "
            f"val_integrated_ADE={format_metric(val_diag['integrated_ade'])} "
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


def make_loader(
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    delta_scaler: LabelScaler,
    start_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        HybridDeltaDataset(args.data_root.resolve(), metadata, delta_scaler, start_scaler, centroid_scaler),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    delta_scaler: LabelScaler,
    start_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            loss, _ = compute_batch_losses(model, batch, delta_scaler, start_scaler, centroid_scaler, device, args)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = batch["image"].shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
    return total_loss / max(1, total_samples)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    delta_scaler: LabelScaler,
    start_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    device: torch.device,
    args: argparse.Namespace,
    compute_metrics: bool,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "loss_delta": 0.0,
        "loss_start": 0.0,
        "loss_centroid": 0.0,
        "loss_abs": 0.0,
        "loss_heading": 0.0,
        "loss_turn": 0.0,
    }
    total_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            loss, parts = compute_batch_losses(model, batch, delta_scaler, start_scaler, centroid_scaler, device, args)
            batch_size = batch["image"].shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key in ["loss_delta", "loss_start", "loss_centroid", "loss_abs", "loss_heading", "loss_turn"]:
                totals[key] += float(parts[key].detach().cpu()) * batch_size
            if compute_metrics:
                point_errors = parts["integrated_errors"]
                ade_sum += float(point_errors.mean(dim=1).sum().detach().cpu())
                fde_sum += float(point_errors[:, -1].sum().detach().cpu())
            total_samples += batch_size
    return {
        key: value / max(1, total_samples)
        for key, value in totals.items()
    } | {
        "integrated_ade": ade_sum / max(1, total_samples) if compute_metrics else float("nan"),
        "integrated_fde": fde_sum / max(1, total_samples) if compute_metrics else float("nan"),
    }


def compute_batch_losses(
    model: nn.Module,
    batch: dict[str, object],
    delta_scaler: LabelScaler,
    start_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = batch["image"].to(device, non_blocking=True)
    gt_delta_norm = batch["label_delta_norm"].to(device, non_blocking=True)
    gt_start_norm = batch["start_norm"].to(device, non_blocking=True)
    gt_centroid_norm = batch["centroid_norm"].to(device, non_blocking=True)
    gt_absolute = batch["label_absolute"].to(device, non_blocking=True)

    output = model(images)
    pred_delta_norm = output["delta"]
    pred_start_norm = output["start"]
    pred_centroid_norm = output["centroid"]
    loss_delta = torch.nn.functional.mse_loss(pred_delta_norm, gt_delta_norm)
    loss_start = torch.nn.functional.mse_loss(pred_start_norm, gt_start_norm)
    loss_centroid = torch.nn.functional.mse_loss(pred_centroid_norm, gt_centroid_norm)

    pred_delta = delta_scaler.inverse_transform_tensor(pred_delta_norm)
    pred_start = start_scaler.inverse_transform_tensor(pred_start_norm)
    pred_abs = integrate_delta_tensor(pred_start, pred_delta)
    point_error = torch.linalg.norm(pred_abs - gt_absolute, dim=2)
    loss_abs = point_error.mean()
    loss_heading, loss_turn = compute_geometry_losses(
        pred_abs,
        gt_absolute,
        args.speed_eps,
        args.turn_sharp_weight,
        args.turn_weight_max,
    )
    loss = (
        loss_delta
        + args.lambda_start * loss_start
        + args.lambda_centroid * loss_centroid
        + args.lambda_abs * loss_abs
        + args.lambda_heading * loss_heading
        + args.lambda_turn * loss_turn
    )
    return loss, {
        "loss_delta": loss_delta,
        "loss_start": loss_start,
        "loss_centroid": loss_centroid,
        "loss_abs": loss_abs,
        "loss_heading": loss_heading,
        "loss_turn": loss_turn,
        "integrated_errors": point_error,
    }


def integrate_delta_tensor(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    relative = torch.cumsum(delta, dim=1)
    relative[:, 0, :] = 0.0
    return start.unsqueeze(1) + relative


def compute_geometry_losses(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    speed_eps: float,
    turn_sharp_weight: float,
    turn_weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_delta = pred_points[:, 1:, :] - pred_points[:, :-1, :]
    gt_delta = gt_points[:, 1:, :] - gt_points[:, :-1, :]
    pred_heading = torch.atan2(pred_delta[:, :, 1], pred_delta[:, :, 0])
    gt_heading = torch.atan2(gt_delta[:, :, 1], gt_delta[:, :, 0])
    pred_speed = torch.linalg.norm(pred_delta, dim=2)
    gt_speed = torch.linalg.norm(gt_delta, dim=2)
    valid_heading = (pred_speed > speed_eps) & (gt_speed > speed_eps)

    heading_diff = wrap_angle_tensor(pred_heading - gt_heading)
    if bool(valid_heading.any().item()):
        loss_heading = (1.0 - torch.cos(heading_diff[valid_heading])).mean()
    else:
        loss_heading = pred_points.sum() * 0.0

    pred_turn = wrap_angle_tensor(pred_heading[:, 1:] - pred_heading[:, :-1])
    gt_turn = wrap_angle_tensor(gt_heading[:, 1:] - gt_heading[:, :-1])
    valid_turn = valid_heading[:, 1:] & valid_heading[:, :-1]
    turn_diff = wrap_angle_tensor(pred_turn - gt_turn)
    base_turn_loss = 1.0 - torch.cos(turn_diff)
    gt_turn_magnitude = torch.abs(gt_turn)
    turn_weight = 1.0 + turn_sharp_weight * gt_turn_magnitude / torch.pi
    turn_weight = torch.clamp(turn_weight, max=turn_weight_max)
    if bool(valid_turn.any().item()):
        loss_turn = (turn_weight[valid_turn] * base_turn_loss[valid_turn]).mean()
    else:
        loss_turn = pred_points.sum() * 0.0
    return loss_heading, loss_turn


def wrap_angle_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def write_train_log(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_eval_loss",
        "val_loss",
        "train_ADE",
        "train_FDE",
        "val_ADE",
        "val_FDE",
        "train_delta_loss",
        "train_start_loss",
        "train_centroid_loss",
        "train_abs_loss",
        "train_heading_loss",
        "train_turn_loss",
        "val_delta_loss",
        "val_start_loss",
        "val_centroid_loss",
        "val_abs_loss",
        "val_heading_loss",
        "val_turn_loss",
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
            "model": "HybridMultiHeadResNet18Decoder",
            "backbone": "torchvision.models.resnet18(weights=None)",
            "pretrained": False,
            "delta_head": "feature_dim -> 1024 -> dropout -> 448 -> reshape[224,2]",
            "start_head": "feature_dim -> 256 -> dropout -> 2",
            "centroid_head": "feature_dim -> 256 -> dropout -> 2",
            "optimizer": "Adam",
            "scheduler": "none",
            "image_normalization": "image / 255.0",
            "label_normalization": "train-only mean/std for delta, start, and centroid labels",
            "loss": "delta_MSE + lambda_start * start_MSE + lambda_centroid * centroid_MSE + lambda_abs * ADE-style absolute loss + lambda_heading * heading_loss + lambda_turn * turn_loss",
            "heading_loss": "mean(1 - cos(wrap(pred_heading - gt_heading))) over valid speed mask",
            "turn_loss": "mean(weight * (1 - cos(wrap(pred_turn - gt_turn)))) over valid turn mask",
            "turn_loss_weight": "weight = clamp(1 + turn_sharp_weight * abs(gt_turn) / pi, max=turn_weight_max)",
            "turn_sharp_weight": args.turn_sharp_weight,
            "turn_weight_max": args.turn_weight_max,
            "integration": "relative=cumsum(delta); relative[0]=0; absolute=start+relative",
            "centroid_head_usage": "auxiliary supervision only; not used for direct trajectory correction",
            "geometry_supervision": "output-side only; GAF image remains the only model input",
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
    pred_delta_dir = evaluation_dir / "predictions_delta"
    pred_start_dir = evaluation_dir / "predictions_start"
    pred_absolute_dir = evaluation_dir / "predictions_absolute_integrated"
    plots_dir = evaluation_dir / "plots"
    for directory in [pred_delta_dir, pred_start_dir, pred_absolute_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    delta_scaler = LabelScaler(**json.loads((output_dir / "label_delta_scaler.json").read_text()))
    start_scaler = LabelScaler(**json.loads((output_dir / "label_start_scaler.json").read_text()))
    centroid_scaler = LabelScaler(**json.loads((output_dir / "label_centroid_scaler.json").read_text()))
    model = HybridMultiHeadResNet18Decoder(dropout=args.dropout).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    timestep_rows: list[dict[str, object]] = []
    predictions_abs: dict[str, dict[str, np.ndarray]] = {}
    truths_abs: dict[str, dict[str, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = make_loader(
            args, split_frame, delta_scaler, start_scaler, centroid_scaler, shuffle=False, device=device
        )
        split_pred_abs, split_true_abs, split_rows, timestep = predict_split(
            model,
            loader,
            delta_scaler,
            start_scaler,
            centroid_scaler,
            device,
            pred_delta_dir,
            pred_start_dir,
            pred_absolute_dir,
            split,
            args.speed_eps,
        )
        predictions_abs[split] = split_pred_abs
        truths_abs[split] = split_true_abs
        rows.extend(split_rows)
        timestep_rows.extend(
            {
                "split": split,
                "timestep": index,
                "delta_ade": float(timestep["delta"][index]),
                "integrated_ade": float(timestep["integrated"][index]),
                "oracle_integrated_ade": float(timestep["oracle"][index]),
            }
            for index in range(224)
        )

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.DataFrame(timestep_rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_metrics(per_sample)
    oracle_summary = summarize_oracle_vs_learned_start(per_sample)
    correlation_summary = summarize_error_correlations(per_sample, plots_dir)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    write_json(evaluation_dir / "oracle_vs_learned_start_summary.json", oracle_summary)
    write_json(evaluation_dir / "error_correlation_summary.json", correlation_summary)
    plot_evaluation_outputs(per_sample, per_timestep, predictions_abs["test"], truths_abs["test"], plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "oracle_vs_learned_start_summary": oracle_summary,
        "error_correlation_summary": correlation_summary,
    }


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    delta_scaler: LabelScaler,
    start_scaler: LabelScaler,
    centroid_scaler: LabelScaler,
    device: torch.device,
    pred_delta_dir: Path,
    pred_start_dir: Path,
    pred_absolute_dir: Path,
    split: str,
    speed_eps: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]], dict[str, np.ndarray]]:
    pred_abs_by_id: dict[str, np.ndarray] = {}
    true_abs_by_id: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    timestep_error_sum = {
        "delta": np.zeros(224, dtype=np.float64),
        "integrated": np.zeros(224, dtype=np.float64),
        "oracle": np.zeros(224, dtype=np.float64),
    }
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            output = model(images)
            pred_delta = delta_scaler.inverse_transform_tensor(output["delta"]).cpu().numpy().astype(np.float32)
            pred_start = start_scaler.inverse_transform_tensor(output["start"]).cpu().numpy().astype(np.float32)
            pred_centroid = centroid_scaler.inverse_transform_tensor(output["centroid"]).cpu().numpy().astype(np.float32)
            true_delta = batch["label_delta_raw"].cpu().numpy().astype(np.float32)
            true_abs = batch["label_absolute"].cpu().numpy().astype(np.float32)
            true_start = batch["start_raw"].cpu().numpy().astype(np.float32)
            true_centroid = batch["centroid_raw"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred_delta = pred_delta[index]
                sample_pred_start = pred_start[index]
                sample_pred_centroid = pred_centroid[index]
                sample_true_delta = true_delta[index]
                sample_true_abs = true_abs[index]
                sample_true_start = true_start[index]
                sample_true_centroid = true_centroid[index]
                sample_pred_abs = integrate_delta(sample_pred_start, sample_pred_delta)
                sample_oracle_abs = integrate_delta(sample_true_start, sample_pred_delta)

                np.save(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_pred_delta)
                np.save(pred_start_dir / f"{sample_id}_pred_start.npy", sample_pred_start)
                np.save(pred_absolute_dir / f"{sample_id}_pred_absolute_integrated.npy", sample_pred_abs)

                delta_metrics, delta_errors = compute_basic_metrics(sample_pred_delta, sample_true_delta, "delta")
                integrated_metrics, integrated_errors = compute_basic_metrics(
                    sample_pred_abs, sample_true_abs, "integrated"
                )
                oracle_metrics, oracle_errors = compute_basic_metrics(
                    sample_oracle_abs, sample_true_abs, "oracle_integrated"
                )
                pred_length = trajectory_length(sample_pred_abs)
                true_length = trajectory_length(sample_true_abs)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **delta_metrics,
                        **integrated_metrics,
                        "start_error": float(np.linalg.norm(sample_pred_start - sample_true_start)),
                        "centroid_head_error": float(np.linalg.norm(sample_pred_centroid - sample_true_centroid)),
                        "integrated_centroid_error": float(
                            np.linalg.norm(sample_pred_abs.mean(axis=0) - sample_true_abs.mean(axis=0))
                        ),
                        "end_error": float(integrated_errors[-1]),
                        "trajectory_length_error": float(abs(pred_length - true_length)),
                        "trajectory_length_ratio": float(pred_length / true_length) if true_length else np.nan,
                        "cumulative_drift_final": float(
                            np.linalg.norm((sample_pred_abs[-1] - sample_true_abs[-1]) - (sample_pred_abs[0] - sample_true_abs[0]))
                        ),
                        **compute_geometry_diagnostics(sample_pred_abs, sample_true_abs, speed_eps),
                        "oracle_integrated_ADE": float(oracle_metrics["oracle_integrated_ADE"]),
                        "oracle_integrated_FDE": float(oracle_metrics["oracle_integrated_FDE"]),
                    }
                )
                pred_abs_by_id[sample_id] = sample_pred_abs
                true_abs_by_id[sample_id] = sample_true_abs
                timestep_error_sum["delta"] += delta_errors
                timestep_error_sum["integrated"] += integrated_errors
                timestep_error_sum["oracle"] += oracle_errors
                total += 1

    timestep = {key: value / max(1, total) for key, value in timestep_error_sum.items()}
    return pred_abs_by_id, true_abs_by_id, rows, timestep


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


def compute_geometry_diagnostics(pred: np.ndarray, true: np.ndarray, speed_eps: float) -> dict[str, float]:
    pred_heading, pred_valid_heading = compute_heading_np(pred, speed_eps)
    true_heading, true_valid_heading = compute_heading_np(true, speed_eps)
    valid_heading = pred_valid_heading & true_valid_heading
    heading_diff = wrap_angle_np(pred_heading - true_heading)
    pred_turn = wrap_angle_np(pred_heading[1:] - pred_heading[:-1])
    true_turn = wrap_angle_np(true_heading[1:] - true_heading[:-1])
    pred_valid_turn = pred_valid_heading[1:] & pred_valid_heading[:-1]
    true_valid_turn = true_valid_heading[1:] & true_valid_heading[:-1]
    valid_turn = pred_valid_turn & true_valid_turn
    turn_diff = wrap_angle_np(pred_turn - true_turn)
    mean_abs_turn_gt_deg = mean_abs_degrees(true_turn, true_valid_turn)
    mean_abs_turn_pred_deg = mean_abs_degrees(pred_turn, pred_valid_turn)
    return {
        "heading_error_deg": mean_abs_degrees(heading_diff, valid_heading),
        "turn_error_deg": mean_abs_degrees(turn_diff, valid_turn),
        "mean_abs_turn_gt_deg": mean_abs_turn_gt_deg,
        "mean_abs_turn_pred_deg": mean_abs_turn_pred_deg,
        "turn_sharpness_ratio": compute_turn_sharpness_ratio(mean_abs_turn_pred_deg, mean_abs_turn_gt_deg),
    }


def compute_heading_np(points: np.ndarray, speed_eps: float) -> tuple[np.ndarray, np.ndarray]:
    delta = points[1:] - points[:-1]
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    speed = np.linalg.norm(delta, axis=1)
    return heading, speed > speed_eps


def wrap_angle_np(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def mean_abs_degrees(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return float("nan")
    return float(np.degrees(np.mean(np.abs(values[mask]))))


def compute_turn_sharpness_ratio(pred_turn_deg: float, gt_turn_deg: float) -> float:
    if not np.isfinite(gt_turn_deg) or abs(gt_turn_deg) < 1e-12:
        return float("nan")
    if not np.isfinite(pred_turn_deg):
        return float("nan")
    return float(pred_turn_deg / gt_turn_deg)


def trajectory_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def summarize_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in DELTA_METRICS + INTEGRATED_METRICS + GEOMETRY_METRICS:
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


def summarize_oracle_vs_learned_start(per_sample: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        learned_ade = frame["integrated_ADE"].to_numpy(dtype=float)
        oracle_ade = frame["oracle_integrated_ADE"].to_numpy(dtype=float)
        learned_fde = frame["integrated_FDE"].to_numpy(dtype=float)
        oracle_fde = frame["oracle_integrated_FDE"].to_numpy(dtype=float)
        summary[split] = {
            "learned_start_integrated_ADE_mean": float(np.nanmean(learned_ade)),
            "oracle_start_integrated_ADE_mean": float(np.nanmean(oracle_ade)),
            "learned_minus_oracle_ADE_mean": float(np.nanmean(learned_ade - oracle_ade)),
            "oracle_better_ADE_sample_count": int(np.sum(oracle_ade < learned_ade)),
            "learned_start_integrated_FDE_mean": float(np.nanmean(learned_fde)),
            "oracle_start_integrated_FDE_mean": float(np.nanmean(oracle_fde)),
            "learned_minus_oracle_FDE_mean": float(np.nanmean(learned_fde - oracle_fde)),
            "oracle_better_FDE_sample_count": int(np.sum(oracle_fde < learned_fde)),
        }
    return summary


def summarize_error_correlations(per_sample: pd.DataFrame, plots_dir: Path) -> dict[str, float]:
    test = per_sample[per_sample["split"] == "test"].copy()
    pairs = [
        ("start_error", "integrated_ADE", "start_error_vs_ADE.png"),
        ("centroid_head_error", "integrated_ADE", "centroid_head_error_vs_ADE.png"),
        ("integrated_centroid_error", "integrated_ADE", "integrated_centroid_error_vs_ADE.png"),
        ("start_error", "integrated_FDE", "start_error_vs_FDE.png"),
        ("centroid_head_error", "integrated_FDE", "centroid_head_error_vs_FDE.png"),
        ("integrated_centroid_error", "integrated_FDE", "integrated_centroid_error_vs_FDE.png"),
    ]
    summary: dict[str, float] = {}
    for x_name, y_name, filename in pairs:
        x_values = test[x_name].to_numpy(dtype=float)
        y_values = test[y_name].to_numpy(dtype=float)
        pearson_r = pearson_corr(x_values, y_values)
        key = f"{x_name}_corr_{y_name}"
        summary[key] = pearson_r
        plot_error_scatter(
            x_values,
            y_values,
            x_name,
            y_name,
            pearson_r,
            plots_dir / filename,
        )
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
    axis.set_title(f"{x_label} vs {y_label} Pearson r={pearson_r:.4f}")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plot_hist(
        test_rows["integrated_ADE"],
        plots_dir / "hist_test_learned_vs_oracle_ADE.png",
        "Test learned-start vs oracle-start ADE",
        overlay=test_rows["oracle_integrated_ADE"],
        overlay_label="oracle start",
        values_label="learned start",
    )
    plot_per_timestep(per_timestep, plots_dir / "per_timestep_ADE_curve.png")
    plot_groups(test_rows, test_predictions_abs, test_truths_abs, plots_dir)


def plot_hist(
    values: pd.Series,
    path: Path,
    title: str,
    overlay: pd.Series | None = None,
    values_label: str = "value",
    overlay_label: str = "overlay",
) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=30, alpha=0.65, label=values_label if overlay is not None else None)
    if overlay is not None:
        axis.hist(overlay, bins=30, alpha=0.55, label=overlay_label)
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel(title)
    axis.set_ylabel("count")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_timestep(per_timestep: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4))
    test = per_timestep[per_timestep["split"] == "test"]
    axis.plot(test["timestep"], test["integrated_ade"], label="learned start")
    axis.plot(test["timestep"], test["oracle_integrated_ade"], label="oracle start")
    axis.set_xlabel("timestep")
    axis.set_ylabel("test integrated ADE")
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
                float(row.integrated_ADE),
                float(row.integrated_FDE),
                float(row.oracle_integrated_ADE),
                path,
            )
            paths.append(path)
        make_contact_sheet(paths, plots_dir / f"{group_name}_10_grid.png")


def plot_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    pred: np.ndarray,
    integrated_ade: float,
    integrated_fde: float,
    oracle_ade: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
    axis.plot(pred[:, 0], pred[:, 1], label="learned-start integrated", linewidth=1.5)
    axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", label="gt start")
    axis.scatter(pred[0, 0], pred[0, 1], s=22, marker="x", label="pred start")
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
        f"ADE={integrated_ade:.2f} FDE={integrated_fde:.2f} oracleADE={oracle_ade:.2f}"
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_chinese_report(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    metrics = summary["metrics_summary"]
    oracle = summary["oracle_vs_learned_start_summary"]
    correlations = summary["error_correlation_summary"]
    best_path = output_dir / "best_summary.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    test = metrics["test"]
    oracle_test = oracle["test"]

    report = f"""
Hybrid Multi-task Delta Decoder V3 Geometry 中文报告

实验设置：
- 数据集：{args.data_root.resolve()}
- 输入：RGB image / 255.0，shape [3,224,224]。
- 模型：scratch ResNet18 shared backbone，delta head 512 -> 1024 -> 448，start head 512 -> 256 -> 2，centroid head 512 -> 256 -> 2。
- 归一化：delta、start、centroid 都只用 train split 拟合 mean/std。
- 训练目标：loss_delta + {args.lambda_start} * loss_start + {args.lambda_centroid} * loss_centroid + {args.lambda_abs} * ADE-style loss_abs + {args.lambda_heading} * heading_loss + {args.lambda_turn} * turn_loss。
- centroid head 仅作为 auxiliary supervision，不用于直接修正 pred_abs 或轨迹坐标。
- V3 增加 geometry-aware supervision，但 GAF image 仍然是唯一模型输入；heading/turn loss 只在输出轨迹与 GT 轨迹之间计算。
- heading loss 用于保持局部运动方向；turn loss 用于惩罚把直角/急转弯圆滑化的预测。
- turn loss 使用 GT turn magnitude 加权：weight = clamp(1 + {args.turn_sharp_weight} * abs(gt_turn) / pi, max={args.turn_weight_max})。
- turn weight clipping 用于避免 strong geometry loss 过度支配 ADE/start/centroid losses。
- 更大的转弯角会得到更高 penalty，目标是减少 right-angle 或 high-curvature turns 被平滑化的问题。
- 预期效果：在保持较低 drift 的同时，让 high-curvature turns 更尖锐、更接近真实 maneuver geometry。
- 积分约定：relative=cumsum(delta)，relative[0]=0，absolute=start+relative，避免 timestep off-by-one 歧义。

训练结果：
- best epoch={best.get('best_epoch', 'NA')}, final epoch={best.get('final_epoch', 'NA')}, early stopping={best.get('early_stopping_triggered', 'NA')}。
- best validation loss={best.get('best_val_loss', 'NA')}。

测试集结果：
- delta ADE/FDE={test['delta_ADE']['mean']:.4f} / {test['delta_FDE']['mean']:.4f}
- learned-start integrated ADE/FDE={test['integrated_ADE']['mean']:.4f} / {test['integrated_FDE']['mean']:.4f}
- oracle-start integrated ADE/FDE={test['oracle_integrated_ADE']['mean']:.4f} / {test['oracle_integrated_FDE']['mean']:.4f}
- start error={test['start_error']['mean']:.4f}
- centroid head error={test['centroid_head_error']['mean']:.4f}
- integrated centroid error={test['integrated_centroid_error']['mean']:.4f}
- trajectory length error={test['trajectory_length_error']['mean']:.4f}
- trajectory length ratio={test['trajectory_length_ratio']['mean']:.4f}
- cumulative drift final={test['cumulative_drift_final']['mean']:.4f}
- heading error={test['heading_error_deg']['mean']:.4f} deg
- turn error={test['turn_error_deg']['mean']:.4f} deg
- mean abs turn GT={test['mean_abs_turn_gt_deg']['mean']:.4f} deg
- mean abs turn pred={test['mean_abs_turn_pred_deg']['mean']:.4f} deg
- turn sharpness ratio={test['turn_sharpness_ratio']['mean']:.4f}
- turn sharpness ratio < 1 表示预测转弯比 GT 更平滑，约等于 1 表示转弯 sharpness 被保留，大于 1 表示可能过尖锐或 overshoot。

Learned-start vs oracle-start：
- ADE learned - oracle mean={oracle_test['learned_minus_oracle_ADE_mean']:.4f}
- FDE learned - oracle mean={oracle_test['learned_minus_oracle_FDE_mean']:.4f}
- oracle better ADE samples={oracle_test['oracle_better_ADE_sample_count']}
- oracle better FDE samples={oracle_test['oracle_better_FDE_sample_count']}

Centroid supervision 与相关性分析：
- corr(start_error, integrated_ADE)={correlations['start_error_corr_integrated_ADE']:.4f}
- corr(centroid_head_error, integrated_ADE)={correlations['centroid_head_error_corr_integrated_ADE']:.4f}
- corr(integrated_centroid_error, integrated_ADE)={correlations['integrated_centroid_error_corr_integrated_ADE']:.4f}
- corr(start_error, integrated_FDE)={correlations['start_error_corr_integrated_FDE']:.4f}
- corr(centroid_head_error, integrated_FDE)={correlations['centroid_head_error_corr_integrated_FDE']:.4f}
- corr(integrated_centroid_error, integrated_FDE)={correlations['integrated_centroid_error_corr_integrated_FDE']:.4f}
- 如果 integrated_centroid_error 与 integrated ADE/FDE 高相关，说明主要失败模式更可能是 global localization drift / centroid shift，而不是局部 delta 形状本身完全不可解码。

输出文件：
- config.json
- label_delta_scaler.json, label_start_scaler.json, label_centroid_scaler.json
- train_log.csv
- checkpoints/best_model.pt, checkpoints/last_model.pt
- evaluation/per_sample_metrics.csv
- evaluation/per_timestep_ade.csv
- evaluation/metrics_summary.json
- evaluation/oracle_vs_learned_start_summary.json
- evaluation/error_correlation_summary.json
- evaluation/delta0_sanity_summary.json
- evaluation/predictions_delta/
- evaluation/predictions_start/
- evaluation/predictions_absolute_integrated/
- evaluation/plots/
"""
    path = output_dir / "hybrid_multitask_delta_decoder_report_zh.txt"
    path.write_text(report.strip() + "\n")
    print(report.strip())


if __name__ == "__main__":
    main()

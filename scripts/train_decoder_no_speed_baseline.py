"""Train the original-style CNN-FC decoder on no-speed trajectory images."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
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


@dataclass(frozen=True)
class LabelScaler:
    mean: list[float]
    std: list[float]

    def transform(self, labels: np.ndarray) -> np.ndarray:
        return (labels - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.std, dtype=np.float32
        )

    def inverse_transform_tensor(self, labels: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=labels.dtype, device=labels.device).view(1, 1, 2)
        std = torch.tensor(self.std, dtype=labels.dtype, device=labels.device).view(1, 1, 2)
        return labels * std + mean


class NoSpeedTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        metadata: pd.DataFrame,
        scaler: LabelScaler,
    ) -> None:
        self.data_root = data_root
        self.metadata = metadata.reset_index(drop=True)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        label = np.load(self.data_root / row.label_path).astype(np.float32)
        label = self.scaler.transform(label)
        return {
            "image": image_tensor,
            "label": torch.from_numpy(label),
            "sample_id": row.sample_id,
            "vehicle_id": row.vehicle_id,
        }


class OriginalStyleCNNFCDecoder(nn.Module):
    """Notebook-style CNN-FC decoder: RGB image -> 224x2 trajectory."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 1024, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 448),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        values = self.regressor(features)
        return values.view(-1, 224, 2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    split_metadata = pd.read_csv(data_root / "splits" / "split_metadata.csv")
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    if train_metadata.empty or val_metadata.empty:
        raise ValueError("Both train and val splits are required for training.")

    scaler = fit_label_scaler(data_root, train_metadata)
    write_json(output_dir / "label_scaler.json", asdict(scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")

    config = make_jsonable_config(args)
    config["device_resolved"] = str(device)
    config["model"] = "OriginalStyleCNNFCDecoder"
    config["label_normalization"] = "train_mean_std"
    config["train_samples"] = len(train_metadata)
    config["val_samples"] = len(val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = DataLoader(
        NoSpeedTrajectoryDataset(data_root, train_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        NoSpeedTrajectoryDataset(data_root, val_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = OriginalStyleCNNFCDecoder(dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, args)
    criterion = nn.MSELoss()

    log_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        compute_metric_diag = epoch == 1 or epoch % args.metrics_every == 0
        train_diag = evaluate_loader(
            model, train_loader, criterion, scaler, device, compute_metrics=compute_metric_diag
        )
        val_diag = evaluate_loader(
            model, val_loader, criterion, scaler, device, compute_metrics=compute_metric_diag
        )
        val_loss = val_diag["loss"]
        elapsed = time.perf_counter() - start_time
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_diag["loss"],
            "val_loss": val_loss,
            "train_ADE": train_diag["ade"],
            "train_FDE": train_diag["fde"],
            "val_ADE": val_diag["ade"],
            "val_FDE": val_diag["fde"],
            "learning_rate": lr,
            "elapsed_time": elapsed,
        }
        log_rows.append(row)
        write_train_log(output_dir / "train_log.csv", log_rows)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "config": config,
            "label_scaler": asdict(scaler),
        }
        torch.save(checkpoint, checkpoints_dir / "last_model.pt")
        improved = val_loss < (best_val_loss - args.early_stopping_min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoints_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        plot_training_curves(log_rows, plots_dir)
        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_ADE={format_metric(train_diag['ade'])} "
            f"val_ADE={format_metric(val_diag['ade'])} "
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
    print(f"Training complete. Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Outputs saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_baseline"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def make_jsonable_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


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


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def fit_label_scaler(data_root: Path, metadata: pd.DataFrame) -> LabelScaler:
    labels = [np.load(data_root / row.label_path).astype(np.float32) for row in metadata.itertuples()]
    stacked = np.concatenate(labels, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return LabelScaler(mean=mean.tolist(), std=std.tolist())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            predictions = model(images)
            loss = criterion(predictions, labels)

            if is_train:
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
    criterion: nn.Module,
    scaler: LabelScaler,
    device: torch.device,
    compute_metrics: bool = True,
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
            predictions_norm = model(images)
            loss = criterion(predictions_norm, labels_norm)

            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            if compute_metrics:
                predictions = scaler.inverse_transform_tensor(predictions_norm)
                labels = scaler.inverse_transform_tensor(labels_norm)
                point_errors = torch.linalg.norm(predictions - labels, dim=2)
                batch_ade = point_errors.mean(dim=1)
                batch_fde = point_errors[:, -1]
                ade_sum += float(batch_ade.sum().detach().cpu())
                fde_sum += float(batch_fde.sum().detach().cpu())
            total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "ade": ade_sum / max(1, total_samples) if compute_metrics else float("nan"),
        "fde": fde_sum / max(1, total_samples) if compute_metrics else float("nan"),
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2))


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


def plot_training_curves(rows: list[dict[str, object]], plots_dir: Path) -> None:
    plot_curve(rows, "train_loss", "val_loss", "MSE loss", plots_dir / "loss_curves.png")
    plot_curve(rows, "train_ADE", "val_ADE", "ADE", plots_dir / "ade_curves.png")
    plot_curve(rows, "train_FDE", "val_FDE", "FDE", plots_dir / "fde_curves.png")


def plot_curve(
    rows: list[dict[str, object]],
    train_key: str,
    val_key: str,
    ylabel: str,
    path: Path,
) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    train_values = np.asarray([float(row[train_key]) for row in rows], dtype=float)
    val_values = np.asarray([float(row[val_key]) for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, train_values, label="train")
    axis.plot(epochs, val_values, label="val")
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()

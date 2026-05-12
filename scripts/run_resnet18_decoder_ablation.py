"""Run a scratch ResNet18 no-speed decoder ablation.

This is a controlled sibling of the original-config decoder experiment:
same paired dataset, same fixed split, absolute x/y labels, train-only
mean/std label normalization, and original-coordinate evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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

from run_original_config_decoder_experiment import (  # noqa: E402
    compute_metrics,
    compute_naive_mean_baseline,
    compute_variance_diagnostics,
    make_contact_sheet,
    plot_evaluation_outputs,
    plot_prediction,
    summarize_metrics,
    verify_data_and_split,
)


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
        label = np.load(self.data_root / row.label_path).astype(np.float32)
        return {
            "image": image_tensor.contiguous(),
            "label": torch.from_numpy(self.scaler.transform(label)),
            "label_raw": torch.from_numpy(label),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id),
        }


class ResNet18TrajectoryDecoder(nn.Module):
    """Scratch ResNet18 image encoder with an MLP trajectory regression head."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        if backbone.conv1.in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                3,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False,
            )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 448),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.regressor(self.backbone(images))
        return values.view(-1, 224, 2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    _, split_metadata = verify_data_and_split(data_root)
    if not args.skip_training:
        train(args, split_metadata, device)

    summary = evaluate(args, split_metadata, device)
    compare_with_original(args, summary)
    write_chinese_report(args, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_resnet18_ablation"),
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_original_config"),
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
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    scaler = fit_label_scaler(data_root, train_metadata)

    write_json(output_dir / "label_scaler.json", asdict(scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
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
        train_diag = evaluate_loader(model, train_loader, criterion, scaler, device, compute_diag)
        val_diag = evaluate_loader(model, val_loader, criterion, scaler, device, compute_diag)
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


def evaluate(
    args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device
) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    predictions_dir = evaluation_dir / "predictions"
    plots_dir = evaluation_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    scaler_data = json.loads((output_dir / "label_scaler.json").read_text())
    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    model = ResNet18TrajectoryDecoder(dropout=args.dropout).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    per_timestep_rows: list[dict[str, object]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    truths: dict[str, dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = DataLoader(
            NoSpeedTrajectoryDataset(data_root, split_frame, scaler),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        split_preds, split_truths, split_rows, timestep_ade = predict_split(
            model, loader, scaler, device, predictions_dir, split
        )
        predictions[split] = split_preds
        truths[split] = split_truths
        rows.extend(split_rows)
        per_timestep_rows.extend(
            {"split": split, "timestep": index, "ade": float(value)}
            for index, value in enumerate(timestep_ade)
        )

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.DataFrame(per_timestep_rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_metrics(per_sample)
    naive_summary = compute_naive_mean_baseline(data_root, split_metadata)
    variance_summary = compute_variance_diagnostics(predictions, truths)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    write_json(evaluation_dir / "naive_mean_baseline_metrics.json", naive_summary)
    write_json(evaluation_dir / "prediction_variance_diagnostics.json", variance_summary)
    plot_evaluation_outputs(per_sample, per_timestep, predictions["test"], truths["test"], plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "naive_summary": naive_summary,
        "variance_summary": variance_summary,
    }


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    scaler: LabelScaler,
    device: torch.device,
    predictions_dir: Path,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, object]], np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    timestep_error_sum = np.zeros(224, dtype=np.float64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            pred_norm = model(images)
            pred = scaler.inverse_transform_tensor(pred_norm).cpu().numpy().astype(np.float32)
            true = batch["label_raw"].cpu().numpy().astype(np.float32)
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred = pred[index]
                sample_true = true[index]
                np.save(predictions_dir / f"{sample_id}_pred.npy", sample_pred)
                metrics, point_errors = compute_metrics(sample_pred, sample_true)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **metrics,
                    }
                )
                predictions[sample_id] = sample_pred
                truths[sample_id] = sample_true
                timestep_error_sum += point_errors
                total += 1
    return predictions, truths, rows, timestep_error_sum / max(1, total)


def compare_with_original(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    original_dir = args.original_dir.resolve()
    comparison_dir = output_dir / "comparison_with_original"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    resnet_metrics = summary["metrics_summary"]
    original_metrics = json.loads((original_dir / "evaluation" / "metrics_summary.json").read_text())
    rows = []
    for split in ["train", "val", "test"]:
        for metric in ["ADE", "FDE", "RMSE", "MAE", "start_error", "end_error", "trajectory_length_error"]:
            original_mean = original_metrics[split][metric]["mean"]
            resnet_mean = resnet_metrics[split][metric]["mean"]
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "original_mean": original_mean,
                    "resnet18_mean": resnet_mean,
                    "absolute_delta_resnet_minus_original": resnet_mean - original_mean,
                    "percent_change_vs_original": (resnet_mean - original_mean) / original_mean * 100.0
                    if original_mean
                    else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(comparison_dir / "comparison_table.csv", index=False)

    original_samples = pd.read_csv(original_dir / "evaluation" / "per_sample_metrics.csv")
    resnet_samples = pd.read_csv(output_dir / "evaluation" / "per_sample_metrics.csv")
    comparison_summary = {
        "original_dir": str(original_dir),
        "resnet18_dir": str(output_dir),
        "mean_metric_table": table.to_dict(orient="records"),
        "test": make_test_comparison(original_samples, resnet_samples),
    }
    write_json(comparison_dir / "comparison_summary.json", comparison_summary)
    plot_distribution_comparisons(original_samples, resnet_samples, comparison_dir)
    plot_side_by_side_examples(original_dir, output_dir, comparison_dir, original_samples, resnet_samples)


def make_test_comparison(original_samples: pd.DataFrame, resnet_samples: pd.DataFrame) -> dict[str, object]:
    original_test = original_samples[original_samples["split"] == "test"].copy()
    resnet_test = resnet_samples[resnet_samples["split"] == "test"].copy()
    merged = original_test.merge(
        resnet_test,
        on=["sample_id", "vehicle_id", "split"],
        suffixes=("_original", "_resnet18"),
    )
    result: dict[str, object] = {}
    for metric in ["ADE", "FDE"]:
        before = merged[f"{metric}_original"].to_numpy(dtype=float)
        after = merged[f"{metric}_resnet18"].to_numpy(dtype=float)
        result[metric] = {
            "original_mean": float(before.mean()),
            "resnet18_mean": float(after.mean()),
            "absolute_delta_resnet_minus_original": float(after.mean() - before.mean()),
            "percent_change_vs_original": float((after.mean() - before.mean()) / before.mean() * 100.0),
            "samples_improved_count": int((after < before).sum()),
            "samples_worse_count": int((after > before).sum()),
            "original_p95": float(np.percentile(before, 95)),
            "resnet18_p95": float(np.percentile(after, 95)),
            "original_max": float(before.max()),
            "resnet18_max": float(after.max()),
        }
    return result


def plot_distribution_comparisons(
    original_samples: pd.DataFrame, resnet_samples: pd.DataFrame, comparison_dir: Path
) -> None:
    original_test = original_samples[original_samples["split"] == "test"]
    resnet_test = resnet_samples[resnet_samples["split"] == "test"]
    for metric in ["ADE", "FDE"]:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(original_test[metric], bins=30, alpha=0.55, label="Original CNN-FC")
        axis.hist(resnet_test[metric], bins=30, alpha=0.55, label="ResNet18")
        axis.set_xlabel(f"Test {metric}")
        axis.set_ylabel("count")
        axis.legend()
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(comparison_dir / f"test_{metric}_distribution_comparison.png", dpi=150)
        plt.close(fig)


def plot_side_by_side_examples(
    original_dir: Path,
    output_dir: Path,
    comparison_dir: Path,
    original_samples: pd.DataFrame,
    resnet_samples: pd.DataFrame,
) -> None:
    test_rows = resnet_samples[resnet_samples["split"] == "test"].sort_values("ADE").reset_index(drop=True)
    groups = {
        "best": test_rows.head(3),
        "median": test_rows.iloc[
            max(0, len(test_rows) // 2 - 1) : min(len(test_rows), len(test_rows) // 2 + 2)
        ],
        "worst": test_rows.tail(3).sort_values("ADE", ascending=False),
    }
    original_by_id = original_samples.set_index("sample_id")
    paths = []
    for group, frame in groups.items():
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            true_path = args_path_data_label(output_dir, sample_id)
            if true_path is None:
                continue
            true = np.load(true_path).astype(np.float32)
            original_pred = np.load(original_dir / "evaluation" / "predictions" / f"{sample_id}_pred.npy")
            resnet_pred = np.load(output_dir / "evaluation" / "predictions" / f"{sample_id}_pred.npy")
            original_ade = float(original_by_id.loc[sample_id, "ADE"])
            original_fde = float(original_by_id.loc[sample_id, "FDE"])
            path = comparison_dir / f"{group}_{sample_id}_side_by_side.png"
            plot_side_by_side(
                sample_id,
                str(row.vehicle_id),
                true,
                original_pred,
                resnet_pred,
                original_ade,
                original_fde,
                float(row.ADE),
                float(row.FDE),
                path,
            )
            paths.append(path)
    make_contact_sheet(paths, comparison_dir / "best_median_worst_side_by_side_grid.png")


def args_path_data_label(output_dir: Path, sample_id: str) -> Path | None:
    split_copy = pd.read_csv(output_dir / "split_copy.csv")
    row = split_copy[split_copy["sample_id"] == sample_id]
    if row.empty:
        return None
    return PROJECT_ROOT / "data_no_speed_paired" / str(row.iloc[0]["label_path"])


def plot_side_by_side(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    original_pred: np.ndarray,
    resnet_pred: np.ndarray,
    original_ade: float,
    original_fde: float,
    resnet_ade: float,
    resnet_fde: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    panels = [
        ("Original CNN-FC", original_pred, original_ade, original_fde),
        ("ResNet18", resnet_pred, resnet_ade, resnet_fde),
    ]
    all_xy = np.vstack([true, original_pred, resnet_pred])
    pad_x = max(1.0, float(np.ptp(all_xy[:, 0])) * 0.05)
    pad_y = max(1.0, float(np.ptp(all_xy[:, 1])) * 0.05)
    for axis, (title, pred, ade, fde) in zip(axes, panels):
        axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
        axis.plot(pred[:, 0], pred[:, 1], label="prediction", linewidth=1.5)
        axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", label="gt start")
        axis.scatter(true[-1, 0], true[-1, 1], s=22, marker="s", label="gt end")
        axis.scatter(pred[0, 0], pred[0, 1], s=26, marker="x", label="pred start")
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
    comparison = json.loads(
        (output_dir / "comparison_with_original" / "comparison_summary.json").read_text()
    )
    metrics = summary["metrics_summary"]
    best = json.loads((output_dir / "best_summary.json").read_text())
    variance = summary["variance_summary"]
    test_ade = comparison["test"]["ADE"]
    test_fde = comparison["test"]["FDE"]
    train_gap = metrics["test"]["ADE"]["mean"] - metrics["train"]["ADE"]["mean"]
    original_table = pd.read_csv(output_dir / "comparison_with_original" / "comparison_table.csv")
    original_train_ade = float(
        original_table.query("split == 'train' and metric == 'ADE'")["original_mean"].iloc[0]
    )
    original_test_ade = float(
        original_table.query("split == 'test' and metric == 'ADE'")["original_mean"].iloc[0]
    )
    original_gap = original_test_ade - original_train_ade
    ade_better = test_ade["resnet18_mean"] < test_ade["original_mean"]
    fde_better = test_fde["resnet18_mean"] < test_fde["original_mean"]
    significant = (
        test_ade["percent_change_vs_original"] <= -5.0
        and test_fde["percent_change_vs_original"] <= -5.0
    )
    worst_reduced = (
        test_ade["resnet18_p95"] < test_ade["original_p95"]
        and test_ade["resnet18_max"] < test_ade["original_max"]
    )

    report = f"""
ResNet18 stronger-decoder ablation 中文报告

实验设置：
- 数据集：data_no_speed_paired/，未重新生成。
- Split：data_no_speed_paired/splits/split_metadata.csv，未改变。
- 输入：no-speed 三通道图像；未使用 speed、map、diffusion、GAN。
- 标签：与 original decoder 相同的 absolute x/y，使用 train-only mean/std normalization，评估时反归一化回原始 x/y 坐标。
- 模型：scratch ResNet18 backbone + MLP head 512 -> 1024 -> 448，dropout=0.3，Adam lr=1e-4，batch_size=8，patience=80。

1. ResNet18 是否改善 test ADE/FDE？
   Original test mean ADE={test_ade['original_mean']:.4f}, FDE={test_fde['original_mean']:.4f}
   ResNet18 test mean ADE={test_ade['resnet18_mean']:.4f}, FDE={test_fde['resnet18_mean']:.4f}
   ADE change={test_ade['percent_change_vs_original']:.2f}%, FDE change={test_fde['percent_change_vs_original']:.2f}%。
   结论：{'ResNet18 同时改善了 test ADE 和 FDE。' if ade_better and fde_better else 'ResNet18 没有同时改善 test ADE 和 FDE。'}

2. 是否减少 worst-case failures？
   Original test ADE p95={test_ade['original_p95']:.4f}, max={test_ade['original_max']:.4f}
   ResNet18 test ADE p95={test_ade['resnet18_p95']:.4f}, max={test_ade['resnet18_max']:.4f}
   结论：{'worst-case 有下降。' if worst_reduced else 'worst-case 没有稳定下降，极端失败仍然存在。'}

3. 是否减少 train-test gap 或增加 overfitting？
   Original ADE train-test gap={original_gap:.4f}
   ResNet18 ADE train-test gap={train_gap:.4f}
   ResNet18 train mean ADE={metrics['train']['ADE']['mean']:.4f}, val mean ADE={metrics['val']['ADE']['mean']:.4f}, test mean ADE={metrics['test']['ADE']['mean']:.4f}
   结论：{'ResNet18 的 train-test gap 更小，泛化差距减少。' if train_gap < original_gap else 'ResNet18 的 train-test gap 更大，说明更强 decoder 可能增加了过拟合或 split 泛化压力。'}

4. Decoder bottleneck 判断：
   {'ResNet18 相对 original 有显著改善，因此 original CNN-FC decoder 很可能是瓶颈。' if significant else 'ResNet18 没有显著改善 ADE/FDE，因此 no-speed 图像表示本身可能限制了精确重建；original decoder 不是唯一瓶颈。'}

5. 其他诊断：
   best epoch={int(best['best_epoch'])}, final epoch={int(best['final_epoch'])}, early stopping={'triggered' if best['early_stopping_triggered'] else 'not triggered'}。
   ResNet18 test predicted/GT variance ratio={variance['test']['predicted_to_ground_truth_variance_ratio']:.4f}。
   测试集中 ADE 改善样本数={test_ade['samples_improved_count']}，变差样本数={test_ade['samples_worse_count']}。

输出文件：
- evaluation/metrics_summary.json, per_sample_metrics.csv, per_timestep_ade.csv, predictions/*.npy
- evaluation/plots/ best/median/worst qualitative plots
- comparison_with_original/comparison_summary.json, comparison_table.csv, distribution plots, side-by-side plots
"""
    (output_dir / "resnet18_ablation_report_zh.txt").write_text(report.strip() + "\n")
    print(report.strip())


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
            predictions_norm = model(images)
            loss = criterion(predictions_norm, labels_norm)
            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            if compute_metrics:
                predictions = scaler.inverse_transform_tensor(predictions_norm)
                labels = scaler.inverse_transform_tensor(labels_norm)
                point_errors = torch.linalg.norm(predictions - labels, dim=2)
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
            "label_normalization": "train_mean_std",
            "coordinate_space": "absolute_xy",
            "device_resolved": str(device),
            "train_samples": len(train_metadata),
            "val_samples": len(val_metadata),
        }
    )
    return config


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


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2))


def format_metric(value: float) -> str:
    return "NA" if np.isnan(value) else f"{value:.3f}"


if __name__ == "__main__":
    main()

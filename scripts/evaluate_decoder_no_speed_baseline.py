"""Evaluate the no-speed image-to-trajectory decoder on the fixed test split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LabelScaler:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = mean
        self.std = std

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
        image = image.astype(np.float32) / 255.0
        label = np.load(self.data_root / row.label_path).astype(np.float32)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "label_norm": torch.from_numpy(self.scaler.transform(label)),
            "label": torch.from_numpy(label),
            "sample_id": row.sample_id,
            "vehicle_id": row.vehicle_id,
        }


class OriginalStyleCNNFCDecoder(nn.Module):
    def __init__(self) -> None:
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
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 448),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.regressor(self.features(images))
        return values.view(-1, 224, 2)


def main() -> None:
    args = parse_args()
    from argparse import Namespace

    from run_original_config_decoder_experiment import (
        evaluate_best_checkpoint,
        verify_data_and_split,
        write_chinese_report,
    )

    _, split_metadata = verify_data_and_split(args.data_root.resolve())
    delegated_args = Namespace(
        data_root=args.data_root,
        output_dir=args.experiment_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout=args.dropout,
    )
    summary = evaluate_best_checkpoint(delegated_args, split_metadata)
    write_chinese_report(args.experiment_dir.resolve(), summary)
    return

    data_root = args.data_root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    evaluation_dir = experiment_dir / "evaluation"
    predictions_dir = evaluation_dir / "predictions"
    plots_dir = evaluation_dir / "plots"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    scaler_data = json.loads((experiment_dir / "label_scaler.json").read_text())
    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])

    split_metadata = pd.read_csv(data_root / "splits" / "split_metadata.csv")
    checkpoint_path = experiment_dir / "checkpoints" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = OriginalStyleCNNFCDecoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    predictions_by_id: dict[str, np.ndarray] = {}
    labels_by_id: dict[str, np.ndarray] = {}
    vehicle_by_id: dict[str, str] = {}

    for split in args.splits:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        if split_frame.empty:
            raise ValueError(f"No {split} samples found in split_metadata.csv.")
        dataset = NoSpeedTrajectoryDataset(data_root, split_frame, scaler)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        split_rows = evaluate_split(
            model=model,
            loader=loader,
            scaler=scaler,
            device=device,
            predictions_dir=predictions_dir,
            split=split,
            predictions_by_id=predictions_by_id,
            labels_by_id=labels_by_id,
            vehicle_by_id=vehicle_by_id,
        )
        rows.extend(split_rows)

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    summary = summarize_metrics(per_sample)
    (evaluation_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    test_rows = per_sample[per_sample["split"] == "test"].copy()
    if test_rows.empty:
        raise ValueError("Evaluation must include the test split for qualitative plots.")
    plot_qualitative(test_rows, predictions_by_id, labels_by_id, vehicle_by_id, plots_dir, count=30)
    plot_summary(test_rows, plots_dir)

    print(f"Evaluated checkpoint {checkpoint_path}")
    for split in args.splits:
        split_summary = summary[split]
        print(
            f"{split}: "
            f"mean ADE={split_summary['ade']['mean']:.4f}, "
            f"median ADE={split_summary['ade']['median']:.4f}, "
            f"mean FDE={split_summary['fde']['mean']:.4f}, "
            f"median FDE={split_summary['fde']['median']:.4f}"
        )
    print(f"Evaluation outputs saved to {evaluation_dir}")


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    scaler: LabelScaler,
    device: torch.device,
    predictions_dir: Path,
    split: str,
    predictions_by_id: dict[str, np.ndarray],
    labels_by_id: dict[str, np.ndarray],
    vehicle_by_id: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split_predictions_dir = predictions_dir / split
    split_predictions_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pred_norm = model(images)
            pred = scaler.inverse_transform_tensor(pred_norm).cpu().numpy()
            labels = batch["label"].cpu().numpy()

            for index, sample_id in enumerate(batch["sample_id"]):
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred = pred[index].astype(np.float32)
                sample_true = labels[index].astype(np.float32)
                metrics = compute_sample_metrics(sample_pred, sample_true)
                np.save(split_predictions_dir / f"{sample_id}_pred.npy", sample_pred)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **metrics,
                    }
                )
                predictions_by_id[str(sample_id)] = sample_pred
                labels_by_id[str(sample_id)] = sample_true
                vehicle_by_id[str(sample_id)] = vehicle_id
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/decoder_no_speed_baseline"),
    )
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def compute_sample_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = pred - true
    point_errors = np.linalg.norm(diff, axis=1)
    return {
        "ade": float(point_errors.mean()),
        "fde": float(point_errors[-1]),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "dtw": float(dtw_distance(pred, true)),
        "hausdorff": float(hausdorff_distance(pred, true)),
    }


def dtw_distance(pred: np.ndarray, true: np.ndarray) -> float:
    n, m = len(pred), len(true)
    previous = np.full(m + 1, np.inf, dtype=np.float64)
    current = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current[0] = np.inf
        distances = np.linalg.norm(pred[i - 1] - true, axis=1)
        for j in range(1, m + 1):
            current[j] = distances[j - 1] + min(previous[j], current[j - 1], previous[j - 1])
        previous, current = current, previous
    return float(previous[m])


def hausdorff_distance(pred: np.ndarray, true: np.ndarray) -> float:
    distances = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def summarize_metrics(per_sample: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split, frame in per_sample.groupby("split", sort=False):
        summary[split] = {}
        for metric in ["ade", "fde", "rmse", "mae", "dtw", "hausdorff"]:
            values = frame[metric].to_numpy(dtype=float)
            summary[split][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def plot_qualitative(
    per_sample: pd.DataFrame,
    predictions_by_id: dict[str, np.ndarray],
    labels_by_id: dict[str, np.ndarray],
    vehicle_by_id: dict[str, str],
    plots_dir: Path,
    count: int,
) -> None:
    selected = per_sample.sort_values("ade").iloc[
        np.linspace(0, len(per_sample) - 1, num=min(count, len(per_sample)), dtype=int)
    ]
    qualitative_dir = plots_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    for row in selected.itertuples(index=False):
        sample_id = str(row.sample_id)
        plot_prediction(
            sample_id,
            vehicle_by_id[sample_id],
            labels_by_id[sample_id],
            predictions_by_id[sample_id],
            float(row.ade),
            float(row.fde),
            qualitative_dir / f"{sample_id}_trajectory.png",
        )


def plot_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    pred: np.ndarray,
    ade: float,
    fde: float,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.5)
    axis.plot(pred[:, 0], pred[:, 1], label="prediction", linewidth=1.5)
    axis.scatter(true[0, 0], true[0, 1], s=24, marker="o", label="gt start")
    axis.scatter(true[-1, 0], true[-1, 1], s=24, marker="s", label="gt end")
    axis.scatter(pred[0, 0], pred[0, 1], s=24, marker="x", label="pred start")
    axis.scatter(pred[-1, 0], pred[-1, 1], s=24, marker="^", label="pred end")
    all_xy = np.vstack([true, pred])
    pad_x = max(1.0, 0.05 * float(np.ptp(all_xy[:, 0])))
    pad_y = max(1.0, 0.05 * float(np.ptp(all_xy[:, 1])))
    axis.set_xlim(float(all_xy[:, 0].min() - pad_x), float(all_xy[:, 0].max() + pad_x))
    axis.set_ylim(float(all_xy[:, 1].min() - pad_y), float(all_xy[:, 1].max() + pad_y))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{sample_id} {vehicle_id} ADE={ade:.2f} FDE={fde:.2f}")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(per_sample: pd.DataFrame, plots_dir: Path) -> None:
    for metric in ["ade", "fde"]:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(per_sample[metric], bins=30)
        axis.set_xlabel(metric.upper())
        axis.set_ylabel("count")
        axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"hist_{metric}.png", dpi=150)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(per_sample["ade"], per_sample["fde"], s=16, alpha=0.8)
    axis.set_xlabel("ADE")
    axis.set_ylabel("FDE")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "scatter_ade_vs_fde.png", dpi=150)
    plt.close(fig)

    for name, frame in [
        ("worst_10_by_ade", per_sample.sort_values("ade", ascending=False).head(10)),
        ("best_10_by_ade", per_sample.sort_values("ade", ascending=True).head(10)),
    ]:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(frame["sample_id"], frame["ade"])
        axis.set_ylabel("ADE")
        axis.set_title(name)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()

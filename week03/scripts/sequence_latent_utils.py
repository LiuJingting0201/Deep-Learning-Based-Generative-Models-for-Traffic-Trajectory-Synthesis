"""Data, metrics, and IO helpers for Week3 sequence-latent experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


TRAJECTORY_STEPS = 223


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_train_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    epoch: int,
    config: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
        "model_state_dict": model_state,
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    torch.save(payload, path)


def load_split_metadata(data_root: Path, seed: int = 42) -> pd.DataFrame:
    split_path = data_root / "splits" / "split_metadata.csv"
    if split_path.exists():
        return pd.read_csv(split_path)

    metadata_path = data_root / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "valid_or_skipped" in metadata.columns:
            metadata = metadata[metadata["valid_or_skipped"] == "valid"].copy()
    else:
        label_dir = data_root / "labels_delta_displacement"
        rows = []
        for label_path in sorted(label_dir.glob("*.npy")):
            sample_id = label_path.stem
            rows.append(
                {
                    "sample_id": sample_id,
                    "vehicle_id": "",
                    "image_path": f"images/{sample_id}.png",
                    "label_delta_displacement_path": f"labels_delta_displacement/{sample_id}.npy",
                }
            )
        metadata = pd.DataFrame(rows)

    if metadata.empty:
        raise ValueError(f"No samples found under {data_root}")
    rng = random.Random(seed)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    n_train = int(0.8 * len(indices))
    n_val = int(0.1 * len(indices))
    split_by_index = {}
    for position, index in enumerate(indices):
        if position < n_train:
            split_by_index[index] = "train"
        elif position < n_train + n_val:
            split_by_index[index] = "val"
        else:
            split_by_index[index] = "test"
    frame = metadata.reset_index(drop=True).copy()
    frame["split"] = [split_by_index[index] for index in range(len(frame))]
    return frame


def normalize_delta_length(delta: np.ndarray, steps: int = TRAJECTORY_STEPS) -> np.ndarray:
    """Return [steps,2]; existing [224,2] labels drop delta[0] placeholder."""

    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError(f"Expected delta [T,2], got {delta.shape}")
    if delta.shape[0] == steps:
        return delta.astype(np.float32)
    if delta.shape[0] == steps + 1:
        return delta[1:].astype(np.float32)
    if delta.shape[0] > steps:
        return delta[:steps].astype(np.float32)
    raise ValueError(f"Need at least {steps} delta steps, got {delta.shape[0]}")


class TrajectoryDeltaDataset(Dataset):
    """Load only delta-displacement labels for Stage 1."""

    def __init__(self, data_root: Path, metadata: pd.DataFrame, steps: int = TRAJECTORY_STEPS) -> None:
        self.data_root = Path(data_root)
        self.metadata = metadata.reset_index(drop=True)
        self.steps = int(steps)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        delta = np.load(self.data_root / row.label_delta_displacement_path).astype(np.float32)
        delta = normalize_delta_length(delta, self.steps)
        return {
            "delta": torch.from_numpy(delta),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id) if "vehicle_id" in row.index else "",
        }


class PairedImageDeltaDataset(TrajectoryDeltaDataset):
    """Load paired RGB GAF/MTF images and delta labels for Stage 2/evaluation."""

    MTF_CHANNEL_INDEX = 2

    def __init__(
        self,
        data_root: Path,
        metadata: pd.DataFrame,
        steps: int = TRAJECTORY_STEPS,
        image_channel_mode: str = "rgb",
    ) -> None:
        super().__init__(data_root, metadata, steps=steps)
        if image_channel_mode not in {"rgb", "mtf"}:
            raise ValueError(f"Unsupported image_channel_mode: {image_channel_mode}")
        self.image_channel_mode = image_channel_mode

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        if self.image_channel_mode == "mtf":
            # RGB channel convention is R=GASF, G=GADF, B=MTF.
            image_tensor = image_tensor[self.MTF_CHANNEL_INDEX : self.MTF_CHANNEL_INDEX + 1]
        item["image"] = image_tensor
        return item


def integrated_trajectory(delta: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(delta, dim=1)


def sequence_losses(delta_hat: torch.Tensor, delta_gt: torch.Tensor, beta: float) -> dict[str, torch.Tensor]:
    delta_loss = torch.nn.functional.mse_loss(delta_hat, delta_gt)
    integrated_loss = torch.nn.functional.mse_loss(integrated_trajectory(delta_hat), integrated_trajectory(delta_gt))
    return {
        "loss": delta_loss + beta * integrated_loss,
        "delta_loss": delta_loss,
        "integrated_loss": integrated_loss,
    }


def batch_metric_sums(
    delta_hat: torch.Tensor,
    delta_gt: torch.Tensor,
    latent_mse: torch.Tensor | None = None,
) -> dict[str, float]:
    batch_size = delta_gt.shape[0]
    delta_errors = torch.linalg.norm(delta_hat - delta_gt, dim=2)
    traj_errors = torch.linalg.norm(integrated_trajectory(delta_hat) - integrated_trajectory(delta_gt), dim=2)
    metrics = {
        "delta_mse": float(torch.nn.functional.mse_loss(delta_hat, delta_gt).detach().cpu()) * batch_size,
        "delta_ADE": float(delta_errors.mean(dim=1).sum().detach().cpu()),
        "delta_FDE": float(delta_errors[:, -1].sum().detach().cpu()),
        "integrated_ADE": float(traj_errors.mean(dim=1).sum().detach().cpu()),
        "integrated_FDE": float(traj_errors[:, -1].sum().detach().cpu()),
    }
    if latent_mse is not None:
        metrics["latent_mse"] = float(latent_mse.detach().cpu()) * batch_size
    return metrics


def average_sums(totals: dict[str, float], sample_count: int) -> dict[str, float]:
    if sample_count <= 0:
        raise ValueError("No samples were loaded in this epoch.")
    return {key: value / sample_count for key, value in totals.items()}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    summary: dict[str, dict[str, float]] = {}
    for column in frame.columns:
        if column in {"sample_id", "vehicle_id", "split"}:
            continue
        if not np.issubdtype(frame[column].dtype, np.number):
            continue
        values = frame[column].to_numpy(dtype=float)
        summary[column] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "p90": float(np.nanpercentile(values, 90)),
            "max": float(np.nanmax(values)),
        }
    return summary

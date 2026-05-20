"""Delta-displacement decoder utilities.

The delta-displacement paired dataset stores pseudo-modal RGB images:
R = GASF, G = GADF, and B = MTF. These are not natural image color
channels; they are three encodings of the same Hilbert-mapped delta sequence.
The training target is the physical, unnormalized delta displacement.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


ChannelMode = Literal["rgb", "r", "g", "b"]
EncoderType = Literal["small_gap", "small_spatial"]


class DeltaDisplacementPairedDataset(Dataset):
    """Load no-speed delta-displacement paired samples.

    Images are loaded as RGB PNG and normalized to [0, 1] with image / 255.0,
    matching the existing decoder scripts in this repository. Labels remain in
    physical coordinate units.
    """

    def __init__(
        self,
        data_root: str | Path,
        metadata: pd.DataFrame,
        channels: ChannelMode = "rgb",
    ) -> None:
        self.data_root = Path(data_root)
        self.metadata = metadata.reset_index(drop=True)
        self.channels = channels
        if channels not in {"rgb", "r", "g", "b"}:
            raise ValueError(f"Unsupported channel mode: {channels}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        image = np.asarray(Image.open(self.data_root / row.image_path).convert("RGB"))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        if self.channels != "rgb":
            channel_index = {"r": 0, "g": 1, "b": 2}[self.channels]
            image_tensor = image_tensor[channel_index : channel_index + 1]

        delta_gt = np.load(self.data_root / row.label_delta_displacement_path).astype(
            np.float32
        )
        absolute_gt = np.load(self.data_root / row.label_absolute_path).astype(np.float32)
        start_xy = np.load(self.data_root / row.start_path).astype(np.float32)
        return {
            "image": image_tensor,
            "delta_gt": torch.from_numpy(delta_gt),
            "absolute_gt": torch.from_numpy(absolute_gt),
            "start_xy": torch.from_numpy(start_xy),
            "sample_id": str(row.sample_id),
            "vehicle_id": str(row.vehicle_id) if "vehicle_id" in row.index else "",
        }


class SingleChannelEncoder(nn.Module):
    """Configurable CNN encoder for one pseudo-modal channel."""

    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.1,
        encoder_type: EncoderType = "small_spatial",
    ) -> None:
        super().__init__()
        if encoder_type not in {"small_gap", "small_spatial"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if encoder_type == "small_gap":
            pooled_size = 1
            pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            pooled_size = 4
            pool = nn.AdaptiveAvgPool2d((4, 4))
        self.projection = nn.Sequential(
            pool,
            nn.Flatten(),
            nn.Linear(256 * pooled_size * pooled_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(image))


class MultiBranchDeltaDecoder(nn.Module):
    """Three-branch decoder: GASF/GADF/MTF image -> physical delta [B, 224, 2]."""

    def __init__(
        self,
        sequence_length: int = 224,
        feature_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        channels: ChannelMode = "rgb",
        encoder_type: EncoderType = "small_spatial",
    ) -> None:
        super().__init__()
        if channels not in {"rgb", "r", "g", "b"}:
            raise ValueError(f"Unsupported channel mode: {channels}")
        if encoder_type not in {"small_gap", "small_spatial"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.sequence_length = sequence_length
        self.channels = channels
        self.encoder_type = encoder_type

        if channels == "rgb":
            self.encoder_r = SingleChannelEncoder(feature_dim, dropout, encoder_type)
            self.encoder_g = SingleChannelEncoder(feature_dim, dropout, encoder_type)
            self.encoder_b = SingleChannelEncoder(feature_dim, dropout, encoder_type)
            fusion_in = feature_dim * 3
        else:
            self.encoder = SingleChannelEncoder(feature_dim, dropout, encoder_type)
            fusion_in = feature_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, sequence_length * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.channels == "rgb":
            if image.shape[1] != 3:
                raise ValueError(f"Expected 3 pseudo-modal channels, got {image.shape[1]}")
            # R=GASF, G=GADF, B=MTF. Encoders intentionally do not share weights.
            feature_r = self.encoder_r(image[:, 0:1])
            feature_g = self.encoder_g(image[:, 1:2])
            feature_b = self.encoder_b(image[:, 2:3])
            features = torch.cat([feature_r, feature_g, feature_b], dim=1)
        else:
            if image.shape[1] != 1:
                raise ValueError(f"Expected 1 selected pseudo-modal channel, got {image.shape[1]}")
            features = self.encoder(image)
        delta = self.fusion_head(features)
        return delta.view(-1, self.sequence_length, 2)


class SimpleCNNDeltaDecoder(nn.Module):
    """Single-stream CNN baseline for ablations on the same delta target."""

    def __init__(
        self,
        in_channels: int = 3,
        sequence_length: int = 224,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        encoder_type: EncoderType = "small_spatial",
    ) -> None:
        super().__init__()
        if encoder_type not in {"small_gap", "small_spatial"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.sequence_length = sequence_length
        self.encoder_type = encoder_type
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if encoder_type == "small_gap":
            pooled_size = 1
            pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            pooled_size = 4
            pool = nn.AdaptiveAvgPool2d((4, 4))
        self.features = nn.Sequential(
            pool,
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * pooled_size * pooled_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sequence_length * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        delta = self.head(self.features(self.conv(image)))
        return delta.view(-1, self.sequence_length, 2)


def build_delta_decoder(
    model_name: str,
    channels: ChannelMode,
    sequence_length: int = 224,
    dropout: float = 0.2,
    encoder_type: EncoderType = "small_spatial",
) -> nn.Module:
    if model_name == "multi_branch_delta":
        return MultiBranchDeltaDecoder(
            sequence_length=sequence_length,
            dropout=dropout,
            channels=channels,
            encoder_type=encoder_type,
        )
    if model_name == "simple_cnn_delta":
        in_channels = 3 if channels == "rgb" else 1
        return SimpleCNNDeltaDecoder(
            in_channels=in_channels,
            sequence_length=sequence_length,
            dropout=dropout,
            encoder_type=encoder_type,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def integrate_delta_torch(start_xy: torch.Tensor, delta_xy: torch.Tensor) -> torch.Tensor:
    """Integrate deltas with delta[0] treated as a placeholder, not a motion."""

    absolute = torch.empty_like(delta_xy)
    absolute[:, 0, :] = start_xy
    absolute[:, 1:, :] = start_xy[:, None, :] + torch.cumsum(delta_xy[:, 1:, :], dim=1)
    return absolute


def integrate_delta_numpy(start_xy: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    absolute = np.empty_like(delta_xy, dtype=np.float32)
    absolute[0] = start_xy.astype(np.float32)
    absolute[1:] = start_xy.astype(np.float32) + np.cumsum(delta_xy[1:], axis=0)
    return absolute


def compute_basic_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
    diff = pred - gt
    point_errors = np.linalg.norm(diff, axis=1)
    return {
        f"{prefix}_MSE": float(np.mean(diff**2)),
        f"{prefix}_MAE": float(np.mean(np.abs(diff))),
        f"{prefix}_ADE": float(point_errors.mean()),
        f"{prefix}_FDE": float(point_errors[-1]),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(diff**2))),
    }


def compute_alignment_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    raw_errors = np.linalg.norm(pred - gt, axis=1)
    start_aligned = pred + (gt[0] - pred[0])
    start_errors = np.linalg.norm(start_aligned - gt, axis=1)

    pred_centroid = pred.mean(axis=0)
    gt_centroid = gt.mean(axis=0)
    centroid_aligned = pred + (gt_centroid - pred_centroid)
    centroid_errors = np.linalg.norm(centroid_aligned - gt, axis=1)

    pred_centered = pred - pred_centroid
    gt_centered = gt - gt_centroid
    pred_scale = float(np.sqrt(np.mean(np.sum(pred_centered**2, axis=1))))
    gt_scale = float(np.sqrt(np.mean(np.sum(gt_centered**2, axis=1))))
    if pred_scale > 0.0 and gt_scale > 0.0:
        scale_normalized = pred_centered * (gt_scale / pred_scale) + gt_centroid
    else:
        scale_normalized = centroid_aligned
    scale_errors = np.linalg.norm(scale_normalized - gt, axis=1)

    return {
        "start_aligned_ADE": float(start_errors.mean()),
        "start_aligned_FDE": float(start_errors[-1]),
        "centroid_aligned_ADE": float(centroid_errors.mean()),
        "centroid_aligned_FDE": float(centroid_errors[-1]),
        "scale_normalized_shape_ADE": float(scale_errors.mean()),
        "scale_normalized_shape_FDE": float(scale_errors[-1]),
        "raw_ADE": float(raw_errors.mean()),
        "raw_FDE": float(raw_errors[-1]),
    }


def summarize_metric_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"sample_id", "vehicle_id", "split"}
        and np.issubdtype(frame[column].dtype, np.number)
    ]
    summary: dict[str, dict[str, float]] = {}
    for column in metric_columns:
        values = frame[column].to_numpy(dtype=float)
        summary[column] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "p75": float(np.nanpercentile(values, 75)),
            "p90": float(np.nanpercentile(values, 90)),
            "p95": float(np.nanpercentile(values, 95)),
            "max": float(np.nanmax(values)),
        }
    return summary


def load_or_create_split_metadata(
    data_root: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    data_root = Path(data_root)
    if split_metadata_path is not None:
        return pd.read_csv(split_metadata_path)

    split_path = data_root / "splits" / "split_metadata.csv"
    if split_path.exists():
        return pd.read_csv(split_path)

    metadata = pd.read_csv(data_root / "metadata.csv")
    if "valid_or_skipped" in metadata.columns:
        metadata = metadata[metadata["valid_or_skipped"] == "valid"].copy()
    rng = random.Random(seed)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    split_by_index = {}
    for position, index in enumerate(indices):
        if position < n_train:
            split_by_index[index] = "train"
        elif position < n_train + n_val:
            split_by_index[index] = "val"
        else:
            split_by_index[index] = "test"
    split_metadata = metadata.reset_index(drop=True).copy()
    split_metadata["split"] = [split_by_index[i] for i in range(len(split_metadata))]
    return split_metadata


def write_split_json(path: Path, split_metadata: pd.DataFrame, source: str, seed: int) -> None:
    payload = {
        "source": source,
        "seed": seed,
        "counts": {
            split: int((split_metadata["split"] == split).sum())
            for split in ["train", "val", "test"]
        },
        "samples": split_metadata[["sample_id", "vehicle_id", "split"]].to_dict(
            orient="records"
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")

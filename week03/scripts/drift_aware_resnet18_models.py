"""Drift-aware raw RGB ResNet18 decoder for Week3 Experiment E1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


TRAJECTORY_STEPS = 224
TRAJECTORY_DIMS = 2
IMAGE_SIZE = 224


def build_resnet18_backbone(in_channels: int = 3) -> tuple[nn.Module, int]:
    """Build a scratch ResNet18 feature extractor with configurable input channels."""
    backbone = resnet18(weights=None)
    if in_channels != backbone.conv1.in_channels:
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def _validate_rgb_input(images: torch.Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected input ndim=4 [B,3,224,224], got shape {tuple(images.shape)}")
    if images.shape[1] != 3:
        raise ValueError(f"Expected 3 input channels, got {images.shape[1]}")
    if images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} input, got {tuple(images.shape[-2:])}")


class DriftAwareResNet18DeltaDecoder(nn.Module):
    """Raw RGB ResNet18 with a delta head and a low-frequency absolute correction head."""

    def __init__(self, dropout: float = 0.3, correction_anchors: int = 16) -> None:
        super().__init__()
        if correction_anchors < 2:
            raise ValueError("correction_anchors must be at least 2 for linear interpolation.")
        self.correction_anchors = int(correction_anchors)
        self.backbone, feature_dim = build_resnet18_backbone(in_channels=3)
        self.delta_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, TRAJECTORY_STEPS * TRAJECTORY_DIMS),
        )
        self.correction_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, self.correction_anchors * TRAJECTORY_DIMS),
        )
        last_layer = self.correction_head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    @property
    def conv1_in_channels(self) -> int:
        return int(self.backbone.conv1.in_channels)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_rgb_input(images)
        features = self.backbone(images)
        pred_delta_norm = self.delta_head(features).view(-1, TRAJECTORY_STEPS, TRAJECTORY_DIMS)
        corr_anchor_norm = self.correction_head(features).view(-1, self.correction_anchors, TRAJECTORY_DIMS)
        corr_anchor_norm = corr_anchor_norm.clone()
        corr_anchor_norm[:, 0, :] = 0.0
        corr_full_norm = F.interpolate(
            corr_anchor_norm.transpose(1, 2),
            size=TRAJECTORY_STEPS,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        corr_full_norm = corr_full_norm.clone()
        corr_full_norm[:, 0, :] = 0.0
        return {
            "pred_delta_norm": pred_delta_norm,
            "corr_anchor_norm": corr_anchor_norm,
            "corr_full_norm": corr_full_norm,
        }

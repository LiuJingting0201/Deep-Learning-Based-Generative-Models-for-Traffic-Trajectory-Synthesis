"""Structure-aware ResNet18 decoders for Week3 delta-displacement experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


TRAJECTORY_STEPS = 224
TRAJECTORY_DIMS = 2
IMAGE_SIZE = 224


def build_resnet18_backbone(in_channels: int) -> tuple[nn.Module, int]:
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


def _validate_image_input(x: torch.Tensor, expected_channels: int = 3) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"Expected input ndim=4 [B,{expected_channels},224,224], got shape {tuple(x.shape)}"
        )
    if x.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} input channels, got {x.shape[1]}")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError(f"Expected square input, got H={x.shape[-2]} W={x.shape[-1]}")
    if x.shape[-2] != IMAGE_SIZE or x.shape[-1] != IMAGE_SIZE:
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} input, got {x.shape[-2]}x{x.shape[-1]}")


class _ResNet18DeltaDecoderBase(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone, feature_dim = build_resnet18_backbone(in_channels)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, TRAJECTORY_STEPS * TRAJECTORY_DIMS),
        )

    @property
    def conv1_in_channels(self) -> int:
        return int(self.backbone.conv1.in_channels)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        values = self.regressor(self.backbone(x))
        pred = values.view(-1, TRAJECTORY_STEPS, TRAJECTORY_DIMS)
        if pred.shape[1:] != (TRAJECTORY_STEPS, TRAJECTORY_DIMS):
            raise RuntimeError(f"Unexpected decoder output shape: {tuple(pred.shape)}")
        return pred


class RawResNet18DeltaDecoder(_ResNet18DeltaDecoderBase):
    """Raw RGB scratch ResNet18 delta-displacement decoder baseline."""

    def __init__(self, dropout: float = 0.3, in_channels: int = 3) -> None:
        super().__init__(in_channels=in_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_image_input(x, expected_channels=self.conv1_in_channels)
        return self._decode(x)


class StructureDecomposedResNet18DeltaDecoder(_ResNet18DeltaDecoderBase):
    """Experiment C decoder using GASF/GADF theory components plus residuals."""

    def __init__(self, dropout: float = 0.3, return_debug: bool = False) -> None:
        super().__init__(in_channels=5, dropout=dropout)
        self.return_debug = return_debug

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        _validate_image_input(x)
        r = x[:, 0:1] * 2.0 - 1.0
        g = x[:, 1:2] * 2.0 - 1.0
        b = x[:, 2:3]

        r_sym = 0.5 * (r + r.transpose(-1, -2))
        r_res = r - r_sym
        g_antisym = 0.5 * (g - g.transpose(-1, -2))
        g_res = g - g_antisym
        structured = torch.cat([r_sym, r_res, g_antisym, g_res, b], dim=1)
        pred = self._decode(structured)
        if not self.return_debug:
            return pred
        debug = {
            "input_min": float(x.detach().min().cpu()),
            "input_max": float(x.detach().max().cpu()),
            "r_theory_min": float(r.detach().min().cpu()),
            "r_theory_max": float(r.detach().max().cpu()),
            "g_theory_min": float(g.detach().min().cpu()),
            "g_theory_max": float(g.detach().max().cpu()),
            "b_raw_min": float(b.detach().min().cpu()),
            "b_raw_max": float(b.detach().max().cpu()),
            "r_res_mean_abs": float(r_res.detach().abs().mean().cpu()),
            "g_res_mean_abs": float(g_res.detach().abs().mean().cpu()),
            "structured_input_shape": list(structured.shape),
            "structured_input_min": float(structured.detach().min().cpu()),
            "structured_input_max": float(structured.detach().max().cpu()),
        }
        return pred, debug


class HardStructureResNet18DeltaDecoder(_ResNet18DeltaDecoderBase):
    """Hard projection baseline using only GASF symmetric and GADF anti-symmetric parts."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__(in_channels=3, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_image_input(x)
        r = x[:, 0:1] * 2.0 - 1.0
        g = x[:, 1:2] * 2.0 - 1.0
        b = x[:, 2:3]
        r_sym = 0.5 * (r + r.transpose(-1, -2))
        g_antisym = 0.5 * (g - g.transpose(-1, -2))
        return self._decode(torch.cat([r_sym, g_antisym, b], dim=1))


class DecompMtfLocalResNet18DeltaDecoder(_ResNet18DeltaDecoderBase):
    """Structure decomposition plus a 3x3 local-average MTF channel for later Experiment D."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__(in_channels=6, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_image_input(x)
        r = x[:, 0:1] * 2.0 - 1.0
        g = x[:, 1:2] * 2.0 - 1.0
        b = x[:, 2:3]
        r_sym = 0.5 * (r + r.transpose(-1, -2))
        r_res = r - r_sym
        g_antisym = 0.5 * (g - g.transpose(-1, -2))
        g_res = g - g_antisym
        b_padded = F.pad(b, pad=(1, 1, 1, 1), mode="replicate")
        b_local = F.avg_pool2d(b_padded, kernel_size=3, stride=1, padding=0)
        return self._decode(torch.cat([r_sym, r_res, g_antisym, g_res, b, b_local], dim=1))

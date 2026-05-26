"""Sequence-latent models for Week3 trajectory-defined alignment experiments."""

from __future__ import annotations

import torch
from torch import nn


TRAJECTORY_STEPS = 223
TRAJECTORY_DIMS = 2
IMAGE_SIZE = 224


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TrajectorySequenceEncoder(nn.Module):
    """Encode delta displacement [B, T, 2] into sequence tokens [B, T, latent_dim]."""

    def __init__(
        self,
        latent_dim: int = 64,
        activation: str = "gelu",
        expected_steps: int | None = TRAJECTORY_STEPS,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.expected_steps = expected_steps
        self.net = nn.Sequential(
            nn.Conv1d(TRAJECTORY_DIMS, 64, kernel_size=5, padding=2),
            _activation(activation),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            _activation(activation),
            nn.Conv1d(128, self.latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.ndim != 3 or delta.shape[-1] != TRAJECTORY_DIMS:
            expected = self.expected_steps if self.expected_steps is not None else "T"
            raise ValueError(f"Expected delta [B,{expected},2], got {tuple(delta.shape)}")
        if self.expected_steps is not None and delta.shape[1] != self.expected_steps:
            raise ValueError(f"Expected delta [B,{self.expected_steps},2], got {tuple(delta.shape)}")
        # [B, T, 2] -> [B, 2, T] -> [B, latent_dim, T] -> [B, T, latent_dim]
        return self.net(delta.transpose(1, 2)).transpose(1, 2).contiguous()


class TrajectorySequenceDecoder(nn.Module):
    """Decode sequence tokens [B, T, latent_dim] back to delta [B, T, 2]."""

    def __init__(
        self,
        latent_dim: int = 64,
        activation: str = "gelu",
        expected_steps: int | None = TRAJECTORY_STEPS,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.expected_steps = expected_steps
        self.net = nn.Sequential(
            nn.Conv1d(self.latent_dim, 128, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            _activation(activation),
            nn.Conv1d(64, TRAJECTORY_DIMS, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            expected = self.expected_steps if self.expected_steps is not None else "T"
            raise ValueError(f"Expected latent [B,{expected},{self.latent_dim}], got {tuple(z.shape)}")
        if self.expected_steps is not None and z.shape[1] != self.expected_steps:
            raise ValueError(f"Expected latent [B,{self.expected_steps},{self.latent_dim}], got {tuple(z.shape)}")
        if z.shape[-1] != self.latent_dim:
            expected = self.expected_steps if self.expected_steps is not None else "T"
            raise ValueError(f"Expected latent [B,{expected},{self.latent_dim}], got {tuple(z.shape)}")
        # [B, T, D] -> [B, D, T] -> [B, 2, T] -> [B, T, 2]
        return self.net(z.transpose(1, 2)).transpose(1, 2).contiguous()


class TrajectorySequenceAutoencoder(nn.Module):
    """1D CNN trajectory autoencoder preserving timestep-level latent tokens."""

    def __init__(
        self,
        latent_dim: int = 64,
        activation: str = "gelu",
        expected_steps: int | None = TRAJECTORY_STEPS,
    ) -> None:
        super().__init__()
        self.expected_steps = expected_steps
        self.encoder = TrajectorySequenceEncoder(
            latent_dim=latent_dim,
            activation=activation,
            expected_steps=expected_steps,
        )
        self.decoder = TrajectorySequenceDecoder(
            latent_dim=latent_dim,
            activation=activation,
            expected_steps=expected_steps,
        )

    def forward(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(delta)
        delta_hat = self.decoder(z)
        return delta_hat, z


class RowWiseGAFImageEncoder(nn.Module):
    """Map each GAF/MTF image row to one timestep-level latent token."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        activation: str = "gelu",
        image_size: int = IMAGE_SIZE,
        input_channels: int = 3,
        output_steps: int = TRAJECTORY_STEPS,
        use_temporal_refine: bool = False,
        temporal_kernel_size: int = 5,
        temporal_layers: int = 2,
        use_transformer_refine: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        self.output_steps = int(output_steps)
        self.use_temporal_refine = bool(use_temporal_refine)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.temporal_layers = int(temporal_layers)
        self.use_transformer_refine = bool(use_transformer_refine)
        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ff_dim = int(transformer_ff_dim)
        self.transformer_dropout = float(transformer_dropout)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.temporal_kernel_size <= 0 or self.temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer.")
        if self.temporal_layers < 0:
            raise ValueError("temporal_layers must be non-negative.")
        if self.transformer_layers < 0:
            raise ValueError("transformer_layers must be non-negative.")
        if self.transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive.")
        if self.transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive.")
        if self.use_transformer_refine and self.latent_dim % self.transformer_heads != 0:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by "
                f"transformer_heads={self.transformer_heads}."
            )
        self.row_feature_dim = self.input_channels * self.image_size
        self.row_mlp = nn.Sequential(
            nn.Linear(self.row_feature_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        temporal_blocks: list[nn.Module] = []
        if self.use_temporal_refine:
            for _ in range(self.temporal_layers):
                temporal_blocks.extend(
                    [
                        nn.Conv1d(
                            self.latent_dim,
                            self.latent_dim,
                            kernel_size=self.temporal_kernel_size,
                            padding=self.temporal_kernel_size // 2,
                        ),
                        _activation(activation),
                    ]
                )
        self.temporal_refine = nn.Sequential(*temporal_blocks)
        if self.use_transformer_refine and self.transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.transformer_heads,
                dim_feedforward=self.transformer_ff_dim,
                dropout=self.transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_refine = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.transformer_layers,
            )
        else:
            self.transformer_refine = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError(f"Expected images [B,{self.input_channels},H,W], got {tuple(images.shape)}")
        if images.shape[2] < self.output_steps:
            raise ValueError(f"Need at least {self.output_steps} rows, got {images.shape[2]}")
        if images.shape[3] != self.image_size:
            raise ValueError(f"Expected image width {self.image_size}, got {images.shape[3]}")
        # [B, C, 224, 224] -> [B, 224, C, 224] -> [B, 224, C*224]
        rows = images.permute(0, 2, 1, 3).contiguous().view(images.shape[0], images.shape[2], -1)
        z = self.row_mlp(rows[:, : self.output_steps, :])
        if self.use_temporal_refine and self.temporal_layers > 0:
            residual = z
            z_t = z.transpose(1, 2)
            z = residual + self.temporal_refine(z_t).transpose(1, 2).contiguous()
        if self.use_transformer_refine and self.transformer_layers > 0:
            residual = z
            z = residual + self.transformer_refine(z)
        return z


class ImageToSequenceLatentDecoder(nn.Module):
    """Convenience wrapper: image -> sequence latent -> delta."""

    def __init__(
        self,
        image_encoder: RowWiseGAFImageEncoder,
        trajectory_decoder: TrajectorySequenceDecoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.trajectory_decoder = trajectory_decoder

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_img = self.image_encoder(images)
        delta_hat = self.trajectory_decoder(z_img)
        return delta_hat, z_img

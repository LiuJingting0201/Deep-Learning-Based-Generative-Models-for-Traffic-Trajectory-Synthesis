"""Lightweight convolutional autoencoder for GAF trajectory images."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class GAFEncoder(nn.Module):
    """Encode RGB GAF images into compact spatial latents."""

    def __init__(self, latent_channels: int = 16, downsample_blocks: int = 4) -> None:
        super().__init__()
        encoder_channels = (32, 64, 128, 256)[:downsample_blocks]
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in encoder_channels:
            layers.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAFDecoder(nn.Module):
    """Decode spatial latents back to RGB GAF images in [-1, 1]."""

    def __init__(self, latent_channels: int = 16, downsample_blocks: int = 4) -> None:
        super().__init__()
        encoder_channels = (32, 64, 128, 256)[:downsample_blocks]
        decoder_channels = tuple(reversed(encoder_channels[:-1]))
        layers: list[nn.Module] = []
        in_channels = latent_channels
        for out_channels in decoder_channels:
            layers.append(deconv_block(in_channels, out_channels))
            in_channels = out_channels
        layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GAFAutoencoder(nn.Module):
    """Autoencoder used as the V1 latent diffusion first stage."""

    def __init__(
        self,
        latent_channels: int = 16,
        latent_spatial_size: int = 14,
        downsample_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.downsample_blocks = downsample_blocks
        self.encoder = GAFEncoder(latent_channels=latent_channels, downsample_blocks=downsample_blocks)
        self.decoder = GAFDecoder(latent_channels=latent_channels, downsample_blocks=downsample_blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def save_pretrained(self, path: str | Path) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "model.pt")
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    "latent_channels": self.latent_channels,
                    "latent_spatial_size": self.latent_spatial_size,
                    "downsample_blocks": self.downsample_blocks,
                },
                indent=2,
            )
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "GAFAutoencoder":
        checkpoint_dir = Path(path)
        config_path = checkpoint_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        model = cls(
            latent_channels=int(config.get("latent_channels", 16)),
            latent_spatial_size=int(config.get("latent_spatial_size", 14)),
            downsample_blocks=int(config.get("downsample_blocks", 4)),
        )
        state = torch.load(checkpoint_dir / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


def deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )

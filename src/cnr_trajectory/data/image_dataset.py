"""Minimal image dataset utilities for diffusion smoke tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DiffusionImageDataset(Dataset):
    """Load PNG images as tensors normalized to the diffusion range [-1, 1]."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int,
        max_samples: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        self.paths = sorted(self.image_dir.glob("*.png"))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise ValueError(f"No PNG images found in {self.image_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)

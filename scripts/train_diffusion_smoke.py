"""Run a tiny real DDPM training smoke test on encoded images.

This script intentionally performs only a few optimizer steps and saves no
model checkpoint. It verifies that the encoded image dataset can feed a
diffusion model training loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DiffusionImageDataset(
        args.input_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False)

    model = build_tiny_model(args.image_size)
    scheduler = DDPMScheduler(num_train_timesteps=10)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    batches = cycle(dataloader)
    model.train()
    for _ in range(args.train_steps):
        clean_images = next(batches)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            dtype=torch.long,
        )
        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))

    artifact = {
        "status": "ok",
        "num_images": len(dataset),
        "image_size": args.image_size,
        "train_steps": args.train_steps,
        "losses": losses,
    }
    artifact_path = output_dir / "smoke_log.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))
    print(
        f"Diffusion smoke training completed: steps={args.train_steps}, "
        f"images={len(dataset)}, artifact={artifact_path}"
    )


def build_tiny_model(image_size: int) -> UNet2DModel:
    """Build a very small UNet for smoke testing, not thesis-scale training."""
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of PNG images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Smoke output folder.")
    parser.add_argument("--image-size", type=int, default=32, help="Resize images to this size.")
    parser.add_argument("--max-samples", type=int, default=2, help="Maximum images to load.")
    parser.add_argument("--train-steps", type=int, default=1, help="Optimizer steps to run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Tiny smoke batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Smoke learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

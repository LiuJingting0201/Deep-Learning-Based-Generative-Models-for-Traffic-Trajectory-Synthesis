"""Train a lightweight GAF image autoencoder."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from cnr_trajectory.models.gaf_autoencoder import GAFAutoencoder


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    checkpoint_final = output_dir / "checkpoint-final"
    loss_curve_path = output_dir / "loss_curve.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    args.latent_spatial_size = resolve_latent_spatial_size(args)
    downsample_blocks = resolve_downsample_blocks(args.image_size, args.latent_spatial_size)
    model = GAFAutoencoder(
        latent_channels=args.latent_channels,
        latent_spatial_size=args.latent_spatial_size,
        downsample_blocks=downsample_blocks,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_training_state(args.resume, model, optimizer, device)

    if not loss_curve_path.exists() or start_epoch == 0:
        with loss_curve_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "step", "loss", "l1_loss", "mse_loss"])
            writer.writeheader()

    global_step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for images in dataloader:
            images = images.to(device)
            recon = model(images)
            l1_loss = F.l1_loss(recon, images)
            mse_loss = F.mse_loss(recon, images)
            loss = l1_loss + mse_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += float(loss.detach().cpu())

        avg_loss = epoch_loss / max(1, len(dataloader))
        with loss_curve_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "step", "loss", "l1_loss", "mse_loss"])
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": avg_loss,
                    "l1_loss": float(l1_loss.detach().cpu()),
                    "mse_loss": float(mse_loss.detach().cpu()),
                }
            )
        print(f"epoch={epoch + 1} step={global_step} loss={avg_loss:.6f}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_training_state(output_dir / f"checkpoint-{epoch + 1}", model, optimizer, epoch + 1, args)
            save_reconstruction_samples(model, dataset, samples_dir, epoch + 1, device, args)

    save_training_state(checkpoint_final, model, optimizer, args.epochs, args)
    save_reconstruction_samples(model, dataset, samples_dir, args.epochs, device, args)
    metrics = compute_reconstruction_metrics(model, dataset, device, args)
    (output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "num_images": len(dataset),
                "image_size": args.image_size,
                "latent_channels": args.latent_channels,
                "latent_spatial_size": args.latent_spatial_size,
                "latent_shape_chw": [args.latent_channels, args.latent_spatial_size, args.latent_spatial_size],
                "downsample_blocks": downsample_blocks,
                "reconstruction_metrics": metrics,
            },
            indent=2,
        )
    )


def save_reconstruction_samples(
    model: GAFAutoencoder,
    dataset: DiffusionImageDataset,
    output_dir: Path,
    epoch: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = min(args.num_sample_images, len(dataset))
    with torch.no_grad():
        images = torch.stack([dataset[index] for index in range(count)]).to(device)
        recon = model(images)
    rows = torch.stack([images.cpu(), recon.cpu()], dim=1).flatten(0, 1)
    grid = make_grid(denormalize(rows), nrow=2)
    save_image(grid, output_dir / f"recon_epoch_{epoch:04d}.png")


def compute_reconstruction_metrics(
    model: GAFAutoencoder,
    dataset: DiffusionImageDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_pixels = 0
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            recon = model(images)
            total_l1 += F.l1_loss(recon, images, reduction="sum").item()
            total_mse += F.mse_loss(recon, images, reduction="sum").item()
            total_pixels += images.numel()
    return {"mae": total_l1 / total_pixels, "mse": total_mse / total_pixels}


def save_training_state(
    path: Path,
    model: GAFAutoencoder,
    optimizer: AdamW,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "autoencoder")
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "args": serialize_args(args),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "training_state.pt",
    )


def load_training_state(
    checkpoint_dir: Path,
    model: GAFAutoencoder,
    optimizer: AdamW,
    device: torch.device,
) -> int:
    checkpoint = checkpoint_dir.resolve()
    loaded_model = GAFAutoencoder.from_pretrained(checkpoint / "autoencoder", map_location=device)
    model.load_state_dict(loaded_model.state_dict())
    state = torch.load(checkpoint / "training_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state.get("epoch", 0))


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_latent_spatial_size(args: argparse.Namespace) -> int:
    if args.latent_spatial_size is not None:
        return args.latent_spatial_size
    return args.image_size // 16


def resolve_downsample_blocks(image_size: int, latent_spatial_size: int) -> int:
    if image_size % latent_spatial_size != 0:
        raise ValueError(f"image_size={image_size} is not divisible by latent_spatial_size={latent_spatial_size}")
    factor = image_size // latent_spatial_size
    if factor < 2 or factor & (factor - 1):
        raise ValueError(
            f"image_size / latent_spatial_size must be a power of two >= 2; got {image_size}/{latent_spatial_size}"
        )
    downsample_blocks = factor.bit_length() - 1
    if downsample_blocks > 4:
        raise ValueError("The lightweight autoencoder supports at most four stride-2 downsampling blocks")
    return downsample_blocks


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing RGB PNG GAF images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and samples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--save-every", type=int, default=5, help="Save and sample every N epochs; 0 disables periodic saves.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--latent-channels", type=int, default=16, help="Latent channel count.")
    parser.add_argument(
        "--latent-spatial-size",
        type=int,
        default=None,
        help="Latent height/width. Defaults to image_size/16, matching the original 224->14 bottleneck.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--num-sample-images", type=int, default=8, help="Images to include in reconstruction grids.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

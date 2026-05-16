"""Generate GAF images by sampling latent DDPM and decoding with the autoencoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.models.gaf_autoencoder import GAFAutoencoder


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet2DModel.from_pretrained(args.latent_checkpoint / "unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.latent_checkpoint / "scheduler")
    autoencoder = GAFAutoencoder.from_pretrained(args.autoencoder_checkpoint / "autoencoder", map_location=device).to(device)
    autoencoder.eval()
    normalization_stats = resolve_normalization_stats(args)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    written = 0
    while written < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - written)
        latents = sample_latents(unet, scheduler, batch_size, device, generator, args)
        latents = inverse_normalize_latents(latents, normalization_stats)
        with torch.no_grad():
            images = autoencoder.decode(latents).detach().cpu()
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        for image in images:
            save_image(image, output_dir / f"sample_{written:06d}.png")
            written += 1

    print(f"Generated {written} decoded latent samples in {output_dir}")


def sample_latents(
    unet: UNet2DModel,
    scheduler: DDPMScheduler,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> torch.Tensor:
    unet.eval()
    latents = torch.randn(
        batch_size,
        args.latent_channels,
        args.sample_size,
        args.sample_size,
        generator=generator,
        device=device,
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            noise_pred = unet(latents, timestep, return_dict=False)[0]
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents


def resolve_normalization_stats(args: argparse.Namespace) -> dict[str, float | bool | str | None]:
    if args.latent_stats is not None:
        stats = json.loads(args.latent_stats.read_text())
        return {
            "enabled": True,
            "path": str(args.latent_stats),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
        }

    summary_path = args.latent_checkpoint.parent / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        stats = summary.get("normalization_stats")
        if stats and stats.get("enabled"):
            return {
                "enabled": True,
                "path": stats.get("path"),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
            }

    return {"enabled": False, "path": None, "mean": 0.0, "std": 1.0}


def inverse_normalize_latents(
    latents: torch.Tensor,
    normalization_stats: dict[str, float | bool | str | None],
) -> torch.Tensor:
    if not normalization_stats["enabled"]:
        return latents
    mean = float(normalization_stats["mean"])
    std = float(normalization_stats["std"])
    return latents * (std + 1e-6) + mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-checkpoint",
        type=Path,
        required=True,
        help="Latent DDPM checkpoint-final directory containing unet/ and scheduler/.",
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        required=True,
        help="Autoencoder checkpoint-final directory containing autoencoder/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where generated PNG images are written.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size.")
    parser.add_argument("--sample-size", type=int, default=28, help="Latent spatial size.")
    parser.add_argument("--latent-channels", type=int, default=32, help="Latent channel count.")
    parser.add_argument("--num-inference-steps", type=int, default=1000, help="Reverse diffusion steps.")
    parser.add_argument(
        "--latent-stats",
        type=Path,
        default=None,
        help="Optional latent_stats.json. If omitted, train_summary.json beside the checkpoint is used when available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

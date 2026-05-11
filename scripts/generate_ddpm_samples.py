"""Generate samples from a DDPM checkpoint saved by scripts/train_ddpm.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = checkpoint / "ema_unet" if args.use_ema else checkpoint / "unet"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    model = UNet2DModel.from_pretrained(model_dir)
    if int(model.config.sample_size) != args.image_size:
        raise ValueError(
            f"Checkpoint image size is {model.config.sample_size}, but --image-size={args.image_size}."
        )

    scheduler_config = checkpoint / "scheduler"
    if args.scheduler == "ddim":
        scheduler = DDIMScheduler.from_pretrained(scheduler_config)
        pipeline_cls = DDIMPipeline
    else:
        scheduler = DDPMScheduler.from_pretrained(scheduler_config)
        pipeline_cls = DDPMPipeline

    pipeline = pipeline_cls(unet=model, scheduler=scheduler).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    remaining = args.num_samples
    written = 0
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        ).images
        for image in images:
            image.save(output_dir / f"sample_{written:06d}.png")
            written += 1
        remaining -= batch_size

    print(f"Generated {written} samples in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory from train_ddpm.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where generated PNG images are written.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--image-size", type=int, default=128, help="Expected checkpoint image size.")
    parser.add_argument("--num-inference-steps", type=int, default=1000, help="Reverse diffusion steps.")
    parser.add_argument("--scheduler", choices=("ddpm", "ddim"), default="ddpm", help="Inference scheduler.")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights saved in the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

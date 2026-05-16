"""Generate images from a DDPM checkpoint-final directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_checkpoint_dir(checkpoint_dir)
    print("===== Sample generation configuration =====")
    print(f"checkpoint-dir={checkpoint_dir}")
    print(f"output-dir={output_dir}")
    print(f"num-samples={args.num_samples}")
    print(f"batch-size={args.batch_size}")
    print(f"image-size={args.image_size}")
    print(f"num-inference-steps={args.num_inference_steps}")
    print(f"seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = checkpoint_dir / "unet"

    model = UNet2DModel.from_pretrained(model_dir)
    if int(model.config.sample_size) != args.image_size:
        raise ValueError(
            f"Checkpoint image size is {model.config.sample_size}, but --image-size={args.image_size}."
        )

    scheduler_config = checkpoint_dir / "scheduler"
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


def validate_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_dir}")

    required_entries = [checkpoint_dir / "unet", checkpoint_dir / "scheduler", checkpoint_dir / "training_state.pt"]
    missing_entries = [entry for entry in required_entries if not entry.exists()]
    if missing_entries:
        missing_text = ", ".join(str(entry) for entry in missing_entries)
        raise FileNotFoundError(
            "checkpoint-dir must point to a checkpoint-final directory containing unet/, scheduler/, and "
            f"training_state.pt; missing: {missing_text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint",
        dest="checkpoint_dir",
        type=Path,
        required=True,
        help="Path to a checkpoint-final directory containing unet/ and scheduler/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where generated PNG images are written.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size.")
    parser.add_argument("--image-size", type=int, default=128, help="Expected checkpoint image size.")
    parser.add_argument("--num-inference-steps", type=int, default=1000, help="Reverse diffusion steps.")
    parser.add_argument("--scheduler", choices=("ddpm", "ddim"), default="ddpm", help="Inference scheduler.")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights saved in the checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

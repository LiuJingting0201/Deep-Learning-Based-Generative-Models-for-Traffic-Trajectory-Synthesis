"""Generate images from a DDPM checkpoint-final directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DDPMPipeline, DDPMScheduler, UNet2DModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_ddpm import inverse_channel_normalization, load_channel_norm_stats, tensors_to_pil


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
    print(f"scheduler={args.scheduler}")
    print(f"eta={args.eta}")
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

    if args.channel_normalization_mode == "standardize":
        load_channel_norm_stats(args)
    else:
        checkpoint_args = load_checkpoint_args(checkpoint_dir)
        channel_norm_mode = checkpoint_args.get("channel_normalization_mode", "none")
        if channel_norm_mode == "standardize":
            args.channel_normalization_mode = "standardize"
            args.channel_norm_mean = checkpoint_args["channel_norm_mean"]
            args.channel_norm_std = checkpoint_args["channel_norm_std"]
        else:
            args.channel_normalization_mode = "none"
            args.channel_norm_mean = None
            args.channel_norm_std = None

    pipeline = pipeline_cls(unet=model, scheduler=scheduler).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    remaining = args.num_samples
    written = 0
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        if args.channel_normalization_mode == "standardize":
            images = generate_channel_normalized_batch(model, scheduler, batch_size, generator, device, args)
        else:
            pipeline_kwargs = {
                "batch_size": batch_size,
                "generator": generator,
                "num_inference_steps": args.num_inference_steps,
            }
            if args.scheduler == "ddim":
                pipeline_kwargs["eta"] = args.eta
            images = pipeline(**pipeline_kwargs).images
        for image in images:
            image.save(output_dir / f"{args.filename_prefix}sample{written:03d}.png")
            written += 1
        remaining -= batch_size

    print(f"Generated {written} samples in {output_dir}")


def generate_channel_normalized_batch(
    model: UNet2DModel,
    scheduler: DDPMScheduler | DDIMScheduler,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    args: argparse.Namespace,
) -> list:
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    sample = torch.randn((batch_size, 3, args.image_size, args.image_size), generator=generator, device=device)
    model.eval()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(sample, timestep)
            noise_pred = model(model_input, timestep, return_dict=False)[0]
            step_kwargs = {"generator": generator}
            if args.scheduler == "ddim":
                step_kwargs["eta"] = args.eta
            sample = scheduler.step(noise_pred, timestep, sample, **step_kwargs).prev_sample
    sample = inverse_channel_normalization(sample, args)
    sample = torch.clamp(sample, -1.0, 1.0)
    return tensors_to_pil(sample)


def load_checkpoint_args(checkpoint_dir: Path) -> dict[str, object]:
    state_path = checkpoint_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    checkpoint_args = state.get("args", {})
    if not isinstance(checkpoint_args, dict):
        return {}
    if checkpoint_args.get("channel_normalization_mode") == "standardize":
        if "channel_norm_mean" not in checkpoint_args or "channel_norm_std" not in checkpoint_args:
            payload = checkpoint_args.get("channel_norm_stats_payload")
            if isinstance(payload, dict):
                checkpoint_args["channel_norm_mean"] = payload.get("mean")
                checkpoint_args["channel_norm_std"] = payload.get("std")
        if checkpoint_args.get("channel_norm_mean") is None or checkpoint_args.get("channel_norm_std") is None:
            raise ValueError(f"Channel-normalized checkpoint is missing channel mean/std in {state_path}")
    return checkpoint_args


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
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta; 0.0 gives deterministic DDIM sampling.")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights saved in the checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="",
        help="Optional prefix for generated filenames, e.g. checkpoint25000_.",
    )
    parser.add_argument(
        "--channel-normalization-mode",
        choices=("none", "standardize"),
        default="none",
        help="Set explicitly for channel-normalized checkpoints to avoid loading training_state.pt.",
    )
    parser.add_argument(
        "--channel-norm-stats",
        type=Path,
        default=None,
        help="Channel stats JSON used when --channel-normalization-mode=standardize.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

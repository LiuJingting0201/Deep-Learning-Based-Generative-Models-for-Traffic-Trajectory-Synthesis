"""Sample Week 5 DDPM checkpoints to float .npy arrays.

Outputs one memmapped .npy per checkpoint, shaped (N, H, W, 3), float32,
with values in [0, 1]. No PNG files are written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_ddpm import build_unet, set_seed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this under a GPU allocation.")

    results_root = args.results_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    jobs = resolve_jobs(results_root, args.configs, args.min_step, args.max_step)
    if not jobs:
        raise ValueError(
            f"No checkpoints found in {results_root} for steps {args.min_step}..{args.max_step}"
        )

    print(f"Sampling {len(jobs)} checkpoints to {output_root}", flush=True)
    for config_dir, step, checkpoint_dir in jobs:
        sample_checkpoint(
            config_dir=config_dir,
            step=step,
            checkpoint_dir=checkpoint_dir,
            output_root=output_root,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_inference_steps=args.sample_inference_steps,
            seed=args.seed,
            device=device,
            prefer_ema=not args.no_ema,
            overwrite=args.overwrite,
        )


def resolve_jobs(
    results_root: Path,
    configs: list[str] | None,
    min_step: int,
    max_step: int,
) -> list[tuple[Path, int, Path]]:
    config_dirs = [results_root / name for name in configs] if configs else sorted(results_root.glob("week05_*"))
    jobs: list[tuple[Path, int, Path]] = []
    for config_dir in config_dirs:
        if not config_dir.is_dir():
            raise FileNotFoundError(config_dir)
        for checkpoint_dir in sorted(config_dir.glob("checkpoint-*"), key=checkpoint_sort_key):
            step = parse_checkpoint_step(checkpoint_dir)
            if step is None or step < min_step or step > max_step:
                continue
            if not (checkpoint_dir / "unet").is_dir():
                raise FileNotFoundError(checkpoint_dir / "unet")
            if not (checkpoint_dir / "scheduler").is_dir():
                raise FileNotFoundError(checkpoint_dir / "scheduler")
            jobs.append((config_dir, step, checkpoint_dir))
    return jobs


def checkpoint_sort_key(path: Path) -> int:
    step = parse_checkpoint_step(path)
    return step if step is not None else 10**12


def parse_checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def sample_checkpoint(
    config_dir: Path,
    step: int,
    checkpoint_dir: Path,
    output_root: Path,
    num_samples: int,
    batch_size: int,
    image_size: int,
    num_inference_steps: int,
    seed: int,
    device: torch.device,
    prefer_ema: bool,
    overwrite: bool,
) -> None:
    config_name = config_dir.name
    output_dir = output_root / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config_name}_step_{step:06d}_samples_{num_samples}.npy"
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and not overwrite:
        print(f"skip existing {output_path}", flush=True)
        return

    started = time.perf_counter()
    print(f"loading {checkpoint_dir}", flush=True)
    model = load_inference_model(checkpoint_dir, image_size, device, prefer_ema)
    scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / "scheduler")
    scheduler.set_timesteps(num_inference_steps, device=device)

    samples = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, image_size, image_size, 3),
    )
    written = 0
    generator = torch.Generator(device=device).manual_seed(seed + step)
    while written < num_samples:
        current_batch = min(batch_size, num_samples - written)
        batch = sample_batch(
            model=model,
            scheduler=scheduler,
            batch_size=current_batch,
            image_size=image_size,
            generator=generator,
            device=device,
        )
        samples[written : written + current_batch] = batch
        written += current_batch
        samples.flush()
        print(f"{config_name} step={step} wrote {written}/{num_samples}", flush=True)

    del samples
    elapsed = time.perf_counter() - started
    metadata = {
        "config": config_name,
        "checkpoint": str(checkpoint_dir),
        "step": step,
        "num_samples": num_samples,
        "shape": [num_samples, image_size, image_size, 3],
        "dtype": "float32",
        "value_range": "[0, 1]",
        "sample_inference_steps": num_inference_steps,
        "batch_size": batch_size,
        "seed": seed,
        "generator_seed": seed + step,
        "used_ema": prefer_ema and (checkpoint_dir / "ema_unet").is_dir(),
        "elapsed_seconds": elapsed,
        "output_path": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {output_path} in {elapsed:.1f}s", flush=True)


def load_inference_model(
    checkpoint_dir: Path,
    image_size: int,
    device: torch.device,
    prefer_ema: bool,
) -> UNet2DModel:
    if prefer_ema and (checkpoint_dir / "ema_unet").is_dir():
        ema_model = EMAModel.from_pretrained(checkpoint_dir / "ema_unet", UNet2DModel)
        model = build_unet(image_size).to(device)
        ema_model.copy_to(model.parameters())
    else:
        model = UNet2DModel.from_pretrained(checkpoint_dir / "unet").to(device)
    model.eval()
    return model


def sample_batch(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    batch_size: int,
    image_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    sample = torch.randn(
        (batch_size, 3, image_size, image_size),
        generator=generator,
        device=device,
    )
    with torch.inference_mode():
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(sample, timestep)
            noise_pred = model(model_input, timestep, return_dict=False)[0]
            sample = scheduler.step(noise_pred, timestep, sample, generator=generator).prev_sample
    sample = torch.clamp((sample + 1.0) / 2.0, 0.0, 1.0)
    return sample.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "week05" / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "week05" / "results_npy_samples")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--min-step", type=int, default=40000)
    parser.add_argument("--max-step", type=int, default=50000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

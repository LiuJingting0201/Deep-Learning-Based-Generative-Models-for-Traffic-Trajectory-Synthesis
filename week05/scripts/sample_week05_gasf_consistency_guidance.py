"""Run Week 5 DDPM inference with differentiable GASF cycle-consistency guidance.

The training checkpoint is unchanged. Guidance is applied only during reverse
sampling, only to channels 0 and 1, and only after the configured start fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEEK05_SCRIPTS_DIR = PROJECT_ROOT / "week05" / "scripts"
for import_dir in (SCRIPTS_DIR, WEEK05_SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from evaluate_week05_npy_fid import (  # noqa: E402
    InceptionFeatureExtractor,
    NpyFIDDataset,
    Week05RealFIDDataset,
    compute_fid,
    compute_stats,
    get_features,
)
from train_ddpm import build_unet, set_seed  # noqa: E402
from train_week05_ddpm import load_normalization  # noqa: E402


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is not available; run this under a GPU allocation or pass --cpu.")

    device = torch.device("cpu" if args.cpu else "cuda")
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "guidance_consistency.log"

    normalization = load_normalization(args.normalization_file.expanduser().resolve())
    model = load_inference_model(checkpoint_dir, args.image_size, device, prefer_ema=not args.no_ema)
    scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / "scheduler")

    fid_model = InceptionFeatureExtractor().to(device)
    real_mu, real_sigma, num_real = compute_or_load_real_fid_stats(
        data_root=data_root,
        image_size=args.image_size,
        sigmoid_k=args.sigmoid_k,
        split=args.real_split,
        batch_size=args.fid_batch_size,
        num_workers=args.num_workers,
        device=device,
        model=fid_model,
        feature_cache_dir=output_dir / "feature_cache",
        overwrite=args.overwrite_features,
    )

    rows = []
    for guidance_scale in args.guidance_scales:
        started = time.perf_counter()
        sample_path = output_dir / "samples" / guidance_sample_name(
            checkpoint_dir=checkpoint_dir,
            num_samples=args.num_samples,
            guidance_scale=guidance_scale,
        )
        metrics_path = output_dir / "metrics" / f"{sample_path.stem}_metrics.json"
        print(f"sampling guidance_scale={guidance_scale:g} -> {sample_path}", flush=True)
        sample_metrics = sample_to_npy(
            model=model,
            scheduler=scheduler,
            output_path=sample_path,
            normalization=normalization,
            num_samples=args.num_samples,
            batch_size=args.sample_batch_size,
            image_size=args.image_size,
            sample_inference_steps=args.sample_inference_steps,
            generator_seed=args.seed + parse_checkpoint_step(checkpoint_dir),
            device=device,
            sigmoid_k=args.sigmoid_k,
            guidance_scale=guidance_scale,
            guidance_grad_clip=args.guidance_grad_clip,
            guidance_start_fraction=args.guidance_start_fraction,
            log_path=log_path,
            overwrite=args.overwrite_samples,
        )
        fid = evaluate_fid_for_npy(
            sample_path=sample_path,
            real_mu=real_mu,
            real_sigma=real_sigma,
            batch_size=args.fid_batch_size,
            num_workers=args.num_workers,
            device=device,
            model=fid_model,
        )
        elapsed = time.perf_counter() - started
        row = {
            "guidance_scale": guidance_scale,
            "FID": fid,
            "mean_consistency_loss": sample_metrics["mean_consistency_loss"],
            "max_consistency_loss": sample_metrics["max_consistency_loss"],
            "num_generated": args.num_samples,
            "elapsed_time": elapsed,
        }
        metrics_payload = {
            **row,
            **sample_metrics,
            "sample_path": str(sample_path),
            "checkpoint_dir": str(checkpoint_dir),
            "normalization": normalization,
            "sigmoid_k": args.sigmoid_k,
            "guidance_grad_clip": args.guidance_grad_clip,
            "real_split": args.real_split,
            "num_real": num_real,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")
        rows.append(row)
        print(
            f"guidance_scale={guidance_scale:g} FID={fid:.6f} "
            f"mean_consistency_loss={row['mean_consistency_loss']:.8f} "
            f"max_consistency_loss={row['max_consistency_loss']:.8f}",
            flush=True,
        )

    write_summary(output_dir / "guidance_consistency_summary.csv", rows)
    (output_dir / "guidance_consistency_summary.json").write_text(
        json.dumps({"results": rows}, indent=2) + "\n"
    )


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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def sample_to_npy(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    output_path: Path,
    normalization: dict[str, float],
    num_samples: int,
    batch_size: int,
    image_size: int,
    sample_inference_steps: int,
    generator_seed: int,
    device: torch.device,
    sigmoid_k: float,
    guidance_scale: float,
    guidance_grad_clip: float,
    guidance_start_fraction: float,
    log_path: Path,
    overwrite: bool,
) -> dict[str, float | int]:
    if output_path.exists() and not overwrite:
        return load_existing_sample_metrics(output_path, normalization, sigmoid_k, batch_size, device)

    scheduler.set_timesteps(sample_inference_steps, device=device)
    guidance_start_index = int(np.ceil(sample_inference_steps * guidance_start_fraction))
    samples = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, image_size, image_size, 3),
    )
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    losses_sum = 0.0
    losses_max = 0.0
    losses_count = 0
    skipped_nan_guidance_steps = 0
    applied_guidance_steps = 0
    written = 0
    while written < num_samples:
        current_batch = min(batch_size, num_samples - written)
        sample = torch.randn(
            (current_batch, 3, image_size, image_size),
            generator=generator,
            device=device,
        )
        for step_index, timestep in enumerate(scheduler.timesteps):
            with torch.no_grad():
                model_input = scheduler.scale_model_input(sample, timestep)
                noise_pred = model(model_input, timestep, return_dict=False)[0]

            should_guide = guidance_scale > 0.0 and step_index >= guidance_start_index
            if should_guide:
                guided = guided_noise_prediction(
                    x_t=sample,
                    noise_pred=noise_pred,
                    timestep=timestep,
                    scheduler=scheduler,
                    normalization=normalization,
                    sigmoid_k=sigmoid_k,
                    guidance_scale=guidance_scale,
                    guidance_grad_clip=guidance_grad_clip,
                )
                if guided["skipped"]:
                    skipped_nan_guidance_steps += 1
                    append_log(
                        log_path,
                        f"skipped guidance due to non-finite consistency loss/gradient "
                        f"sample_offset={written} timestep={int(timestep)}",
                    )
                else:
                    noise_pred = guided["noise_pred"]
                    applied_guidance_steps += 1

            with torch.no_grad():
                sample = scheduler.step(noise_pred, timestep, sample, generator=generator).prev_sample

        sample = torch.clamp(sample, -1.0, 1.0)
        batch_losses = gasf_consistency_loss_per_sample(sample, normalization, sigmoid_k)
        batch_losses = torch.nan_to_num(batch_losses.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        losses_sum += float(batch_losses.sum().cpu())
        losses_max = max(losses_max, float(batch_losses.max().cpu()))
        losses_count += int(batch_losses.numel())
        samples[written : written + current_batch] = (
            ((sample.detach().cpu().permute(0, 2, 3, 1).numpy() + 1.0) / 2.0)
            .clip(0.0, 1.0)
            .astype(np.float32, copy=False)
        )
        written += current_batch
        samples.flush()
        print(f"{output_path.name}: wrote {written}/{num_samples}", flush=True)

    del samples
    return {
        "mean_consistency_loss": losses_sum / max(1, losses_count),
        "max_consistency_loss": losses_max,
        "skipped_nan_guidance_steps": skipped_nan_guidance_steps,
        "applied_guidance_steps": applied_guidance_steps,
        "generator_seed": generator_seed,
    }


def guided_noise_prediction(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: DDPMScheduler,
    normalization: dict[str, float],
    sigmoid_k: float,
    guidance_scale: float,
    guidance_grad_clip: float,
) -> dict[str, torch.Tensor | bool]:
    alpha_t = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[timestep].view(1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0_pred = ((x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t).detach()
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0).requires_grad_(True)
    loss = gasf_consistency_loss(x0_pred, normalization, sigmoid_k)
    if not torch.isfinite(loss):
        return {"noise_pred": noise_pred, "skipped": True}
    grad = torch.autograd.grad(loss, x0_pred, retain_graph=False, create_graph=False)[0]
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad = torch.clamp(grad, -guidance_grad_clip, guidance_grad_clip)
    if not torch.isfinite(grad).all():
        return {"noise_pred": noise_pred, "skipped": True}
    x0_pred_guided = torch.clamp(x0_pred - guidance_scale * grad, -1.0, 1.0).detach()
    noise_pred_guided = (x_t - sqrt_alpha_t * x0_pred_guided) / sqrt_one_minus_alpha_t
    return {"noise_pred": noise_pred_guided.detach(), "skipped": False}


def gasf_consistency_loss(
    images: torch.Tensor,
    normalization: dict[str, float],
    sigmoid_k: float,
) -> torch.Tensor:
    return gasf_consistency_loss_per_sample(images, normalization, sigmoid_k).mean()


def gasf_consistency_loss_per_sample(
    images: torch.Tensor,
    normalization: dict[str, float],
    sigmoid_k: float,
) -> torch.Tensor:
    gasf_channels = images[:, :2, :, :]
    reconstructed = reconstruct_gasf_channels_from_diag(gasf_channels, normalization, sigmoid_k)
    per_channel = (gasf_channels - reconstructed).pow(2).mean(dim=(-1, -2))
    return 0.5 * per_channel[:, 0] + 0.5 * per_channel[:, 1]


def reconstruct_gasf_channels_from_diag(
    gasf_channels: torch.Tensor,
    normalization: dict[str, float],
    sigmoid_k: float,
) -> torch.Tensor:
    dx = reconstruct_one_gasf_channel(
        gasf_channels[:, 0],
        mean=normalization["dx_mean"],
        std=normalization["dx_std"],
        sigmoid_k=sigmoid_k,
    )
    dy = reconstruct_one_gasf_channel(
        gasf_channels[:, 1],
        mean=normalization["dy_mean"],
        std=normalization["dy_std"],
        sigmoid_k=sigmoid_k,
    )
    return torch.stack([dx, dy], dim=1)


def reconstruct_one_gasf_channel(
    gasf_channel: torch.Tensor,
    mean: float,
    std: float,
    sigmoid_k: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    diag = torch.diagonal(gasf_channel, dim1=-2, dim2=-1)
    u = torch.sqrt(torch.clamp((diag + 1.0) / 2.0, min=eps, max=1.0))
    u_for_logit = torch.clamp(u, min=eps, max=1.0 - eps)
    z = torch.log(u_for_logit / (1.0 - u_for_logit))
    delta = mean + std * z / sigmoid_k
    u_reconstructed = torch.sigmoid(sigmoid_k * ((delta - mean) / std))
    return gasf_encode_torch_like_training(u_reconstructed)


def gasf_encode_torch_like_training(values_01: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of train_week05_ddpm.gasf_encode for autograd guidance."""
    phi = torch.arccos(torch.clamp(values_01, 0.0, 1.0))
    return torch.cos(phi[:, :, None] + phi[:, None, :])


def load_existing_sample_metrics(
    output_path: Path,
    normalization: dict[str, float],
    sigmoid_k: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    array = np.load(output_path, mmap_mode="r")
    losses_sum = 0.0
    losses_max = 0.0
    losses_count = 0
    for start in range(0, int(array.shape[0]), batch_size):
        batch = np.array(array[start : start + batch_size], dtype=np.float32, copy=True)
        images = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        images = torch.clamp(images, 0.0, 1.0) * 2.0 - 1.0
        images = images.to(device)
        losses = gasf_consistency_loss_per_sample(images, normalization, sigmoid_k)
        losses = torch.nan_to_num(losses.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        losses_sum += float(losses.sum().cpu())
        losses_max = max(losses_max, float(losses.max().cpu()))
        losses_count += int(losses.numel())
    return {
        "mean_consistency_loss": losses_sum / max(1, losses_count),
        "max_consistency_loss": losses_max,
        "skipped_nan_guidance_steps": 0,
        "applied_guidance_steps": 0,
    }


def compute_or_load_real_fid_stats(
    data_root: Path,
    image_size: int,
    sigmoid_k: float,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: InceptionFeatureExtractor,
    feature_cache_dir: Path,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    stats_path = feature_cache_dir / f"real_{split}_sigmoid_{sigmoid_k:g}_inception_stats.npz"
    if stats_path.exists() and not overwrite:
        cached = np.load(stats_path)
        return cached["mu"], cached["sigma"], int(cached["num_real"])
    dataset = Week05RealFIDDataset(
        data_root=data_root,
        image_size=image_size,
        normalization_file=data_root / "normalization_diagnostics.json",
        sigmoid_k=sigmoid_k,
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = get_features(loader, model, device)
    mu, sigma = compute_stats(features)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma, num_real=np.array(len(dataset), dtype=np.int64))
    return mu, sigma, len(dataset)


def evaluate_fid_for_npy(
    sample_path: Path,
    real_mu: np.ndarray,
    real_sigma: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: InceptionFeatureExtractor,
) -> float:
    dataset = NpyFIDDataset(sample_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = get_features(loader, model, device)
    mu_generated, sigma_generated = compute_stats(features)
    return compute_fid(real_mu, real_sigma, mu_generated, sigma_generated)


def guidance_sample_name(checkpoint_dir: Path, num_samples: int, guidance_scale: float) -> str:
    config_name = checkpoint_dir.parent.name
    step_name = checkpoint_dir.name.replace("checkpoint-", "step_")
    scale_name = f"{guidance_scale:g}".replace(".", "p")
    return f"{config_name}_{step_name}_guidance_{scale_name}_samples_{num_samples}.npy"


def parse_checkpoint_step(checkpoint_dir: Path) -> int:
    if checkpoint_dir.name.startswith("checkpoint-") and checkpoint_dir.name.removeprefix("checkpoint-").isdigit():
        return int(checkpoint_dir.name.removeprefix("checkpoint-"))
    return 0


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a") as handle:
        handle.write(message + "\n")


def write_summary(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "guidance_scale",
        "FID",
        "mean_consistency_loss",
        "max_consistency_loss",
        "num_generated",
        "elapsed_time",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"wrote {path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "week05" / "results" / "week05_sig15_mse_diag" / "checkpoint-40000",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224",
    )
    parser.add_argument(
        "--normalization-file",
        type=Path,
        default=PROJECT_ROOT
        / "week05"
        / "data"
        / "delta_displacement_paired_regularized_bounce224"
        / "normalization_diagnostics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "week05" / "guidance_consistency_results",
    )
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02])
    parser.add_argument("--guidance-start-fraction", type=float, default=0.5)
    parser.add_argument("--guidance-grad-clip", type=float, default=0.05)
    parser.add_argument("--sigmoid-k", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite-samples", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

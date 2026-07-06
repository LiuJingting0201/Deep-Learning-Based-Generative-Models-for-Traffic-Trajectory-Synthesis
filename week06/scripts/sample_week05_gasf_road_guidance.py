"""Sample Week05 GASF DDPM with trajectory-space road/support guidance.

The checkpoint is unchanged. During reverse diffusion, the current x0 estimate
is decoded through the GASF diagonal into a raw XY trajectory, nudged toward a
precomputed empirical-support distance field, then encoded back to GASF space.
"""

from __future__ import annotations

import argparse
import json
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
WEEK05_SCRIPTS_DIR = PROJECT_ROOT / "week05" / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for import_dir in (SCRIPTS_DIR, WEEK05_SCRIPTS_DIR, SRC_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cnr_trajectory.guidance import DistanceFieldRoadGuidance, RoadRasterMetadata  # noqa: E402
from train_ddpm import build_unet, set_seed  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "week05" / "results" / "week05_sig15_mse_diag" / "checkpoint-40000"
DEFAULT_CONFIG = PROJECT_ROOT / "week05" / "results" / "week05_sig15_mse_diag" / "startup_config.json"
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT = PROJECT_ROOT / "week06" / "results" / "week05_sig15_mse_diag_road_guidance_full"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is not available; run under a GPU allocation or pass --cpu for tiny smoke tests.")

    device = torch.device("cpu" if args.cpu else "cuda")
    output_dir = args.output_dir.expanduser().resolve()
    samples_dir = output_dir / "samples"
    metrics_dir = output_dir / "metrics"
    field_dir = output_dir / "distance_field"
    for path in (samples_dir, metrics_dir, field_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.expanduser().resolve().read_text())
    normalization = config["normalization_stats"]
    position_stats = config["position_stats"]
    distance_field, metadata = build_or_load_empirical_distance_field(
        real_data_root=args.real_data_root.expanduser().resolve(),
        position_stats=position_stats,
        field_dir=field_dir,
        raster_size=args.distance_field_size,
        support_radius_m=args.support_radius_m,
        overwrite=args.overwrite_distance_field,
    )
    distance_field_tensor = torch.from_numpy(distance_field).to(device=device, dtype=torch.float32)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    model = load_inference_model(checkpoint_dir, args.image_size, device, prefer_ema=not args.no_ema)
    scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / "scheduler")

    rows = []
    for guidance_strength in args.guidance_strengths:
        started = time.perf_counter()
        sample_path = samples_dir / sample_name(checkpoint_dir, args.num_samples, guidance_strength)
        metrics_path = metrics_dir / f"{sample_path.stem}_metrics.json"
        print(f"sampling road_guidance_strength={guidance_strength:g} -> {sample_path}", flush=True)
        metrics = sample_to_npy(
            model=model,
            scheduler=scheduler,
            output_path=sample_path,
            normalization=normalization,
            position_stats=position_stats,
            distance_field=distance_field_tensor,
            metadata=metadata,
            num_samples=args.num_samples,
            batch_size=args.sample_batch_size,
            image_size=args.image_size,
            sample_inference_steps=args.sample_inference_steps,
            generator_seed=args.seed + parse_checkpoint_step(checkpoint_dir) + int(guidance_strength),
            device=device,
            sigmoid_k=args.sigmoid_k,
            guidance_strength=guidance_strength,
            num_guidance_steps=args.num_guidance_steps,
            guidance_start_fraction=args.guidance_start_fraction,
            guidance_end_fraction=args.guidance_end_fraction,
            distance_scale=args.distance_scale,
            gasf_update_mode=args.gasf_update_mode,
            guidance_x0_blend=args.guidance_x0_blend,
            delta_clip_abs=args.delta_clip_abs,
            endpoint_cap=args.endpoint_cap,
            endpoint_projection_strength=args.endpoint_projection_strength,
            overwrite=args.overwrite_samples,
        )
        elapsed = time.perf_counter() - started
        payload = {
            **metrics,
            "road_guidance_strength": guidance_strength,
            "num_guidance_steps": args.num_guidance_steps,
            "guidance_start_fraction": args.guidance_start_fraction,
            "guidance_end_fraction": args.guidance_end_fraction,
            "distance_scale": args.distance_scale,
            "gasf_update_mode": args.gasf_update_mode,
            "guidance_x0_blend": args.guidance_x0_blend,
            "delta_clip_abs": args.delta_clip_abs,
            "endpoint_cap": args.endpoint_cap,
            "endpoint_projection_strength": args.endpoint_projection_strength,
            "num_generated": args.num_samples,
            "sample_path": str(sample_path),
            "checkpoint_dir": str(checkpoint_dir),
            "elapsed_seconds": elapsed,
            "distance_field_metadata": metadata.__dict__,
        }
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
        rows.append(payload)
        print(
            f"strength={guidance_strength:g} before_mean_dist={metrics['mean_road_dist_before']:.4f} "
            f"after_mean_dist={metrics['mean_road_dist_after']:.4f} "
            f"applied_steps={metrics['applied_guidance_steps']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    (output_dir / "road_guidance_summary.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")


def load_inference_model(checkpoint_dir: Path, image_size: int, device: torch.device, prefer_ema: bool) -> UNet2DModel:
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
    *,
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    output_path: Path,
    normalization: dict[str, float],
    position_stats: dict[str, float],
    distance_field: torch.Tensor,
    metadata: RoadRasterMetadata,
    num_samples: int,
    batch_size: int,
    image_size: int,
    sample_inference_steps: int,
    generator_seed: int,
    device: torch.device,
    sigmoid_k: float,
    guidance_strength: float,
    num_guidance_steps: int,
    guidance_start_fraction: float,
    guidance_end_fraction: float,
    distance_scale: float,
    gasf_update_mode: str,
    guidance_x0_blend: float,
    delta_clip_abs: float | None,
    endpoint_cap: float | None,
    endpoint_projection_strength: float,
    overwrite: bool,
) -> dict[str, float | int]:
    if output_path.exists() and not overwrite:
        return {"loaded_existing": 1, "applied_guidance_steps": 0, "mean_road_dist_before": 0.0, "mean_road_dist_after": 0.0}

    scheduler.set_timesteps(sample_inference_steps, device=device)
    start_index = int(np.ceil(sample_inference_steps * guidance_start_fraction))
    end_index = int(np.floor(sample_inference_steps * guidance_end_fraction))
    guidance = DistanceFieldRoadGuidance(
        metadata=metadata,
        distance_field=distance_field,
        guidance_strength=guidance_strength,
        num_guidance_steps=num_guidance_steps,
        apply_start_step=start_index,
        apply_end_step=end_index,
        distance_scale=distance_scale,
        device=device,
    )
    samples = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, image_size, image_size, 3),
    )
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    written = 0
    applied_steps = 0
    skipped_steps = 0
    before_sum = 0.0
    after_sum = 0.0
    metric_count = 0

    while written < num_samples:
        current_batch = min(batch_size, num_samples - written)
        sample = torch.randn((current_batch, 3, image_size, image_size), generator=generator, device=device)
        for step_index, timestep in enumerate(scheduler.timesteps):
            with torch.no_grad():
                model_input = scheduler.scale_model_input(sample, timestep)
                noise_pred = model(model_input, timestep, return_dict=False)[0]
            if guidance.should_apply(step_index):
                guided = road_guided_noise_prediction(
                    x_t=sample,
                    noise_pred=noise_pred,
                    timestep=timestep,
                    scheduler=scheduler,
                    normalization=normalization,
                    position_stats=position_stats,
                    sigmoid_k=sigmoid_k,
                    guidance=guidance,
                    gasf_update_mode=gasf_update_mode,
                    guidance_x0_blend=guidance_x0_blend,
                    delta_clip_abs=delta_clip_abs,
                    endpoint_cap=endpoint_cap,
                    endpoint_projection_strength=endpoint_projection_strength,
                )
                if guided["skipped"]:
                    skipped_steps += 1
                else:
                    noise_pred = guided["noise_pred"]
                    applied_steps += 1
                    before_sum += float(guided["mean_road_dist_before"])
                    after_sum += float(guided["mean_road_dist_after"])
                    metric_count += 1
            with torch.no_grad():
                sample = scheduler.step(noise_pred, timestep, sample, generator=generator).prev_sample

        sample = torch.clamp(sample, -1.0, 1.0)
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
        "loaded_existing": 0,
        "applied_guidance_steps": applied_steps,
        "skipped_guidance_steps": skipped_steps,
        "mean_road_dist_before": before_sum / max(1, metric_count),
        "mean_road_dist_after": after_sum / max(1, metric_count),
        "generator_seed": generator_seed,
    }


def road_guided_noise_prediction(
    *,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: DDPMScheduler,
    normalization: dict[str, float],
    position_stats: dict[str, float],
    sigmoid_k: float,
    guidance: DistanceFieldRoadGuidance,
    gasf_update_mode: str,
    guidance_x0_blend: float,
    delta_clip_abs: float | None,
    endpoint_cap: float | None,
    endpoint_projection_strength: float,
) -> dict[str, Any]:
    alpha_t = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[timestep].view(1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0_pred = torch.clamp(((x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t).detach(), -1.0, 1.0)
    start_xy, delta_xy = decode_x0_to_start_delta(x0_pred, normalization, position_stats, sigmoid_k)
    trajectory = integrate_delta_torch(start_xy, delta_xy)
    before_energy, before_dist = guidance.road_energy(trajectory)
    guided_trajectory, _, after_dist = guidance(trajectory)
    if not torch.isfinite(guided_trajectory).all():
        return {"noise_pred": noise_pred, "skipped": True}

    guided_trajectory = guided_trajectory - guided_trajectory[:, :1, :] + start_xy[:, None, :]
    if endpoint_cap is not None:
        guided_trajectory = project_endpoint_displacement(
            guided_trajectory,
            max_displacement=float(endpoint_cap),
            strength=float(endpoint_projection_strength),
        )
        _, after_dist = guidance.road_energy(guided_trajectory)
    guided_delta = torch.zeros_like(delta_xy)
    guided_delta[:, 1:, :] = guided_trajectory[:, 1:, :] - guided_trajectory[:, :-1, :]
    if delta_clip_abs is not None:
        guided_delta = torch.clamp(guided_delta, min=-float(delta_clip_abs), max=float(delta_clip_abs))
    x0_guided = x0_pred.clone()
    if gasf_update_mode == "full":
        x0_guided[:, 0] = encode_delta_channel(guided_delta[..., 0], normalization["dx_mean"], normalization["dx_std"], sigmoid_k)
        x0_guided[:, 1] = encode_delta_channel(guided_delta[..., 1], normalization["dy_mean"], normalization["dy_std"], sigmoid_k)
    elif gasf_update_mode == "diagonal":
        x0_guided[:, 0] = replace_channel_diagonal(
            x0_guided[:, 0],
            encode_delta_diagonal(guided_delta[..., 0], normalization["dx_mean"], normalization["dx_std"], sigmoid_k),
        )
        x0_guided[:, 1] = replace_channel_diagonal(
            x0_guided[:, 1],
            encode_delta_diagonal(guided_delta[..., 1], normalization["dy_mean"], normalization["dy_std"], sigmoid_k),
        )
    else:
        raise ValueError(f"Unsupported gasf_update_mode: {gasf_update_mode}")
    if guidance_x0_blend < 1.0:
        x0_guided = x0_pred + float(guidance_x0_blend) * (x0_guided - x0_pred)
    x0_guided = torch.clamp(x0_guided, -1.0, 1.0).detach()
    noise_pred_guided = (x_t - sqrt_alpha_t * x0_guided) / sqrt_one_minus_alpha_t
    return {
        "noise_pred": noise_pred_guided.detach(),
        "skipped": False,
        "mean_road_dist_before": before_dist.detach().cpu(),
        "mean_road_dist_after": after_dist.detach().cpu(),
        "road_energy_before": before_energy.detach().cpu(),
    }


def decode_x0_to_start_delta(
    x0_pred: torch.Tensor,
    normalization: dict[str, float],
    position_stats: dict[str, float],
    sigmoid_k: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = decode_delta_channel(x0_pred[:, 0], normalization["dx_mean"], normalization["dx_std"], sigmoid_k, eps)
    dy = decode_delta_channel(x0_pred[:, 1], normalization["dy_mean"], normalization["dy_std"], sigmoid_k, eps)
    delta = torch.stack([dx, dy], dim=-1)
    delta[:, 0, :] = 0.0
    start = decode_start_heatmap_torch((x0_pred[:, 2] + 1.0) / 2.0, position_stats)
    return start, delta


def decode_delta_channel(channel: torch.Tensor, mean: float, std: float, sigmoid_k: float, eps: float) -> torch.Tensor:
    diag = torch.diagonal(channel, dim1=-2, dim2=-1)
    u = torch.sqrt(torch.clamp((diag + 1.0) / 2.0, min=eps, max=1.0 - eps))
    logits = torch.log(u / (1.0 - u))
    return float(mean) + float(std) * logits / float(sigmoid_k)


def decode_start_heatmap_torch(heatmap_01: torch.Tensor, position_stats: dict[str, float], eps: float = 1e-6) -> torch.Tensor:
    batch, height, width = heatmap_01.shape
    flat = torch.clamp(heatmap_01, min=0.0).reshape(batch, -1)
    indices = torch.argmax(flat, dim=1)
    y_index = torch.div(indices, width, rounding_mode="floor").to(dtype=heatmap_01.dtype)
    x_index = (indices % width).to(dtype=heatmap_01.dtype)
    x_norm = x_index / max(width - 1, 1)
    y_norm = y_index / max(height - 1, 1)
    x = float(position_stats["x_min"]) + x_norm * (float(position_stats["x_max"]) - float(position_stats["x_min"]))
    y = float(position_stats["y_min"]) + y_norm * (float(position_stats["y_max"]) - float(position_stats["y_min"]))
    return torch.stack([x, y], dim=-1)


def integrate_delta_torch(start_xy: torch.Tensor, delta_xy: torch.Tensor) -> torch.Tensor:
    trajectory = torch.zeros_like(delta_xy)
    trajectory[:, 0, :] = start_xy
    trajectory[:, 1:, :] = start_xy[:, None, :] + torch.cumsum(delta_xy[:, 1:, :], dim=1)
    return trajectory


def encode_delta_channel(delta: torch.Tensor, mean: float, std: float, sigmoid_k: float) -> torch.Tensor:
    values_01 = torch.sigmoid(float(sigmoid_k) * ((delta - float(mean)) / float(std)))
    phi = torch.arccos(torch.clamp(values_01, 0.0, 1.0))
    return torch.cos(phi[:, :, None] + phi[:, None, :])


def encode_delta_diagonal(delta: torch.Tensor, mean: float, std: float, sigmoid_k: float) -> torch.Tensor:
    values_01 = torch.sigmoid(float(sigmoid_k) * ((delta - float(mean)) / float(std)))
    return 2.0 * values_01.square() - 1.0


def replace_channel_diagonal(channel: torch.Tensor, diagonal_values: torch.Tensor) -> torch.Tensor:
    updated = channel.clone()
    indices = torch.arange(channel.shape[-1], device=channel.device)
    updated[:, indices, indices] = diagonal_values
    return updated


def project_endpoint_displacement(
    trajectory: torch.Tensor,
    *,
    max_displacement: float,
    strength: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if max_displacement <= 0:
        raise ValueError(f"max_displacement must be positive, got {max_displacement}")
    blend = float(np.clip(strength, 0.0, 1.0))
    if blend == 0.0:
        return trajectory

    start = trajectory[:, :1, :]
    end_offset = trajectory[:, -1:, :] - start
    distance = torch.linalg.norm(end_offset, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_displacement) / torch.clamp(distance, min=eps), max=1.0)
    target_end_offset = end_offset * scale
    correction = (end_offset - target_end_offset) * blend
    ramp = torch.linspace(0.0, 1.0, trajectory.shape[1], dtype=trajectory.dtype, device=trajectory.device)
    ramp = ramp.view(1, trajectory.shape[1], 1)
    return trajectory - ramp * correction


def build_or_load_empirical_distance_field(
    *,
    real_data_root: Path,
    position_stats: dict[str, float],
    field_dir: Path,
    raster_size: int,
    support_radius_m: float,
    overwrite: bool,
) -> tuple[np.ndarray, RoadRasterMetadata]:
    field_path = field_dir / f"empirical_support_distance_{raster_size}.npy"
    metadata_path = field_dir / f"empirical_support_distance_{raster_size}.json"
    if field_path.exists() and metadata_path.exists() and not overwrite:
        metadata_payload = json.loads(metadata_path.read_text())
        return np.load(field_path), RoadRasterMetadata(**metadata_payload["metadata"])

    try:
        from scipy.ndimage import binary_dilation, distance_transform_edt
    except ImportError as exc:
        raise ImportError("Building empirical support distance fields requires scipy.") from exc

    x_min = float(position_stats["x_min"])
    x_max = float(position_stats["x_max"])
    y_min = float(position_stats["y_min"])
    y_max = float(position_stats["y_max"])
    width = height = int(raster_size)
    pixel_x = (x_max - x_min) / max(width - 1, 1)
    pixel_y = (y_max - y_min) / max(height - 1, 1)
    mask = np.zeros((height, width), dtype=bool)
    for path in sorted((real_data_root / "labels_absolute").glob("*.npy")):
        xy = np.load(path).astype(np.float64)
        col = np.rint((xy[:, 0] - x_min) / max(pixel_x, 1e-6)).astype(int)
        row = np.rint((y_max - xy[:, 1]) / max(pixel_y, 1e-6)).astype(int)
        valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        mask[row[valid], col[valid]] = True
    if support_radius_m > 0:
        iterations = int(np.ceil(support_radius_m / max(pixel_x, pixel_y)))
        mask = binary_dilation(mask, iterations=max(1, iterations))
    field = distance_transform_edt(~mask, sampling=(pixel_y, pixel_x)).astype(np.float32)
    np.save(field_path, field)
    metadata = RoadRasterMetadata(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, y_axis_down=True)
    metadata_path.write_text(
        json.dumps(
            {
                "metadata": metadata.__dict__,
                "raster_size": raster_size,
                "support_radius_m": support_radius_m,
                "pixel_size_x": pixel_x,
                "pixel_size_y": pixel_y,
                "support_pixel_ratio": float(mask.mean()),
                "field_path": str(field_path),
            },
            indent=2,
        )
        + "\n"
    )
    return field, metadata


def sample_name(checkpoint_dir: Path, num_samples: int, guidance_strength: float) -> str:
    config_name = checkpoint_dir.parent.name
    step_name = checkpoint_dir.name.replace("checkpoint-", "step_")
    strength_name = f"{guidance_strength:g}".replace(".", "p")
    return f"{config_name}_{step_name}_road_guidance_{strength_name}_samples_{num_samples}.npy"


def parse_checkpoint_step(checkpoint_dir: Path) -> int:
    if checkpoint_dir.name.startswith("checkpoint-") and checkpoint_dir.name.removeprefix("checkpoint-").isdigit():
        return int(checkpoint_dir.name.removeprefix("checkpoint-"))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guidance-strengths", type=float, nargs="+", default=[0.0, 1000.0])
    parser.add_argument("--num-guidance-steps", type=int, default=3)
    parser.add_argument("--guidance-start-fraction", type=float, default=0.5)
    parser.add_argument("--guidance-end-fraction", type=float, default=1.0)
    parser.add_argument("--distance-scale", type=float, default=25.0)
    parser.add_argument("--gasf-update-mode", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--guidance-x0-blend", type=float, default=0.1)
    parser.add_argument("--delta-clip-abs", type=float, default=None)
    parser.add_argument("--endpoint-cap", type=float, default=None)
    parser.add_argument("--endpoint-projection-strength", type=float, default=1.0)
    parser.add_argument("--distance-field-size", type=int, default=1024)
    parser.add_argument("--support-radius-m", type=float, default=5.0)
    parser.add_argument("--sigmoid-k", type=float, default=1.5)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sample-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite-samples", action="store_true")
    parser.add_argument("--overwrite-distance-field", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

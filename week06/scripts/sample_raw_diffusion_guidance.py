"""Sample Week06 raw trajectory DDPM checkpoints with trajectory-space guidance."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from diffusers import DDPMScheduler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
WEEK06_SCRIPTS_DIR = PROJECT_ROOT / "week06" / "scripts"
for import_dir in (SRC_DIR, WEEK06_SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cnr_trajectory.guidance import DistanceFieldRoadGuidance, RoadRasterMetadata  # noqa: E402
from sample_week05_gasf_road_guidance import build_or_load_empirical_distance_field  # noqa: E402
from trajectory_diffusion_common import build_trajectory_model, set_seed  # noqa: E402


DEFAULT_RUNS = (
    "raw_absolute_temporal_resnet",
    "raw_absolute_temporal_resnet_dilated",
    "raw_absolute_unet1d",
    "raw_delta_temporal_resnet",
    "raw_delta_temporal_resnet_dilated",
    "raw_delta_unet1d",
)
DEFAULT_REAL_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is not available; run under a GPU allocation or pass --cpu for smoke tests.")
    device = torch.device("cpu" if args.cpu else "cuda")

    run_dir = args.run_dir.expanduser().resolve()
    run_name = run_dir.name
    checkpoint_dir = (run_dir / args.checkpoint).resolve()
    output_dir = args.output_root.expanduser().resolve() / f"{run_name}_guided_strength{format_float(args.guidance_strength)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories_raw_xy").mkdir(parents=True, exist_ok=True)

    startup = json.loads((run_dir / "startup_config.json").read_text())
    mode = startup["mode"]
    stats = startup["normalization"]
    position_stats = compute_position_stats(args.real_data_root.expanduser().resolve())
    args.distance_field_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    field, metadata = build_or_load_empirical_distance_field(
        real_data_root=args.real_data_root.expanduser().resolve(),
        position_stats=position_stats,
        field_dir=args.distance_field_dir.expanduser().resolve(),
        raster_size=args.distance_field_size,
        support_radius_m=args.support_radius_m,
        overwrite=args.overwrite_distance_field,
    )

    model, scheduler, global_step = load_checkpoint(checkpoint_dir, startup, device, prefer_ema=not args.no_ema)
    guidance = DistanceFieldRoadGuidance(
        metadata=metadata,
        distance_field=torch.from_numpy(field).to(device=device, dtype=torch.float32),
        guidance_strength=args.guidance_strength,
        num_guidance_steps=args.num_guidance_steps,
        apply_start_step=int(np.ceil(args.sample_inference_steps * args.guidance_start_fraction)),
        apply_end_step=int(np.floor(args.sample_inference_steps * args.guidance_end_fraction)),
        distance_scale=args.distance_scale,
        device=device,
    )

    starts = None
    if mode == "delta":
        starts = sample_starts(args.real_data_root.expanduser().resolve(), args.num_samples, args.seed + global_step)

    started = time.perf_counter()
    result = sample_guided(
        model=model,
        scheduler=scheduler,
        guidance=guidance,
        mode=mode,
        stats=stats,
        starts=starts,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sequence_length=int(startup["sequence_length"]),
        sample_inference_steps=args.sample_inference_steps,
        generator_seed=args.seed + global_step,
        device=device,
        overwrite=args.overwrite,
    )
    elapsed = time.perf_counter() - started
    summary = {
        **result,
        "run_name": run_name,
        "mode": mode,
        "checkpoint_dir": str(checkpoint_dir),
        "global_step": global_step,
        "output_dir": str(output_dir),
        "num_samples": args.num_samples,
        "guidance_strength": args.guidance_strength,
        "num_guidance_steps": args.num_guidance_steps,
        "guidance_start_fraction": args.guidance_start_fraction,
        "guidance_end_fraction": args.guidance_end_fraction,
        "distance_scale": args.distance_scale,
        "distance_field_metadata": metadata.__dict__,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def load_checkpoint(
    checkpoint_dir: Path,
    startup: dict[str, Any],
    device: torch.device,
    prefer_ema: bool,
) -> tuple[torch.nn.Module, DDPMScheduler, int]:
    model_config = startup["model_config"]
    model_args = SimpleNamespace(
        model_type=startup["model_type"],
        sequence_length=int(startup["sequence_length"]),
        hidden_channels=int(model_config.get("hidden_channels", 128)),
        num_blocks=int(model_config.get("num_blocks", 8)),
        time_dim=int(model_config.get("time_dim", 256)),
        dropout=float(model_config.get("dropout", 0.0)),
    )
    model = build_trajectory_model(model_args).to(device)
    state = torch.load(checkpoint_dir / "training_state.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if prefer_ema and state.get("ema") is not None:
        shadows = state["ema"]["shadow"]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for parameter, shadow in zip(trainable, shadows):
            parameter.data.copy_(shadow.to(device=device, dtype=parameter.dtype))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(checkpoint_dir / "scheduler")
    return model, scheduler, int(state.get("global_step", 0))


@torch.no_grad()
def sample_starts(real_data_root: Path, count: int, seed: int) -> np.ndarray:
    starts_dir = real_data_root / "starts"
    labels_dir = real_data_root / "labels_absolute"
    sample_ids = sorted(path.stem for path in labels_dir.glob("*.npy"))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(sample_ids), size=count, replace=True)
    starts = []
    for index in indices:
        sample_id = sample_ids[int(index)]
        start_path = starts_dir / f"{sample_id}.npy"
        if start_path.exists():
            starts.append(np.load(start_path).astype(np.float32))
        else:
            starts.append(np.load(labels_dir / f"{sample_id}.npy").astype(np.float32)[0])
    return np.stack(starts, axis=0)


def sample_guided(
    *,
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    guidance: DistanceFieldRoadGuidance,
    mode: str,
    stats: dict[str, Any],
    starts: np.ndarray | None,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    sequence_length: int,
    sample_inference_steps: int,
    generator_seed: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, float | int]:
    batch_path = output_dir / "trajectories_raw_xy.npy"
    if batch_path.exists() and not overwrite:
        return {"loaded_existing": 1, "applied_guidance_steps": 0, "mean_dist_before": 0.0, "mean_dist_after": 0.0}

    scheduler.set_timesteps(sample_inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    trajectory_batch = np.lib.format.open_memmap(
        batch_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, sequence_length, 2),
    )
    raw_sequence_batch = np.lib.format.open_memmap(
        output_dir / f"{mode}_raw_sequences.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, sequence_length, 2),
    )
    if starts is not None:
        np.save(output_dir / "starts_raw_xy.npy", starts.astype(np.float32))

    written = 0
    applied = 0
    skipped = 0
    before_sum = 0.0
    after_sum = 0.0
    metric_count = 0
    while written < num_samples:
        current = min(batch_size, num_samples - written)
        sample = torch.randn((current, 2, sequence_length), generator=generator, device=device)
        starts_batch = None
        if starts is not None:
            starts_batch = torch.from_numpy(starts[written : written + current]).to(device=device, dtype=torch.float32)
        for step_index, timestep in enumerate(scheduler.timesteps):
            with torch.no_grad():
                model_input = scheduler.scale_model_input(sample, timestep)
                noise_pred = model(model_input, torch.full((current,), int(timestep), device=device, dtype=torch.long))
            if guidance.should_apply(step_index):
                guided = guided_noise_prediction(
                    x_t=sample,
                    noise_pred=noise_pred,
                    timestep=timestep,
                    scheduler=scheduler,
                    guidance=guidance,
                    mode=mode,
                    stats=stats,
                    starts=starts_batch,
                )
                if guided["skipped"]:
                    skipped += 1
                else:
                    noise_pred = guided["noise_pred"]
                    applied += 1
                    before_sum += float(guided["mean_dist_before"])
                    after_sum += float(guided["mean_dist_after"])
                    metric_count += 1
            with torch.no_grad():
                sample = scheduler.step(noise_pred, timestep, sample, generator=generator).prev_sample

        raw_sequence = denormalize_tensor(sample.detach().transpose(1, 2), stats)
        if mode == "absolute":
            trajectory = raw_sequence
        else:
            if starts_batch is None:
                raise ValueError("Delta mode requires starts.")
            trajectory = integrate_delta(starts_batch, raw_sequence)
        raw_np = raw_sequence.detach().cpu().numpy().astype(np.float32)
        traj_np = trajectory.detach().cpu().numpy().astype(np.float32)
        raw_sequence_batch[written : written + current] = raw_np
        trajectory_batch[written : written + current] = traj_np
        for local_index, traj in enumerate(traj_np):
            np.save(output_dir / "trajectories_raw_xy" / f"generated_{written + local_index:06d}.npy", traj)
        written += current
        raw_sequence_batch.flush()
        trajectory_batch.flush()
        print(f"{output_dir.name}: wrote {written}/{num_samples}", flush=True)

    del raw_sequence_batch
    del trajectory_batch
    return {
        "loaded_existing": 0,
        "applied_guidance_steps": applied,
        "skipped_guidance_steps": skipped,
        "mean_dist_before": before_sum / max(1, metric_count),
        "mean_dist_after": after_sum / max(1, metric_count),
        "generator_seed": generator_seed,
    }


def guided_noise_prediction(
    *,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: DDPMScheduler,
    guidance: DistanceFieldRoadGuidance,
    mode: str,
    stats: dict[str, Any],
    starts: torch.Tensor | None,
) -> dict[str, Any]:
    alpha_t = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)[timestep].view(1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))
    x0 = ((x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t).detach().transpose(1, 2)
    raw_sequence = denormalize_tensor(x0, stats)
    if mode == "absolute":
        trajectory = raw_sequence
    else:
        if starts is None:
            raise ValueError("Delta mode requires starts.")
        trajectory = integrate_delta(starts, raw_sequence)
    before_energy, before_dist = guidance.road_energy(trajectory)
    guided_trajectory, _, after_dist = guidance(trajectory)
    if not torch.isfinite(guided_trajectory).all():
        return {"noise_pred": noise_pred, "skipped": True}

    if mode == "absolute":
        guided_sequence = guided_trajectory
    else:
        guided_trajectory = guided_trajectory - guided_trajectory[:, :1, :] + starts[:, None, :]
        guided_sequence = torch.zeros_like(raw_sequence)
        guided_sequence[:, 1:, :] = guided_trajectory[:, 1:, :] - guided_trajectory[:, :-1, :]
    x0_guided = normalize_tensor(guided_sequence, stats).transpose(1, 2).detach()
    noise_pred_guided = (x_t - sqrt_alpha_t * x0_guided) / sqrt_one_minus_alpha_t
    return {
        "noise_pred": noise_pred_guided.detach(),
        "skipped": False,
        "mean_dist_before": before_dist.detach().cpu(),
        "mean_dist_after": after_dist.detach().cpu(),
        "energy_before": before_energy.detach().cpu(),
    }


def normalize_tensor(raw: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], dtype=raw.dtype, device=raw.device)
    std = torch.tensor(stats["std"], dtype=raw.dtype, device=raw.device)
    return (raw - mean) / std


def denormalize_tensor(normalized: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], dtype=normalized.dtype, device=normalized.device)
    std = torch.tensor(stats["std"], dtype=normalized.dtype, device=normalized.device)
    return normalized * std + mean


def integrate_delta(starts: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    trajectory = torch.zeros_like(deltas)
    trajectory[:, 0, :] = starts
    trajectory[:, 1:, :] = starts[:, None, :] + torch.cumsum(deltas[:, 1:, :], dim=1)
    return trajectory


def compute_position_stats(real_data_root: Path) -> dict[str, float]:
    mins = np.full(2, np.inf, dtype=np.float64)
    maxs = np.full(2, -np.inf, dtype=np.float64)
    for path in sorted((real_data_root / "labels_absolute").glob("*.npy")):
        xy = np.load(path).astype(np.float64)
        mins = np.minimum(mins, xy.min(axis=0))
        maxs = np.maximum(maxs, xy.max(axis=0))
    return {"x_min": float(mins[0]), "x_max": float(maxs[0]), "y_min": float(mins[1]), "y_max": float(maxs[1])}


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="checkpoint-final")
    parser.add_argument("--real-data-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "week06" / "decoded" / "raw_guided")
    parser.add_argument(
        "--distance-field-dir",
        type=Path,
        default=PROJECT_ROOT / "week06" / "results" / "empirical_support_distance_field",
    )
    parser.add_argument("--guidance-strength", type=float, default=0.1)
    parser.add_argument("--num-guidance-steps", type=int, default=2)
    parser.add_argument("--guidance-start-fraction", type=float, default=0.5)
    parser.add_argument("--guidance-end-fraction", type=float, default=1.0)
    parser.add_argument("--distance-scale", type=float, default=25.0)
    parser.add_argument("--distance-field-size", type=int, default=1024)
    parser.add_argument("--support-radius-m", type=float, default=5.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-inference-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-distance-field", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Shared utilities for raw Week06 trajectory diffusion scripts."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader


@dataclass
class SequenceNormStats:
    mean: list[float]
    std: list[float]
    value_min: list[float]
    value_max: list[float]
    source: str


class RawTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        mode: str,
        split: str,
        sequence_length: int,
        stats: SequenceNormStats | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.mode = mode
        self.split = split
        self.sequence_length = int(sequence_length)
        self.data_dir = self.data_root / data_subdir_for_mode(mode)
        self.start_dir = self.data_root / "starts"
        self.sample_ids = resolve_sample_ids(self.data_root, split, self.data_dir)
        if not self.sample_ids:
            raise ValueError(f"No sample ids found for split={split!r} in {self.data_root}")
        self.stats = stats if stats is not None else compute_sequence_norm_stats(self.data_dir, self.sample_ids, mode)
        self._validate_first_sample()

    def _validate_first_sample(self) -> None:
        array = self.load_raw_xy(self.sample_ids[0])
        if array.shape != (self.sequence_length, 2):
            raise ValueError(
                f"Expected shape ({self.sequence_length}, 2), got {array.shape} "
                f"in {self.data_dir / f'{self.sample_ids[0]}.npy'}"
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        raw_xy = self.load_raw_xy(self.sample_ids[index])
        normalized = normalize_xy(raw_xy, self.stats)
        return torch.from_numpy(normalized.T.astype(np.float32)).contiguous()

    def load_raw_xy(self, sample_id: str) -> np.ndarray:
        path = self.data_dir / f"{sample_id}.npy"
        array = np.load(path).astype(np.float32)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Expected {path} to have shape N,2, got {array.shape}")
        return array

    def load_start_xy(self, sample_id: str) -> np.ndarray:
        start_path = self.start_dir / f"{sample_id}.npy"
        if start_path.exists():
            start_xy = np.load(start_path).astype(np.float32)
            if start_xy.shape == (2,):
                return start_xy
        absolute_path = self.data_root / "labels_absolute" / f"{sample_id}.npy"
        return np.load(absolute_path).astype(np.float32)[0]


class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        args = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LearnableTemporalPositionEmbedding(torch.nn.Module):
    def __init__(self, channels: int, sequence_length: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(1, channels, sequence_length))
        torch.nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x_value: torch.Tensor) -> torch.Tensor:
        if x_value.shape[-1] != self.embedding.shape[-1]:
            raise ValueError(
                f"Position embedding length {self.embedding.shape[-1]} does not match input length {x_value.shape[-1]}"
            )
        return x_value + self.embedding


class ResidualTemporalBlock(torch.nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float, dilation: int = 1) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.norm1 = torch.nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv1 = torch.nn.Conv1d(channels, channels, kernel_size=3, padding=self.dilation, dilation=self.dilation)
        self.time_proj = torch.nn.Linear(time_dim, channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=3, padding=self.dilation, dilation=self.dilation)

    def forward(self, x_value: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = x_value
        x_value = self.conv1(F.silu(self.norm1(x_value)))
        x_value = x_value + self.time_proj(F.silu(time_embedding))[:, :, None]
        x_value = self.conv2(self.dropout(F.silu(self.norm2(x_value))))
        return x_value + residual


class TrajectoryDenoiser1D(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int = 224,
        in_channels: int = 2,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        time_dim: int = 256,
        dropout: float = 0.0,
        dilation_pattern: list[int] | None = None,
        model_type: str = "temporal_resnet",
    ) -> None:
        super().__init__()
        dilations = resolve_dilation_pattern(num_blocks, dilation_pattern)
        self.config = {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_blocks": num_blocks,
            "time_dim": time_dim,
            "dropout": dropout,
            "dilation_pattern": dilations,
            "position_embedding": "learnable",
        }
        self.input_proj = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.position_embedding = LearnableTemporalPositionEmbedding(hidden_channels, sequence_length)
        self.time_embedding = torch.nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, time_dim),
        )
        self.blocks = torch.nn.ModuleList(
            [
                ResidualTemporalBlock(hidden_channels, time_dim, dropout, dilation=dilations[index])
                for index in range(num_blocks)
            ]
        )
        self.output_norm = torch.nn.GroupNorm(num_groups=8, num_channels=hidden_channels)
        self.output_proj = torch.nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        input_shape = sample.shape
        time_embedding = self.time_embedding(timesteps)
        x_value = self.position_embedding(self.input_proj(sample))
        for block in self.blocks:
            x_value = block(x_value, time_embedding)
        output = self.output_proj(F.silu(self.output_norm(x_value)))
        ensure_same_shape(output, input_shape, self.config["model_type"])
        return output


class Downsample1D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x_value: torch.Tensor) -> torch.Tensor:
        return self.conv(x_value)


class Upsample1D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x_value: torch.Tensor) -> torch.Tensor:
        return self.conv(x_value)


class UNetUpBlock1D(torch.nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.upsample = Upsample1D(in_channels, skip_channels)
        self.reduce = torch.nn.Conv1d(skip_channels * 2, skip_channels, kernel_size=1)
        self.block = ResidualTemporalBlock(skip_channels, time_dim, dropout)

    def forward(self, x_value: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x_value = self.upsample(x_value)
        if x_value.shape[-1] != skip.shape[-1]:
            raise ValueError(f"UNet upsample length {x_value.shape[-1]} does not match skip length {skip.shape[-1]}")
        x_value = torch.cat([x_value, skip], dim=1)
        x_value = self.reduce(x_value)
        return self.block(x_value, time_embedding)


class TrajectoryUNet1D(torch.nn.Module):
    def __init__(
        self,
        sequence_length: int = 224,
        in_channels: int = 2,
        hidden_channels: int = 128,
        time_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if sequence_length % 8 != 0:
            raise ValueError(f"TrajectoryUNet1D requires sequence_length divisible by 8, got {sequence_length}")
        channels = [hidden_channels, hidden_channels * 2, hidden_channels * 4, hidden_channels * 4]
        self.config = {
            "model_type": "unet1d",
            "sequence_length": sequence_length,
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "time_dim": time_dim,
            "dropout": dropout,
            "downsample_lengths": [sequence_length, sequence_length // 2, sequence_length // 4, sequence_length // 8],
            "channel_mults": [1, 2, 4, 4],
            "position_embedding": "learnable",
        }
        self.input_proj = torch.nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)
        self.position_embedding = LearnableTemporalPositionEmbedding(channels[0], sequence_length)
        self.time_embedding = torch.nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, time_dim),
        )
        self.down_block1 = ResidualTemporalBlock(channels[0], time_dim, dropout)
        self.downsample1 = Downsample1D(channels[0], channels[1])
        self.down_block2 = ResidualTemporalBlock(channels[1], time_dim, dropout)
        self.downsample2 = Downsample1D(channels[1], channels[2])
        self.down_block3 = ResidualTemporalBlock(channels[2], time_dim, dropout)
        self.downsample3 = Downsample1D(channels[2], channels[3])
        self.bottleneck1 = ResidualTemporalBlock(channels[3], time_dim, dropout, dilation=2)
        self.bottleneck2 = ResidualTemporalBlock(channels[3], time_dim, dropout, dilation=4)
        self.up_block3 = UNetUpBlock1D(channels[3], channels[2], time_dim, dropout)
        self.up_block2 = UNetUpBlock1D(channels[2], channels[1], time_dim, dropout)
        self.up_block1 = UNetUpBlock1D(channels[1], channels[0], time_dim, dropout)
        self.output_norm = torch.nn.GroupNorm(num_groups=8, num_channels=channels[0])
        self.output_proj = torch.nn.Conv1d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        input_shape = sample.shape
        time_embedding = self.time_embedding(timesteps)
        x0 = self.position_embedding(self.input_proj(sample))
        skip1 = self.down_block1(x0, time_embedding)
        x1 = self.downsample1(skip1)
        skip2 = self.down_block2(x1, time_embedding)
        x2 = self.downsample2(skip2)
        skip3 = self.down_block3(x2, time_embedding)
        x3 = self.downsample3(skip3)
        x3 = self.bottleneck1(x3, time_embedding)
        x3 = self.bottleneck2(x3, time_embedding)
        x_value = self.up_block3(x3, skip3, time_embedding)
        x_value = self.up_block2(x_value, skip2, time_embedding)
        x_value = self.up_block1(x_value, skip1, time_embedding)
        output = self.output_proj(F.silu(self.output_norm(x_value)))
        ensure_same_shape(output, input_shape, self.config["model_type"])
        return output


def resolve_dilation_pattern(num_blocks: int, dilation_pattern: list[int] | None) -> list[int]:
    if dilation_pattern is None:
        return [1] * num_blocks
    if not dilation_pattern:
        raise ValueError("dilation_pattern cannot be empty")
    return [int(dilation_pattern[index % len(dilation_pattern)]) for index in range(num_blocks)]


def build_trajectory_model(args: Any) -> torch.nn.Module:
    model_type = getattr(args, "model_type", "temporal_resnet")
    if model_type == "temporal_resnet":
        return TrajectoryDenoiser1D(
            sequence_length=args.sequence_length,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            time_dim=args.time_dim,
            dropout=args.dropout,
            dilation_pattern=None,
            model_type=model_type,
        )
    if model_type == "temporal_resnet_dilated":
        return TrajectoryDenoiser1D(
            sequence_length=args.sequence_length,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            time_dim=args.time_dim,
            dropout=args.dropout,
            dilation_pattern=[1, 2, 4, 8, 16, 32, 1, 2],
            model_type=model_type,
        )
    if model_type == "unet1d":
        return TrajectoryUNet1D(
            sequence_length=args.sequence_length,
            hidden_channels=args.hidden_channels,
            time_dim=args.time_dim,
            dropout=args.dropout,
        )
    raise ValueError(
        f"Unsupported model_type={model_type!r}. "
        "Expected one of: temporal_resnet, temporal_resnet_dilated, unet1d."
    )


def ensure_same_shape(output: torch.Tensor, expected_shape: torch.Size, model_type: str) -> None:
    if output.shape != expected_shape:
        raise RuntimeError(f"{model_type} output shape {tuple(output.shape)} does not match input shape {tuple(expected_shape)}")


class SimpleEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for shadow, parameter in zip(self.shadow, trainable):
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> list[torch.Tensor]:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        backup = [parameter.detach().clone() for parameter in trainable]
        for parameter, shadow in zip(trainable, self.shadow):
            parameter.copy_(shadow)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: list[torch.Tensor]) -> None:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        for parameter, value in zip(trainable, backup):
            parameter.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow = state_dict["shadow"]


def train_raw_sequence_diffusion(args: Any, mode: str) -> None:
    set_seed(args.seed)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    fail_early(args, mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ids = resolve_sample_ids(args.data_root, "train", args.data_root / data_subdir_for_mode(mode))
    stats = load_or_compute_stats(args, mode, train_ids)
    dataset = RawTrajectoryDataset(
        data_root=args.data_root,
        mode=mode,
        split=args.split,
        sequence_length=args.sequence_length,
        stats=stats,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    model = build_trajectory_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    ema = SimpleEMA(model, args.ema_decay) if args.use_ema else None
    write_startup_config(args.output_dir / "startup_config.json", args, mode, dataset, stats, model, device)

    log_path = args.output_dir / "train_log.jsonl"
    start_time = time.perf_counter()
    global_step = 0
    running_loss = 0.0
    steps_since_log = 0
    batches = cycle(dataloader)
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < args.max_train_steps:
        for _ in range(args.gradient_accumulation_steps):
            clean = next(batches).to(device)
            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy = scheduler.add_noise(clean, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            ensure_same_shape(noise_pred, clean.shape, model.config["model_type"])
            loss = F.mse_loss(noise_pred, noise)
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += float(loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)

        global_step += 1
        steps_since_log += 1
        if global_step == 1 or global_step % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            record = {
                "step": global_step,
                "loss": running_loss / max(1, steps_since_log * args.gradient_accumulation_steps),
                "lr": lr_scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": global_step / elapsed if elapsed > 0 else None,
                "device": str(device),
                "mode": mode,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step={global_step} loss={record['loss']:.6f} "
                f"lr={record['lr']:.8g} steps_per_second={record['steps_per_second']:.4f}",
                flush=True,
            )
            running_loss = 0.0
            steps_since_log = 0

        if args.sample_every > 0 and global_step % args.sample_every == 0:
            sample_sequences(args, mode, model, scheduler, ema, stats, dataset, args.output_dir / "samples", global_step, device)
        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(args.output_dir / f"checkpoint-{global_step}", model, scheduler, optimizer, lr_scheduler, ema, global_step, args, mode, stats)

    save_checkpoint(args.output_dir / "checkpoint-final", model, scheduler, optimizer, lr_scheduler, ema, global_step, args, mode, stats)
    sample_sequences(args, mode, model, scheduler, ema, stats, dataset, args.output_dir / "samples", global_step, device)
    write_train_summary(args.output_dir / "train_summary.json", args, mode, dataset, stats, model.config, global_step)


@torch.no_grad()
def sample_sequences(
    args: Any,
    mode: str,
    model: TrajectoryDenoiser1D,
    scheduler: DDPMScheduler,
    ema: SimpleEMA | None,
    stats: SequenceNormStats,
    dataset: RawTrajectoryDataset,
    output_dir: Path,
    global_step: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backup = ema.copy_to(model) if ema is not None else None
    model.eval()
    sample = torch.randn(args.num_sample_sequences, 2, args.sequence_length, device=device)
    inference_scheduler = DDPMScheduler(num_train_timesteps=scheduler.config.num_train_timesteps)
    inference_scheduler.set_timesteps(args.sample_inference_steps)
    for timestep in inference_scheduler.timesteps:
        batch_t = torch.full((sample.shape[0],), int(timestep), device=device, dtype=torch.long)
        noise_pred = model(sample, batch_t)
        ensure_same_shape(noise_pred, sample.shape, model.config["model_type"])
        sample = inference_scheduler.step(noise_pred, timestep, sample).prev_sample
    raw_sequences = denormalize_xy(sample.detach().cpu().numpy().transpose(0, 2, 1), stats)
    np.save(output_dir / f"step_{global_step:06d}_{mode}_raw.npy", raw_sequences.astype(np.float32))
    if mode == "delta":
        starts = sample_starts(dataset, args.num_sample_sequences, args.seed + global_step)
        trajectories = integrate_delta_batch(starts, raw_sequences)
        np.save(output_dir / f"step_{global_step:06d}_delta_integrated_trajectories.npy", trajectories.astype(np.float32))
        np.save(output_dir / f"step_{global_step:06d}_sampled_starts.npy", starts.astype(np.float32))
    model.train()
    if ema is not None and backup is not None:
        ema.restore(model, backup)


def save_checkpoint(
    path: Path,
    model: TrajectoryDenoiser1D,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    ema: SimpleEMA | None,
    global_step: int,
    args: Any,
    mode: str,
    stats: SequenceNormStats,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "global_step": global_step,
            "args": vars(args),
            "mode": mode,
            "model_type": model.config["model_type"],
            "normalization_stats": asdict(stats),
            "scheduler_config": dict(scheduler.config),
        },
        path / "training_state.pt",
    )
    scheduler.save_config(path / "scheduler")


def normalize_xy(xy: np.ndarray, stats: SequenceNormStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return (xy.astype(np.float32) - mean) / std


def denormalize_xy(normalized: np.ndarray, stats: SequenceNormStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return normalized.astype(np.float32) * std + mean


def compute_sequence_norm_stats(data_dir: Path, sample_ids: list[str], mode: str) -> SequenceNormStats:
    sum_xy = np.zeros(2, dtype=np.float64)
    sum_sq_xy = np.zeros(2, dtype=np.float64)
    min_xy = np.full(2, np.inf, dtype=np.float64)
    max_xy = np.full(2, -np.inf, dtype=np.float64)
    count = 0
    for sample_id in sample_ids:
        xy = np.load(data_dir / f"{sample_id}.npy").astype(np.float64)
        sum_xy += xy.sum(axis=0)
        sum_sq_xy += np.square(xy).sum(axis=0)
        min_xy = np.minimum(min_xy, xy.min(axis=0))
        max_xy = np.maximum(max_xy, xy.max(axis=0))
        count += len(xy)
    if count == 0:
        raise ValueError(f"No values found under {data_dir}")
    mean = sum_xy / count
    variance = np.maximum(sum_sq_xy / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return SequenceNormStats(
        mean=mean.tolist(),
        std=std.tolist(),
        value_min=min_xy.tolist(),
        value_max=max_xy.tolist(),
        source=f"{mode}_train_split",
    )


def load_or_compute_stats(args: Any, mode: str, train_ids: list[str]) -> SequenceNormStats:
    if args.normalization_file is not None:
        payload = json.loads(args.normalization_file.expanduser().resolve().read_text())
        if {"dx_mean", "dx_std", "dy_mean", "dy_std"}.issubset(payload):
            return SequenceNormStats(
                mean=[float(payload["dx_mean"]), float(payload["dy_mean"])],
                std=[float(payload["dx_std"]), float(payload["dy_std"])],
                value_min=[float(payload.get("dx_min", 0.0)), float(payload.get("dy_min", 0.0))],
                value_max=[float(payload.get("dx_max", 0.0)), float(payload.get("dy_max", 0.0))],
                source=str(args.normalization_file),
            )
        return SequenceNormStats(**payload)
    if mode == "delta":
        diagnostics_path = args.data_root / "normalization_diagnostics.json"
        if diagnostics_path.exists():
            payload = json.loads(diagnostics_path.read_text())
            stats = SequenceNormStats(
                mean=[float(payload["dx_mean"]), float(payload["dy_mean"])],
                std=[float(payload["dx_std"]), float(payload["dy_std"])],
                value_min=[float(payload["dx_min"]), float(payload["dy_min"])],
                value_max=[float(payload["dx_max"]), float(payload["dy_max"])],
                source=str(diagnostics_path),
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "normalization_stats.json").write_text(json.dumps(asdict(stats), indent=2) + "\n")
            return stats
    stats = compute_sequence_norm_stats(args.data_root / data_subdir_for_mode(mode), train_ids, mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "normalization_stats.json").write_text(json.dumps(asdict(stats), indent=2) + "\n")
    return stats


def resolve_sample_ids(data_root: Path, split: str, data_dir: Path) -> list[str]:
    if split == "all":
        return sorted(path.stem for path in data_dir.glob("*.npy"))
    split_path = data_root / "splits" / f"{split}_ids.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{split_path} must contain a sample_id column.")
        return [row["sample_id"] for row in reader if row.get("sample_id")]


def data_subdir_for_mode(mode: str) -> str:
    if mode == "absolute":
        return "labels_absolute"
    if mode == "delta":
        return "labels_delta_displacement"
    raise ValueError(f"Unsupported mode: {mode}")


def sample_starts(dataset: RawTrajectoryDataset, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset.sample_ids), size=count, replace=True)
    return np.stack([dataset.load_start_xy(dataset.sample_ids[int(index)]) for index in indices], axis=0)


def integrate_delta_batch(starts: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    trajectories = np.zeros_like(deltas, dtype=np.float32)
    trajectories[:, 0, :] = starts.astype(np.float32)
    trajectories[:, 1:, :] = starts[:, None, :].astype(np.float32) + np.cumsum(deltas[:, 1:, :], axis=1)
    return trajectories


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fail_early(args: Any, mode: str) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    data_dir = args.data_root / data_subdir_for_mode(mode)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not (args.data_root / "splits").is_dir():
        raise FileNotFoundError(args.data_root / "splits")
    if args.sequence_length != 224:
        raise ValueError("Week06 raw trajectory diffusion scripts expect sequence_length=224.")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; pass --cpu only for smoke tests.")


def write_startup_config(
    path: Path,
    args: Any,
    mode: str,
    dataset: RawTrajectoryDataset,
    stats: SequenceNormStats,
    model: TrajectoryDenoiser1D,
    device: torch.device,
) -> None:
    payload = {
        "mode": mode,
        "data_format": f"raw_{mode}_trajectory_sequence",
        "gasf_used": False,
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "num_inputs": len(dataset),
        "sequence_length": args.sequence_length,
        "channels": 2,
        "model_type": model.config["model_type"],
        "normalization": asdict(stats),
        "model_config": model.config,
        "scheduler": "DDPMScheduler",
        "num_train_timesteps": args.num_train_timesteps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def write_train_summary(
    path: Path,
    args: Any,
    mode: str,
    dataset: RawTrajectoryDataset,
    stats: SequenceNormStats,
    model_config: dict[str, Any],
    global_step: int,
) -> None:
    payload = {
        "global_step": global_step,
        "mode": mode,
        "data_format": f"raw_{mode}_trajectory_sequence",
        "gasf_used": False,
        "num_inputs": len(dataset),
        "sequence_length": args.sequence_length,
        "channels": 2,
        "model_type": model_config["model_type"],
        "normalization": asdict(stats),
        "model_config": model_config,
        "output_dir": str(args.output_dir),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def copy_script_to_output(script_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, output_dir / script_path.name)

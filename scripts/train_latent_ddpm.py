"""Train a DDPM model directly on autoencoder latent tensors."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


class LatentTensorDataset(Dataset):
    """Load CHW .npy latent tensors."""

    def __init__(self, data_dir: str | Path, latent_channels: int = 16, sample_size: int = 14) -> None:
        root = Path(data_dir)
        if (root / "latent").exists():
            latent_dir = root / "latent"
            stats_root = root
        else:
            latent_dir = root
            stats_root = root.parent
        self.paths = sorted(latent_dir.glob("*.npy"))
        if not self.paths:
            raise ValueError(f"No .npy latent tensors found in {latent_dir}")
        self.latent_channels = latent_channels
        self.sample_size = sample_size
        self.normalization_stats = load_latent_stats(stats_root / "latent_stats.json")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        array = np.load(self.paths[index]).astype(np.float32)
        expected_shape = (self.latent_channels, self.sample_size, self.sample_size)
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected CHW latent tensor with shape {expected_shape}, got {array.shape} in {self.paths[index]}"
            )
        if self.normalization_stats["enabled"]:
            mean = self.normalization_stats["mean"]
            std = self.normalization_stats["std"]
            array = (array - mean) / (std + 1e-6)
        return torch.from_numpy(array)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LatentTensorDataset(args.data_dir, latent_channels=args.latent_channels, sample_size=args.sample_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = build_latent_unet(args).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    global_step = 0
    if args.resume is not None:
        global_step, scheduler = load_checkpoint(args.resume, model, optimizer, lr_scheduler, device)

    recent_logged_losses: deque[float] = deque(maxlen=args.best_loss_window)
    best_rolling_loss: float | None = None
    batches = cycle(dataloader)
    while global_step < args.max_train_steps:
        clean_latents = next(batches).to(device)
        noise = torch.randn_like(clean_latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (clean_latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)
        noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()

        global_step += 1
        if global_step % args.log_every == 0 or global_step == 1:
            record = {"step": global_step, "loss": float(loss.detach().cpu()), "lr": lr_scheduler.get_last_lr()[0]}
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(record) + "\n")
            print(f"step={global_step} loss={record['loss']:.6f} lr={record['lr']:.8f}")
            recent_logged_losses.append(record["loss"])
            rolling_loss = sum(recent_logged_losses) / len(recent_logged_losses)
            if args.save_best and (best_rolling_loss is None or rolling_loss < best_rolling_loss):
                best_rolling_loss = rolling_loss
                save_checkpoint(
                    output_dir / "checkpoint-best",
                    model,
                    scheduler,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    args,
                    best_rolling_loss=best_rolling_loss,
                )

        if args.sample_every > 0 and global_step % args.sample_every == 0:
            save_latent_samples(model, scheduler, output_dir / "samples", global_step, device, args)

        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint-{global_step}",
                model,
                scheduler,
                optimizer,
                lr_scheduler,
                global_step,
                args,
                best_rolling_loss=best_rolling_loss,
            )
            prune_numbered_checkpoints(output_dir, keep_last=args.keep_last_checkpoints)

    save_checkpoint(
        output_dir / "checkpoint-final",
        model,
        scheduler,
        optimizer,
        lr_scheduler,
        global_step,
        args,
        best_rolling_loss=best_rolling_loss,
    )
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "num_latents": len(dataset),
                "sample_size": args.sample_size,
                "latent_channels": args.latent_channels,
                "normalization_stats": dataset.normalization_stats,
                "best_rolling_loss": best_rolling_loss,
                "best_loss_window": args.best_loss_window,
            },
            indent=2,
        )
    )


def build_latent_unet(args: argparse.Namespace) -> UNet2DModel:
    return UNet2DModel(
        sample_size=args.sample_size,
        in_channels=args.latent_channels,
        out_channels=args.latent_channels,
        layers_per_block=args.layers_per_block,
        block_out_channels=parse_int_tuple(args.block_out_channels),
        down_block_types=parse_str_tuple(args.down_block_types),
        up_block_types=parse_str_tuple(args.up_block_types),
        attention_head_dim=args.attention_head_dim,
        norm_num_groups=args.norm_num_groups,
    )


def save_latent_samples(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    output_dir: Path,
    step: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_latents(model, scheduler, args.num_sample_latents, device, args)
    np.save(output_dir / f"latent_step_{step:06d}.npy", samples.cpu().numpy())
    preview = samples[:, :3]
    preview = (preview - preview.amin(dim=(1, 2, 3), keepdim=True)) / (
        preview.amax(dim=(1, 2, 3), keepdim=True) - preview.amin(dim=(1, 2, 3), keepdim=True) + 1e-8
    )
    save_image(make_grid(preview, nrow=min(4, len(preview))), output_dir / f"latent_step_{step:06d}.png")
    model.train()


def sample_latents(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    model.eval()
    latents = torch.randn(batch_size, args.latent_channels, args.sample_size, args.sample_size, device=device)
    scheduler.set_timesteps(args.sample_inference_steps, device=device)
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            noise_pred = model(latents, timestep, return_dict=False)[0]
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents.detach().cpu()


def save_checkpoint(
    path: Path,
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
    args: argparse.Namespace,
    best_rolling_loss: float | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "unet")
    scheduler.save_pretrained(path / "scheduler")
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": serialize_args(args),
            "best_rolling_loss": best_rolling_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "training_state.pt",
    )


def prune_numbered_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last < 0:
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    checkpoints.sort()
    for _, path in checkpoints[:-keep_last]:
        remove_tree(path)


def remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def load_latent_stats(path: Path) -> dict[str, float | bool | str | None]:
    if not path.exists():
        return {"enabled": False, "path": None, "mean": 0.0, "std": 1.0, "min": None, "max": None}
    stats = json.loads(path.read_text())
    return {
        "enabled": True,
        "path": str(path),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "min": float(stats["min"]) if stats.get("min") is not None else None,
        "max": float(stats["max"]) if stats.get("max") is not None else None,
    }


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def load_checkpoint(
    path: Path,
    model: UNet2DModel,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> tuple[int, DDPMScheduler]:
    checkpoint = path.resolve()
    loaded_model = UNet2DModel.from_pretrained(checkpoint / "unet")
    model.load_state_dict(loaded_model.state_dict())
    loaded_scheduler = DDPMScheduler.from_pretrained(checkpoint / "scheduler")
    state = torch.load(checkpoint / "training_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    if "rng_state" in state:
        torch.set_rng_state(state["rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state.get("global_step", 0)), loaded_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing latent/*.npy tensors.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/latent_ddpm"), help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--max-train-steps", type=int, default=10000, help="Total optimizer steps.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps; 0 disables.")
    parser.add_argument("--sample-every", type=int, default=1000, help="Save latent samples every N steps; 0 disables.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--sample-size", type=int, default=28, help="Latent spatial size.")
    parser.add_argument("--latent-channels", type=int, default=32, help="Latent channel count.")
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="DDPM training timesteps.")
    parser.add_argument("--sample-inference-steps", type=int, default=50, help="Reverse diffusion steps for previews.")
    parser.add_argument("--num-sample-latents", type=int, default=4, help="Number of preview latent samples.")
    parser.add_argument("--layers-per-block", type=int, default=2, help="UNet residual layers per block.")
    parser.add_argument("--block-out-channels", type=str, default="64,128", help="Comma-separated UNet block widths.")
    parser.add_argument(
        "--down-block-types",
        type=str,
        default="DownBlock2D,AttnDownBlock2D",
        help="Comma-separated diffusers down block types.",
    )
    parser.add_argument(
        "--up-block-types",
        type=str,
        default="AttnUpBlock2D,UpBlock2D",
        help="Comma-separated diffusers up block types.",
    )
    parser.add_argument("--attention-head-dim", type=int, default=8, help="Attention head dimension.")
    parser.add_argument("--norm-num-groups", type=int, default=8, help="GroupNorm groups.")
    parser.add_argument("--keep-last-checkpoints", type=int, default=8, help="Keep the last N numbered checkpoints.")
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save checkpoint-best using rolling logged loss.",
    )
    parser.add_argument("--best-loss-window", type=int, default=20, help="Number of logged losses in best-loss average.")
    parser.add_argument("--lr-warmup-steps", type=int, default=500, help="Cosine schedule warmup steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

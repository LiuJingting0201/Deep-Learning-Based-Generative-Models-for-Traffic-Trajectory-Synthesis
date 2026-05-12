"""Train a DDPM model on a directory of RGB PNG images.

The notebook this replaces used local checkpoint paths and one-off cells. This
script trains from scratch by default, saves reproducible checkpoints, and can
resume from a checkpoint directory created by this script.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    diag_targets = resolve_diag_targets(dataset, args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    model = build_unet(args.image_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    ema_model = EMAModel(model.parameters(), model_cls=UNet2DModel, model_config=model.config) if args.use_ema else None

    global_step = 0
    if args.resume_from_checkpoint:
        global_step, scheduler = load_checkpoint(
            args.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            device=device,
        )

    model.train()
    running_components = {
        "total_loss": 0.0,
        "mse_loss": 0.0,
        "symmetry_loss": 0.0,
        "diag_loss": 0.0,
    }
    steps_since_log = 0
    batches = cycle(dataloader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for _ in range(args.gradient_accumulation_steps):
            clean_images = next(batches).to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            components = compute_loss_components(
                noisy_images,
                noise_pred,
                noise,
                timesteps,
                scheduler,
                args,
                diag_targets,
            )
            loss = components["total_loss"]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            for name, value in components.items():
                running_components[name] += float(value.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema_model is not None:
            ema_model.step(model.parameters())

        global_step += 1
        steps_since_log += 1
        if global_step % args.log_every == 0 or global_step == 1:
            denom = max(1, steps_since_log * args.gradient_accumulation_steps)
            avg_components = {name: value / denom for name, value in running_components.items()}
            avg_loss = avg_components["total_loss"]
            elapsed = time.perf_counter() - start_time
            log_record = {
                "step": global_step,
                "loss": avg_loss,
                "total_loss": avg_components["total_loss"],
                "mse_loss": avg_components["mse_loss"],
                "symmetry_loss": avg_components["symmetry_loss"],
                "diag_loss": avg_components["diag_loss"],
                "aux_loss_mode": args.aux_loss_mode,
                "diag_targets": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
                "lr": lr_scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": global_step / elapsed if elapsed > 0 else None,
                "device": str(device),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else None,
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2) if torch.cuda.is_available() else None,
            }
            with log_path.open("a") as log_file:
                log_file.write(json.dumps(log_record) + "\n")
            print(
                f"step={global_step} loss={avg_loss:.6f} "
                f"lr={lr_scheduler.get_last_lr()[0]:.8f} "
                f"steps_per_second={log_record['steps_per_second']:.4f}"
            )
            running_components = {name: 0.0 for name in running_components}
            steps_since_log = 0

        if args.sample_every > 0 and global_step % args.sample_every == 0:
            save_samples(model, scheduler, ema_model, output_dir / "samples", global_step, device, args)

        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint-{global_step}", model, scheduler, optimizer, lr_scheduler, ema_model, global_step, args)

    save_checkpoint(output_dir / "checkpoint-final", model, scheduler, optimizer, lr_scheduler, ema_model, global_step, args)
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "num_images": len(dataset),
                "image_size": args.image_size,
                "aux_loss_mode": args.aux_loss_mode,
                "symmetry_weight": args.symmetry_weight,
                "diag_weight": args.diag_weight,
                "disable_symmetry_loss": args.disable_symmetry_loss,
                "diag_targets": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
            },
            indent=2,
        )
    )


def build_unet(image_size: int) -> UNet2DModel:
    if image_size < 64:
        return UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        )
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def compute_loss_components(
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    args: argparse.Namespace | None = None,
    diag_targets: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if args is None:
        args = argparse.Namespace(
            aux_loss_mode="notebook",
            symmetry_weight=0.01,
            diag_weight=0.01,
            disable_symmetry_loss=False,
        )
    return compute_configurable_loss(noisy_images, noise_pred, noise, timesteps, scheduler, args, diag_targets)


def compute_configurable_loss(
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    args: argparse.Namespace,
    diag_targets: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Compute DDPM MSE plus optional structure losses."""
    mse_loss = F.mse_loss(noise_pred, noise)
    zero = mse_loss.new_tensor(0.0)
    if args.aux_loss_mode == "none":
        return {"total_loss": mse_loss, "mse_loss": mse_loss, "symmetry_loss": zero, "diag_loss": zero}

    alpha_t = scheduler.alphas_cumprod.to(noisy_images.device)[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    x0_pred = (noisy_images - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    x0_pred = torch.clamp(x0_pred, -1, 1)

    if args.aux_loss_mode == "notebook":
        second_channel = x0_pred[:, 1, :, :]
        diag_values = torch.diagonal(second_channel, dim1=-2, dim2=-1)
        target = 128 / 127.5 - 1
        diag_loss = ((diag_values - target) ** 2).mean()

        if args.disable_symmetry_loss:
            symmetry_loss = zero
        else:
            first_channel = x0_pred[:, 0, :, :]
            symmetry_loss = torch.mean((first_channel - first_channel.transpose(-1, -2)) ** 2)
        total_loss = mse_loss + args.symmetry_weight * symmetry_loss + args.diag_weight * diag_loss
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "symmetry_loss": symmetry_loss,
            "diag_loss": diag_loss,
        }

    if args.aux_loss_mode == "data_driven_diag":
        if diag_targets is None:
            raise ValueError("diag_targets must be provided for aux_loss_mode=data_driven_diag")
        diag_values = torch.diagonal(x0_pred, dim1=-2, dim2=-1)
        target = diag_targets.to(device=x0_pred.device, dtype=x0_pred.dtype).view(1, 3, 1)
        diag_loss = ((diag_values - target) ** 2).mean()
        total_loss = mse_loss + args.diag_weight * diag_loss
        return {"total_loss": total_loss, "mse_loss": mse_loss, "symmetry_loss": zero, "diag_loss": diag_loss}

    raise ValueError(f"Unsupported aux_loss_mode: {args.aux_loss_mode}")


def compute_notebook_loss(
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
) -> torch.Tensor:
    """Match the original notebook loss exactly; kept for compatibility."""
    return compute_loss_components(noisy_images, noise_pred, noise, timesteps, scheduler)["total_loss"]


def resolve_diag_targets(dataset: DiffusionImageDataset, args: argparse.Namespace) -> torch.Tensor | None:
    if args.aux_loss_mode == "notebook":
        return torch.tensor([128 / 127.5 - 1], dtype=torch.float32)
    if args.aux_loss_mode != "data_driven_diag":
        return None
    channel_sum = torch.zeros(3, dtype=torch.float64)
    total = 0
    for image in dataset:
        diag_values = torch.diagonal(image, dim1=-2, dim2=-1)
        channel_sum += diag_values.double().sum(dim=1)
        total += diag_values.shape[1]
    return (channel_sum / max(1, total)).float()


def save_samples(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    ema_model: EMAModel | None,
    output_dir: Path,
    step: int,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_model = model
    if ema_model is not None:
        inference_model = build_unet(args.image_size).to(device)
        ema_model.copy_to(inference_model.parameters())
        inference_model.eval()

    pipeline = DDPMPipeline(unet=inference_model, scheduler=scheduler).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + step)
    images = pipeline(
        batch_size=args.num_sample_images,
        generator=generator,
        num_inference_steps=args.sample_inference_steps,
    ).images
    for index, image in enumerate(images):
        image.save(output_dir / f"step_{step:06d}_{index:03d}.png")
    model.train()


def save_checkpoint(
    path: Path,
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema_model: EMAModel | None,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "unet")
    scheduler.save_pretrained(path / "scheduler")
    torch.save(
        {
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(args),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "training_state.pt",
    )
    if ema_model is not None:
        ema_model.save_pretrained(path / "ema_unet")


def load_checkpoint(
    path: Path,
    model: UNet2DModel,
    optimizer: AdamW,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema_model: EMAModel | None,
    device: torch.device,
) -> tuple[int, DDPMScheduler]:
    checkpoint = path.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    loaded_model = UNet2DModel.from_pretrained(checkpoint / "unet")
    model.load_state_dict(loaded_model.state_dict())
    loaded_scheduler = DDPMScheduler.from_pretrained(checkpoint / "scheduler")

    state = torch.load(checkpoint / "training_state.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    if ema_model is not None and (checkpoint / "ema_unet").exists():
        loaded_ema = EMAModel.from_pretrained(checkpoint / "ema_unet", UNet2DModel)
        ema_model.load_state_dict(loaded_ema.state_dict())
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
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing PNG training images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, samples, and logs.")
    parser.add_argument("--image-size", type=int, default=128, help="Square image size used for training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--max-train-steps", type=int, default=10000, help="Total optimizer steps to train.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Batches per optimizer step.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save a checkpoint every N optimizer steps.")
    parser.add_argument("--sample-every", type=int, default=1000, help="Generate sample images every N optimizer steps; 0 disables.")
    parser.add_argument("--use-ema", action="store_true", help="Track and save exponential moving average weights.")
    parser.add_argument(
        "--aux-loss-mode",
        choices=("notebook", "none", "data_driven_diag"),
        default="notebook",
        help="Auxiliary structure loss mode. Defaults to the original notebook-style loss.",
    )
    parser.add_argument("--symmetry-weight", type=float, default=0.01, help="Weight for notebook channel-0 symmetry loss.")
    parser.add_argument("--diag-weight", type=float, default=0.01, help="Weight for diagonal auxiliary loss.")
    parser.add_argument(
        "--disable-symmetry-loss",
        action="store_true",
        help="Disable symmetry loss in notebook mode; ignored by none and data_driven_diag modes.",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None, help="Checkpoint directory to resume from.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="DDPM training diffusion timesteps.")
    parser.add_argument("--lr-warmup-steps", type=int, default=500, help="Cosine schedule warmup steps.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--log-every", type=int, default=10, help="Log loss every N optimizer steps.")
    parser.add_argument("--num-sample-images", type=int, default=4, help="Number of images for periodic samples.")
    parser.add_argument("--sample-inference-steps", type=int, default=50, help="Inference steps for periodic samples.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

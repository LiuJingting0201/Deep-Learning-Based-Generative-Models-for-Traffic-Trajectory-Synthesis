"""Train Week 4 DDPM ablations with pure MSE or GASF-only structure loss."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from train_ddpm import build_unet, save_checkpoint, save_samples, set_seed


class FloatArrayDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, image_size: int | None = None):
        self.data_dir = Path(data_dir)
        self.paths = sorted(self.data_dir.glob("*.npy"))
        if not self.paths:
            raise ValueError(f"No .npy files found in {self.data_dir}")

        first = np.load(self.paths[0])
        if first.ndim != 3 or first.shape[-1] != 3:
            raise ValueError(f"Expected [H, W, 3] array, got {first.shape} in {self.paths[0]}")
        if first.shape[0] != first.shape[1]:
            raise ValueError(f"Expected square array, got {first.shape} in {self.paths[0]}")

        self.native_size = int(first.shape[0])
        if image_size is not None and image_size != self.native_size:
            raise ValueError(
                f"Float array resizing is not supported. "
                f"Requested image_size={image_size}, native_size={self.native_size}."
            )
        self.image_size = self.native_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arr = np.load(self.paths[index]).astype(np.float32)
        expected_shape = (self.image_size, self.image_size, 3)
        if arr.shape != expected_shape:
            raise ValueError(f"Unexpected array shape {arr.shape}; expected {expected_shape} in {self.paths[index]}")
        arr = np.clip(arr, 0.0, 1.0)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensor = tensor * 2.0 - 1.0
        return tensor


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.image_size = args.image_size or infer_image_size(args.data_dir, args.data_format)
    args.output_dir = resolve_output_dir(args)
    fail_early(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    start_time = time.perf_counter()
    device = torch.device("cuda")

    dataset = build_dataset(args)
    print_startup_diagnostics(args, dataset)

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

    running = {
        "total_loss": 0.0,
        "mse_loss": 0.0,
        "gasf_symmetry_loss": 0.0,
        "gasf_consistency_loss": 0.0,
    }
    steps_since_log = 0
    global_step = 0
    batches = cycle(dataloader)
    optimizer.zero_grad(set_to_none=True)
    model.train()

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
            components = compute_week04_loss(
                noisy_images=noisy_images,
                noise_pred=noise_pred,
                noise=noise,
                timesteps=timesteps,
                scheduler=scheduler,
                aux_loss_mode=args.aux_loss_mode,
                symmetry_weight=args.symmetry_weight,
                consistency_weight=args.diag_weight,
                gasf_diag_eps=args.gasf_diag_eps,
            )
            if not torch.isfinite(components["total_loss"]):
                write_nonfinite_debug(output_dir, global_step + 1, "loss", components)
                raise RuntimeError(f"Non-finite loss at next_step={global_step + 1}")
            (components["total_loss"] / args.gradient_accumulation_steps).backward()
            for key, value in components.items():
                running[key] += float(value.detach().cpu())

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if has_nonfinite_gradient(model):
            write_nonfinite_debug(output_dir, global_step + 1, "gradient", components)
            raise RuntimeError(f"Non-finite gradient at next_step={global_step + 1}")
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema_model is not None:
            ema_model.step(model.parameters())

        global_step += 1
        steps_since_log += 1
        if global_step % args.log_every == 0 or global_step == 1:
            denom = max(1, steps_since_log * args.gradient_accumulation_steps)
            avg = {key: value / denom for key, value in running.items()}
            elapsed = time.perf_counter() - start_time
            record = {
                "step": global_step,
                "loss": avg["total_loss"],
                "total_loss": avg["total_loss"],
                "mse_loss": avg["mse_loss"],
                "gasf_symmetry_loss": avg["gasf_symmetry_loss"],
                "gasf_consistency_loss": avg["gasf_consistency_loss"],
                "symmetry_weight": args.symmetry_weight,
                "consistency_weight": args.diag_weight,
                "structure_loss_type": structure_loss_type(args.aux_loss_mode),
                "aux_loss_mode": args.aux_loss_mode,
                "data_format": args.data_format,
                "lr": lr_scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": global_step / elapsed if elapsed > 0 else None,
                "device": str(device),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step={global_step} total_loss={avg['total_loss']:.6f} "
                f"mse_loss={avg['mse_loss']:.6f} "
                f"gasf_symmetry_loss={avg['gasf_symmetry_loss']:.6f} "
                f"gasf_consistency_loss={avg['gasf_consistency_loss']:.6f}",
                flush=True,
            )
            running = {key: 0.0 for key in running}
            steps_since_log = 0

        if args.sample_every > 0 and global_step % args.sample_every == 0:
            save_samples(model, scheduler, ema_model, output_dir / "samples", global_step, device, args)
        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint-{global_step}", model, scheduler, optimizer, lr_scheduler, ema_model, global_step, args)

    save_checkpoint(output_dir / "checkpoint-final", model, scheduler, optimizer, lr_scheduler, ema_model, global_step, args)
    write_summary(output_dir, args, dataset, global_step)
    if args.copy_results_to_home:
        copy_results_to_home(args)


def compute_week04_loss(
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    aux_loss_mode: str,
    symmetry_weight: float,
    consistency_weight: float,
    gasf_diag_eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    mse_loss = F.mse_loss(noise_pred, noise)
    zero = mse_loss.new_tensor(0.0)
    if aux_loss_mode == "none":
        return {
            "total_loss": mse_loss,
            "mse_loss": mse_loss,
            "gasf_symmetry_loss": zero,
            "gasf_consistency_loss": zero,
        }

    x0_pred = predict_x0(noisy_images, noise_pred, timesteps, scheduler)
    gasf_symmetry_loss, gasf_consistency_loss = compute_gasf_structure_losses(x0_pred, diag_eps=gasf_diag_eps)
    if aux_loss_mode == "symmetry":
        total_loss = mse_loss + symmetry_weight * gasf_symmetry_loss
    elif aux_loss_mode == "gasf_structure":
        total_loss = (
            mse_loss
            + symmetry_weight * gasf_symmetry_loss
            + consistency_weight * gasf_consistency_loss
        )
    else:
        raise ValueError(f"Unsupported aux_loss_mode: {aux_loss_mode}")
    return {
        "total_loss": total_loss,
        "mse_loss": mse_loss,
        "gasf_symmetry_loss": gasf_symmetry_loss,
        "gasf_consistency_loss": gasf_consistency_loss,
    }


def predict_x0(
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
) -> torch.Tensor:
    alpha_t = scheduler.alphas_cumprod.to(noisy_images.device)[timesteps].view(-1, 1, 1, 1)
    x0_pred = (noisy_images - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    return torch.clamp(x0_pred, -1.0, 1.0)


def compute_gasf_structure_losses(x0_pred: torch.Tensor, diag_eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    gasf_channels = x0_pred[:, :2, :, :]
    gasf_symmetry_loss = torch.mean((gasf_channels - gasf_channels.transpose(-1, -2)) ** 2)
    diag = torch.diagonal(gasf_channels, dim1=-2, dim2=-1)
    # Clamp to the interior of arccos/sqrt domains. The old [0, 1] clamp had
    # finite forward values but singular gradients when predicted diagonals hit
    # exactly -1 or 1 after x0 clipping.
    diag01 = torch.clamp((diag.float() + 1.0) / 2.0, min=diag_eps, max=1.0 - diag_eps)
    u = torch.sqrt(diag01)
    phi = torch.arccos(u)
    gasf_rebuild = torch.cos(phi[:, :, :, None] + phi[:, :, None, :]).to(gasf_channels.dtype)
    gasf_consistency_loss = torch.mean((gasf_channels - gasf_rebuild) ** 2)
    return gasf_symmetry_loss, gasf_consistency_loss


def has_nonfinite_gradient(model: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def write_nonfinite_debug(
    output_dir: Path,
    step: int,
    kind: str,
    components: dict[str, torch.Tensor],
) -> None:
    payload = {
        "step": step,
        "kind": kind,
        "components": {
            name: float(value.detach().cpu()) if value.numel() == 1 else None
            for name, value in components.items()
        },
    }
    (output_dir / f"nonfinite_debug_step_{step:06d}.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--data-format",
        choices=("png", "npy"),
        default="png",
        help="Input format: png uses DiffusionImageDataset; npy uses float arrays in [0,1].",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to native input size from --data-dir.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=5000)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--aux-loss-mode", choices=("none", "symmetry", "gasf_structure"), default="none")
    parser.add_argument(
        "--diag-target-file",
        type=Path,
        default=None,
        help="Deprecated compatibility option; Week 4 training no longer uses dataset mean diagonal targets.",
    )
    parser.add_argument("--symmetry-weight", type=float, default=0.01)
    parser.add_argument(
        "--diag-weight",
        type=float,
        default=0.01,
        help="Weight for GASF self-consistency loss reconstructed from predicted diagonal.",
    )
    parser.add_argument(
        "--gasf-diag-eps",
        type=float,
        default=1e-4,
        help="Interior clamp epsilon for diagonal-derived sqrt/arccos in gasf_structure loss.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-sample-images", type=int, default=16)
    parser.add_argument("--sample-inference-steps", type=int, default=50)
    parser.add_argument("--copy-results-to-home", action="store_true")
    parser.add_argument("--home-results-dir", type=Path, default=None)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    if args.data_format == "png":
        return DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    if args.data_format == "npy":
        return FloatArrayDiffusionDataset(args.data_dir, image_size=args.image_size)
    raise ValueError(f"Unsupported data_format: {args.data_format}")


def infer_image_size(data_dir: Path, data_format: str) -> int:
    if data_format == "png":
        from PIL import Image

        first = next(iter(sorted(data_dir.glob("*.png"))), None)
        if first is None:
            raise ValueError(f"No PNG images found in {data_dir}")
        with Image.open(first) as image:
            width, height = image.size
        if width != height:
            raise ValueError(f"Expected square images, got {width}x{height}: {first}")
        return int(width)
    if data_format == "npy":
        first = next(iter(sorted(data_dir.glob("*.npy"))), None)
        if first is None:
            raise ValueError(f"No NPY arrays found in {data_dir}")
        arr = np.load(first)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected [H, W, 3] array, got {arr.shape} in {first}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected square array, got {arr.shape} in {first}")
        return int(arr.shape[0])
    raise ValueError(f"Unsupported data_format: {data_format}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if args.run_name is None:
        raise ValueError("Provide --output-dir or --run-name")
    scratch_root = args.scratch_root
    if scratch_root is None:
        scratch_env = os.environ.get("SCRATCH")
        if not scratch_env:
            raise ValueError("SCRATCH is not set. Provide --scratch-root or --output-dir.")
        scratch_root = Path(scratch_env)
    args.scratch_root = scratch_root.expanduser().resolve()
    return (args.scratch_root / "Thesis_runs" / args.run_name).resolve()


def fail_early(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if args.data_format == "png":
        if not any(args.data_dir.glob("*.png")):
            raise ValueError(f"No PNG images found in {args.data_dir}")
    elif args.data_format == "npy":
        if not any(args.data_dir.glob("*.npy")):
            raise ValueError(f"No NPY arrays found in {args.data_dir}")
    else:
        raise ValueError(f"Unsupported data_format: {args.data_format}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this under a GPU Slurm allocation.")


def print_startup_diagnostics(args: argparse.Namespace, dataset: torch.utils.data.Dataset) -> None:
    print("Week04 DDPM training startup diagnostics")
    print(f"hostname: {socket.gethostname()}")
    print(f"resolved_data_dir: {args.data_dir}")
    print(f"resolved_output_dir: {args.output_dir}")
    print(f"data_format: {args.data_format}")
    print(f"native_or_requested_image_size: {args.image_size}")
    input_count_label = "num_input_arrays" if args.data_format == "npy" else "num_input_images"
    print(f"{input_count_label}: {len(dataset)}")
    print(f"torch_version: {torch.__version__}")
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    for name in (
        "batch_size",
        "data_format",
        "gradient_accumulation_steps",
        "max_train_steps",
        "aux_loss_mode",
        "structure_loss_type",
        "symmetry_weight",
        "diag_weight",
        "gasf_diag_eps",
        "sample_every",
        "save_every",
        "num_sample_images",
        "sample_inference_steps",
        "run_name",
    ):
        value = structure_loss_type(args.aux_loss_mode) if name == "structure_loss_type" else getattr(args, name)
        print(f"  {name}: {value}")
    print(f"  consistency_weight: {args.diag_weight}", flush=True)


def structure_loss_type(aux_loss_mode: str) -> str:
    if aux_loss_mode == "none":
        return "none"
    if aux_loss_mode == "symmetry":
        return "symmetry_only"
    if aux_loss_mode == "gasf_structure":
        return "gasf_self_consistency_from_predicted_diagonal"
    raise ValueError(f"Unsupported aux_loss_mode: {aux_loss_mode}")


def write_summary(output_dir: Path, args: argparse.Namespace, dataset: torch.utils.data.Dataset, global_step: int) -> None:
    payload = {
        "global_step": global_step,
        "num_images": len(dataset),
        "num_inputs": len(dataset),
        "data_format": args.data_format,
        "image_size": args.image_size,
        "aux_loss_mode": args.aux_loss_mode,
        "structure_loss_type": structure_loss_type(args.aux_loss_mode),
        "diag_target_file_unused": str(args.diag_target_file) if args.diag_target_file is not None else None,
        "symmetry_weight": args.symmetry_weight,
        "consistency_weight": args.diag_weight,
        "gasf_diag_eps": args.gasf_diag_eps,
        "run_name": args.run_name,
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def copy_results_to_home(args: argparse.Namespace) -> None:
    destination_name = args.run_name if args.run_name is not None else args.output_dir.name
    if args.home_results_dir is None:
        destination = Path(os.environ.get("HOME", "~")).expanduser() / "Thesis" / "results_hpc" / destination_name
    else:
        destination = args.home_results_dir.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying results to home: {args.output_dir} -> {destination}", flush=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", f"{args.output_dir}/", f"{destination}/"], check=True)
    else:
        shutil.copytree(args.output_dir, destination, dirs_exist_ok=True)


if __name__ == "__main__":
    main()

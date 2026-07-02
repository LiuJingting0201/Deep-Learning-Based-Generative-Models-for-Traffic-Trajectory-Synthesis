"""Train Week 5 DDPM ablations on regularized delta-displacement float arrays.

The dataset root contains 224-step delta trajectories, absolute labels, starts,
splits, and normalization diagnostics. This script builds DDPM training inputs
as float array tensors on the fly:

R = GASF(sigmoid-normalized dx)
G = GASF(sigmoid-normalized dy)
B = start-position heatmap

No PNG inputs are used for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any

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

from train_ddpm import build_unet, save_checkpoint, save_samples, set_seed


class Week05DeltaGASFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: Path,
        image_size: int,
        normalization_file: Path,
        sigmoid_k: float,
        split: str = "train",
        heatmap_sigma: float = 2.5,
    ):
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        self.normalization_file = Path(normalization_file)
        self.sigmoid_k = float(sigmoid_k)
        self.split = split
        self.heatmap_sigma = float(heatmap_sigma)
        self.delta_dir = self.data_root / "labels_delta_displacement"
        self.absolute_dir = self.data_root / "labels_absolute"
        self.starts_dir = self.data_root / "starts"
        self.sample_ids = resolve_sample_ids(self.data_root, split)
        if not self.sample_ids:
            raise ValueError(f"No sample ids found for split={split!r} in {self.data_root}")

        self.normalization = load_normalization(self.normalization_file)
        train_sample_ids = resolve_sample_ids(self.data_root, "train")
        self.position_stats = compute_position_stats(self.absolute_dir, train_sample_ids)
        self._validate_first_sample()

    def _validate_first_sample(self) -> None:
        sample_id = self.sample_ids[0]
        delta_xy = np.load(self.delta_dir / f"{sample_id}.npy")
        if delta_xy.shape != (self.image_size, 2):
            raise ValueError(
                f"Expected delta shape ({self.image_size}, 2), got {delta_xy.shape} "
                f"in {self.delta_dir / f'{sample_id}.npy'}"
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample_id = self.sample_ids[index]
        delta_xy = np.load(self.delta_dir / f"{sample_id}.npy").astype(np.float32)
        if delta_xy.shape != (self.image_size, 2):
            raise ValueError(
                f"Expected delta shape ({self.image_size}, 2), got {delta_xy.shape} "
                f"in {self.delta_dir / f'{sample_id}.npy'}"
            )
        start_xy = self._load_start(sample_id)
        array = encode_delta_as_float_array(
            delta_xy=delta_xy,
            start_xy=start_xy,
            normalization=self.normalization,
            position_stats=self.position_stats,
            sigmoid_k=self.sigmoid_k,
            image_size=self.image_size,
            heatmap_sigma=self.heatmap_sigma,
        )
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor * 2.0 - 1.0

    def _load_start(self, sample_id: str) -> np.ndarray:
        start_path = self.starts_dir / f"{sample_id}.npy"
        if start_path.exists():
            start_xy = np.load(start_path).astype(np.float32)
            if start_xy.shape == (2,):
                return start_xy
        absolute_path = self.absolute_dir / f"{sample_id}.npy"
        absolute_xy = np.load(absolute_path).astype(np.float32)
        if absolute_xy.ndim != 2 or absolute_xy.shape[1] != 2:
            raise ValueError(f"Expected absolute shape (N, 2), got {absolute_xy.shape} in {absolute_path}")
        return absolute_xy[0]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.normalization_file = resolve_normalization_file(args)
    fail_early(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    start_time = time.perf_counter()
    device = torch.device("cuda")

    dataset = Week05DeltaGASFDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        normalization_file=args.normalization_file,
        sigmoid_k=args.sigmoid_k,
        split=args.split,
        heatmap_sigma=args.heatmap_sigma,
    )
    print_startup_diagnostics(args, dataset)
    write_startup_config(output_dir / "startup_config.json", args, dataset)

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
        "diag_loss": 0.0,
        "symmetry_loss": 0.0,
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
            components = compute_week05_loss(
                clean_images=clean_images,
                noisy_images=noisy_images,
                noise_pred=noise_pred,
                noise=noise,
                timesteps=timesteps,
                scheduler=scheduler,
                aux_loss_mode=args.aux_loss_mode,
                diag_weight=args.diag_weight,
                symmetry_weight=args.symmetry_weight,
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
                "diag_loss": avg["diag_loss"],
                "symmetry_loss": avg["symmetry_loss"],
                "aux_loss_mode": args.aux_loss_mode,
                "normalization_mode": "sigmoid",
                "sigmoid_k": args.sigmoid_k,
                "diag_weight": args.diag_weight,
                "symmetry_weight": args.symmetry_weight,
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
                f"diag_loss={avg['diag_loss']:.6f} "
                f"symmetry_loss={avg['symmetry_loss']:.6f}",
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


def compute_week05_loss(
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    aux_loss_mode: str,
    diag_weight: float,
    symmetry_weight: float,
) -> dict[str, torch.Tensor]:
    mse_loss = F.mse_loss(noise_pred, noise)
    zero = mse_loss.new_tensor(0.0)
    if aux_loss_mode == "none":
        return {
            "total_loss": mse_loss,
            "mse_loss": mse_loss,
            "diag_loss": zero,
            "symmetry_loss": zero,
        }

    x0_pred = predict_x0(noisy_images, noise_pred, timesteps, scheduler)
    if aux_loss_mode == "diag":
        diag_loss = compute_diag_loss(x0_pred, clean_images)
        total_loss = mse_loss + diag_weight * diag_loss
        symmetry_loss = zero
    elif aux_loss_mode == "symmetry":
        symmetry_loss = compute_symmetry_loss(x0_pred)
        total_loss = mse_loss + symmetry_weight * symmetry_loss
        diag_loss = zero
    else:
        raise ValueError(f"Unsupported aux_loss_mode: {aux_loss_mode}")

    return {
        "total_loss": total_loss,
        "mse_loss": mse_loss,
        "diag_loss": diag_loss,
        "symmetry_loss": symmetry_loss,
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


def compute_diag_loss(x0_pred: torch.Tensor, clean_images: torch.Tensor) -> torch.Tensor:
    pred_diag = torch.diagonal(x0_pred[:, :2, :, :], dim1=-2, dim2=-1)
    clean_diag = torch.diagonal(clean_images[:, :2, :, :], dim1=-2, dim2=-1)
    return F.mse_loss(pred_diag, clean_diag)


def compute_symmetry_loss(x0_pred: torch.Tensor) -> torch.Tensor:
    gasf_channels = x0_pred[:, :2, :, :]
    return torch.mean((gasf_channels - gasf_channels.transpose(-1, -2)) ** 2)


def encode_delta_as_float_array(
    delta_xy: np.ndarray,
    start_xy: np.ndarray,
    normalization: dict[str, float],
    position_stats: dict[str, float],
    sigmoid_k: float,
    image_size: int,
    heatmap_sigma: float,
) -> np.ndarray:
    dx_norm = sigmoid_normalize(delta_xy[:, 0], normalization["dx_mean"], normalization["dx_std"], sigmoid_k)
    dy_norm = sigmoid_normalize(delta_xy[:, 1], normalization["dy_mean"], normalization["dy_std"], sigmoid_k)
    gasf_dx = gasf_encode(dx_norm)
    gasf_dy = gasf_encode(dy_norm)
    heatmap = make_start_heatmap(start_xy, position_stats, image_size, heatmap_sigma)
    image_float = np.stack([(gasf_dx + 1.0) / 2.0, (gasf_dy + 1.0) / 2.0, heatmap], axis=-1)
    return np.clip(image_float, 0.0, 1.0).astype(np.float32)


def sigmoid_normalize(values: np.ndarray, mean: float, std: float, k_value: float) -> np.ndarray:
    if std == 0.0:
        raise ValueError("Cannot apply sigmoid normalization with zero std.")
    logits = np.clip(k_value * ((values - mean) / std), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def gasf_encode(values_01: np.ndarray) -> np.ndarray:
    phi = np.arccos(np.clip(values_01, 0.0, 1.0))
    return np.cos(phi[:, None] + phi[None, :])


def make_start_heatmap(
    start_xy: np.ndarray,
    position_stats: dict[str, float],
    image_size: int,
    sigma: float,
) -> np.ndarray:
    x0, y0 = start_xy
    px = normalize_position(x0, position_stats["x_min"], position_stats["x_max"]) * (image_size - 1)
    py = normalize_position(y0, position_stats["y_min"], position_stats["y_max"]) * (image_size - 1)
    grid_y, grid_x = np.mgrid[0:image_size, 0:image_size]
    dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
    heatmap = np.exp(-dist2 / (2.0 * sigma**2))
    max_value = float(heatmap.max())
    if max_value > 0.0:
        heatmap = heatmap / max_value
    return heatmap.astype(np.float32)


def normalize_position(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.5
    return float(np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0))


def resolve_sample_ids(data_root: Path, split: str) -> list[str]:
    if split == "all":
        return sorted(path.stem for path in (data_root / "labels_delta_displacement").glob("*.npy"))
    split_path = data_root / "splits" / f"{split}_ids.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{split_path} must contain a sample_id column.")
        return [row["sample_id"] for row in reader if row.get("sample_id")]


def load_normalization(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    required = ("dx_mean", "dx_std", "dy_mean", "dy_std")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing normalization keys: {missing}")
    return {key: float(payload[key]) for key in required}


def compute_position_stats(absolute_dir: Path, sample_ids: list[str]) -> dict[str, float]:
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for sample_id in sample_ids:
        absolute_path = absolute_dir / f"{sample_id}.npy"
        absolute_xy = np.load(absolute_path).astype(np.float64)
        if absolute_xy.ndim != 2 or absolute_xy.shape[1] != 2:
            raise ValueError(f"Expected absolute shape (N, 2), got {absolute_xy.shape} in {absolute_path}")
        x_min = min(x_min, float(absolute_xy[:, 0].min()))
        x_max = max(x_max, float(absolute_xy[:, 0].max()))
        y_min = min(y_min, float(absolute_xy[:, 1].min()))
        y_max = max(y_max, float(absolute_xy[:, 1].max()))
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sigmoid-k", type=float, required=True)
    parser.add_argument("--heatmap-sigma", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--aux-loss-mode", choices=("none", "diag", "symmetry"), default="none")
    parser.add_argument("--diag-weight", type=float, default=0.05)
    parser.add_argument("--symmetry-weight", type=float, default=0.05)
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


def resolve_normalization_file(args: argparse.Namespace) -> Path:
    if args.normalization_file is not None:
        return args.normalization_file.expanduser().resolve()
    return (args.data_root / "normalization_diagnostics.json").resolve()


def fail_early(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(args.data_root)
    if not (args.data_root / "labels_delta_displacement").is_dir():
        raise FileNotFoundError(args.data_root / "labels_delta_displacement")
    if not (args.data_root / "labels_absolute").is_dir():
        raise FileNotFoundError(args.data_root / "labels_absolute")
    if not args.normalization_file.exists():
        raise FileNotFoundError(args.normalization_file)
    if args.image_size != 224:
        raise ValueError("Week05 DDPM experiments require image_size=224.")
    if args.batch_size != 8:
        raise ValueError("Week05 matrix is configured for batch_size=8.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this under a GPU Slurm allocation.")


def print_startup_diagnostics(args: argparse.Namespace, dataset: Week05DeltaGASFDataset) -> None:
    config = {
        "hostname": socket.gethostname(),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "normalization_file": str(args.normalization_file),
        "normalization_mode": "sigmoid",
        "sigmoid_k": args.sigmoid_k,
        "data_format": "float_gasf_generated_on_the_fly_from_delta_npy",
        "png_inputs_used": False,
        "image_size": args.image_size,
        "channels": 3,
        "split": args.split,
        "num_input_arrays": len(dataset),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train_steps": args.max_train_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "num_train_timesteps": args.num_train_timesteps,
        "lr_warmup_steps": args.lr_warmup_steps,
        "scheduler": "DDPMScheduler",
        "model_architecture": "scripts.train_ddpm.build_unet",
        "aux_loss_mode": args.aux_loss_mode,
        "diag_weight": args.diag_weight,
        "symmetry_weight": args.symmetry_weight,
        "save_every": args.save_every,
        "sample_every": args.sample_every,
        "use_ema": args.use_ema,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "normalization_stats": dataset.normalization,
        "position_stats": dataset.position_stats,
    }
    print("Week05 DDPM training startup config")
    print(json.dumps(config, indent=2), flush=True)


def write_startup_config(
    path: Path,
    args: argparse.Namespace,
    dataset: Week05DeltaGASFDataset,
) -> None:
    path.write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "normalization_file": str(args.normalization_file),
                "normalization_stats": dataset.normalization,
                "position_stats": dataset.position_stats,
                "num_inputs": len(dataset),
                "sigmoid_k": args.sigmoid_k,
                "aux_loss_mode": args.aux_loss_mode,
                "diag_weight": args.diag_weight,
                "symmetry_weight": args.symmetry_weight,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "max_train_steps": args.max_train_steps,
                "seed": args.seed,
                "data_format": "float_gasf_generated_on_the_fly_from_delta_npy",
                "png_inputs_used": False,
            },
            indent=2,
        )
        + "\n"
    )


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: Week05DeltaGASFDataset,
    global_step: int,
) -> None:
    payload = {
        "global_step": global_step,
        "num_inputs": len(dataset),
        "data_root": str(args.data_root),
        "data_format": "float_gasf_generated_on_the_fly_from_delta_npy",
        "png_inputs_used": False,
        "image_size": args.image_size,
        "channels": 3,
        "split": args.split,
        "normalization_mode": "sigmoid",
        "sigmoid_k": args.sigmoid_k,
        "normalization_file": str(args.normalization_file),
        "aux_loss_mode": args.aux_loss_mode,
        "diag_weight": args.diag_weight,
        "symmetry_weight": args.symmetry_weight,
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def copy_results_to_home(args: argparse.Namespace) -> None:
    destination_name = args.output_dir.name
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

"""Train a DDPM model with HPC scratch-friendly path handling.

This entrypoint keeps the training loop aligned with scripts/train_ddpm.py,
while adding safer defaults for Polito HPC runs:

- input data should live on BeeGFS scratch
- outputs default to $SCRATCH/Thesis_runs/<run_name>
- selected results can be copied back to $HOME/Thesis/results_hpc/<run_name>
"""

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

import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from train_ddpm import (
    apply_channel_normalization,
    build_unet,
    compute_loss_components,
    load_channel_norm_stats,
    load_checkpoint,
    resolve_diag_targets,
    save_checkpoint,
    save_samples,
    set_seed,
    standardize_diag_targets_if_needed,
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = resolve_output_dir(args)

    fail_early(args)
    output_dir = args.output_dir
    log_path = output_dir / "train_log.jsonl"
    start_time = time.perf_counter()
    device = torch.device("cuda")

    dataset = DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    diag_target_info = resolve_diag_targets_for_scratch(dataset, args)
    diag_targets = diag_target_info["targets"]
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

    global_step = 0
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = args.resume_from_checkpoint.expanduser().resolve()
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
        "mtf_symmetry_loss": 0.0,
    }
    steps_since_log = 0
    batches = cycle(dataloader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for _ in range(args.gradient_accumulation_steps):
            clean_images = next(batches).to(device)
            clean_images = apply_channel_normalization(clean_images, args)
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
                "mtf_symmetry_loss": avg_components["mtf_symmetry_loss"],
                "aux_loss_mode": args.aux_loss_mode,
                "diag_targets": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
                "diag_target_rgb": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
                "diag_weight": args.diag_weight,
                "symmetry_weight": args.symmetry_weight,
                "mtf_symmetry_weight": args.mtf_symmetry_weight,
                "lr": lr_scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed,
                "steps_per_second": global_step / elapsed if elapsed > 0 else None,
                "device": str(device),
                "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
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
            save_checkpoint(
                output_dir / f"checkpoint-{global_step}",
                model,
                scheduler,
                optimizer,
                lr_scheduler,
                ema_model,
                global_step,
                args,
            )

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
                "mtf_symmetry_weight": args.mtf_symmetry_weight,
                "disable_symmetry_loss": args.disable_symmetry_loss,
                "diag_targets": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
                "diag_target_rgb": diag_targets.detach().cpu().tolist() if diag_targets is not None else None,
                "diag_target_source": diag_target_info["source"],
                "diag_target_num_images": diag_target_info["num_images"],
                "diag_target_file": str(args.diag_target_file) if args.diag_target_file is not None else None,
                "split_metadata": str(args.split_metadata) if args.split_metadata is not None else None,
                "channel_normalization_mode": args.channel_normalization_mode,
                "channel_norm_stats": args.channel_norm_stats_payload,
                "channel_norm_stats_file": str(args.channel_norm_stats) if args.channel_norm_stats is not None else None,
                "scratch_root": str(args.scratch_root) if args.scratch_root is not None else None,
                "run_name": args.run_name,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    if args.copy_results_to_home:
        copy_results_to_home(args)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()

    if args.run_name is None:
        raise ValueError("Provide --output-dir, or provide --run-name so output can default to $SCRATCH/Thesis_runs/<run_name>.")

    scratch_root = args.scratch_root
    if scratch_root is None:
        scratch_env = os.environ.get("SCRATCH")
        if not scratch_env:
            raise ValueError("SCRATCH is not set. Provide --scratch-root or --output-dir explicitly.")
        scratch_root = Path(scratch_env)
    args.scratch_root = scratch_root.expanduser().resolve()
    return (args.scratch_root / "Thesis_runs" / args.run_name).resolve()


def resolve_diag_targets_for_scratch(dataset: DiffusionImageDataset, args: argparse.Namespace) -> dict[str, object]:
    if args.aux_loss_mode not in ("data_driven_diag", "data_driven_diag_mtf_symmetry") or args.diag_target_file is None:
        return resolve_diag_targets(dataset, args)

    diag_target_file = args.diag_target_file.expanduser().resolve()
    if not diag_target_file.exists():
        raise FileNotFoundError(f"diag_target_file does not exist: {diag_target_file}")

    with diag_target_file.open() as json_file:
        payload = json.load(json_file)

    values = payload.get("diag_targets")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"diag_target_file must contain a diag_targets list of three RGB values: {diag_target_file}")

    try:
        targets = torch.tensor([float(value) for value in values], dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"diag_targets must contain numeric RGB values: {diag_target_file}") from exc
    targets = standardize_diag_targets_if_needed(targets, args)

    args.diag_target_file = diag_target_file
    return {
        "targets": targets,
        "source": payload.get("source", "diag_target_file"),
        "num_images": payload.get("num_images"),
    }


def fail_early(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    image_paths = sorted(args.data_dir.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in data_dir: {args.data_dir}")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output_dir {args.output_dir}: {exc}") from exc

    if not args.output_dir.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {args.output_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This scratch HPC entrypoint requires a GPU allocation.")


def print_startup_diagnostics(args: argparse.Namespace, dataset: DiffusionImageDataset) -> None:
    print("DDPM scratch training startup diagnostics")
    print(f"hostname: {socket.gethostname()}")
    print(f"cwd: {Path.cwd()}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
        print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"resolved_data_dir: {args.data_dir}")
    print(f"resolved_output_dir: {args.output_dir}")
    print(f"num_input_images: {len(dataset)}")
    print("training_hyperparameters:")
    for name in (
        "image_size",
        "batch_size",
        "lr",
        "max_train_steps",
        "gradient_accumulation_steps",
        "save_every",
        "sample_every",
        "use_ema",
        "aux_loss_mode",
        "diag_target_file",
        "symmetry_weight",
        "diag_weight",
        "mtf_symmetry_weight",
        "disable_symmetry_loss",
        "resume_from_checkpoint",
        "seed",
        "num_train_timesteps",
        "lr_warmup_steps",
        "max_grad_norm",
        "num_workers",
        "log_every",
        "num_sample_images",
        "sample_inference_steps",
        "run_name",
        "scratch_root",
        "copy_results_to_home",
        "home_results_dir",
        "channel_normalization_mode",
        "channel_norm_stats",
    ):
        print(f"  {name}: {getattr(args, name)}")
    print("", flush=True)


def copy_results_to_home(args: argparse.Namespace) -> None:
    if args.run_name is None:
        destination_name = args.output_dir.name
    else:
        destination_name = args.run_name

    if args.home_results_dir is None:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        destination = home / "Thesis" / "results_hpc" / destination_name
    else:
        destination = args.home_results_dir.expanduser()

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying results to home: {args.output_dir} -> {destination}", flush=True)
    if shutil.which("rsync"):
        subprocess.run(
            ["rsync", "-a", f"{args.output_dir}/", f"{destination}/"],
            check=True,
        )
    else:
        shutil.copytree(args.output_dir, destination, dirs_exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing PNG training images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, samples, and logs. If omitted, use $SCRATCH/Thesis_runs/<run_name>.",
    )
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
        choices=("notebook", "none", "data_driven_diag", "mtf_symmetry", "data_driven_diag_mtf_symmetry"),
        default="notebook",
        help="Auxiliary structure loss mode. Defaults to the original notebook-style loss.",
    )
    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=None,
        help="Optional split_metadata.csv used to compute data-driven diagonal targets from train images only.",
    )
    parser.add_argument(
        "--diag-target-file",
        type=Path,
        default=None,
        help="Optional JSON file with precomputed RGB diagonal targets for data_driven_diag mode.",
    )
    parser.add_argument("--symmetry-weight", type=float, default=0.01, help="Weight for notebook channel-0 symmetry loss.")
    parser.add_argument("--diag-weight", type=float, default=0.01, help="Weight for diagonal auxiliary loss.")
    parser.add_argument("--mtf-symmetry-weight", type=float, default=0.01, help="Weight for MTF/B-channel symmetry loss.")
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
    parser.add_argument("--run-name", type=str, default=None, help="Run name used under $SCRATCH/Thesis_runs when --output-dir is omitted.")
    parser.add_argument("--scratch-root", type=Path, default=None, help="Scratch root. Defaults to environment variable SCRATCH.")
    parser.add_argument(
        "--copy-results-to-home",
        action="store_true",
        help="Copy final output directory back to $HOME/Thesis/results_hpc/<run_name> after training.",
    )
    parser.add_argument(
        "--home-results-dir",
        type=Path,
        default=None,
        help="Optional explicit destination for --copy-results-to-home.",
    )
    parser.add_argument("--channel-norm-stats", type=Path, default=None, help="JSON file with per-channel mean/std in [-1, 1].")
    parser.add_argument(
        "--channel-normalization-mode",
        choices=("none", "standardize"),
        default="none",
        help="Optionally train DDPM in channel-wise standardized space.",
    )
    args = parser.parse_args()
    load_channel_norm_stats(args)
    return args


if __name__ == "__main__":
    main()

"""Evaluate Week3 image-to-sequence-latent alignment checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
if str(WEEK03_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK03_SCRIPTS_DIR))

from sequence_latent_models import (  # noqa: E402
    RowWiseGAFImageEncoder,
    TrajectorySequenceAutoencoder,
)
from sequence_latent_utils import (  # noqa: E402
    PairedImageDeltaDataset,
    integrated_trajectory,
    load_split_metadata,
    resolve_device,
    set_seed,
    summarize_rows,
    write_json,
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    for subdir in ["predictions_delta", "predictions_integrated", "teacher_predictions_delta", "plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = load_split_metadata(args.data_root.resolve(), seed=args.seed)
    teacher = load_teacher(args.trajectory_ae_checkpoint, args.latent_dim, device)
    image_encoder = load_image_encoder(args.image_encoder_checkpoint, args.latent_dim, device)
    checkpoint_config = load_checkpoint_config(args.image_encoder_checkpoint)
    image_channel_mode = str(checkpoint_config.get("image_channel_mode", "rgb"))

    all_rows: list[dict[str, Any]] = []
    plotted = 0
    for split in args.splits:
        frame = split_metadata[split_metadata["split"] == split].copy()
        if frame.empty:
            continue
        loader = DataLoader(
            PairedImageDeltaDataset(args.data_root.resolve(), frame, image_channel_mode=image_channel_mode),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        rows, plotted = evaluate_split(args, split, loader, image_encoder, teacher, device, plotted)
        all_rows.extend(rows)
        write_json(output_dir / f"metrics_{split}.json", summarize_rows(rows))

    pd.DataFrame(all_rows).to_csv(output_dir / "per_sample_metrics.csv", index=False)
    summary = summarize_rows(all_rows)
    write_json(output_dir / "metrics_summary.json", summary)
    print(f"Done. Evaluation outputs saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_delta_displacement_paired"))
    parser.add_argument("--trajectory-ae-checkpoint", type=Path, required=True)
    parser.add_argument("--image-encoder-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "week03" / "results" / "sequence_latent" / "stage2_rowwise_img_latent64" / "eval",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    return parser.parse_args()


def load_teacher(checkpoint_path: Path, latent_dim: int, device: torch.device) -> TrajectorySequenceAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TrajectorySequenceAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return {}
    return config


def load_image_encoder(checkpoint_path: Path, latent_dim: int, device: torch.device) -> RowWiseGAFImageEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    input_channels = int(config.get("input_channels", 3))
    model = RowWiseGAFImageEncoder(
        latent_dim=latent_dim,
        input_channels=input_channels,
        use_temporal_refine=bool(config.get("use_temporal_refine", False)),
        temporal_kernel_size=int(config.get("temporal_kernel_size", 5)),
        temporal_layers=int(config.get("temporal_layers", 2)),
        use_transformer_refine=bool(config.get("use_transformer_refine", False)),
        transformer_layers=int(config.get("transformer_layers", 2)),
        transformer_heads=int(config.get("transformer_heads", 4)),
        transformer_ff_dim=int(config.get("transformer_ff_dim", 256)),
        transformer_dropout=float(config.get("transformer_dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_split(
    args: argparse.Namespace,
    split: str,
    loader: DataLoader,
    image_encoder: RowWiseGAFImageEncoder,
    teacher: TrajectorySequenceAutoencoder,
    device: torch.device,
    plotted: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    output_dir = args.output_dir.resolve()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            delta_gt = batch["delta"].to(device, non_blocking=True)
            z_traj = teacher.encoder(delta_gt)
            teacher_delta = teacher.decoder(z_traj)
            z_img = image_encoder(images)
            delta_hat = teacher.decoder(z_img)

            gt_traj = integrated_trajectory(delta_gt).cpu().numpy().astype(np.float32)
            pred_traj = integrated_trajectory(delta_hat).cpu().numpy().astype(np.float32)
            teacher_traj = integrated_trajectory(teacher_delta).cpu().numpy().astype(np.float32)
            delta_np = delta_gt.cpu().numpy().astype(np.float32)
            pred_delta_np = delta_hat.cpu().numpy().astype(np.float32)
            teacher_delta_np = teacher_delta.cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                np.save(output_dir / "predictions_delta" / f"{sample_id}_pred_delta.npy", pred_delta_np[index])
                np.save(output_dir / "predictions_integrated" / f"{sample_id}_pred_integrated.npy", pred_traj[index])
                np.save(output_dir / "teacher_predictions_delta" / f"{sample_id}_teacher_delta.npy", teacher_delta_np[index])

                row = {
                    "sample_id": sample_id,
                    "vehicle_id": vehicle_id,
                    "split": split,
                    **metric_row(pred_delta_np[index], delta_np[index], prefix="delta"),
                    **metric_row(pred_traj[index], gt_traj[index], prefix="integrated"),
                    "latent_mse": float(torch.nn.functional.mse_loss(z_img[index], z_traj[index]).cpu()),
                    **metric_row(teacher_traj[index], gt_traj[index], prefix="teacher_integrated"),
                }
                rows.append(row)
                if plotted < args.num_plots:
                    plot_comparison(
                        output_dir / "plots" / f"{split}_{sample_id}.png",
                        sample_id,
                        gt_traj[index],
                        pred_traj[index],
                        teacher_traj[index],
                        delta_np[index],
                        pred_delta_np[index],
                    )
                    plotted += 1
    return rows, plotted


def metric_row(pred: np.ndarray, gt: np.ndarray, prefix: str) -> dict[str, float]:
    diff = pred - gt
    errors = np.linalg.norm(diff, axis=1)
    return {
        f"{prefix}_mse": float(np.mean(diff**2)),
        f"{prefix}_ADE": float(errors.mean()),
        f"{prefix}_FDE": float(errors[-1]),
        f"{prefix}_MAE": float(np.mean(np.abs(diff))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(diff**2))),
    }


def plot_comparison(
    path: Path,
    sample_id: str,
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    teacher_traj: np.ndarray,
    gt_delta: np.ndarray,
    pred_delta: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(gt_traj[:, 0], gt_traj[:, 1], label="gt integrated", linewidth=1.5)
    axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], label="image latent", linewidth=1.5)
    axes[0].plot(teacher_traj[:, 0], teacher_traj[:, 1], label="teacher latent", linewidth=1.0, alpha=0.8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(sample_id)
    axes[0].legend(fontsize=8)

    axes[1].plot(gt_delta[:, 0], label="gt dx", linewidth=1.0)
    axes[1].plot(pred_delta[:, 0], label="pred dx", linewidth=1.0)
    axes[1].set_title("dx")
    axes[1].legend(fontsize=8)

    axes[2].plot(gt_delta[:, 1], label="gt dy", linewidth=1.0)
    axes[2].plot(pred_delta[:, 1], label="pred dy", linewidth=1.0)
    axes[2].set_title("dy")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

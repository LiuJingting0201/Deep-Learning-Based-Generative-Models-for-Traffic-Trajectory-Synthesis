"""Train/evaluate delta-displacement decoders for no-speed GAF paired data."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cnr_trajectory.reconstruction.delta_displacement import (  # noqa: E402
    DeltaDisplacementPairedDataset,
    build_delta_decoder,
    compute_alignment_metrics,
    compute_basic_metrics,
    integrate_delta_numpy,
    integrate_delta_torch,
    load_or_create_split_metadata,
    summarize_metric_rows,
    write_split_json,
)


IMAGE_NORMALIZATION = "PNG RGB converted to float32 and divided by 255.0"
TARGET_TYPE = "physical unnormalized delta displacement [B,224,2]"
INTEGRATION_FORMULA = (
    "xy[:,0,:]=start_xy; "
    "xy[:,1:,:]=start_xy[:,None,:]+cumsum(delta[:,1:,:], dim=1)"
)


@dataclass(frozen=True)
class TargetDeltaNormalizer:
    mean: list[float]
    std: list[float]

    def tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, dtype=dtype, device=device).view(1, 1, 2)
        std = torch.tensor(self.std, dtype=dtype, device=device).view(1, 1, 2)
        return mean, std

    def normalize(self, delta: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(delta.device, delta.dtype)
        return (delta - mean) / std

    def unnormalize(self, delta: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors(delta.device, delta.dtype)
        return delta * std + mean


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_root = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    for directory in [
        output_dir,
        output_dir / "checkpoints",
        output_dir / "predictions_delta",
        output_dir / "predictions_integrated_xy",
        output_dir / "comparison_plots",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    split_metadata, split_source = load_split_metadata(args, data_root)
    split_metadata.to_csv(output_dir / "split_metadata.csv", index=False)
    write_split_json(output_dir / "split_metadata.json", split_metadata, split_source, args.seed)

    if args.eval_only and not args.normalize_target_delta:
        config_path = output_dir / "config.json"
        if config_path.exists():
            previous_config = json.loads(config_path.read_text())
            args.normalize_target_delta = bool(previous_config.get("normalize_target_delta", False))

    normalizer = prepare_target_normalizer(args, data_root, split_metadata, output_dir)
    config = make_config(args, device, split_metadata, split_source, normalizer)
    write_json(output_dir / "config.json", config)
    print_config_summary(config)

    if not args.eval_only:
        train(args, split_metadata, device, normalizer, config)

    checkpoint_path = args.checkpoint or output_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train first or pass --checkpoint."
        )
    evaluate(args, split_metadata, device, checkpoint_path, normalizer)
    print(f"Done. Outputs saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["multi_branch_delta", "simple_cnn_delta"],
        default="multi_branch_delta",
    )
    parser.add_argument(
        "--channels",
        choices=["rgb", "r", "g", "b"],
        default="rgb",
        help="Pseudo-modal channels: rgb=GASF/GADF/MTF, r=GASF, g=GADF, b=MTF.",
    )
    parser.add_argument(
        "--encoder-type",
        choices=["small_gap", "small_spatial"],
        default="small_spatial",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_no_speed_delta_displacement_paired"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decoder_multibranch_delta"),
    )
    parser.add_argument(
        "--split-metadata",
        type=Path,
        default=None,
        help="Optional split CSV. Priority: explicit path, data_dir/splits, deterministic split.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta-integrated-loss", type=float, default=0.2)
    parser.add_argument("--normalize-target-delta", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_split_metadata(args: argparse.Namespace, data_root: Path) -> tuple[pd.DataFrame, str]:
    if args.split_metadata is not None:
        path = args.split_metadata.resolve()
        return load_or_create_split_metadata(data_root, split_metadata_path=path), str(path)
    default_path = data_root / "splits" / "split_metadata.csv"
    if default_path.exists():
        return load_or_create_split_metadata(data_root), str(default_path)
    return (
        load_or_create_split_metadata(data_root, seed=args.seed),
        f"deterministic_80_10_10_seed_{args.seed}",
    )


def prepare_target_normalizer(
    args: argparse.Namespace,
    data_root: Path,
    split_metadata: pd.DataFrame,
    output_dir: Path,
) -> TargetDeltaNormalizer | None:
    stats_path = output_dir / "target_delta_normalization.json"
    if not args.normalize_target_delta:
        return None
    if args.eval_only:
        if not stats_path.exists():
            raise FileNotFoundError(
                f"{stats_path} is required for eval-only normalized-target decoding."
            )
        payload = json.loads(stats_path.read_text())
        return TargetDeltaNormalizer(mean=payload["mean"], std=payload["std"])

    train_frame = split_metadata[split_metadata["split"] == "train"].copy()
    normalizer = fit_target_delta_normalizer(data_root, train_frame)
    write_json(
        stats_path,
        {
            "mean": normalizer.mean,
            "std": normalizer.std,
            "shape": "[1, 1, 2]",
            "epsilon": 1e-8,
            "fit_scope": "train split over all samples and timesteps, separately for dx/dy",
        },
    )
    return normalizer


def fit_target_delta_normalizer(
    data_root: Path, train_frame: pd.DataFrame, eps: float = 1e-8
) -> TargetDeltaNormalizer:
    if train_frame.empty:
        raise ValueError("Cannot fit target delta normalization without train samples.")
    arrays = [
        np.load(data_root / row.label_delta_displacement_path).astype(np.float32)
        for row in train_frame.itertuples(index=False)
    ]
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), eps)
    return TargetDeltaNormalizer(mean=mean.tolist(), std=std.tolist())


def train(
    args: argparse.Namespace,
    split_metadata: pd.DataFrame,
    device: torch.device,
    normalizer: TargetDeltaNormalizer | None,
    config: dict[str, object],
) -> None:
    output_dir = args.output_dir.resolve()
    train_frame = split_metadata[split_metadata["split"] == "train"].copy()
    val_frame = split_metadata[split_metadata["split"] == "val"].copy()
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and val splits are required.")

    train_loader = make_loader(args, train_frame, shuffle=True, device=device)
    val_loader = make_loader(args, val_frame, shuffle=False, device=device)
    model = build_delta_decoder(
        args.model,
        args.channels,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(args, model, train_loader, device, normalizer, optimizer)
        val_metrics = run_epoch(args, model, val_loader, device, normalizer, optimizer=None)
        row = {
            "epoch": epoch,
            "elapsed_seconds": time.perf_counter() - start_time,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        append_jsonl(log_path, row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_total_loss": val_metrics["total_loss"],
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last_model.pt")
        improved = val_metrics["total_loss"] < best_val_loss - args.early_stopping_min_delta
        if improved:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "checkpoints" / "best_model.pt")
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch:04d}/{args.epochs} "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_delta_ADE={val_metrics['delta_ADE']:.6f} "
            f"val_integrated_ADE={val_metrics['integrated_ADE']:.6f} "
            f"best_epoch={best_epoch}"
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"Early stopping after {stale_epochs} stale validation epochs.")
            break

    write_json(
        output_dir / "best_summary.json",
        {"best_epoch": best_epoch, "best_val_total_loss": best_val_loss},
    )


def run_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: TargetDeltaNormalizer | None,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "total_loss": 0.0,
        "delta_loss": 0.0,
        "integrated_loss": 0.0,
        "delta_ADE": 0.0,
        "delta_FDE": 0.0,
        "integrated_ADE": 0.0,
        "integrated_FDE": 0.0,
    }
    sample_count = 0
    mse = nn.MSELoss()

    for batch_index, batch in enumerate(loader):
        if args.max_train_batches is not None and training and batch_index >= args.max_train_batches:
            break
        if args.max_eval_batches is not None and not training and batch_index >= args.max_eval_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        delta_gt = batch["delta_gt"].to(device, non_blocking=True)
        absolute_gt = batch["absolute_gt"].to(device, non_blocking=True)
        start_xy = batch["start_xy"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            model_output = model(images)
            if normalizer is None:
                delta_pred = model_output
                delta_loss = mse(delta_pred, delta_gt)
            else:
                delta_pred_norm = model_output
                delta_gt_norm = normalizer.normalize(delta_gt)
                delta_loss = mse(delta_pred_norm, delta_gt_norm)
                delta_pred = normalizer.unnormalize(delta_pred_norm)
            abs_pred = integrate_delta_torch(start_xy, delta_pred)
            integrated_loss = mse(abs_pred, absolute_gt)
            total_loss = delta_loss + args.beta_integrated_loss * integrated_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        delta_errors = torch.linalg.norm(delta_pred.detach() - delta_gt, dim=2)
        integrated_errors = torch.linalg.norm(abs_pred.detach() - absolute_gt, dim=2)
        totals["total_loss"] += float(total_loss.detach().cpu()) * batch_size
        totals["delta_loss"] += float(delta_loss.detach().cpu()) * batch_size
        totals["integrated_loss"] += float(integrated_loss.detach().cpu()) * batch_size
        totals["delta_ADE"] += float(delta_errors.mean(dim=1).sum().cpu())
        totals["delta_FDE"] += float(delta_errors[:, -1].sum().cpu())
        totals["integrated_ADE"] += float(integrated_errors.mean(dim=1).sum().cpu())
        totals["integrated_FDE"] += float(integrated_errors[:, -1].sum().cpu())
        sample_count += batch_size

    return {key: value / max(1, sample_count) for key, value in totals.items()}


def evaluate(
    args: argparse.Namespace,
    split_metadata: pd.DataFrame,
    device: torch.device,
    checkpoint_path: Path,
    normalizer: TargetDeltaNormalizer | None,
) -> None:
    print("Evaluation uses physical unnormalized delta predictions for saved arrays and metrics.")
    output_dir = args.output_dir.resolve()
    model = build_delta_decoder(
        args.model,
        args.channels,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_rows: list[dict[str, object]] = []
    plotted = 0
    for split in ["train", "val", "test"]:
        frame = split_metadata[split_metadata["split"] == split].copy()
        if frame.empty:
            continue
        loader = make_loader(args, frame, shuffle=False, device=device)
        rows, plotted = predict_split(args, model, loader, device, split, plotted, normalizer)
        all_rows.extend(rows)
        summary = summarize_metric_rows(rows)
        if split == "val":
            write_json(output_dir / "metrics_val.json", summary)
        if split == "test":
            write_json(output_dir / "metrics_test.json", summary)

    pd.DataFrame(all_rows).to_csv(output_dir / "per_sample_metrics.csv", index=False)


def predict_split(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    plotted: int,
    normalizer: TargetDeltaNormalizer | None,
) -> tuple[list[dict[str, object]], int]:
    output_dir = args.output_dir.resolve()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            model_output = model(images)
            if normalizer is None:
                delta_pred_tensor = model_output
            else:
                delta_pred_tensor = normalizer.unnormalize(model_output)
            delta_pred = delta_pred_tensor.cpu().numpy().astype(np.float32)
            delta_gt = batch["delta_gt"].cpu().numpy().astype(np.float32)
            absolute_gt = batch["absolute_gt"].cpu().numpy().astype(np.float32)
            starts = batch["start_xy"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                pred_delta = delta_pred[index]
                pred_xy = integrate_delta_numpy(starts[index], pred_delta)
                gt_delta = delta_gt[index]
                gt_xy = absolute_gt[index]
                np.save(output_dir / "predictions_delta" / f"{sample_id}_pred_delta.npy", pred_delta)
                np.save(
                    output_dir / "predictions_integrated_xy" / f"{sample_id}_pred_xy.npy",
                    pred_xy,
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **compute_basic_metrics(pred_delta, gt_delta, "delta"),
                        **compute_basic_metrics(pred_xy, gt_xy, "integrated"),
                        **compute_alignment_metrics(pred_xy, gt_xy),
                    }
                )
                if split in {"val", "test"} and plotted < args.num_plots:
                    plot_comparison(
                        output_dir / "comparison_plots" / f"{split}_{sample_id}.png",
                        sample_id,
                        gt_xy,
                        pred_xy,
                        gt_delta,
                        pred_delta,
                    )
                    plotted += 1
    return rows, plotted


def plot_comparison(
    path: Path,
    sample_id: str,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    gt_delta: np.ndarray,
    pred_delta: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(gt_xy[:, 0], gt_xy[:, 1], label="gt absolute", linewidth=1.5)
    axes[0].plot(pred_xy[:, 0], pred_xy[:, 1], label="pred integrated", linewidth=1.5)
    axes[0].scatter(gt_xy[0, 0], gt_xy[0, 1], s=18, label="start")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8)
    axes[0].set_title(sample_id)

    axes[1].plot(gt_delta[:, 0], label="gt dx", linewidth=1.0)
    axes[1].plot(pred_delta[:, 0], label="pred dx", linewidth=1.0)
    axes[1].legend(fontsize=8)
    axes[1].set_title("dx")

    axes[2].plot(gt_delta[:, 1], label="gt dy", linewidth=1.0)
    axes[2].plot(pred_delta[:, 1], label="pred dy", linewidth=1.0)
    axes[2].legend(fontsize=8)
    axes[2].set_title("dy")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_loader(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = DeltaDisplacementPairedDataset(args.data_dir, frame, channels=args.channels)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def make_config(
    args: argparse.Namespace,
    device: torch.device,
    split_metadata: pd.DataFrame,
    split_source: str,
    normalizer: TargetDeltaNormalizer | None,
) -> dict[str, object]:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "device_resolved": str(device),
            "image_normalization": IMAGE_NORMALIZATION,
            "pseudo_modal_channels": {"R": "GASF", "G": "GADF", "B": "MTF"},
            "target": TARGET_TYPE,
            "normalize_target_delta": normalizer is not None,
            "target_delta_mean": normalizer.mean if normalizer is not None else None,
            "target_delta_std": normalizer.std if normalizer is not None else None,
            "loss_without_target_normalization": (
                "MSE(delta_pred, delta_gt) + beta_integrated_loss * "
                "MSE(integrate_delta(start_xy, delta_pred), absolute_gt)"
            ),
            "loss_with_target_normalization": (
                "MSE(delta_pred_norm, delta_gt_norm) + beta_integrated_loss * "
                "MSE(integrate_delta(start_xy, unnormalize(delta_pred_norm)), absolute_gt)"
            ),
            "integration": INTEGRATION_FORMULA,
            "recommended_beta_ablations": [0.0, 0.05, 0.2],
            "split_source": split_source,
            "split_counts": {
                split: int((split_metadata["split"] == split).sum())
                for split in ["train", "val", "test"]
            },
        }
    )
    return config


def print_config_summary(config: dict[str, object]) -> None:
    print("Delta displacement decoder configuration")
    for key in [
        "model",
        "channels",
        "encoder_type",
        "data_dir",
        "output_dir",
        "image_normalization",
        "target",
        "normalize_target_delta",
        "beta_integrated_loss",
        "batch_size",
        "epochs",
        "lr",
        "split_counts",
        "split_source",
        "device_resolved",
    ]:
        print(f"  {key}: {config.get(key)}")
    print("  beta ablations to consider: 0.0, 0.05, 0.2")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()


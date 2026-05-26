"""Train Stage 1 trajectory sequence autoencoder for Week3 latent alignment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
if str(WEEK03_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK03_SCRIPTS_DIR))

from sequence_latent_models import TrajectorySequenceAutoencoder  # noqa: E402
from sequence_latent_utils import (  # noqa: E402
    TrajectoryDeltaDataset,
    average_sums,
    batch_metric_sums,
    load_split_metadata,
    resolve_device,
    save_checkpoint,
    sequence_losses,
    set_seed,
    write_json,
    write_train_log,
)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    split_metadata = load_split_metadata(args.data_root.resolve(), seed=args.seed)
    split_metadata.to_csv(output_dir / "split_copy.csv", index=False)
    config = make_config(args, device, split_metadata)
    write_json(output_dir / "config.json", config)

    train_frame = split_metadata[split_metadata["split"] == "train"].copy()
    val_frame = split_metadata[split_metadata["split"] == "val"].copy()
    if train_frame.empty or val_frame.empty:
        raise ValueError("Stage 1 requires non-empty train and val splits.")

    train_loader = make_loader(args, train_frame, shuffle=True, device=device)
    val_loader = make_loader(args, val_frame, shuffle=False, device=device)
    model = TrajectorySequenceAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.beta_integrated,
            args.grad_clip_norm,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.beta_integrated,
            args.grad_clip_norm,
            optimizer=None,
        )
        if args.best_metric not in val_metrics:
            available = ", ".join(sorted(val_metrics))
            raise ValueError(f"Unknown --best-metric '{args.best_metric}'. Available validation metrics: {available}")
        row = {
            "epoch": epoch,
            "elapsed_time": time.perf_counter() - start_time,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        log_rows.append(row)
        write_train_log(output_dir / "train_log.csv", log_rows)

        checkpoint_metrics = {"train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"], **val_metrics}
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            epoch,
            config,
            model.state_dict(),
            optimizer.state_dict(),
            checkpoint_metrics,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                epoch,
                config,
                model.state_dict(),
                optimizer.state_dict(),
                checkpoint_metrics,
            )
        current_score = val_metrics[args.best_metric]
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(output_dir / "best.pt", epoch, config, model.state_dict(), optimizer.state_dict(), checkpoint_metrics)
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                epoch,
                config,
                model.state_dict(),
                optimizer.state_dict(),
                checkpoint_metrics,
            )

        print(
            f"epoch={epoch:04d}/{args.epochs} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_integrated_ADE={val_metrics['integrated_ADE']:.6f} "
            f"best_{args.best_metric}={best_score:.6f} best_epoch={best_epoch}"
        )

    if best_metrics is None:
        raise ValueError("Training finished without selecting a best checkpoint.")
    final_metrics = {"best_epoch": best_epoch, "best_metric": args.best_metric, "best_score": best_score, **val_metrics}
    save_checkpoint(output_dir / "final.pt", args.epochs, config, model.state_dict(), optimizer.state_dict(), final_metrics)
    best_payload = {
        "best_epoch": best_epoch,
        "best_metric": args.best_metric,
        "best_score": best_score,
        **best_metrics,
    }
    write_json(output_dir / "best_metrics.json", best_payload)
    write_json(output_dir / "final_validation_metrics.json", val_metrics)
    write_json(output_dir / "validation_metrics.json", val_metrics)
    write_json(output_dir / "best_summary.json", best_payload)
    print("Wrote final_validation_metrics.json; validation_metrics.json is kept as a final-epoch alias.")
    print(f"Done. Stage 1 outputs saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_delta_displacement_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "week03" / "results" / "sequence_latent" / "stage1_traj_ae_latent64",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta-integrated", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--best-metric", default="integrated_ADE")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    return parser.parse_args()


def make_loader(args: argparse.Namespace, frame: pd.DataFrame, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        TrajectoryDeltaDataset(args.data_root.resolve(), frame),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: TrajectorySequenceAutoencoder,
    loader: DataLoader,
    device: torch.device,
    beta_integrated: float,
    grad_clip_norm: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "delta_loss": 0.0, "integrated_loss": 0.0, "delta_mse": 0.0, "delta_ADE": 0.0, "delta_FDE": 0.0, "integrated_ADE": 0.0, "integrated_FDE": 0.0}
    sample_count = 0
    for batch in loader:
        delta_gt = batch["delta"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            delta_hat, _ = model(delta_gt)
            losses = sequence_losses(delta_hat, delta_gt, beta_integrated)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        batch_size = delta_gt.shape[0]
        for key in ["loss", "delta_loss", "integrated_loss"]:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        metric_sums = batch_metric_sums(delta_hat.detach(), delta_gt.detach())
        for key, value in metric_sums.items():
            totals[key] += value
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("No samples were loaded in this epoch.")
    return average_sums(totals, sample_count)


def make_config(args: argparse.Namespace, device: torch.device, split_metadata: pd.DataFrame) -> dict[str, Any]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["data_root"] = str(args.data_root.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    return {
        **config,
        "device_resolved": str(device),
        "stage": "trajectory_sequence_autoencoder",
        "input_delta_shape": "[B,223,2]",
        "label_length_policy": "if stored label is [224,2], drop delta[0] placeholder",
        "loss": "MSE(delta_hat, delta_gt) + beta_integrated * MSE(cumsum(delta_hat), cumsum(delta_gt))",
        "split_counts": {split: int((split_metadata["split"] == split).sum()) for split in ["train", "val", "test"]},
    }


if __name__ == "__main__":
    main()

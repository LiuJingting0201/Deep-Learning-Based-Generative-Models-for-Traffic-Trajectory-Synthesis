"""Train Week3 structure-aware ResNet18 delta-displacement decoders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
for path in [SCRIPTS_DIR, WEEK03_SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_delta_displacement_resnet18_decoder_ablation import (  # noqa: E402
    DeltaDisplacementDataset,
    compute_basic_metrics,
    compute_variance_diagnostics,
    evaluate_delta_loader,
    fit_delta_label_scaler,
    integrate_delta,
    plot_delta_distribution_diagnostics,
    plot_evaluation_outputs,
    summarize_delta_distribution,
    summarize_delta_experiment_metrics,
    trajectory_length,
    verify_delta_data_and_split,
    write_train_log,
)
from run_resnet18_decoder_ablation import (  # noqa: E402
    format_metric,
    plot_training_curves,
    resolve_device,
    run_epoch,
    set_seed,
    write_json,
)
from structure_aware_resnet18_models import (  # noqa: E402
    DecompMtfLocalResNet18DeltaDecoder,
    HardStructureResNet18DeltaDecoder,
    RawResNet18DeltaDecoder,
    StructureDecomposedResNet18DeltaDecoder,
)


MODEL_TYPES = {
    "raw_resnet18": RawResNet18DeltaDecoder,
    "decomp_resnet18": StructureDecomposedResNet18DeltaDecoder,
    "hard_resnet18": HardStructureResNet18DeltaDecoder,
    "decomp_mtf_local_resnet18": DecompMtfLocalResNet18DeltaDecoder,
}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.resolve()
    for subdir in ["checkpoints", "plots", "evaluation", "evaluation/plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(args.data_root.resolve())
    if not args.skip_training:
        train(args, split_metadata, device)
    summary = evaluate(args, split_metadata, device)
    write_week3_report(args, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/jliu/data_no_speed_delta_displacement_paired"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "week03" / "results" / "structure_resnet18_decoder",
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="decomp_resnet18")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--metrics-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--near-zero-eps", type=float, default=1e-3)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def build_model(model_type: str, dropout: float) -> nn.Module:
    model_class = MODEL_TYPES[model_type]
    if model_type == "decomp_resnet18":
        return model_class(dropout=dropout, return_debug=False)
    return model_class(dropout=dropout)


def train(args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device) -> None:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    val_metadata = split_metadata[split_metadata["split"] == "val"].copy()
    scaler = fit_delta_label_scaler(data_root, train_metadata)

    write_json(output_dir / "label_scaler.json", asdict(scaler))
    shutil.copy2(data_root / "splits" / "split_metadata.csv", output_dir / "split_copy.csv")
    config = make_config(args, device, train_metadata, val_metadata)
    write_json(output_dir / "config.json", config)

    train_loader = DataLoader(
        DeltaDisplacementDataset(data_root, train_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_metadata) % args.batch_size == 1,
    )
    val_loader = DataLoader(
        DeltaDisplacementDataset(data_root, val_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_type, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    log_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        compute_diag = epoch == 1 or epoch % args.metrics_every == 0
        train_diag = evaluate_delta_loader(model, train_loader, criterion, scaler, device, compute_diag)
        val_diag = evaluate_delta_loader(model, val_loader, criterion, scaler, device, compute_diag)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_eval_loss": train_diag["loss"],
            "val_loss": val_diag["loss"],
            "train_ADE": train_diag["ade"],
            "train_FDE": train_diag["fde"],
            "val_ADE": val_diag["ade"],
            "val_FDE": val_diag["fde"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_time": time.perf_counter() - start_time,
        }
        log_rows.append(row)
        write_train_log(output_dir / "train_log.csv", log_rows)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_diag["loss"],
            "train_loss": train_loss,
            "config": config,
            "label_scaler": asdict(scaler),
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last_model.pt")
        if val_diag["loss"] < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_diag["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "checkpoints" / "best_model.pt")
        else:
            epochs_without_improvement += 1

        plot_training_curves(log_rows, output_dir / "plots")
        print(
            f"epoch={epoch:04d}/{args.epochs} model={args.model_type} "
            f"train_loss={train_loss:.6f} val_loss={val_diag['loss']:.6f} "
            f"train_delta_ADE={format_metric(train_diag['ade'])} "
            f"val_delta_ADE={format_metric(val_diag['ade'])} "
            f"best_val_loss={best_val_loss:.6f} best_epoch={best_epoch}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopping_triggered = True
            print(
                "Early stopping triggered after "
                f"{epochs_without_improvement} epochs without validation improvement."
            )
            break

    write_json(
        output_dir / "best_summary.json",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_epoch": int(log_rows[-1]["epoch"]) if log_rows else 0,
            "early_stopping_triggered": early_stopping_triggered,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
    )


def make_config(
    args: argparse.Namespace,
    device: torch.device,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    config.update(
        {
            "model": model_class_name(args.model_type),
            "backbone": "torchvision.models.resnet18(weights=None)",
            "pretrained": False,
            "conv1_input_channels": conv1_input_channels(args.model_type),
            "head": "feature_dim -> 1024 -> 448 -> reshape[224,2]",
            "optimizer": "Adam",
            "scheduler": "none",
            "image_normalization": "[0, 1]",
            "label_normalization": "train_mean_std_delta_displacement_xy",
            "loss": "MSELoss in normalized delta-displacement space",
            "coordinate_space": "delta_displacement_xy",
            "integrated_absolute_evaluation": "oracle true start point",
            "input_representation": describe_input_representation(args.model_type),
            "device_resolved": str(device),
            "train_samples": len(train_metadata),
            "val_samples": len(val_metadata),
        }
    )
    return config


def evaluate(
    args: argparse.Namespace, split_metadata: pd.DataFrame, device: torch.device
) -> dict[str, object]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    evaluation_dir = output_dir / "evaluation"
    pred_delta_dir = evaluation_dir / "predictions_delta_displacement"
    pred_absolute_dir = evaluation_dir / "predictions_absolute_integrated"
    plots_dir = evaluation_dir / "plots"
    for directory in [pred_delta_dir, pred_absolute_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    scaler_data = json.loads((output_dir / "label_scaler.json").read_text())
    from run_resnet18_decoder_ablation import LabelScaler  # noqa: PLC0415

    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    model = build_model(args.model_type, args.dropout).to(device)
    checkpoint = torch.load(output_dir / "checkpoints" / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    delta_timestep_rows: list[dict[str, object]] = []
    integrated_timestep_rows: list[dict[str, object]] = []
    predictions_abs: dict[str, dict[str, np.ndarray]] = {}
    truths_abs: dict[str, dict[str, np.ndarray]] = {}
    predictions_delta: dict[str, dict[str, np.ndarray]] = {}
    truths_delta: dict[str, dict[str, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        split_frame = split_metadata[split_metadata["split"] == split].copy()
        loader = DataLoader(
            DeltaDisplacementDataset(data_root, split_frame, scaler),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        (
            split_pred_delta,
            split_true_delta,
            split_pred_abs,
            split_true_abs,
            split_rows,
            delta_timestep_ade,
            integrated_timestep_ade,
        ) = predict_split(model, loader, scaler, device, pred_delta_dir, pred_absolute_dir, split, args.near_zero_eps)
        predictions_delta[split] = split_pred_delta
        truths_delta[split] = split_true_delta
        predictions_abs[split] = split_pred_abs
        truths_abs[split] = split_true_abs
        rows.extend(split_rows)
        delta_timestep_rows.extend(
            {"split": split, "timestep": index, "delta_ade": float(value)}
            for index, value in enumerate(delta_timestep_ade)
        )
        integrated_timestep_rows.extend(
            {"split": split, "timestep": index, "integrated_ade": float(value)}
            for index, value in enumerate(integrated_timestep_ade)
        )

    per_sample = pd.DataFrame(rows)
    per_timestep = pd.merge(
        pd.DataFrame(delta_timestep_rows),
        pd.DataFrame(integrated_timestep_rows),
        on=["split", "timestep"],
    )
    per_sample.to_csv(evaluation_dir / "per_sample_metrics.csv", index=False)
    per_timestep.to_csv(evaluation_dir / "per_timestep_ade.csv", index=False)

    metrics_summary = summarize_delta_experiment_metrics(per_sample)
    distribution_summary = summarize_delta_distribution(per_sample)
    variance_summary = compute_variance_diagnostics(predictions_delta, truths_delta, predictions_abs, truths_abs)
    write_json(evaluation_dir / "metrics_summary.json", metrics_summary)
    write_json(evaluation_dir / "delta_distribution_diagnostics.json", distribution_summary)
    write_json(evaluation_dir / "prediction_variance_diagnostics.json", variance_summary)
    plot_evaluation_outputs(per_sample, per_timestep, predictions_abs["test"], truths_abs["test"], plots_dir)
    plot_delta_distribution_diagnostics(per_sample, plots_dir)
    return {
        "checkpoint": checkpoint,
        "metrics_summary": metrics_summary,
        "distribution_summary": distribution_summary,
        "variance_summary": variance_summary,
    }


def predict_split(
    model: nn.Module,
    loader: DataLoader,
    scaler: object,
    device: torch.device,
    pred_delta_dir: Path,
    pred_absolute_dir: Path,
    split: str,
    near_zero_eps: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
]:
    pred_delta_by_id: dict[str, np.ndarray] = {}
    true_delta_by_id: dict[str, np.ndarray] = {}
    pred_abs_by_id: dict[str, np.ndarray] = {}
    true_abs_by_id: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    delta_timestep_error_sum = np.zeros(224, dtype=np.float64)
    integrated_timestep_error_sum = np.zeros(224, dtype=np.float64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            pred_norm = model(images)
            pred_delta = scaler.inverse_transform_tensor(pred_norm).cpu().numpy().astype(np.float32)
            true_delta = batch["label_raw"].cpu().numpy().astype(np.float32)
            true_abs = batch["absolute"].cpu().numpy().astype(np.float32)
            starts = batch["start"].cpu().numpy().astype(np.float32)

            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                vehicle_id = str(batch["vehicle_id"][index])
                sample_pred_delta = pred_delta[index]
                sample_true_delta = true_delta[index]
                sample_true_abs = true_abs[index]
                sample_start = starts[index]
                sample_pred_abs = integrate_delta(sample_start, sample_pred_delta)

                np.save(pred_delta_dir / f"{sample_id}_pred_delta.npy", sample_pred_delta)
                np.save(pred_absolute_dir / f"{sample_id}_pred_absolute_integrated.npy", sample_pred_abs)

                delta_metrics, delta_errors = compute_basic_metrics(sample_pred_delta, sample_true_delta, "delta")
                integrated_metrics, integrated_errors = compute_basic_metrics(
                    sample_pred_abs, sample_true_abs, "integrated"
                )
                pred_length = trajectory_length(sample_pred_abs)
                true_length = trajectory_length(sample_true_abs)
                pred_delta_mag = np.linalg.norm(sample_pred_delta, axis=1)
                true_delta_mag = np.linalg.norm(sample_true_delta, axis=1)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "vehicle_id": vehicle_id,
                        "split": split,
                        **delta_metrics,
                        **integrated_metrics,
                        "start_error": float(integrated_errors[0]),
                        "end_error": float(integrated_errors[-1]),
                        "trajectory_length_error": float(abs(pred_length - true_length)),
                        "trajectory_length_ratio": float(pred_length / true_length) if true_length else np.nan,
                        "cumulative_drift_final": float(integrated_errors[-1]),
                        "pred_delta_magnitude_mean": float(pred_delta_mag.mean()),
                        "true_delta_magnitude_mean": float(true_delta_mag.mean()),
                        "pred_delta_magnitude_median": float(np.median(pred_delta_mag)),
                        "true_delta_magnitude_median": float(np.median(true_delta_mag)),
                        "pred_near_zero_delta_ratio": float(np.mean(pred_delta_mag <= near_zero_eps)),
                        "true_near_zero_delta_ratio": float(np.mean(true_delta_mag <= near_zero_eps)),
                    }
                )
                pred_delta_by_id[sample_id] = sample_pred_delta
                true_delta_by_id[sample_id] = sample_true_delta
                pred_abs_by_id[sample_id] = sample_pred_abs
                true_abs_by_id[sample_id] = sample_true_abs
                delta_timestep_error_sum += delta_errors
                integrated_timestep_error_sum += integrated_errors
                total += 1

    return (
        pred_delta_by_id,
        true_delta_by_id,
        pred_abs_by_id,
        true_abs_by_id,
        rows,
        delta_timestep_error_sum / max(1, total),
        integrated_timestep_error_sum / max(1, total),
    )


def describe_input_representation(model_type: str) -> str:
    descriptions = {
        "raw_resnet18": "raw RGB [GASF,GADF,MTF] in [0,1]",
        "decomp_resnet18": "[R_sym,R_res,G_antisym,G_res,B_raw], with R/G converted to [-1,1]",
        "hard_resnet18": "[R_sym,G_antisym,B_raw], with R/G converted to [-1,1]",
        "decomp_mtf_local_resnet18": "[R_sym,R_res,G_antisym,G_res,B_raw,B_local]",
    }
    return descriptions[model_type]


def model_class_name(model_type: str) -> str:
    return MODEL_TYPES[model_type].__name__


def conv1_input_channels(model_type: str) -> int:
    channels = {
        "raw_resnet18": 3,
        "decomp_resnet18": 5,
        "hard_resnet18": 3,
        "decomp_mtf_local_resnet18": 6,
    }
    return channels[model_type]


def write_week3_report(args: argparse.Namespace, summary: dict[str, object]) -> None:
    output_dir = args.output_dir.resolve()
    metrics = summary["metrics_summary"]
    best = json.loads((output_dir / "best_summary.json").read_text())
    config = json.loads((output_dir / "config.json").read_text())
    report = {
        "model_type": args.model_type,
        "model": config["model"],
        "input_representation": config["input_representation"],
        "conv1_input_channels": config["conv1_input_channels"],
        "best_epoch": best["best_epoch"],
        "best_val_loss": best["best_val_loss"],
        "delta_ADE_FDE": {
            split: {
                "delta_ADE": metrics[split]["delta_ADE"]["mean"],
                "delta_FDE": metrics[split]["delta_FDE"]["mean"],
            }
            for split in ["train", "val", "test"]
        },
        "integrated_ADE_FDE": {
            split: {
                "integrated_ADE": metrics[split]["integrated_ADE"]["mean"],
                "integrated_FDE": metrics[split]["integrated_FDE"]["mean"],
            }
            for split in ["train", "val", "test"]
        },
        "trajectory_length_ratio": {
            split: metrics[split]["trajectory_length_ratio"]["mean"] for split in ["train", "val", "test"]
        },
    }
    write_json(output_dir / "week3_report.json", report)
    lines = [
        "# Week3 Structure ResNet18 Report",
        "",
        f"- model_type: {report['model_type']}",
        f"- model: {report['model']}",
        f"- input_representation: {report['input_representation']}",
        f"- conv1_input_channels: {report['conv1_input_channels']}",
        f"- best_epoch: {report['best_epoch']}",
        f"- best_val_loss: {report['best_val_loss']}",
        "",
        "| split | delta ADE | delta FDE | integrated ADE | integrated FDE | trajectory length ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        lines.append(
            f"| {split} | {report['delta_ADE_FDE'][split]['delta_ADE']:.6f} | "
            f"{report['delta_ADE_FDE'][split]['delta_FDE']:.6f} | "
            f"{report['integrated_ADE_FDE'][split]['integrated_ADE']:.6f} | "
            f"{report['integrated_ADE_FDE'][split]['integrated_FDE']:.6f} | "
            f"{report['trajectory_length_ratio'][split]:.6f} |"
        )
    (output_dir / "week3_report.md").write_text("\n".join(lines) + "\n")
    print((output_dir / "week3_report.md").read_text())


if __name__ == "__main__":
    main()

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
import matplotlib.pyplot as plt
from PIL import Image
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week03" / "results" / "structure_resnet18_decoder"
DEFAULT_MAP_DIR = Path("/home/jliu/data_no_speed_delta_displacement_paired/maps_oracle_bbox")


def main() -> None:
    args = parse_args()
    finalize_output_dir(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir.resolve()
    for subdir in ["checkpoints", "plots", "evaluation", "evaluation/plots"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(args.data_root.resolve())
    if args.use_map:
        preflight_map_inputs(args, split_metadata)
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
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="decomp_resnet18")
    parser.add_argument("--use-map", action="store_true")
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--map-fusion", choices=["concat"], default="concat")
    parser.add_argument("--run-name", default="")
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


class MapConcatDeltaDisplacementDataset(DeltaDisplacementDataset):
    def __init__(self, data_root: Path, metadata: pd.DataFrame, scaler: object, map_dir: Path) -> None:
        super().__init__(data_root, metadata, scaler)
        self.map_dir = map_dir

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        sample_id = str(item["sample_id"])
        map_path = self.map_dir / f"{sample_id}_map.png"
        if not map_path.exists():
            raise FileNotFoundError(map_path)
        map_image = Image.open(map_path).convert("L")
        if map_image.size != (224, 224):
            raise ValueError(f"Expected 224x224 map for {sample_id}, got {map_image.size}")
        map_tensor = torch.from_numpy(np.asarray(map_image, dtype=np.float32) / 255.0).unsqueeze(0)
        item["map"] = map_tensor.contiguous()
        item["map_path"] = str(map_path)
        item["image"] = torch.cat([item["image"], item["map"]], dim=0).contiguous()
        return item


def make_dataset(args: argparse.Namespace, data_root: Path, metadata: pd.DataFrame, scaler: object) -> DeltaDisplacementDataset:
    if not args.use_map:
        return DeltaDisplacementDataset(data_root, metadata, scaler)
    if args.map_fusion != "concat":
        raise ValueError(f"Unsupported map fusion: {args.map_fusion}")
    return MapConcatDeltaDisplacementDataset(data_root, metadata, scaler, args.map_dir.resolve())


def build_model(model_type: str, dropout: float, input_channels: int = 3) -> nn.Module:
    model_class = MODEL_TYPES[model_type]
    if input_channels != 3 and model_type != "raw_resnet18":
        raise ValueError("Map concat is currently implemented only for --model-type raw_resnet18.")
    if model_type == "raw_resnet18":
        return model_class(dropout=dropout, in_channels=input_channels)
    if model_type == "decomp_resnet18":
        return model_class(dropout=dropout, return_debug=False)
    return model_class(dropout=dropout)


def input_channels(args: argparse.Namespace) -> int:
    return 4 if args.use_map and args.map_fusion == "concat" else 3


def infer_map_crop_mode(map_dir: Path) -> str:
    name = map_dir.expanduser().resolve().name
    if "oracle_bbox" in name:
        return "oracle_bbox"
    if "start_center" in name:
        return "start_center"
    return name.replace("maps_", "") or "unknown_crop"


def finalize_output_dir(args: argparse.Namespace) -> None:
    if not args.use_map:
        return
    if args.model_type != "raw_resnet18":
        raise ValueError("Map concat is currently implemented only with --model-type raw_resnet18.")
    crop_mode = infer_map_crop_mode(args.map_dir)
    run_name = args.run_name or args.model_type
    suffix = f"map_concat_{crop_mode}_{run_name}"
    output_dir = args.output_dir
    if suffix not in output_dir.name:
        args.output_dir = output_dir / suffix


def preflight_map_inputs(args: argparse.Namespace, split_metadata: pd.DataFrame) -> None:
    map_dir = args.map_dir.resolve()
    sample_ids = split_metadata["sample_id"].astype(str).tolist()
    missing = [sample_id for sample_id in sample_ids if not (map_dir / f"{sample_id}_map.png").exists()]
    found = len(sample_ids) - len(missing)
    print(f"Map preflight: found {found}/{len(sample_ids)} maps in {map_dir}; missing {len(missing)}.")
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} map raster(s) in {map_dir}. "
            f"Expected files like {{sample_id}}_map.png. First missing: {preview}"
        )

    extents_path = extent_metadata_path(map_dir)
    if extents_path.exists():
        extents = pd.read_csv(extents_path)
        required = {"sample_id", "min_x", "max_x", "min_y", "max_y", "crs", "crop_mode"}
        missing_columns = required.difference(extents.columns)
        if missing_columns:
            print(
                f"WARNING: map extent metadata is missing columns {sorted(missing_columns)}; "
                "map-background plots will be skipped."
            )
            return
        missing_extent_ids = sorted(set(sample_ids) - set(extents["sample_id"].astype(str)))
        print(
            f"Map extent preflight: found metadata for {len(sample_ids) - len(missing_extent_ids)}/"
            f"{len(sample_ids)} samples in {extents_path}."
        )
        if missing_extent_ids:
            print(
                "WARNING: map extent metadata is incomplete; map-background plots will be skipped. "
                f"First missing extents: {', '.join(missing_extent_ids[:10])}"
            )
    else:
        print(f"WARNING: map extent metadata not found at {extents_path}; map-background plots will be skipped.")


def extent_metadata_path(map_dir: Path) -> Path:
    name = map_dir.resolve().name
    if "oracle_bbox" in name:
        return map_dir.parent / "maps_oracle_bbox_extents.csv"
    if "start_center" in name:
        return map_dir.parent / "maps_start_center_extents.csv"
    return map_dir.parent / f"{name}_extents.csv"


def load_extent_lookup(map_dir: Path) -> dict[str, dict[str, object]] | None:
    path = extent_metadata_path(map_dir)
    if not path.exists():
        print(f"WARNING: skipping map-background plots because extent metadata is missing: {path}")
        return None
    frame = pd.read_csv(path)
    required = {"sample_id", "min_x", "max_x", "min_y", "max_y", "crs", "crop_mode"}
    missing = required.difference(frame.columns)
    if missing:
        print(f"WARNING: skipping map-background plots because {path} is missing columns: {sorted(missing)}")
        return None
    return {str(row.sample_id): row._asdict() for row in frame.itertuples(index=False)}


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
        make_dataset(args, data_root, train_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_metadata) % args.batch_size == 1,
    )
    val_loader = DataLoader(
        make_dataset(args, data_root, val_metadata, scaler),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_type, args.dropout, input_channels(args)).to(device)
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
            "conv1_input_channels": input_channels(args) if args.use_map else conv1_input_channels(args.model_type),
            "head": "feature_dim -> 1024 -> 448 -> reshape[224,2]",
            "optimizer": "Adam",
            "scheduler": "none",
            "image_normalization": "[0, 1]",
            "label_normalization": "train_mean_std_delta_displacement_xy",
            "loss": "MSELoss in normalized delta-displacement space",
            "coordinate_space": "delta_displacement_xy",
            "integrated_absolute_evaluation": "oracle true start point",
            "input_representation": describe_input_representation(args.model_type),
            "map_crop_mode": infer_map_crop_mode(args.map_dir) if args.use_map else "none",
            "map_fusion_effective": args.map_fusion if args.use_map else "none",
            "map_extent_metadata": str(extent_metadata_path(args.map_dir)) if args.use_map else "none",
            "trajectory_coordinate_frame": "raw local x/y from labels_absolute, not projected lon/lat",
            "map_coordinate_frame": "projected OSM CRS from extent metadata when --use-map is enabled",
            "map_background_overlay_warning": (
                "Map raster extents are projected OSM coordinates, while decoder trajectories are raw local x/y; "
                "map-background plots are diagnostic only and are not geometrically valid overlays without an "
                "explicit x/y-to-projected transform."
                if args.use_map
                else "none"
            ),
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
    model = build_model(args.model_type, args.dropout, input_channels(args)).to(device)
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
            make_dataset(args, data_root, split_frame, scaler),
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
    if args.use_map:
        plot_map_background_groups(
            per_sample,
            predictions_abs["test"],
            truths_abs["test"],
            args.map_dir.resolve(),
            infer_map_crop_mode(args.map_dir),
            plots_dir / "map_background",
        )
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


def plot_map_background_groups(
    per_sample: pd.DataFrame,
    predictions_abs: dict[str, np.ndarray],
    truths_abs: dict[str, np.ndarray],
    map_dir: Path,
    crop_mode: str,
    output_dir: Path,
) -> None:
    extent_lookup = load_extent_lookup(map_dir)
    if extent_lookup is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = per_sample[per_sample["split"] == "test"].copy()
    sorted_rows = test_rows.sort_values("integrated_ADE").reset_index(drop=True)
    groups = {
        "best": sorted_rows.head(5),
        "median": sorted_rows.iloc[
            max(0, len(sorted_rows) // 2 - 2) : min(len(sorted_rows), len(sorted_rows) // 2 + 3)
        ],
        "worst": sorted_rows.tail(5).sort_values("integrated_ADE", ascending=False),
    }
    for group_name, frame in groups.items():
        group_dir = output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            plot_map_background_prediction(
                sample_id,
                str(row.vehicle_id),
                truths_abs[sample_id],
                predictions_abs[sample_id],
                map_dir / f"{sample_id}_map.png",
                extent_lookup.get(sample_id),
                float(row.delta_ADE),
                float(row.integrated_ADE),
                float(row.integrated_FDE),
                group_dir / f"{sample_id}_map_background_trajectory.png",
            )


def plot_map_background_prediction(
    sample_id: str,
    vehicle_id: str,
    true: np.ndarray,
    pred: np.ndarray,
    map_path: Path,
    extent_row: dict[str, object] | None,
    delta_ade: float,
    integrated_ade: float,
    integrated_fde: float,
    path: Path,
) -> None:
    if not map_path.exists():
        return
    if extent_row is None:
        print(f"WARNING: skipping map-background plot for {sample_id}; extent metadata row is missing.")
        return
    map_image = np.asarray(Image.open(map_path).convert("L"), dtype=np.float32) / 255.0
    extent = (
        float(extent_row["min_x"]),
        float(extent_row["max_x"]),
        float(extent_row["min_y"]),
        float(extent_row["max_y"]),
    )
    crop_mode = str(extent_row["crop_mode"])
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(map_image, cmap="gray", origin="upper", extent=extent, alpha=0.45)
    axis.plot(true[:, 0], true[:, 1], label="ground truth", linewidth=1.6, color="#1b9e77")
    axis.plot(pred[:, 0], pred[:, 1], label="pred integrated", linewidth=1.6, color="#d95f02")
    axis.scatter(true[0, 0], true[0, 1], s=22, marker="o", color="#1b9e77", label="gt/start")
    axis.scatter(true[-1, 0], true[-1, 1], s=22, marker="s", color="#1b9e77", label="gt end")
    axis.scatter(pred[-1, 0], pred[-1, 1], s=26, marker="^", color="#d95f02", label="pred end")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{sample_id} {vehicle_id} map={crop_mode}\n"
        f"delta ADE={delta_ade:.2f} int ADE={integrated_ade:.2f} int FDE={integrated_fde:.2f}"
    )
    axis.legend(fontsize=7)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
        "map_background_overlay_warning": config.get("map_background_overlay_warning", "none"),
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
        f"- map_background_overlay_warning: {report['map_background_overlay_warning']}",
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

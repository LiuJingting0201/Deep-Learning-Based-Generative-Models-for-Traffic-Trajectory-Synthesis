"""Smoke test Week3 structure-aware ResNet18 delta decoders without full training."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    fit_delta_label_scaler,
    verify_delta_data_and_split,
)
from run_resnet18_decoder_ablation import resolve_device, set_seed  # noqa: E402
from structure_aware_resnet18_models import (  # noqa: E402
    DecompMtfLocalResNet18DeltaDecoder,
    HardStructureResNet18DeltaDecoder,
    RawResNet18DeltaDecoder,
    StructureDecomposedResNet18DeltaDecoder,
)


DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
OUTPUT_DIR = PROJECT_ROOT / "week03" / "results" / "smoke_structure_resnet18"


def main() -> None:
    set_seed(42)
    device = resolve_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(DATA_ROOT)
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    scaler = fit_delta_label_scaler(DATA_ROOT, train_metadata)
    subset_metadata = train_metadata.head(4).copy()
    loader = DataLoader(
        DeltaDisplacementDataset(DATA_ROOT, subset_metadata, scaler),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    expected_keys = {"image", "label", "label_raw", "absolute", "start", "sample_id", "vehicle_id"}
    missing_keys = expected_keys.difference(batch)
    if missing_keys:
        raise RuntimeError(f"Smoke batch missing keys: {sorted(missing_keys)}")

    image = batch["image"]
    labels = batch["label"]
    raw_labels = batch["label_raw"]
    absolute = batch["absolute"]
    start = batch["start"]
    batch_size = int(image.shape[0])

    assert list(image.shape) == [2, 3, 224, 224]
    assert list(labels.shape) == [2, 224, 2]
    assert list(raw_labels.shape) == [2, 224, 2]
    assert list(absolute.shape) == [2, 224, 2]
    assert list(start.shape) == [2, 2]
    image_in_range = bool(image.min() >= 0.0 and image.max() <= 1.0)
    assert image_in_range

    image_stats = tensor_stats(image)
    channel_stats = {
        name: tensor_stats(image[:, index])
        for index, name in enumerate(["R_GASF", "G_GADF", "B_MTF"])
    }

    models: list[tuple[str, nn.Module]] = [
        ("RawResNet18DeltaDecoder", RawResNet18DeltaDecoder(dropout=0.3)),
        (
            "StructureDecomposedResNet18DeltaDecoder",
            StructureDecomposedResNet18DeltaDecoder(dropout=0.3, return_debug=True),
        ),
        ("HardStructureResNet18DeltaDecoder", HardStructureResNet18DeltaDecoder(dropout=0.3)),
        ("DecompMtfLocalResNet18DeltaDecoder", DecompMtfLocalResNet18DeltaDecoder(dropout=0.3)),
    ]

    criterion = nn.MSELoss()
    model_reports: list[dict[str, Any]] = []
    structure_pred_norm: torch.Tensor | None = None
    structure_debug: dict[str, Any] = {}
    for name, model in models:
        model = model.to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(image.to(device))
        if isinstance(output, tuple):
            pred, debug = output
            structure_debug = debug
        else:
            pred = output
        assert list(pred.shape) == list(labels.shape)
        loss = criterion(pred, labels.to(device))
        loss.backward()
        grad_ok = any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        assert grad_ok
        model_reports.append(
            {
                "model_name": name,
                "parameter_count": count_parameters(model),
                "conv1_input_channels": int(model.conv1_in_channels),
                "output_shape": list(pred.shape),
                "loss": float(loss.detach().cpu()),
                "finite_gradient_found": grad_ok,
            }
        )
        if name == "StructureDecomposedResNet18DeltaDecoder":
            structure_pred_norm = pred.detach()

    if structure_pred_norm is None:
        raise RuntimeError("Structure model did not run.")
    expected_debug_keys = {
        "input_min",
        "input_max",
        "r_theory_min",
        "r_theory_max",
        "g_theory_min",
        "g_theory_max",
        "b_raw_min",
        "b_raw_max",
        "r_res_mean_abs",
        "g_res_mean_abs",
        "structured_input_shape",
        "structured_input_min",
        "structured_input_max",
    }
    missing_debug_keys = expected_debug_keys.difference(structure_debug)
    if missing_debug_keys:
        raise RuntimeError(f"Structure debug missing keys: {sorted(missing_debug_keys)}")
    assert structure_debug["structured_input_shape"][1] == 5

    pred_delta = scaler.inverse_transform_tensor(structure_pred_norm)
    pred_abs = integrate_delta_torch(start.to(pred_delta.device), pred_delta)
    assert list(pred_abs.shape) == [batch_size, 224, 2]

    true_abs_from_delta = integrate_delta_torch(start, raw_labels)
    integration_error = float((true_abs_from_delta - absolute).abs().max().cpu())
    integration_tolerance = 1e-3
    integration_sanity_passed = integration_error < integration_tolerance
    assert integration_sanity_passed

    report: dict[str, Any] = {
        "status": "passed",
        "dataset_path": str(DATA_ROOT),
        "split_used": "train first 4 samples",
        "subset_size": int(len(subset_metadata)),
        "batch_size": batch_size,
        "device": str(device),
        "image_shape": list(image.shape),
        "label_shape": list(labels.shape),
        "raw_label_shape": list(raw_labels.shape),
        "absolute_shape": list(absolute.shape),
        "start_shape": list(start.shape),
        "image_stats": image_stats,
        "per_channel_stats": channel_stats,
        "image_values_in_0_1": image_in_range,
        "label_scaler": asdict(scaler),
        "models": model_reports,
        "structure_debug": structure_debug,
        "structure_pred_delta_shape": list(pred_delta.shape),
        "structure_pred_abs_shape": list(pred_abs.shape),
        "true_delta_integration_max_error": integration_error,
        "true_delta_integration_tolerance": integration_tolerance,
        "true_delta_integration_sanity_passed": integration_sanity_passed,
        "conclusion": (
            "Smoke test passed. Models produce [B,224,2], backpropagate, and integration convention was recorded."
        ),
    }
    write_outputs(report)
    print_summary(report)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(tensor.min().cpu()),
        "max": float(tensor.max().cpu()),
        "mean": float(tensor.mean().cpu()),
        "std": float(tensor.std(unbiased=False).cpu()),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def integrate_delta_torch(start: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    absolute = torch.empty_like(delta)
    absolute[:, 0, :] = start
    absolute[:, 1:, :] = start[:, None, :] + torch.cumsum(delta[:, 1:, :], dim=1)
    return absolute


def write_outputs(report: dict[str, Any]) -> None:
    json_path = OUTPUT_DIR / "smoke_report.json"
    md_path = OUTPUT_DIR / "smoke_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    lines = [
        "# Structure ResNet18 Smoke Report",
        "",
        f"- Status: {report['status']}",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Split: {report['split_used']}",
        f"- Subset size: {report['subset_size']}",
        f"- Batch size: {report['batch_size']}",
        f"- Device: {report['device']}",
        f"- Image shape: {report['image_shape']}",
        f"- Label shape: {report['label_shape']}",
        f"- Raw label shape: {report['raw_label_shape']}",
        f"- Absolute shape: {report['absolute_shape']}",
        f"- Start shape: {report['start_shape']}",
        f"- Image stats: {report['image_stats']}",
        f"- Per-channel stats: {report['per_channel_stats']}",
        f"- Image values in [0,1]: {report['image_values_in_0_1']}",
        f"- Label scaler mean: {report['label_scaler']['mean']}",
        f"- Label scaler std: {report['label_scaler']['std']}",
        "",
        "## Models",
    ]
    for model in report["models"]:
        lines.extend(
            [
                "",
                f"### {model['model_name']}",
                f"- Parameter count: {model['parameter_count']}",
                f"- Conv1 input channels: {model['conv1_input_channels']}",
                f"- Output shape: {model['output_shape']}",
                f"- Loss: {model['loss']:.8f}",
                f"- Finite gradient found: {model['finite_gradient_found']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Structure Debug",
            "```json",
            json.dumps(report["structure_debug"], indent=2),
            "```",
            "",
            f"- True-delta integration sanity max error: {report['true_delta_integration_max_error']:.10f}",
            f"- True-delta integration tolerance: {report['true_delta_integration_tolerance']:.10f}",
            f"- True-delta integration sanity passed: {report['true_delta_integration_sanity_passed']}",
            f"- Conclusion: {report['conclusion']}",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))


def print_summary(report: dict[str, Any]) -> None:
    print("Structure ResNet18 smoke test passed")
    print(f"image_shape={report['image_shape']} label_shape={report['label_shape']}")
    print(f"image_stats={report['image_stats']}")
    print(f"per_channel_stats={report['per_channel_stats']}")
    for model in report["models"]:
        print(
            f"{model['model_name']}: conv1_in={model['conv1_input_channels']} "
            f"params={model['parameter_count']} loss={model['loss']:.8f} "
            f"grad_ok={model['finite_gradient_found']}"
        )
    print(f"structure_debug={report['structure_debug']}")
    print(f"true_delta_integration_max_error={report['true_delta_integration_max_error']:.10f}")
    print(f"true_delta_integration_sanity_passed={report['true_delta_integration_sanity_passed']}")
    print(f"reports={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

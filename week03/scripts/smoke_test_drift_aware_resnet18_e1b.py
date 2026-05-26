"""Smoke test Week3 E1b drift-aware raw ResNet18 with correction-target loss."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/home/jliu/Thesis/week03/logs/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
for path in [SCRIPTS_DIR, WEEK03_SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drift_aware_resnet18_models import DriftAwareResNet18DeltaDecoder  # noqa: E402
from run_delta_displacement_resnet18_decoder_ablation import (  # noqa: E402
    DeltaDisplacementDataset,
    fit_delta_label_scaler,
    verify_delta_data_and_split,
)
from run_resnet18_decoder_ablation import resolve_device, set_seed  # noqa: E402
from train_drift_aware_resnet18_decoder import (  # noqa: E402
    fit_absolute_label_scaler,
    integrate_delta_torch,
    scaler_transform_tensor,
)


DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
OUTPUT_DIR = PROJECT_ROOT / "week03" / "results" / "smoke_e1b_drift_aware_corrtarget"


def main() -> None:
    set_seed(42)
    device = resolve_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_metadata = verify_delta_data_and_split(DATA_ROOT)
    train_metadata = split_metadata[split_metadata["split"] == "train"].copy()
    delta_scaler = fit_delta_label_scaler(DATA_ROOT, train_metadata)
    absolute_scaler = fit_absolute_label_scaler(DATA_ROOT, train_metadata)
    subset_metadata = train_metadata.head(4).copy()
    loader = DataLoader(
        DeltaDisplacementDataset(DATA_ROOT, subset_metadata, delta_scaler),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    image = batch["image"].to(device)
    true_delta_norm = batch["label"].to(device)
    true_abs_raw = batch["absolute"].to(device)
    start = batch["start"].to(device)
    assert list(image.shape) == [2, 3, 224, 224]
    assert list(true_delta_norm.shape) == [2, 224, 2]
    assert list(true_abs_raw.shape) == [2, 224, 2]
    assert list(start.shape) == [2, 2]

    model = DriftAwareResNet18DeltaDecoder(dropout=0.3, correction_anchors=16).to(device)
    model.train()
    outputs = model(image)
    expected_shapes = {
        "pred_delta_norm": [2, 224, 2],
        "corr_anchor_norm": [2, 16, 2],
        "corr_full_norm": [2, 224, 2],
    }
    for key, shape in expected_shapes.items():
        assert list(outputs[key].shape) == shape

    criterion = nn.MSELoss()
    pred_delta_raw = delta_scaler.inverse_transform_tensor(outputs["pred_delta_norm"])
    pred_abs_raw = integrate_delta_torch(start, pred_delta_raw)
    pred_abs_raw_norm = scaler_transform_tensor(absolute_scaler, pred_abs_raw)
    true_abs_norm = scaler_transform_tensor(absolute_scaler, true_abs_raw)
    pred_abs_corrected_norm = pred_abs_raw_norm + outputs["corr_full_norm"]
    corr_target_norm = true_abs_norm - pred_abs_raw_norm.detach()
    pred_abs_corrected_raw = absolute_scaler.inverse_transform_tensor(pred_abs_corrected_norm)

    delta_loss = criterion(outputs["pred_delta_norm"], true_delta_norm)
    abs_loss = criterion(pred_abs_corrected_norm, true_abs_norm)
    corr_target_loss = criterion(outputs["corr_full_norm"], corr_target_norm)
    smooth_loss = torch.mean((outputs["corr_full_norm"][:, 1:, :] - outputs["corr_full_norm"][:, :-1, :]) ** 2)
    mag_loss = torch.mean(outputs["corr_full_norm"] ** 2)
    total_loss = delta_loss + 0.1 * abs_loss + 0.1 * corr_target_loss + 0.001 * smooth_loss + 0.00005 * mag_loss
    total_loss.backward()
    finite_gradient_found = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert finite_gradient_found
    assert list(pred_abs_raw.shape) == [2, 224, 2]
    assert list(pred_abs_corrected_raw.shape) == [2, 224, 2]
    assert list(corr_target_norm.shape) == [2, 224, 2]
    assert torch.allclose(outputs["corr_anchor_norm"][:, 0, :], torch.zeros_like(outputs["corr_anchor_norm"][:, 0, :]))
    assert torch.allclose(outputs["corr_full_norm"][:, 0, :], torch.zeros_like(outputs["corr_full_norm"][:, 0, :]))
    assert torch.isfinite(total_loss)

    report: dict[str, Any] = {
        "status": "passed",
        "dataset_path": str(DATA_ROOT),
        "device": str(device),
        "image_shape": list(image.shape),
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "raw_absolute_shape": list(pred_abs_raw.shape),
        "corrected_absolute_shape": list(pred_abs_corrected_raw.shape),
        "corr_target_norm_shape": list(corr_target_norm.shape),
        "corr_anchor_t0_zero": bool(
            torch.allclose(outputs["corr_anchor_norm"][:, 0, :], torch.zeros_like(outputs["corr_anchor_norm"][:, 0, :]))
        ),
        "corr_full_t0_zero": bool(
            torch.allclose(outputs["corr_full_norm"][:, 0, :], torch.zeros_like(outputs["corr_full_norm"][:, 0, :]))
        ),
        "conv1_input_channels": int(model.conv1_in_channels),
        "correction_anchors": int(model.correction_anchors),
        "loss_components": {
            "total_loss": float(total_loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
            "abs_loss": float(abs_loss.detach().cpu()),
            "corr_target_loss": float(corr_target_loss.detach().cpu()),
            "smooth_loss": float(smooth_loss.detach().cpu()),
            "mag_loss": float(mag_loss.detach().cpu()),
        },
        "finite_gradient_found": finite_gradient_found,
        "delta_scaler": asdict(delta_scaler),
        "absolute_scaler": asdict(absolute_scaler),
    }
    write_outputs(report)
    print_summary(report)


def write_outputs(report: dict[str, Any]) -> None:
    (OUTPUT_DIR / "smoke_report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# E1b Drift-Aware ResNet18 Smoke Report",
        "",
        f"- Status: {report['status']}",
        f"- Device: {report['device']}",
        f"- Image shape: {report['image_shape']}",
        f"- Output shapes: {report['output_shapes']}",
        f"- Raw absolute shape: {report['raw_absolute_shape']}",
        f"- Corrected absolute shape: {report['corrected_absolute_shape']}",
        f"- Correction target shape: {report['corr_target_norm_shape']}",
        f"- Correction anchor t0 zero: {report['corr_anchor_t0_zero']}",
        f"- Correction full t0 zero: {report['corr_full_t0_zero']}",
        f"- Conv1 input channels: {report['conv1_input_channels']}",
        f"- Correction anchors: {report['correction_anchors']}",
        f"- Loss components: {report['loss_components']}",
        f"- Finite gradient found: {report['finite_gradient_found']}",
    ]
    (OUTPUT_DIR / "smoke_report.md").write_text("\n".join(lines) + "\n")


def print_summary(report: dict[str, Any]) -> None:
    print("E1b drift-aware ResNet18 smoke test passed")
    print(f"output_shapes={report['output_shapes']}")
    print(f"raw_absolute_shape={report['raw_absolute_shape']}")
    print(f"corrected_absolute_shape={report['corrected_absolute_shape']}")
    print(f"corr_target_norm_shape={report['corr_target_norm_shape']}")
    print(f"corr_anchor_t0_zero={report['corr_anchor_t0_zero']} corr_full_t0_zero={report['corr_full_t0_zero']}")
    print(f"loss_components={report['loss_components']}")
    print(f"finite_gradient_found={report['finite_gradient_found']}")
    print(f"reports={OUTPUT_DIR}")


if __name__ == "__main__":
    main()

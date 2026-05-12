"""Smoke tests for the extracted DDPM training, sampling, and FID scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("diffusers")
pytest.importorskip("PIL")

from PIL import Image


def write_tiny_images(path: Path, count: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (16, 16), color=(index * 60, 20, 200 - index * 30))
        image.save(path / f"image_{index}.png")


def run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, *args],
        cwd=Path.cwd(),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_train_and_generate_ddpm_one_step_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "real"
    train_dir = tmp_path / "train"
    samples_dir = tmp_path / "samples"
    write_tiny_images(data_dir)

    run_script(
        [
            "scripts/train_ddpm.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(train_dir),
            "--image-size",
            "16",
            "--batch-size",
            "1",
            "--lr",
            "0.0001",
            "--max-train-steps",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--save-every",
            "1",
            "--sample-every",
            "0",
        ]
    )

    checkpoint = train_dir / "checkpoint-final"
    assert (checkpoint / "unet").exists()
    assert (checkpoint / "scheduler").exists()
    assert (checkpoint / "training_state.pt").exists()
    summary = json.loads((train_dir / "train_summary.json").read_text())
    assert summary["global_step"] == 1

    run_script(
        [
            "scripts/generate_ddpm_samples.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(samples_dir),
            "--num-samples",
            "1",
            "--image-size",
            "16",
            "--num-inference-steps",
            "1",
            "--scheduler",
            "ddpm",
            "--batch-size",
            "1",
        ]
    )

    assert (samples_dir / "sample_000000.png").exists()


@pytest.mark.parametrize("aux_loss_mode", ["notebook", "none", "data_driven_diag"])
def test_train_ddpm_aux_loss_modes_one_step_smoke(tmp_path: Path, aux_loss_mode: str) -> None:
    data_dir = tmp_path / "real"
    train_dir = tmp_path / aux_loss_mode
    write_tiny_images(data_dir, count=3)

    run_script(
        [
            "scripts/train_ddpm.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(train_dir),
            "--image-size",
            "16",
            "--batch-size",
            "1",
            "--lr",
            "0.0001",
            "--max-train-steps",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--save-every",
            "0",
            "--sample-every",
            "0",
            "--aux-loss-mode",
            aux_loss_mode,
        ]
    )

    log_record = json.loads((train_dir / "train_log.jsonl").read_text().splitlines()[0])
    assert log_record["aux_loss_mode"] == aux_loss_mode
    assert "total_loss" in log_record
    assert "mse_loss" in log_record
    assert "symmetry_loss" in log_record
    assert "diag_loss" in log_record
    summary = json.loads((train_dir / "train_summary.json").read_text())
    assert summary["aux_loss_mode"] == aux_loss_mode
    if aux_loss_mode == "data_driven_diag":
        assert len(summary["diag_targets"]) == 3
    elif aux_loss_mode == "none":
        assert summary["diag_targets"] is None


def test_train_ddpm_data_driven_diag_uses_split_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "paired"
    data_dir = data_root / "images"
    split_dir = data_root / "splits"
    train_dir = tmp_path / "data_driven_split"
    write_tiny_images(data_dir, count=4)
    split_dir.mkdir(parents=True)
    (split_dir / "split_metadata.csv").write_text(
        "\n".join(
            [
                "sample_id,image_path,split",
                "sample_000000,images/image_0.png,train",
                "sample_000001,images/image_1.png,train",
                "sample_000002,images/image_2.png,val",
                "sample_000003,images/image_3.png,test",
            ]
        )
        + "\n"
    )

    run_script(
        [
            "scripts/train_ddpm.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(train_dir),
            "--image-size",
            "16",
            "--batch-size",
            "1",
            "--lr",
            "0.0001",
            "--max-train-steps",
            "1",
            "--gradient-accumulation-steps",
            "1",
            "--save-every",
            "0",
            "--sample-every",
            "0",
            "--aux-loss-mode",
            "data_driven_diag",
            "--split-metadata",
            str(split_dir / "split_metadata.csv"),
        ]
    )

    summary = json.loads((train_dir / "train_summary.json").read_text())
    assert summary["diag_target_source"] == "train_split_metadata"
    assert summary["diag_target_num_images"] == 2
    assert len(summary["diag_target_rgb"]) == 3


def test_fid_formula_smoke() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None
    from scripts.evaluate_fid import compute_fid

    mu = np_array([0.0, 1.0])
    sigma = np_eye(2)
    assert compute_fid(mu, sigma, mu, sigma) == pytest.approx(0.0)


def np_array(values: list[float]):
    import numpy as np

    return np.array(values)


def np_eye(size: int):
    import numpy as np

    return np.eye(size)

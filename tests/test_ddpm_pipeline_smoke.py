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

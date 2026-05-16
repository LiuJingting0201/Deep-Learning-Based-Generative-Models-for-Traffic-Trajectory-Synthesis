"""Smoke tests for the GAF autoencoder first-stage baseline."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

import numpy as np
from PIL import Image


def write_tiny_images(path: Path, count: int = 2, size: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (size, size), color=(index * 50, 40, 200 - index * 40))
        image.save(path / f"image_{index:03d}.png")


def run_script(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run([sys.executable, *args], cwd=Path.cwd(), env=env, check=True, text=True, capture_output=True)


def test_train_autoencoder_and_encode_latents_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    output_dir = tmp_path / "autoencoder"
    latent_dir = tmp_path / "data_latent"
    write_tiny_images(data_dir)

    run_script(
        [
            "scripts/train_autoencoder.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--save-every",
            "1",
            "--image-size",
            "32",
            "--latent-channels",
            "32",
            "--latent-spatial-size",
            "4",
        ]
    )

    checkpoint = output_dir / "checkpoint-final"
    assert (checkpoint / "autoencoder" / "model.pt").exists()
    assert (checkpoint / "training_state.pt").exists()
    assert (output_dir / "loss_curve.csv").exists()
    assert (output_dir / "reconstruction_metrics.json").exists()
    assert (output_dir / "train_summary.json").exists()
    assert (output_dir / "samples" / "recon_epoch_0001.png").exists()

    run_script(
        [
            "scripts/encode_dataset_to_latent.py",
            "--data-dir",
            str(data_dir),
            "--autoencoder-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(latent_dir),
            "--batch-size",
            "2",
            "--image-size",
            "32",
        ]
    )

    latent_paths = sorted((latent_dir / "latent").glob("*.npy"))
    assert len(latent_paths) == 2
    latent = np.load(latent_paths[0])
    assert latent.shape == (32, 4, 4)

    rows = list(csv.DictReader((latent_dir / "metadata.csv").open()))
    assert len(rows) == 2
    assert rows[0]["index"] == "0"
    assert rows[0]["latent_path"].startswith("latent/")
    assert rows[0]["latent_shape"] == "32x4x4"

    stats = json.loads((latent_dir / "latent_stats.json").read_text())
    assert stats["count"] == 2 * 32 * 4 * 4
    assert {"mean", "std", "min", "max"}.issubset(stats)

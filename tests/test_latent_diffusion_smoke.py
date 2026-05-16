"""Smoke tests for latent DDPM training and decoded sample generation."""

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

import numpy as np
import torch
from PIL import Image


def run_script(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run([sys.executable, *args], cwd=Path.cwd(), env=env, check=True, text=True, capture_output=True)


def write_latents(path: Path, count: int = 2) -> None:
    latent_dir = path / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)
    values = []
    for index in range(count):
        latent = np.random.default_rng(index).normal(size=(32, 28, 28)).astype(np.float32)
        values.append(latent.reshape(-1))
        np.save(latent_dir / f"latent_{index:03d}.npy", latent)
    all_values = np.concatenate(values)
    (path / "latent_stats.json").write_text(
        json.dumps(
            {
                "count": int(all_values.size),
                "mean": float(all_values.mean()),
                "std": float(all_values.std()),
                "min": float(all_values.min()),
                "max": float(all_values.max()),
            },
            indent=2,
        )
    )


def write_images(path: Path, count: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (32, 32), color=(index * 40, 70, 180)).save(path / f"image_{index:03d}.png")


def test_train_generate_and_evaluate_latent_ddpm_smoke(tmp_path: Path) -> None:
    latent_data = tmp_path / "data_latent"
    latent_train = tmp_path / "latent_ddpm"
    generated_dir = tmp_path / "generated"
    image_dir = tmp_path / "images"
    autoencoder_dir = tmp_path / "autoencoder"
    eval_dir = tmp_path / "eval"
    write_latents(latent_data)
    write_images(image_dir)

    run_script(
        [
            "scripts/train_autoencoder.py",
            "--data-dir",
            str(image_dir),
            "--output-dir",
            str(autoencoder_dir),
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--save-every",
            "0",
            "--image-size",
            "32",
            "--latent-channels",
            "32",
            "--latent-spatial-size",
            "4",
        ]
    )

    run_script(
        [
            "scripts/train_latent_ddpm.py",
            "--data-dir",
            str(latent_data),
            "--output-dir",
            str(latent_train),
            "--batch-size",
            "2",
            "--max-train-steps",
            "1",
            "--save-every",
            "1",
            "--sample-every",
            "1",
            "--sample-inference-steps",
            "1",
            "--num-sample-latents",
            "2",
            "--keep-last-checkpoints",
            "1",
            "--sample-size",
            "28",
            "--latent-channels",
            "32",
        ]
    )

    checkpoint = latent_train / "checkpoint-final"
    best_checkpoint = latent_train / "checkpoint-best"
    assert (checkpoint / "unet").exists()
    assert (checkpoint / "scheduler").exists()
    assert (checkpoint / "training_state.pt").exists()
    assert (best_checkpoint / "unet").exists()
    summary = json.loads((latent_train / "train_summary.json").read_text())
    assert summary["global_step"] == 1
    assert summary["sample_size"] == 28
    assert summary["latent_channels"] == 32
    assert summary["normalization_stats"]["enabled"] is True
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["args"]["block_out_channels"] == "64,128"
    assert state["args"]["down_block_types"] == "DownBlock2D,AttnDownBlock2D"
    assert state["args"]["up_block_types"] == "AttnUpBlock2D,UpBlock2D"
    assert (latent_train / "samples" / "latent_step_000001.npy").exists()

    run_script(
        [
            "scripts/generate_latent_samples.py",
            "--latent-checkpoint",
            str(best_checkpoint),
            "--autoencoder-checkpoint",
            str(autoencoder_dir / "checkpoint-final"),
            "--output-dir",
            str(generated_dir),
            "--num-samples",
            "1",
            "--batch-size",
            "1",
            "--num-inference-steps",
            "1",
            "--sample-size",
            "28",
            "--latent-channels",
            "32",
        ]
    )
    assert (generated_dir / "sample_000000.png").exists()

    run_script(
        [
            "scripts/evaluate_latent_generation.py",
            "--real-dir",
            str(image_dir),
            "--generated-dir",
            str(generated_dir),
            "--autoencoder-checkpoint",
            str(autoencoder_dir / "checkpoint-final"),
            "--output-dir",
            str(eval_dir),
            "--image-size",
            "32",
            "--batch-size",
            "2",
        ]
    )
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "side_by_side.png").exists()

"""Smoke tests for the currently extractable trajectory-generation pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from diffusers import DDPMScheduler, UNet2DModel
from pyts.image import GramianAngularField, MarkovTransitionField

from cnr_trajectory.encoding import duplicate, squeeze
from cnr_trajectory.reconstruction.decoder import decode_image_to_trajectory


def test_original_notebooks_are_valid_json() -> None:
    notebook_paths = sorted(Path("notebooks/original").glob("*/*.ipynb"))

    assert len(notebook_paths) == 5
    for path in notebook_paths:
        notebook = json.loads(path.read_text())
        assert "cells" in notebook


def test_raw_trajectory_file_is_readable() -> None:
    path = Path("Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls")

    frame = pd.read_csv(path, sep=",", low_memory=False, nrows=5)

    assert list(frame.columns) == [
        "time",
        "vehicle_id",
        "x",
        "y",
        "type",
        "lon",
        "lat",
        "speed",
    ]


def test_synthetic_encoding_pipeline_smoke() -> None:
    sequence = [0.1, 0.1, 0.2, 0.3, 0.3, 0.4]
    sequence = duplicate(squeeze(sequence, 4), 8)
    values = np.array([sequence], dtype=float)

    gasf = GramianAngularField(method="summation", sample_range=(0, 1)).fit_transform(
        values
    )
    gadf = GramianAngularField(method="difference", sample_range=(0, 1)).fit_transform(
        values
    )
    mtf = MarkovTransitionField(strategy="uniform", n_bins=4).fit_transform(values)

    image = np.stack((gasf[0], gadf[0], mtf[0]), axis=-1)

    assert image.shape == (8, 8, 3)
    assert np.isfinite(image).all()


def test_tiny_diffusion_model_forward_smoke() -> None:
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=10)
    clean_image = torch.randn(1, 3, 32, 32)
    noise = torch.randn_like(clean_image)
    timestep = torch.tensor([1])

    noisy_image = scheduler.add_noise(clean_image, noise, timestep)
    prediction = model(noisy_image, timestep).sample

    assert tuple(prediction.shape) == (1, 3, 32, 32)
    assert torch.isfinite(prediction).all()


def test_reconstruction_pipeline_is_still_todo() -> None:
    with pytest.raises(NotImplementedError, match="TODO"):
        decode_image_to_trajectory()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_SCRIPTS_DIR = PROJECT_ROOT / "week03" / "scripts"
if str(WEEK03_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK03_SCRIPTS_DIR))

from sequence_latent_models import (  # noqa: E402
    RowWiseGAFImageEncoder,
    TrajectorySequenceAutoencoder,
    TrajectorySequenceDecoder,
    TrajectorySequenceEncoder,
)
from sequence_latent_utils import PairedImageDeltaDataset, sequence_losses  # noqa: E402


def test_trajectory_sequence_encoder_shape() -> None:
    delta = torch.randn(2, 223, 2)
    encoder = TrajectorySequenceEncoder(latent_dim=16)
    z = encoder(delta)
    assert z.shape == (2, 223, 16)


def test_trajectory_sequence_decoder_shape() -> None:
    decoder = TrajectorySequenceDecoder(latent_dim=16)
    z = torch.randn(2, 223, 16)
    delta_hat = decoder(z)
    assert delta_hat.shape == (2, 223, 2)


def test_trajectory_autoencoder_forward() -> None:
    model = TrajectorySequenceAutoencoder(latent_dim=8)
    delta_hat, z = model(torch.randn(3, 223, 2))
    assert delta_hat.shape == (3, 223, 2)
    assert z.shape == (3, 223, 8)


def test_rowwise_image_encoder_shape_without_temporal_refinement() -> None:
    model = RowWiseGAFImageEncoder(latent_dim=12)
    z = model(torch.randn(4, 3, 224, 224))
    assert z.shape == (4, 223, 12)


def test_rowwise_image_encoder_shape_with_temporal_refinement() -> None:
    model = RowWiseGAFImageEncoder(
        latent_dim=12,
        hidden_dim=32,
        use_temporal_refine=True,
        temporal_kernel_size=5,
        temporal_layers=2,
    )
    z = model(torch.randn(4, 3, 224, 224))
    assert z.shape == (4, 223, 12)


def test_rowwise_image_encoder_shape_with_transformer_refinement() -> None:
    model = RowWiseGAFImageEncoder(
        latent_dim=64,
        hidden_dim=32,
        use_transformer_refine=True,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=256,
    )
    z = model(torch.randn(2, 3, 224, 224))
    assert z.shape == (2, 223, 64)


def test_rowwise_image_encoder_mtf_only_transformer_shape() -> None:
    model = RowWiseGAFImageEncoder(
        latent_dim=64,
        hidden_dim=32,
        input_channels=1,
        use_transformer_refine=True,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=256,
    )
    z = model(torch.randn(2, 1, 224, 224))
    assert z.shape == (2, 223, 64)


def test_rowwise_image_encoder_mtf_only_rejects_rgb_input() -> None:
    model = RowWiseGAFImageEncoder(latent_dim=64, hidden_dim=32, input_channels=1)
    with pytest.raises(ValueError, match=r"Expected images \[B,1,H,W\]"):
        model(torch.randn(2, 3, 224, 224))


def test_rowwise_transformer_heads_validation() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        RowWiseGAFImageEncoder(
            latent_dim=64,
            use_transformer_refine=True,
            transformer_heads=3,
        )


def test_expected_steps_validation() -> None:
    encoder = TrajectorySequenceEncoder(latent_dim=16, expected_steps=223)
    delta_bad = torch.randn(2, 224, 2)
    with pytest.raises(ValueError, match=r"Expected delta \[B,223,2\]"):
        encoder(delta_bad)


def test_decoder_latent_dim_validation() -> None:
    decoder = TrajectorySequenceDecoder(latent_dim=64)
    z_bad = torch.randn(2, 223, 128)
    with pytest.raises(ValueError, match=r"Expected latent \[B,223,64\]"):
        decoder(z_bad)


def test_stage2_frozen_decoder_gradient_flow() -> None:
    image_encoder = RowWiseGAFImageEncoder(latent_dim=64, hidden_dim=32)
    decoder = TrajectorySequenceDecoder(latent_dim=64)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    images = torch.randn(2, 3, 224, 224)
    delta_gt = torch.randn(2, 223, 2)
    z_img = image_encoder(images)
    delta_hat = decoder(z_img)
    loss = torch.nn.functional.mse_loss(delta_hat, delta_gt)
    loss.backward()

    assert any(parameter.grad is not None for parameter in image_encoder.parameters())
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_stage1_tiny_training_step() -> None:
    model = TrajectorySequenceAutoencoder(latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    delta = torch.randn(2, 223, 2)
    delta_hat, _ = model(delta)
    losses = sequence_losses(delta_hat, delta, beta=0.1)
    losses["loss"].backward()
    optimizer.step()
    assert torch.isfinite(losses["loss"])


def test_stage2_tiny_training_step() -> None:
    traj_encoder = TrajectorySequenceEncoder(latent_dim=8)
    traj_decoder = TrajectorySequenceDecoder(latent_dim=8)
    for module in [traj_encoder, traj_decoder]:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    image_encoder = RowWiseGAFImageEncoder(latent_dim=8, hidden_dim=32)
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=1e-3)
    images = torch.randn(2, 3, 224, 224)
    delta = torch.randn(2, 223, 2)
    with torch.no_grad():
        z_traj = traj_encoder(delta)
    z_img = image_encoder(images)
    delta_hat = traj_decoder(z_img)
    losses = sequence_losses(delta_hat, delta, beta=0.1)
    latent_loss = torch.nn.functional.mse_loss(z_img, z_traj)
    total_loss = losses["loss"] + latent_loss
    total_loss.backward()
    optimizer.step()
    assert torch.isfinite(total_loss)


def test_paired_image_delta_dataset_mtf_mode_returns_single_mtf_channel(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels_delta_displacement"
    images_dir.mkdir()
    labels_dir.mkdir()
    sample_id = "sample_000001"
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    Image.fromarray(image).save(images_dir / f"{sample_id}.png")
    np.save(labels_dir / f"{sample_id}.npy", np.zeros((223, 2), dtype=np.float32))
    metadata = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "vehicle_id": "veh",
                "image_path": f"images/{sample_id}.png",
                "label_delta_displacement_path": f"labels_delta_displacement/{sample_id}.npy",
            }
        ]
    )

    dataset = PairedImageDeltaDataset(tmp_path, metadata, image_channel_mode="mtf")
    item = dataset[0]

    assert item["image"].shape == (1, 224, 224)
    assert torch.allclose(item["image"], torch.full((1, 224, 224), 30.0 / 255.0))

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cnr_trajectory.reconstruction.delta_displacement import (  # noqa: E402
    DeltaDisplacementPairedDataset,
    MultiBranchDeltaDecoder,
    integrate_delta_torch,
)


def test_delta_dataset_multibranch_forward_and_step(tmp_path: Path) -> None:
    data_root = tmp_path / "delta_data"
    for subdir in ["images", "labels_delta_displacement", "labels_absolute", "starts"]:
        (data_root / subdir).mkdir(parents=True)

    rows = []
    for index in range(2):
        sample_id = f"sample_{index:06d}"
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[..., 0] = 40 + index
        image[..., 1] = 80 + index
        image[..., 2] = 120 + index
        Image.fromarray(image, mode="RGB").save(data_root / "images" / f"{sample_id}.png")

        start = np.array([float(index), float(index + 1)], dtype=np.float32)
        delta = np.zeros((224, 2), dtype=np.float32)
        delta[1:, 0] = 0.1
        delta[1:, 1] = -0.05
        absolute = np.empty_like(delta)
        absolute[0] = start
        absolute[1:] = start + np.cumsum(delta[1:], axis=0)
        np.save(data_root / "labels_delta_displacement" / f"{sample_id}.npy", delta)
        np.save(data_root / "labels_absolute" / f"{sample_id}.npy", absolute)
        np.save(data_root / "starts" / f"{sample_id}.npy", start)
        rows.append(
            {
                "sample_id": sample_id,
                "vehicle_id": f"car{index}",
                "image_path": f"images/{sample_id}.png",
                "label_delta_displacement_path": f"labels_delta_displacement/{sample_id}.npy",
                "label_absolute_path": f"labels_absolute/{sample_id}.npy",
                "start_path": f"starts/{sample_id}.npy",
                "valid_or_skipped": "valid",
            }
        )

    metadata = pd.DataFrame(rows)
    dataset = DeltaDisplacementPairedDataset(data_root, metadata, channels="rgb")
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["image"].shape == (2, 3, 224, 224)
    assert batch["delta_gt"].shape == (2, 224, 2)
    assert batch["absolute_gt"].shape == (2, 224, 2)
    assert batch["start_xy"].shape == (2, 2)

    model = MultiBranchDeltaDecoder(
        feature_dim=16,
        hidden_dim=64,
        dropout=0.0,
        channels="rgb",
        encoder_type="small_spatial",
    )
    pred_delta = model(batch["image"])
    assert pred_delta.shape == (2, 224, 2)

    pred_xy = integrate_delta_torch(batch["start_xy"], pred_delta)
    assert pred_xy.shape == (2, 224, 2)
    torch.testing.assert_close(pred_xy[:, 0, :], batch["start_xy"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.mse_loss(pred_delta, batch["delta_gt"])
    loss.backward()
    optimizer.step()


def test_multibranch_shapes_and_delta_zero_convention() -> None:
    rgb_model = MultiBranchDeltaDecoder(
        feature_dim=16,
        hidden_dim=64,
        dropout=0.0,
        channels="rgb",
        encoder_type="small_spatial",
    )
    rgb_output = rgb_model(torch.zeros(2, 3, 224, 224))
    assert rgb_output.shape == (2, 224, 2)

    single_model = MultiBranchDeltaDecoder(
        feature_dim=16,
        hidden_dim=64,
        dropout=0.0,
        channels="r",
        encoder_type="small_spatial",
    )
    single_output = single_model(torch.zeros(2, 1, 224, 224))
    assert single_output.shape == (2, 224, 2)

    start_xy = torch.tensor([[10.0, 20.0], [-5.0, 3.0]])
    delta = torch.zeros(2, 224, 2)
    delta[:, 0, :] = torch.tensor([999.0, -999.0])
    delta[:, 1:, 0] = 1.0
    delta[:, 1:, 1] = -2.0

    absolute = integrate_delta_torch(start_xy, delta)
    assert absolute.shape == (2, 224, 2)
    torch.testing.assert_close(absolute[:, 0, :], start_xy)
    expected_tail = start_xy[:, None, :] + torch.cumsum(delta[:, 1:, :], dim=1)
    torch.testing.assert_close(absolute[:, 1:, :], expected_tail)

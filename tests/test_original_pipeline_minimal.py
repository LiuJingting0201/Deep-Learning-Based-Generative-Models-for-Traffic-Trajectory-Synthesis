"""Minimal real checks for the original notebook-based encoding pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from cnr_trajectory.data.loader import load_trajectory_csv
from cnr_trajectory.encoding.gaf import encode_trajectory_frame


RAW_FILE = Path("Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls")


def test_original_encoding_pipeline_minimal(tmp_path: Path) -> None:
    """Run the real preprocessing and image encoding logic on a tiny subset."""
    frame = load_trajectory_csv(RAW_FILE, nrows=1000)

    encoded = encode_trajectory_frame(frame, max_samples=1)
    output_file = tmp_path / "encoded_sample.npy"
    np.save(output_file, encoded.images)

    assert output_file.exists()
    assert encoded.images.shape == (1, 224, 224, 3)
    assert encoded.images.dtype == np.uint8
    assert np.isfinite(encoded.images).all()
    assert encoded.sequences.shape == (1, 224)
    assert np.isfinite(encoded.sequences).all()
    assert len(encoded.labels) == 1
    assert len(encoded.vehicle_ids) == 1


def test_encode_trajectories_script_subprocess_smoke(tmp_path: Path) -> None:
    """Run the minimal CLI without hard-coded output paths."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/encode_trajectories.py",
            "--input",
            str(RAW_FILE),
            "--output",
            str(tmp_path),
            "--max-samples",
            "2",
            "--nrows",
            "1000",
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    images = np.load(tmp_path / "images.npy")
    metadata = (tmp_path / "metadata.csv").read_text().splitlines()
    png_files = sorted((tmp_path / "all").glob("*.png"))

    assert "Encoded 2 trajectories" in result.stdout
    assert images.shape == (2, 224, 224, 3)
    assert images.dtype == np.uint8
    assert np.isfinite(images).all()
    assert len(metadata) == 3
    assert metadata[0] == "filename,vehicle_id,sequence_length,image_shape,source_file"
    assert len(png_files) == 2

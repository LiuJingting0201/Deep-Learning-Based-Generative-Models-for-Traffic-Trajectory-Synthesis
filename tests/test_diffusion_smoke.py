"""End-to-end smoke test from encoded images to one tiny diffusion step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RAW_FILE = Path("Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls")


def test_encoding_to_tiny_diffusion_training_smoke(tmp_path: Path) -> None:
    encoded_dir = tmp_path / "gaf_rgb"
    smoke_dir = tmp_path / "diffusion_smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    subprocess.run(
        [
            sys.executable,
            "scripts/encode_trajectories.py",
            "--input",
            str(RAW_FILE),
            "--output",
            str(encoded_dir),
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

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_diffusion_smoke.py",
            "--input-dir",
            str(encoded_dir / "all"),
            "--output-dir",
            str(smoke_dir),
            "--image-size",
            "32",
            "--max-samples",
            "2",
            "--train-steps",
            "1",
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    artifact_path = smoke_dir / "smoke_log.json"
    artifact = json.loads(artifact_path.read_text())

    assert "Diffusion smoke training completed" in result.stdout
    assert artifact_path.exists()
    assert artifact["status"] == "ok"
    assert artifact["num_images"] == 2
    assert artifact["image_size"] == 32
    assert artifact["train_steps"] == 1
    assert len(artifact["losses"]) == 1

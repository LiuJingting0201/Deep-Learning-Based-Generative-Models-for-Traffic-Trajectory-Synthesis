"""Smoke tests for the Map_Matching integration wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cnr_trajectory.reconstruction.map_matching import (
    load_predicted_trajectories,
    rescale_prediction,
    write_trace_csv,
)


PREDICTIONS = Path("Map_Matching/trajectories_prediction.pkl")
RAW_FILE = Path("Map_Matching/gps_with_speed_224_UPDATED.xls")


def test_map_matching_prediction_to_trace_csv_smoke(tmp_path: Path) -> None:
    predictions = load_predicted_trajectories(PREDICTIONS)

    trace_frame = rescale_prediction(
        predictions,
        RAW_FILE,
        trajectory_index=0,
        sequence_length=224,
    )
    output_path = write_trace_csv(trace_frame, tmp_path / "trace.csv")
    reloaded = pd.read_csv(output_path)

    assert predictions.shape == (20, 448)
    assert list(reloaded.columns) == ["latitude", "longitude"]
    assert reloaded.shape == (224, 2)
    assert np.isfinite(reloaded["latitude"]).all()
    assert np.isfinite(reloaded["longitude"]).all()


def test_real_mappymatch_dependency_is_available_or_skipped() -> None:
    if importlib.util.find_spec("mappymatch") is None:
        pytest.skip("mappymatch is not installed; real map matching requires it.")


def test_reconstruct_trajectories_script_prediction_to_trace_smoke(
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reconstruct_trajectories.py",
            "--predictions",
            str(PREDICTIONS),
            "--reference-csv",
            str(RAW_FILE),
            "--output-dir",
            str(tmp_path),
            "--max-samples",
            "1",
        ],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    metadata_path = tmp_path / "reconstruction_metadata.json"
    trace_path = tmp_path / "traces" / "trace_0000.csv"
    metadata = json.loads(metadata_path.read_text())
    trace = pd.read_csv(trace_path)

    assert "Wrote 1 trace CSV" in result.stdout
    assert metadata[0]["trajectory_index"] == 0
    assert metadata[0]["num_points"] == 224
    assert list(trace.columns) == ["latitude", "longitude"]
    assert trace.shape == (224, 2)

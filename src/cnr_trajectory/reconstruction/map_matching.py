"""Map matching helpers extracted from the original Map_Matching notebook.

The original notebook starts from predicted normalized coordinate trajectories,
not from generated images. This module keeps that contract explicit and avoids
running map downloads unless the caller asks for it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_predicted_trajectories(path: str | Path) -> np.ndarray:
    """Load `trajectories_prediction.pkl` and return it as a NumPy array."""
    with Path(path).open("rb") as file:
        trajectories = pickle.load(file)
    return _to_numpy(trajectories)


def load_reference_coordinate_bounds(raw_csv_path: str | Path) -> dict[str, float]:
    """Load lon/lat bounds using the same raw CSV contract as the notebook."""
    frame = pd.read_csv(raw_csv_path)
    return {
        "lon_min": float(frame["lon"].min()),
        "lon_max": float(frame["lon"].max()),
        # Preserve the notebook naming, even though min/max variable names are inverted.
        "lat_min_notebook": float(frame["lat"].max()),
        "lat_max_notebook": float(frame["lat"].min()),
    }


def split_prediction(
    predicted_trajectories: np.ndarray,
    trajectory_index: int,
    sequence_length: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one prediction row into normalized longitude and latitude arrays."""
    row = predicted_trajectories[trajectory_index]
    if row.shape[0] < 2 * sequence_length:
        raise ValueError(
            f"Prediction row has length {row.shape[0]}, expected at least {2 * sequence_length}."
        )
    longitudes = row[:sequence_length].astype(float)
    latitudes = row[-sequence_length:].astype(float)
    return longitudes, latitudes


def rescale_prediction(
    predicted_trajectories: np.ndarray,
    raw_csv_path: str | Path,
    trajectory_index: int = 0,
    sequence_length: int = 224,
) -> pd.DataFrame:
    """Convert one normalized prediction row to longitude/latitude coordinates."""
    normalized_lon, normalized_lat = split_prediction(
        predicted_trajectories,
        trajectory_index=trajectory_index,
        sequence_length=sequence_length,
    )
    bounds = load_reference_coordinate_bounds(raw_csv_path)

    lon = (bounds["lon_max"] - bounds["lon_min"]) * normalized_lon + bounds["lon_min"]
    lat = bounds["lat_max_notebook"] - (
        normalized_lat * (bounds["lat_max_notebook"] - bounds["lat_min_notebook"])
    )
    return pd.DataFrame({"latitude": lat, "longitude": lon})


def write_trace_csv(trace_frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a trace CSV compatible with `mappymatch.constructs.trace.Trace.from_csv`."""
    required = {"latitude", "longitude"}
    missing = required.difference(trace_frame.columns)
    if missing:
        raise ValueError(f"Missing required trace columns: {sorted(missing)}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_frame.to_csv(output, index=False)
    return output


def remove_loops(path: list[Any]) -> list[Any]:
    """Preserve the loop-removal helper from the Map_Matching notebook."""
    visited = {}
    cleaned = []

    for i, road in enumerate(path):
        road_tuple = (road.road_id.start, road.road_id.end)

        if road_tuple in visited:
            loop_start = visited[road_tuple]
            cleaned = cleaned[: loop_start + 1]
            visited = {
                (item.road_id.start, item.road_id.end): j
                for j, item in enumerate(cleaned)
            }
            continue

        if cleaned and cleaned[-1].road_id.end != road.road_id.start:
            expected_start = cleaned[-1].road_id.end
            connector_found = False
            for j in range(i + 1, len(path)):
                next_road = path[j]
                if next_road.road_id.start == expected_start:
                    connector_found = True
                    visited[(next_road.road_id.start, next_road.road_id.end)] = len(
                        cleaned
                    )
                    cleaned.append(next_road)
                    break

            if not connector_found:
                continue

        visited[road_tuple] = len(cleaned)
        cleaned.append(road)

    return cleaned


def match_trace_csv(
    trace_csv_path: str | Path,
    padding: float = 1e3,
    network_type: str = "DRIVE",
) -> Any:
    """Run mappymatch on a trace CSV.

    This may download a road network through the mappymatch/NxMap stack, so it
    is intentionally not used by default tests.

    TODO: Add explicit offline map/network inputs if the thesis pipeline requires
    reproducible map matching without network access.
    """
    from mappymatch.constructs.geofence import Geofence
    from mappymatch.constructs.trace import Trace
    from mappymatch.maps.nx.nx_map import NetworkType, NxMap
    from mappymatch.matchers.lcss.lcss import LCSSMatcher

    trace = Trace.from_csv(
        str(trace_csv_path),
        lat_column="latitude",
        lon_column="longitude",
        xy=True,
    )
    geofence = Geofence.from_trace(trace, padding=padding)
    selected_network_type = getattr(NetworkType, network_type)
    nx_map = NxMap.from_geofence(geofence, network_type=selected_network_type)
    matcher = LCSSMatcher(nx_map)
    return matcher.match_trace(trace)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

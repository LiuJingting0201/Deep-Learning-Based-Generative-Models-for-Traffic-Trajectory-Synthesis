"""Gramian Angular Field encoding utilities.

These helpers preserve the core trajectory image encoding path from
`23_Speed-1_folder.ipynb`:

1. normalize longitude and latitude globally,
2. compute Hilbert index per point,
3. group points by vehicle,
4. squeeze and duplicate each sequence to a fixed length,
5. encode GASF, GADF, and MTF channels,
6. scale channels to 0-255.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField

from cnr_trajectory.encoding.hilbert import hilbert_index
from cnr_trajectory.encoding.sequence import duplicate, squeeze


@dataclass(frozen=True)
class EncodedTrajectories:
    """Container for encoded trajectory images and their labels."""

    images: np.ndarray
    labels: np.ndarray
    vehicle_ids: np.ndarray
    sequences: np.ndarray


def normalize_lon_lat(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize `lon` and `lat` with the original global min/max method."""
    required = {"vehicle_id", "type", "lon", "lat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    normalized = frame.copy()
    lon_min = normalized["lon"].min()
    lon_max = normalized["lon"].max()
    lat_min = normalized["lat"].min()
    lat_max = normalized["lat"].max()

    if lon_max == lon_min or lat_max == lat_min:
        raise ValueError("Cannot normalize coordinates with zero longitude/latitude range.")

    normalized["lon"] = (normalized["lon"] - lon_min) / (lon_max - lon_min)
    normalized["lat"] = (normalized["lat"] - lat_min) / (lat_max - lat_min)
    return normalized


def extract_hilbert_sequences(
    frame: pd.DataFrame,
    sequence_length: int = 224,
    eps: float = 0.0037,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Extract fixed-length Hilbert sequences grouped by vehicle."""
    normalized = normalize_lon_lat(frame)
    normalized["Hilbert"] = [
        hilbert_index(x, y, eps) for x, y in zip(normalized["lon"], normalized["lat"])
    ]

    grouped = (
        normalized.groupby("vehicle_id")
        .agg(labels=("type", list), Hilbert=("Hilbert", list))
        .reset_index()
    )
    grouped["label"] = grouped["labels"].apply(lambda labels: labels[0])
    grouped["squeezed_Hilbert"] = grouped["Hilbert"].apply(
        squeeze, final_length=sequence_length
    )
    grouped["squeezed_Hilbert"] = grouped["squeezed_Hilbert"].apply(
        duplicate, k=sequence_length
    )

    grouped = _sort_vehicle_ids_like_notebook(grouped)
    if max_samples is not None:
        grouped = grouped.head(max_samples)

    if grouped.empty:
        raise ValueError("No trajectory sequences were extracted.")

    return grouped


def encode_hilbert_sequences(
    sequences: list[list[float]] | np.ndarray,
    n_bins: int = 8,
) -> np.ndarray:
    """Encode fixed-length Hilbert sequences as RGB GASF/GADF/MTF images."""
    values = np.vstack(np.array(sequences, dtype=object)).astype(float)

    gaf_s = GramianAngularField(method="summation", sample_range=(0, 1))
    gaf_d = GramianAngularField(method="difference", sample_range=(0, 1))
    mtf = MarkovTransitionField(strategy="uniform", n_bins=n_bins)

    gasf = gaf_s.fit_transform(values)
    gadf = gaf_d.fit_transform(values)
    mtf_images = mtf.fit_transform(values)

    red = _scale_to_uint8_range(gasf)
    green = _scale_to_uint8_range(gadf)
    blue = _scale_to_uint8_range(mtf_images)
    return np.round(np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def encode_trajectory_frame(
    frame: pd.DataFrame,
    sequence_length: int = 224,
    eps: float = 0.0037,
    max_samples: int | None = None,
) -> EncodedTrajectories:
    """Run the minimal original preprocessing and image encoding pipeline."""
    grouped = extract_hilbert_sequences(
        frame,
        sequence_length=sequence_length,
        eps=eps,
        max_samples=max_samples,
    )
    images = encode_hilbert_sequences(grouped["squeezed_Hilbert"].tolist())
    return EncodedTrajectories(
        images=images,
        labels=grouped["label"].to_numpy(),
        vehicle_ids=grouped["vehicle_id"].to_numpy(),
        sequences=np.vstack(grouped["squeezed_Hilbert"].to_numpy()).astype(float),
    )


def _scale_to_uint8_range(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        raise ValueError("Cannot scale a constant encoded channel to 0-255.")
    return ((values - minimum) / (maximum - minimum)) * 255


def _sort_vehicle_ids_like_notebook(grouped: pd.DataFrame) -> pd.DataFrame:
    vehicle_numbers = grouped["vehicle_id"].str.extract(r"car(\d+)").astype(float)
    if vehicle_numbers.isna().any().any():
        return grouped.reset_index(drop=True)
    sorted_grouped = grouped.assign(vehicle_num=vehicle_numbers[0].astype(int))
    return (
        sorted_grouped.sort_values("vehicle_num")
        .drop(columns="vehicle_num")
        .reset_index(drop=True)
    )

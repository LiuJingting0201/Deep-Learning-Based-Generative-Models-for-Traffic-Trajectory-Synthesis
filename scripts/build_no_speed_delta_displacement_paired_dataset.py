"""Build a no-speed delta-displacement trajectory-image paired dataset.

The labels keep the physical delta displacement values. For image construction
only, dx and dy are normalized with dataset-level min/max values into [0, 1]
before Hilbert/GASF/GADF/MTF encoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IMAGE_GENERATION_DIR = PROJECT_ROOT / "Image_Generation" / "Image_Generation"
if str(ORIGINAL_IMAGE_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_IMAGE_GENERATION_DIR))

from Duplicate_function import duplicate  # noqa: E402
from Hilbert_function import hilbIndex  # noqa: E402
from Squeeze_function import squeeze  # noqa: E402


DEFAULT_INPUT = ORIGINAL_IMAGE_GENERATION_DIR / "gps_with_speed_224_UPDATED.xls"
DEFAULT_UNIQUE_IDS = ORIGINAL_IMAGE_GENERATION_DIR / "unique_vehicle_ids.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_no_speed_delta_displacement_paired"
EXPECTED_COLUMNS = ["time", "vehicle_id", "x", "y", "type", "lon", "lat", "speed"]


@dataclass(frozen=True)
class DeltaSample:
    sample_id: str
    vehicle_id: str
    original_num_points: int
    absolute_xy: np.ndarray
    delta_xy: np.ndarray
    start_xy: np.ndarray
    image_path: Path
    label_delta_path: Path
    label_absolute_path: Path
    start_path: Path
    hilbert_sequence: list[float]
    sequence_length: int


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    images_dir = output_dir / "images"
    labels_delta_dir = output_dir / "labels_delta_displacement"
    labels_absolute_dir = output_dir / "labels_absolute"
    starts_dir = output_dir / "starts"
    sanity_dir = output_dir / "sanity_checks"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    for directory in [images_dir, labels_delta_dir, labels_absolute_dir, starts_dir, sanity_dir]:
        directory.mkdir(parents=True)

    frame = read_raw_gps(input_path)
    unique_ids_frame = read_unique_ids(args.unique_vehicle_ids)
    print_raw_inspection(frame, unique_ids_frame)
    validate_raw_inputs(frame, unique_ids_frame, args.sequence_length)

    frame = frame.copy()
    frame["vehicle_id"] = frame["vehicle_id"].astype(str)

    ordered_vehicle_ids = sort_vehicle_ids(frame["vehicle_id"].unique())
    trajectories: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_deltas: list[np.ndarray] = []
    for vehicle_id in ordered_vehicle_ids:
        group = frame.loc[frame["vehicle_id"] == vehicle_id].sort_values(
            "time", kind="mergesort"
        )
        absolute_xy = group[["x", "y"]].to_numpy(dtype=np.float64)
        delta_xy = compute_delta_displacement(absolute_xy)
        trajectories.append((vehicle_id, absolute_xy, delta_xy))
        all_deltas.append(delta_xy)

    stacked_delta = np.concatenate(all_deltas, axis=0)
    delta_x_min = float(stacked_delta[:, 0].min())
    delta_x_max = float(stacked_delta[:, 0].max())
    delta_y_min = float(stacked_delta[:, 1].min())
    delta_y_max = float(stacked_delta[:, 1].max())
    if delta_x_max == delta_x_min or delta_y_max == delta_y_min:
        raise ValueError("Cannot normalize delta values with a zero delta range.")

    abs_x_min = float(frame["x"].min())
    abs_x_max = float(frame["x"].max())
    abs_y_min = float(frame["y"].min())
    abs_y_max = float(frame["y"].max())

    write_normalization_json(
        output_dir / "encoding_normalization.json",
        delta_x_min,
        delta_x_max,
        delta_y_min,
        delta_y_max,
    )

    samples: list[DeltaSample] = []
    metadata_rows: list[dict[str, Any]] = []
    for sample_index, (vehicle_id, absolute_xy, delta_xy) in enumerate(trajectories):
        sample_id = f"sample_{sample_index:06d}"
        image_path = images_dir / f"{sample_id}.png"
        label_delta_path = labels_delta_dir / f"{sample_id}.npy"
        label_absolute_path = labels_absolute_dir / f"{sample_id}.npy"
        start_path = starts_dir / f"{sample_id}.npy"
        start_xy = absolute_xy[0].astype(np.float64)

        normalized_delta = normalize_delta_for_encoding(
            delta_xy, delta_x_min, delta_x_max, delta_y_min, delta_y_max
        )
        hilbert_sequence = [
            hilbIndex(float(x), float(y), args.hilbert_eps)
            for x, y in normalized_delta
        ]
        hilbert_sequence = squeeze(hilbert_sequence, args.sequence_length)
        hilbert_sequence = duplicate(hilbert_sequence, args.sequence_length)

        np.save(label_absolute_path, absolute_xy.astype(np.float64))
        np.save(label_delta_path, delta_xy.astype(np.float64))
        np.save(start_path, start_xy)

        samples.append(
            DeltaSample(
                sample_id=sample_id,
                vehicle_id=vehicle_id,
                original_num_points=len(absolute_xy),
                absolute_xy=absolute_xy,
                delta_xy=delta_xy,
                start_xy=start_xy,
                image_path=image_path,
                label_delta_path=label_delta_path,
                label_absolute_path=label_absolute_path,
                start_path=start_path,
                hilbert_sequence=hilbert_sequence,
                sequence_length=args.sequence_length,
            )
        )
        metadata_rows.append(
            metadata_row(
                sample_id=sample_id,
                vehicle_id=vehicle_id,
                original_num_points=len(absolute_xy),
                image_path=image_path.relative_to(output_dir).as_posix(),
                label_delta_path=label_delta_path.relative_to(output_dir).as_posix(),
                label_absolute_path=label_absolute_path.relative_to(output_dir).as_posix(),
                start_path=start_path.relative_to(output_dir).as_posix(),
                encoding_method=args.encoding_method,
                delta_x_min=delta_x_min,
                delta_x_max=delta_x_max,
                delta_y_min=delta_y_min,
                delta_y_max=delta_y_max,
                absolute_x_min=abs_x_min,
                absolute_x_max=abs_x_max,
                absolute_y_min=abs_y_min,
                absolute_y_max=abs_y_max,
                valid_or_skipped="valid",
                skip_reason="",
            )
        )

    image_shapes = encode_and_save_images(samples, args.n_bins)
    write_metadata(output_dir / "metadata.csv", metadata_rows)
    sanity_summary = run_sanity_checks(samples, image_shapes, sanity_dir, args.seed)
    write_summary(output_dir / "dataset_summary.json", frame, samples, sanity_summary)

    print(f"Created {len(samples)} valid delta-displacement paired samples in {output_dir}")
    print(f"Images: {len(list(images_dir.glob('*.png')))}")
    print(f"Delta labels: {len(list(labels_delta_dir.glob('*.npy')))}")
    print(f"Absolute labels: {len(list(labels_absolute_dir.glob('*.npy')))}")
    print(f"Starts: {len(list(starts_dir.glob('*.npy')))}")
    print(f"Metadata rows: {len(metadata_rows)}")
    print(f"Max integration reconstruction error in sanity checks: {sanity_summary['max_abs_error']:.10g}")
    print(f"Sanity check plots: {sanity_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unique-vehicle-ids", type=Path, default=DEFAULT_UNIQUE_IDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=224)
    parser.add_argument("--hilbert-eps", type=float, default=0.0037)
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--encoding-method",
        default="hilbert_gasf_gadf_mtf_no_speed_delta_displacement_xy",
    )
    return parser.parse_args()


def read_raw_gps(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, sep=",", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_excel(path)


def read_unique_ids(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def print_raw_inspection(frame: pd.DataFrame, unique_ids_frame: pd.DataFrame) -> None:
    print("Raw GPS file inspection")
    print(f"shape: {frame.shape}")
    print(f"columns: {list(frame.columns)}")
    print("sample rows:")
    print(frame.head(8).to_string(index=False))
    print(f"unique vehicle_id count: {frame['vehicle_id'].nunique() if 'vehicle_id' in frame.columns else 'missing'}")
    if "vehicle_id" in frame.columns:
        rows_per_vehicle = frame.groupby("vehicle_id").size()
        print("rows per vehicle_id distribution:")
        print(rows_per_vehicle.describe().to_string())
        print(rows_per_vehicle.value_counts().sort_index().to_string())
    print("unique_vehicle_ids.csv inspection")
    print(f"shape: {unique_ids_frame.shape}")
    print(f"columns: {list(unique_ids_frame.columns)}")
    print("sample rows:")
    print(unique_ids_frame.head(8).to_string(index=False))


def validate_raw_inputs(
    frame: pd.DataFrame, unique_ids_frame: pd.DataFrame, sequence_length: int
) -> None:
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Raw GPS file missing expected columns: {missing_columns}")

    null_counts = frame[["vehicle_id", "time", "x", "y"]].isna().sum()
    if int(null_counts.sum()) != 0:
        raise ValueError(f"Raw GPS file has nulls in required columns: {null_counts.to_dict()}")

    rows_per_vehicle = frame.groupby("vehicle_id").size()
    bad_counts = rows_per_vehicle[rows_per_vehicle != sequence_length]
    if not bad_counts.empty:
        raise ValueError(
            "Found vehicle_id groups without exactly "
            f"{sequence_length} rows: {bad_counts.head(20).to_dict()}"
        )

    id_column = "vehicle_id" if "vehicle_id" in unique_ids_frame.columns else unique_ids_frame.columns[0]
    csv_ids = set(unique_ids_frame[id_column].astype(str))
    raw_ids = set(frame["vehicle_id"].astype(str))
    only_csv = sorted(csv_ids - raw_ids)
    only_raw = sorted(raw_ids - csv_ids)
    print(f"unique IDs in raw GPS: {len(raw_ids)}")
    print(f"unique IDs in unique_vehicle_ids.csv: {len(csv_ids)}")
    print(f"IDs only in CSV: {len(only_csv)}")
    print(f"IDs only in raw GPS: {len(only_raw)}")
    if only_csv or only_raw:
        raise ValueError(
            "Vehicle ID mismatch between raw GPS and unique_vehicle_ids.csv. "
            f"only_csv={only_csv[:20]}, only_raw={only_raw[:20]}"
        )


def sort_vehicle_ids(vehicle_ids: Any) -> list[str]:
    def key(vehicle_id: Any) -> tuple[int, Any]:
        text = str(vehicle_id)
        if text.startswith("car") and text[3:].isdigit():
            return (0, int(text[3:]))
        return (1, text)

    return [str(vehicle_id) for vehicle_id in sorted(vehicle_ids, key=key)]


def compute_delta_displacement(absolute_xy: np.ndarray) -> np.ndarray:
    delta_xy = np.zeros_like(absolute_xy, dtype=np.float64)
    delta_xy[1:] = absolute_xy[1:] - absolute_xy[:-1]
    return delta_xy


def normalize_delta_for_encoding(
    delta_xy: np.ndarray,
    delta_x_min: float,
    delta_x_max: float,
    delta_y_min: float,
    delta_y_max: float,
) -> np.ndarray:
    normalized = delta_xy.astype(np.float64).copy()
    normalized[:, 0] = (normalized[:, 0] - delta_x_min) / (delta_x_max - delta_x_min)
    normalized[:, 1] = (normalized[:, 1] - delta_y_min) / (delta_y_max - delta_y_min)
    return np.clip(normalized, 0.0, 1.0)


def encode_and_save_images(samples: list[DeltaSample], n_bins: int) -> list[tuple[int, ...]]:
    image_shapes: list[tuple[int, ...]] = []
    gaf_s = GramianAngularField(method="summation", sample_range=(0, 1))
    gaf_d = GramianAngularField(method="difference", sample_range=(0, 1))
    mtf = MarkovTransitionField(strategy="uniform", n_bins=n_bins)

    for sample in samples:
        values = np.asarray(sample.hilbert_sequence, dtype=np.float64).reshape(1, -1)
        red = scale_from_range(gaf_s.fit_transform(values)[0], -1.0, 1.0)
        green = scale_from_range(gaf_d.fit_transform(values)[0], -1.0, 1.0)
        blue = scale_from_range(mtf.fit_transform(values)[0], 0.0, 1.0)
        image = np.round(np.stack((red, green, blue), axis=-1)).astype(np.uint8)
        Image.fromarray(image, mode="RGB").save(sample.image_path)
        image_shapes.append(tuple(image.shape))
    return image_shapes


def scale_from_range(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    clipped = np.clip(values, minimum, maximum)
    return ((clipped - minimum) / (maximum - minimum)) * 255.0


def metadata_row(
    sample_id: str,
    vehicle_id: str,
    original_num_points: int,
    image_path: str,
    label_delta_path: str,
    label_absolute_path: str,
    start_path: str,
    encoding_method: str,
    delta_x_min: float,
    delta_x_max: float,
    delta_y_min: float,
    delta_y_max: float,
    absolute_x_min: float,
    absolute_x_max: float,
    absolute_y_min: float,
    absolute_y_max: float,
    valid_or_skipped: str,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "vehicle_id": vehicle_id,
        "original_num_points": original_num_points,
        "image_path": image_path,
        "label_delta_displacement_path": label_delta_path,
        "label_absolute_path": label_absolute_path,
        "start_path": start_path,
        "encoding_method": encoding_method,
        "coordinate_type": "delta_displacement_xy",
        "delta_x_min": delta_x_min,
        "delta_x_max": delta_x_max,
        "delta_y_min": delta_y_min,
        "delta_y_max": delta_y_max,
        "absolute_x_min": absolute_x_min,
        "absolute_x_max": absolute_x_max,
        "absolute_y_min": absolute_y_min,
        "absolute_y_max": absolute_y_max,
        "valid_or_skipped": valid_or_skipped,
        "skip_reason": skip_reason,
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "vehicle_id",
        "original_num_points",
        "image_path",
        "label_delta_displacement_path",
        "label_absolute_path",
        "start_path",
        "encoding_method",
        "coordinate_type",
        "delta_x_min",
        "delta_x_max",
        "delta_y_min",
        "delta_y_max",
        "absolute_x_min",
        "absolute_x_max",
        "absolute_y_min",
        "absolute_y_max",
        "valid_or_skipped",
        "skip_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_normalization_json(
    path: Path,
    delta_x_min: float,
    delta_x_max: float,
    delta_y_min: float,
    delta_y_max: float,
) -> None:
    payload = {
        "delta_x_global_min": delta_x_min,
        "delta_x_global_max": delta_x_max,
        "delta_y_global_min": delta_y_min,
        "delta_y_global_max": delta_y_max,
        "normalization_method": "dataset_level_min_max_to_unit_interval_for_image_encoding_only",
        "notes": (
            "Physical delta_displacement_xy labels are saved unnormalized. "
            "For Hilbert/GASF/GADF/MTF image construction, dx is mapped as "
            "(dx - delta_x_global_min) / (delta_x_global_max - delta_x_global_min), "
            "and dy is mapped analogously. Values are clipped to [0, 1] before "
            "calling hilbIndex. Speed and map channels are not used."
        ),
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_sanity_checks(
    samples: list[DeltaSample],
    image_shapes: list[tuple[int, ...]],
    sanity_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    count = min(5, len(samples))
    indices = rng.choice(len(samples), size=count, replace=False)
    max_abs_error = 0.0
    checked_sample_ids: list[str] = []

    for check_index, sample_index in enumerate(indices):
        sample = samples[int(sample_index)]
        image = np.asarray(Image.open(sample.image_path))
        absolute_xy = np.load(sample.label_absolute_path)
        delta_xy = np.load(sample.label_delta_path)
        start_xy = np.load(sample.start_path)
        integrated_xy = integrate_delta(start_xy, delta_xy)
        sample_max_error = float(np.max(np.abs(integrated_xy - absolute_xy)))
        max_abs_error = max(max_abs_error, sample_max_error)
        checked_sample_ids.append(sample.sample_id)

        if image.shape != (sample.sequence_length, sample.sequence_length, 3):
            raise AssertionError(f"Unexpected image shape for {sample.sample_id}: {image.shape}")
        if absolute_xy.shape != (sample.sequence_length, 2):
            raise AssertionError(f"Unexpected absolute label shape for {sample.sample_id}: {absolute_xy.shape}")
        if delta_xy.shape != (sample.sequence_length, 2):
            raise AssertionError(f"Unexpected delta label shape for {sample.sample_id}: {delta_xy.shape}")
        if start_xy.shape != (2,):
            raise AssertionError(f"Unexpected start shape for {sample.sample_id}: {start_xy.shape}")
        if sample_max_error > 1e-5:
            raise AssertionError(
                f"Integrated delta does not reconstruct absolute trajectory for "
                f"{sample.sample_id}: max_abs_error={sample_max_error}"
            )

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(absolute_xy[:, 0], absolute_xy[:, 1], linewidth=1.5)
        axes[0, 0].scatter(absolute_xy[0, 0], absolute_xy[0, 1], s=24, label="start")
        axes[0, 0].scatter(absolute_xy[-1, 0], absolute_xy[-1, 1], s=24, label="end")
        axes[0, 0].set_title("original absolute trajectory")
        axes[0, 0].set_aspect("equal", adjustable="box")
        axes[0, 0].legend(loc="best", fontsize=8)

        axes[0, 1].plot(delta_xy[:, 0], label="dx", linewidth=1.2)
        axes[0, 1].plot(delta_xy[:, 1], label="dy", linewidth=1.2)
        axes[0, 1].set_title("delta displacement over timestep")
        axes[0, 1].legend(loc="best", fontsize=8)

        axes[1, 0].plot(integrated_xy[:, 0], integrated_xy[:, 1], linewidth=1.5)
        axes[1, 0].scatter(integrated_xy[0, 0], integrated_xy[0, 1], s=24, label="start")
        axes[1, 0].scatter(integrated_xy[-1, 0], integrated_xy[-1, 1], s=24, label="end")
        axes[1, 0].set_title(f"integrated delta, max error={sample_max_error:.3g}")
        axes[1, 0].set_aspect("equal", adjustable="box")
        axes[1, 0].legend(loc="best", fontsize=8)

        axes[1, 1].imshow(image)
        axes[1, 1].set_title("generated RGB image")
        axes[1, 1].axis("off")
        for axis in axes.ravel():
            axis.grid(False)
        fig.suptitle(f"{sample.sample_id} | {sample.vehicle_id}", fontsize=11)
        fig.tight_layout()
        fig.savefig(sanity_dir / f"sanity_{check_index}_{sample.sample_id}.png", dpi=150)
        plt.close(fig)

    expected_shape = (samples[0].sequence_length, samples[0].sequence_length, 3)
    bad_shapes = [shape for shape in image_shapes if shape != expected_shape]
    if bad_shapes:
        raise AssertionError(f"Unexpected encoded image shapes: {bad_shapes[:5]}")

    return {
        "num_checked": count,
        "checked_sample_ids": checked_sample_ids,
        "max_abs_error": max_abs_error,
        "integrated_delta_reconstructs_absolute": max_abs_error <= 1e-5,
    }


def integrate_delta(start_xy: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    integrated_xy = np.empty_like(delta_xy, dtype=np.float64)
    integrated_xy[0] = start_xy.astype(np.float64)
    integrated_xy[1:] = start_xy.astype(np.float64) + np.cumsum(delta_xy[1:], axis=0)
    return integrated_xy


def write_summary(
    path: Path, frame: pd.DataFrame, samples: list[DeltaSample], sanity_summary: dict[str, Any]
) -> None:
    payload = {
        "raw_shape": list(frame.shape),
        "raw_columns": list(frame.columns),
        "num_vehicle_ids": int(frame["vehicle_id"].nunique()),
        "num_samples": len(samples),
        "sanity_checks": sanity_summary,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

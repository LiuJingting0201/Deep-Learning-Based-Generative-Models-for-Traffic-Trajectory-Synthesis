"""Build a strictly paired no-speed trajectory-image dataset.

This script rebuilds image/label pairs directly from raw GPS trajectory rows.
It does not depend on any pre-existing generated images.
"""

from __future__ import annotations

import argparse
import csv
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
from US_function import uniform_Sampling  # noqa: E402,F401


DEFAULT_INPUT = ORIGINAL_IMAGE_GENERATION_DIR / "gps_with_speed_224_UPDATED.xls"
DEFAULT_UNIQUE_IDS = ORIGINAL_IMAGE_GENERATION_DIR / "unique_vehicle_ids.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_no_speed_paired"


@dataclass(frozen=True)
class FieldNames:
    vehicle_id: str
    x: str
    y: str
    order: str | None
    speed: str | None


@dataclass(frozen=True)
class ValidSample:
    sample_id: str
    vehicle_id: str
    original_num_points: int
    label_path: Path
    image_path: Path
    coords: np.ndarray
    hilbert_sequence: list[float]
    sequence_length: int


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    sanity_dir = output_dir / "sanity_checks"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    sanity_dir.mkdir(parents=True)

    frame = read_raw_gps(input_path)
    fields = infer_fields(frame, args.coordinate_columns)
    unique_ids_frame = read_unique_ids(args.unique_vehicle_ids)
    print_inspection(frame, fields, unique_ids_frame)

    frame = frame.dropna(subset=[fields.vehicle_id, fields.x, fields.y]).copy()
    frame[fields.vehicle_id] = frame[fields.vehicle_id].astype(str)

    x_min, x_max = frame[fields.x].min(), frame[fields.x].max()
    y_min, y_max = frame[fields.y].min(), frame[fields.y].max()
    if x_max == x_min or y_max == y_min:
        raise ValueError("Cannot normalize coordinates with a zero coordinate range.")

    samples: list[ValidSample] = []
    metadata_rows: list[dict[str, Any]] = []

    grouped = frame.groupby(fields.vehicle_id, sort=False)
    ordered_vehicle_ids = sort_vehicle_ids(grouped.groups.keys())
    for vehicle_id in ordered_vehicle_ids:
        group = grouped.get_group(vehicle_id).copy()
        original_num_points = len(group)

        if fields.order is not None:
            group = group.sort_values(fields.order, kind="mergesort")

        if original_num_points < args.sequence_length:
            metadata_rows.append(
                metadata_row(
                    sample_id="",
                    vehicle_id=vehicle_id,
                    original_num_points=original_num_points,
                    image_path="",
                    label_path="",
                    encoding_method=args.encoding_method,
                    skipped_or_valid="skipped",
                    skip_reason=(
                        f"too_few_points:{original_num_points}<"
                        f"{args.sequence_length}"
                    ),
                )
            )
            continue

        coords = group[[fields.x, fields.y]].to_numpy(dtype=np.float32)
        coords_224 = resample_trajectory(coords, args.sequence_length)
        sample_id = f"sample_{len(samples):06d}"
        label_path = labels_dir / f"{sample_id}.npy"
        image_path = images_dir / f"{sample_id}.png"

        np.save(label_path, coords_224.astype(np.float32))

        normalized = normalize_coords(coords_224, x_min, x_max, y_min, y_max)
        hilbert_sequence = [
            hilbIndex(float(x), float(y), args.hilbert_eps) for x, y in normalized
        ]
        hilbert_sequence = squeeze(hilbert_sequence, args.sequence_length)
        hilbert_sequence = duplicate(hilbert_sequence, args.sequence_length)

        samples.append(
            ValidSample(
                sample_id=sample_id,
                vehicle_id=vehicle_id,
                original_num_points=original_num_points,
                label_path=label_path,
                image_path=image_path,
                coords=coords_224,
                hilbert_sequence=hilbert_sequence,
                sequence_length=args.sequence_length,
            )
        )
        metadata_rows.append(
            metadata_row(
                sample_id=sample_id,
                vehicle_id=vehicle_id,
                original_num_points=original_num_points,
                image_path=image_path.relative_to(output_dir).as_posix(),
                label_path=label_path.relative_to(output_dir).as_posix(),
                encoding_method=args.encoding_method,
                skipped_or_valid="valid",
                skip_reason="",
            )
        )

    if not samples:
        write_metadata(output_dir / "metadata.csv", metadata_rows)
        raise ValueError("No valid samples were created.")

    image_shapes = encode_and_save_images(samples, n_bins=args.n_bins)

    write_metadata(output_dir / "metadata.csv", metadata_rows)
    run_sanity_checks(samples, image_shapes, sanity_dir, args.seed)

    valid_count = sum(row["skipped_or_valid"] == "valid" for row in metadata_rows)
    skipped_count = sum(row["skipped_or_valid"] == "skipped" for row in metadata_rows)
    print(f"Created {valid_count} valid paired samples in {output_dir}")
    print(f"Skipped {skipped_count} trajectories")
    print(
        "Images: "
        f"({valid_count}, {args.sequence_length}, {args.sequence_length}, 3); "
        f"labels: ({valid_count}, {args.sequence_length}, 2)"
    )
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
        "--coordinate-columns",
        nargs=2,
        metavar=("X_COLUMN", "Y_COLUMN"),
        default=None,
        help="Position columns to use for labels and image encoding. Defaults to x y.",
    )
    parser.add_argument(
        "--encoding-method",
        default="hilbert_gasf_gadf_mtf_no_speed",
        help="Metadata label for the image encoding method.",
    )
    return parser.parse_args()


def read_raw_gps(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path, sep=",", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_excel(path)


def read_unique_ids(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def infer_fields(frame: pd.DataFrame, coordinate_columns: list[str] | None) -> FieldNames:
    columns = set(frame.columns)
    vehicle_id = first_present(columns, ["vehicle_id", "vehicle", "id"])
    if vehicle_id is None:
        raise ValueError("Could not infer vehicle_id column.")

    if coordinate_columns is not None:
        missing = [column for column in coordinate_columns if column not in columns]
        if missing:
            raise ValueError(f"Requested coordinate columns not found: {missing}")
        x_col, y_col = coordinate_columns
    elif {"x", "y"}.issubset(columns):
        x_col, y_col = "x", "y"
    elif {"lon", "lat"}.issubset(columns):
        x_col, y_col = "lon", "lat"
    else:
        raise ValueError("Could not infer coordinate columns from x/y or lon/lat.")

    order_col = first_present(columns, ["time", "timestamp", "frame", "step", "order"])
    speed_col = first_present(columns, ["speed", "velocity", "v"])
    return FieldNames(
        vehicle_id=vehicle_id,
        x=x_col,
        y=y_col,
        order=order_col,
        speed=speed_col,
    )


def first_present(columns: set[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def print_inspection(
    frame: pd.DataFrame, fields: FieldNames, unique_ids_frame: pd.DataFrame | None
) -> None:
    print("Raw GPS file inspection")
    print(f"shape: {frame.shape}")
    print(f"columns: {list(frame.columns)}")
    print("sample rows:")
    print(frame.head(8).to_string(index=False))
    print(f"unique vehicle_id count: {frame[fields.vehicle_id].nunique()}")
    print(f"coordinate columns: {fields.x}, {fields.y}")
    print(f"time/order column: {fields.order}")
    print(f"speed column ignored: {fields.speed}")

    if unique_ids_frame is None:
        print("unique_vehicle_ids.csv: not found")
        return

    id_column = (
        "vehicle_id"
        if "vehicle_id" in unique_ids_frame.columns
        else unique_ids_frame.columns[0]
    )
    csv_ids = set(unique_ids_frame[id_column].astype(str))
    raw_ids = set(frame[fields.vehicle_id].astype(str))
    print("unique_vehicle_ids.csv inspection")
    print(f"shape: {unique_ids_frame.shape}")
    print(f"columns: {list(unique_ids_frame.columns)}")
    print(f"ids also present in raw GPS file: {len(csv_ids.intersection(raw_ids))}")
    print(f"ids only in CSV: {len(csv_ids.difference(raw_ids))}")
    print(f"ids only in raw GPS file: {len(raw_ids.difference(csv_ids))}")


def sort_vehicle_ids(vehicle_ids: Any) -> list[str]:
    def key(vehicle_id: Any) -> tuple[int, Any]:
        text = str(vehicle_id)
        if text.startswith("car") and text[3:].isdigit():
            return (0, int(text[3:]))
        return (1, text)

    return [str(vehicle_id) for vehicle_id in sorted(vehicle_ids, key=key)]


def resample_trajectory(coords: np.ndarray, target_length: int) -> np.ndarray:
    if len(coords) == target_length:
        return coords.astype(np.float32)
    if len(coords) < target_length:
        raise ValueError("Cannot resample a trajectory shorter than target_length.")

    old_t = np.linspace(0.0, 1.0, num=len(coords), dtype=np.float64)
    new_t = np.linspace(0.0, 1.0, num=target_length, dtype=np.float64)
    x = np.interp(new_t, old_t, coords[:, 0])
    y = np.interp(new_t, old_t, coords[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def normalize_coords(
    coords: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    normalized = coords.astype(np.float64).copy()
    normalized[:, 0] = (normalized[:, 0] - x_min) / (x_max - x_min)
    normalized[:, 1] = (normalized[:, 1] - y_min) / (y_max - y_min)
    return np.clip(normalized, 0.0, 1.0)


def encode_and_save_images(samples: list[ValidSample], n_bins: int) -> list[tuple[int, ...]]:
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
    label_path: str,
    encoding_method: str,
    skipped_or_valid: str,
    skip_reason: str,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "vehicle_id": vehicle_id,
        "original_num_points": original_num_points,
        "image_path": image_path,
        "label_path": label_path,
        "encoding_method": encoding_method,
        "skipped_or_valid": skipped_or_valid,
        "skip_reason": skip_reason,
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "vehicle_id",
        "original_num_points",
        "image_path",
        "label_path",
        "encoding_method",
        "skipped_or_valid",
        "skip_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_sanity_checks(
    samples: list[ValidSample],
    image_shapes: list[tuple[int, ...]],
    sanity_dir: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    count = min(5, len(samples))
    indices = rng.choice(len(samples), size=count, replace=False)

    for check_index, sample_index in enumerate(indices):
        sample = samples[int(sample_index)]
        image = np.asarray(Image.open(sample.image_path))
        label = np.load(sample.label_path)

        if sample.sample_id not in sample.image_path.name:
            raise AssertionError(f"Image filename does not match {sample.sample_id}")
        if sample.sample_id not in sample.label_path.name:
            raise AssertionError(f"Label filename does not match {sample.sample_id}")
        expected_image_shape = (sample.sequence_length, sample.sequence_length, 3)
        expected_label_shape = (sample.sequence_length, 2)
        if image.shape != expected_image_shape:
            raise AssertionError(f"Unexpected image shape for {sample.sample_id}: {image.shape}")
        if label.shape != expected_label_shape:
            raise AssertionError(f"Unexpected label shape for {sample.sample_id}: {label.shape}")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(label[:, 0], label[:, 1], linewidth=1.5)
        axes[0].scatter(label[0, 0], label[0, 1], s=20, label="start")
        axes[0].scatter(label[-1, 0], label[-1, 1], s=20, label="end")
        axes[0].set_title(f"{sample.sample_id} trajectory")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].legend(loc="best", fontsize=8)
        axes[1].imshow(image)
        axes[1].set_title("encoded image")
        for axis in axes:
            axis.grid(False)
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(sanity_dir / f"sanity_{check_index}_{sample.sample_id}.png", dpi=150)
        plt.close(fig)

    expected_shape = (samples[0].sequence_length, samples[0].sequence_length, 3)
    bad_shapes = [shape for shape in image_shapes if shape != expected_shape]
    if bad_shapes:
        raise AssertionError(f"Unexpected encoded image shapes: {bad_shapes[:5]}")


if __name__ == "__main__":
    main()

"""Create fixed splits for the delta-displacement paired dataset.

The split assignment is copied from data_no_speed_paired/splits/split_metadata.csv
by sample_id. No reshuffling is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_DELTA_COLUMNS = {
    "sample_id",
    "vehicle_id",
    "image_path",
    "label_delta_displacement_path",
    "label_absolute_path",
    "start_path",
    "valid_or_skipped",
}


def main() -> None:
    args = parse_args()
    delta_root = args.delta_root.resolve()
    absolute_split_path = args.absolute_split_metadata.resolve()
    splits_dir = delta_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    delta_metadata = pd.read_csv(delta_root / "metadata.csv")
    absolute_split_metadata = pd.read_csv(absolute_split_path)
    validate_columns(delta_metadata, absolute_split_metadata)

    valid_delta = delta_metadata[delta_metadata["valid_or_skipped"] == "valid"].copy()
    validate_files(delta_root, valid_delta, splits_dir / "missing_samples_report.csv")

    split_metadata = copy_splits_by_sample_id(valid_delta, absolute_split_metadata)
    verify_splits(split_metadata, absolute_split_metadata)

    split_metadata.to_csv(splits_dir / "split_metadata.csv", index=False)
    for split in ["train", "val", "test"]:
        split_metadata.loc[
            split_metadata["split"] == split, ["sample_id", "vehicle_id"]
        ].to_csv(splits_dir / f"{split}_ids.csv", index=False)

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    absolute_counts = absolute_split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    print(f"Delta valid samples: {len(split_metadata)}")
    print(f"Absolute split samples: {len(absolute_split_metadata)}")
    print(f"train: {int(counts['train'])} (absolute: {int(absolute_counts['train'])})")
    print(f"val: {int(counts['val'])} (absolute: {int(absolute_counts['val'])})")
    print(f"test: {int(counts['test'])} (absolute: {int(absolute_counts['test'])})")
    print("sample_id alignment with absolute dataset: preserved")
    print(f"Saved splits to {splits_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-root",
        type=Path,
        default=Path("data_no_speed_delta_displacement_paired"),
    )
    parser.add_argument(
        "--absolute-split-metadata",
        type=Path,
        default=Path("data_no_speed_paired/splits/split_metadata.csv"),
    )
    return parser.parse_args()


def validate_columns(delta_metadata: pd.DataFrame, absolute_split_metadata: pd.DataFrame) -> None:
    missing_delta = REQUIRED_DELTA_COLUMNS.difference(delta_metadata.columns)
    if missing_delta:
        raise ValueError(f"Delta metadata missing required columns: {sorted(missing_delta)}")
    missing_absolute = {"sample_id", "vehicle_id", "split"}.difference(
        absolute_split_metadata.columns
    )
    if missing_absolute:
        raise ValueError(
            f"Absolute split metadata missing required columns: {sorted(missing_absolute)}"
        )
    if delta_metadata["sample_id"].duplicated().any():
        raise ValueError("Delta metadata contains duplicate sample_id values.")
    if absolute_split_metadata["sample_id"].duplicated().any():
        raise ValueError("Absolute split metadata contains duplicate sample_id values.")


def validate_files(delta_root: Path, metadata: pd.DataFrame, report_path: Path) -> None:
    issues: list[dict[str, object]] = []
    expected_images = set()
    expected_delta_labels = set()
    expected_absolute_labels = set()
    expected_starts = set()

    for row in metadata.itertuples(index=False):
        sample_id = str(row.sample_id)
        image_path = delta_root / row.image_path
        delta_path = delta_root / row.label_delta_displacement_path
        absolute_path = delta_root / row.label_absolute_path
        start_path = delta_root / row.start_path
        expected_images.add(Path(row.image_path).name)
        expected_delta_labels.add(Path(row.label_delta_displacement_path).name)
        expected_absolute_labels.add(Path(row.label_absolute_path).name)
        expected_starts.add(Path(row.start_path).name)

        if not image_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, "missing_image", row.image_path))
            continue
        if not delta_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, "missing_delta_label", row.label_delta_displacement_path))
            continue
        if not absolute_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, "missing_absolute_label", row.label_absolute_path))
            continue
        if not start_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, "missing_start", row.start_path))
            continue

        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                issues.append(issue(sample_id, row.vehicle_id, f"bad_image:{image.mode}:{image.size}", row.image_path))
        if np.load(delta_path).shape != (224, 2):
            issues.append(issue(sample_id, row.vehicle_id, "bad_delta_shape", row.label_delta_displacement_path))
        if np.load(absolute_path).shape != (224, 2):
            issues.append(issue(sample_id, row.vehicle_id, "bad_absolute_shape", row.label_absolute_path))
        if np.load(start_path).shape != (2,):
            issues.append(issue(sample_id, row.vehicle_id, "bad_start_shape", row.start_path))

    actual_images = {path.name for path in (delta_root / "images").glob("*.png")}
    actual_delta_labels = {
        path.name for path in (delta_root / "labels_delta_displacement").glob("*.npy")
    }
    actual_absolute_labels = {
        path.name for path in (delta_root / "labels_absolute").glob("*.npy")
    }
    actual_starts = {path.name for path in (delta_root / "starts").glob("*.npy")}
    for label, actual, expected in [
        ("image_count_or_name_mismatch", actual_images, expected_images),
        ("delta_label_count_or_name_mismatch", actual_delta_labels, expected_delta_labels),
        ("absolute_label_count_or_name_mismatch", actual_absolute_labels, expected_absolute_labels),
        ("start_count_or_name_mismatch", actual_starts, expected_starts),
    ]:
        for filename in sorted(actual.symmetric_difference(expected)):
            issues.append(issue("", "", label, filename))

    if issues:
        pd.DataFrame(issues).to_csv(report_path, index=False)
        raise RuntimeError(
            "Delta dataset consistency check failed. "
            f"Report written to {report_path}"
        )


def issue(sample_id: object, vehicle_id: object, reason: str, path: object) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "vehicle_id": vehicle_id,
        "reason": reason,
        "path": path,
    }


def copy_splits_by_sample_id(
    delta_metadata: pd.DataFrame, absolute_split_metadata: pd.DataFrame
) -> pd.DataFrame:
    absolute_by_sample = absolute_split_metadata.set_index("sample_id")
    delta_ids = set(delta_metadata["sample_id"])
    absolute_ids = set(absolute_split_metadata["sample_id"])
    missing_in_absolute = sorted(delta_ids - absolute_ids)
    missing_in_delta = sorted(absolute_ids - delta_ids)
    if missing_in_absolute or missing_in_delta:
        raise ValueError(
            "Cannot preserve absolute split assignment because sample_ids differ. "
            f"missing_in_absolute={missing_in_absolute[:20]}, "
            f"missing_in_delta={missing_in_delta[:20]}"
        )

    split_metadata = delta_metadata.copy()
    split_metadata["absolute_vehicle_id"] = split_metadata["sample_id"].map(
        absolute_by_sample["vehicle_id"]
    )
    vehicle_mismatch = split_metadata[
        split_metadata["vehicle_id"].astype(str)
        != split_metadata["absolute_vehicle_id"].astype(str)
    ]
    if not vehicle_mismatch.empty:
        raise ValueError(
            "sample_id alignment exists, but vehicle_id assignment differs. "
            f"Examples: {vehicle_mismatch[['sample_id', 'vehicle_id', 'absolute_vehicle_id']].head(20).to_dict('records')}"
        )
    split_metadata["split"] = split_metadata["sample_id"].map(absolute_by_sample["split"])
    return split_metadata.drop(columns=["absolute_vehicle_id"])


def verify_splits(
    split_metadata: pd.DataFrame, absolute_split_metadata: pd.DataFrame
) -> None:
    split_sets = {
        split: set(split_metadata.loc[split_metadata["split"] == split, "sample_id"])
        for split in ["train", "val", "test"]
    }
    if split_sets["train"] & split_sets["val"]:
        raise AssertionError("train and val splits overlap")
    if split_sets["train"] & split_sets["test"]:
        raise AssertionError("train and test splits overlap")
    if split_sets["val"] & split_sets["test"]:
        raise AssertionError("val and test splits overlap")
    combined = split_sets["train"] | split_sets["val"] | split_sets["test"]
    if combined != set(split_metadata["sample_id"]):
        raise AssertionError("Some delta samples are missing from split assignment.")

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    absolute_counts = absolute_split_metadata["split"].value_counts().reindex(
        ["train", "val", "test"]
    )
    if not counts.equals(absolute_counts):
        raise AssertionError(
            f"Delta split counts do not match absolute counts: "
            f"delta={counts.to_dict()}, absolute={absolute_counts.to_dict()}"
        )


if __name__ == "__main__":
    main()

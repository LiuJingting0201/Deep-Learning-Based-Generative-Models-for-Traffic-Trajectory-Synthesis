"""Create deterministic train/val/test splits for the no-speed paired dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_COLUMNS = {
    "sample_id",
    "vehicle_id",
    "image_path",
    "label_path",
    "skipped_or_valid",
}


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    metadata_path = data_root / "metadata.csv"
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    validate_metadata_columns(metadata)
    valid_metadata = metadata[metadata["skipped_or_valid"] == "valid"].copy()
    validate_pairs(data_root, valid_metadata, splits_dir / "missing_samples_report.csv")

    split_metadata = assign_splits(valid_metadata, args.seed)
    verify_splits(split_metadata)

    split_metadata.to_csv(splits_dir / "split_metadata.csv", index=False)
    for split in ["train", "val", "test"]:
        columns = ["sample_id", "vehicle_id"]
        split_metadata.loc[split_metadata["split"] == split, columns].to_csv(
            splits_dir / f"{split}_ids.csv", index=False
        )

    counts = split_metadata["split"].value_counts().reindex(["train", "val", "test"])
    print(f"Total valid samples: {len(split_metadata)}")
    print(f"train: {int(counts['train'])}")
    print(f"val: {int(counts['val'])}")
    print(f"test: {int(counts['test'])}")
    print(f"Saved splits to {splits_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_paired"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_metadata_columns(metadata: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")
    if metadata["sample_id"].duplicated().any():
        duplicates = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values found: {duplicates[:10]}")


def validate_pairs(data_root: Path, metadata: pd.DataFrame, report_path: Path) -> None:
    image_files = set(path.name for path in (data_root / "images").glob("*.png"))
    label_files = set(path.name for path in (data_root / "labels").glob("*.npy"))

    issues: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        image_path = data_root / row.image_path
        label_path = data_root / row.label_path
        sample_id = str(row.sample_id)

        if not image_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, row.image_path, row.label_path, "missing_image"))
            continue
        if not label_path.exists():
            issues.append(issue(sample_id, row.vehicle_id, row.image_path, row.label_path, "missing_label"))
            continue

        label = np.load(label_path)
        if label.shape != (224, 2):
            issues.append(
                issue(
                    sample_id,
                    row.vehicle_id,
                    row.image_path,
                    row.label_path,
                    f"bad_label_shape:{label.shape}",
                )
            )

        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                issues.append(
                    issue(
                        sample_id,
                        row.vehicle_id,
                        row.image_path,
                        row.label_path,
                        f"bad_image:{image.mode}:{image.size}",
                    )
                )

    expected_images = {Path(path).name for path in metadata["image_path"]}
    expected_labels = {Path(path).name for path in metadata["label_path"]}
    if image_files != expected_images:
        for filename in sorted(image_files.symmetric_difference(expected_images)):
            issues.append(issue("", "", filename, "", "image_count_or_name_mismatch"))
    if label_files != expected_labels:
        for filename in sorted(label_files.symmetric_difference(expected_labels)):
            issues.append(issue("", "", "", filename, "label_count_or_name_mismatch"))

    if issues:
        pd.DataFrame(issues).to_csv(report_path, index=False)
        raise RuntimeError(
            "Dataset image/label/metadata consistency check failed. "
            f"Report written to {report_path}"
        )


def issue(
    sample_id: object,
    vehicle_id: object,
    image_path: object,
    label_path: object,
    reason: str,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "vehicle_id": vehicle_id,
        "image_path": image_path,
        "label_path": label_path,
        "reason": reason,
    }


def assign_splits(metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    metadata = metadata.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(metadata))
    train_end = int(len(indices) * 0.8)
    val_end = train_end + int(len(indices) * 0.1)

    split_by_index = {}
    for index in indices[:train_end]:
        split_by_index[int(index)] = "train"
    for index in indices[train_end:val_end]:
        split_by_index[int(index)] = "val"
    for index in indices[val_end:]:
        split_by_index[int(index)] = "test"

    metadata["split"] = [split_by_index[index] for index in range(len(metadata))]
    return metadata


def verify_splits(split_metadata: pd.DataFrame) -> None:
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
    all_ids = set(split_metadata["sample_id"])
    if combined != all_ids:
        raise AssertionError("Some samples are missing from the split assignment")
    if split_metadata["sample_id"].duplicated().any():
        raise AssertionError("A sample appears more than once in split_metadata")


if __name__ == "__main__":
    main()

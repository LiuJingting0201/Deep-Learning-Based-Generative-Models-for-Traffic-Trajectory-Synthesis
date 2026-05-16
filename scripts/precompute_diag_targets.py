"""Precompute RGB diagonal targets for data-driven DDPM auxiliary loss."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for import_dir in (SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from train_ddpm import resolve_diag_target_paths


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.split_metadata = args.split_metadata.expanduser().resolve()
    args.output_file = args.output_file.expanduser().resolve()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")
    if not args.split_metadata.exists():
        raise FileNotFoundError(f"split_metadata does not exist: {args.split_metadata}")

    try:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create output directory {args.output_file.parent}: {exc}") from exc

    dataset = DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    target_paths = resolve_diag_target_paths(dataset, args)
    if not target_paths:
        raise ValueError(f"No train images from {args.split_metadata} matched {args.data_dir}")

    path_to_index = {path.resolve(): index for index, path in enumerate(dataset.paths)}
    channel_sum = torch.zeros(3, dtype=torch.float64)
    total = 0
    for path in target_paths:
        image = dataset[path_to_index[path.resolve()]]
        diag_values = torch.diagonal(image, dim1=-2, dim2=-1)
        channel_sum += diag_values.double().sum(dim=1)
        total += diag_values.shape[1]

    diag_targets = (channel_sum / max(1, total)).float()
    payload = {
        "diag_targets": diag_targets.tolist(),
        "source": "train_split_metadata_precomputed",
        "num_images": len(target_paths),
        "split_metadata": str(args.split_metadata),
        "data_dir": str(args.data_dir),
        "image_size": args.image_size,
    }
    args.output_file.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"data_dir: {args.data_dir}")
    print(f"split_metadata: {args.split_metadata}")
    print(f"output_file: {args.output_file}")
    print(f"number_of_train_images_used: {len(target_paths)}")
    print(f"diag_targets: {diag_targets.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing PNG training images.")
    parser.add_argument("--split-metadata", type=Path, required=True, help="split_metadata.csv with train rows.")
    parser.add_argument("--output-file", type=Path, required=True, help="JSON file to write precomputed targets to.")
    parser.add_argument("--image-size", type=int, required=True, help="Square image size used by DiffusionImageDataset.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Compute Week 4 GASF diagonal targets from real training PNG images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset


def main() -> None:
    args = parse_args()
    image_size = args.image_size or infer_image_size(args.data_dir)
    dataset = DiffusionImageDataset(args.data_dir, image_size=image_size)
    channel_sum = torch.zeros(3, dtype=torch.float64)
    total_diag_values = 0
    for image in dataset:
        diag = torch.diagonal(image, dim1=-2, dim2=-1)
        channel_sum += diag.double().sum(dim=1)
        total_diag_values += diag.shape[1]
    targets = (channel_sum / max(1, total_diag_values)).tolist()
    payload = {
        "source": "computed_from_training_images",
        "data_dir": str(args.data_dir.resolve()),
        "image_size": image_size,
        "num_images": len(dataset),
        "diag_targets": targets,
        "gasf_channels": [0, 1],
        "notes": "Targets are in diffusion tensor space [-1,1]. Gasf_structure uses channels 0 and 1 only.",
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output_file}")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def infer_image_size(data_dir: Path) -> int:
    from PIL import Image

    first = next(iter(sorted(data_dir.glob("*.png"))), None)
    if first is None:
        raise ValueError(f"No PNG images found in {data_dir}")
    with Image.open(first) as image:
        width, height = image.size
    if width != height:
        raise ValueError(f"Expected square images, got {width}x{height} for {first}")
    return int(width)


if __name__ == "__main__":
    main()

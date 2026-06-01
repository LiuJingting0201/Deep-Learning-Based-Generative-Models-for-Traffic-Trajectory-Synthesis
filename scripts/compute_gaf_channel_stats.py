"""Compute per-channel GAF image statistics in normalized [-1, 1] space."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


CHANNELS = ("GASF_R", "GADF_G", "MTF_B")


def list_paths(image_dir: Path, split_metadata: Path | None, split: str) -> tuple[list[Path], str]:
    if split_metadata is None:
        return sorted(image_dir.glob("*.png")), "all_images_no_split_metadata"
    split_metadata = split_metadata.resolve()
    data_root = split_metadata.parent.parent
    paths: list[Path] = []
    with split_metadata.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"split", "image_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split metadata missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] == split:
                paths.append((data_root / row["image_path"]).resolve())
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise ValueError(f"No images found for split={split} using {split_metadata}")
    return sorted(paths), f"{split}_split_metadata"


def compute_stats(paths: list[Path]) -> dict[str, object]:
    hist = np.zeros((3, 256), dtype=np.int64)
    count = 0
    sums = np.zeros(3, dtype=np.float64)
    sumsqs = np.zeros(3, dtype=np.float64)
    sat = np.zeros(3, dtype=np.int64)
    for path in paths:
        raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        norm = raw.astype(np.float32) / 127.5 - 1.0
        flat = norm.reshape(-1, 3).astype(np.float64)
        count += flat.shape[0]
        sums += flat.sum(axis=0)
        sumsqs += (flat * flat).sum(axis=0)
        sat += (np.abs(flat) > 0.98).sum(axis=0)
        for channel in range(3):
            hist[channel] += np.bincount(raw[:, :, channel].ravel(), minlength=256)

    values = np.arange(256, dtype=np.float64) / 127.5 - 1.0
    mean = sums / count
    std = np.sqrt(np.maximum((sumsqs - count * mean * mean) / max(count - 1, 1), 0.0))

    def quantile(channel_hist: np.ndarray, q: float) -> float:
        idx = int(np.searchsorted(np.cumsum(channel_hist), int(np.ceil(channel_hist.sum() * q)), side="left"))
        return float(values[min(idx, 255)])

    q01 = [quantile(hist[channel], 0.01) for channel in range(3)]
    q50 = [quantile(hist[channel], 0.50) for channel in range(3)]
    q99 = [quantile(hist[channel], 0.99) for channel in range(3)]
    saturation = (sat / count).astype(float)

    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "q01": q01,
        "q50": q50,
        "q99": q99,
        "saturation_ratio_abs_gt_0_98": saturation.astype(float).tolist(),
        "channel_stats": {
            name: {
                "mean": float(mean[index]),
                "std": float(std[index]),
                "q01": q01[index],
                "q50": q50[index],
                "q99": q99[index],
                "saturation_ratio_abs_gt_0_98": float(saturation[index]),
            }
            for index, name in enumerate(CHANNELS)
        },
    }


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    split_metadata = args.split_metadata.expanduser().resolve() if args.split_metadata is not None else None
    paths, source = list_paths(image_dir, split_metadata, args.split)
    stats = compute_stats(paths)
    payload = {
        "source": source,
        "image_dir": str(image_dir),
        "split_metadata": str(split_metadata) if split_metadata is not None else None,
        "split": args.split if split_metadata is not None else None,
        "num_images": len(paths),
        "normalized_range": "[-1, 1]",
        "channel_order": CHANNELS,
        **stats,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output_json": str(args.output_json), "num_images": len(paths), "source": source}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

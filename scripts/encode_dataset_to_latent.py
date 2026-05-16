"""Encode a GAF image dataset into autoencoder latent tensors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from cnr_trajectory.models.gaf_autoencoder import GAFAutoencoder


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    latent_dir = output_dir / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAFAutoencoder.from_pretrained(args.autoencoder_checkpoint / "autoencoder", map_location=device).to(device)
    model.eval()

    dataset = DiffusionImageDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metadata_path = output_dir / "metadata.csv"
    stats = RunningLatentStats()
    with metadata_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["index", "sample_id", "image_path", "latent_path", "latent_shape"],
        )
        writer.writeheader()
        offset = 0
        with torch.no_grad():
            for images in dataloader:
                images = images.to(device)
                latents = model.encode(images).detach().cpu().numpy()
                for batch_index, latent in enumerate(latents):
                    index = offset + batch_index
                    source_path = dataset.paths[index]
                    sample_id = source_path.stem
                    latent_chw = latent.astype(np.float32)
                    stats.update(latent_chw)
                    latent_path = latent_dir / f"{sample_id}.npy"
                    np.save(latent_path, latent_chw)
                    writer.writerow(
                        {
                            "index": index,
                            "sample_id": sample_id,
                            "image_path": str(source_path),
                            "latent_path": str(latent_path.relative_to(output_dir)),
                            "latent_shape": "x".join(str(dim) for dim in latent_chw.shape),
                        }
                    )
                offset += len(images)

    (output_dir / "latent_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    print(f"Encoded {len(dataset)} images to {latent_dir}")


class RunningLatentStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def update(self, latent: np.ndarray) -> None:
        latent64 = latent.astype(np.float64, copy=False)
        self.count += latent64.size
        self.sum += float(latent64.sum())
        self.sum_sq += float(np.square(latent64).sum())
        current_min = float(latent64.min())
        current_max = float(latent64.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    def to_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("Cannot compute latent stats for an empty dataset")
        mean = self.sum / self.count
        variance = max(0.0, self.sum_sq / self.count - mean**2)
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "min": self.min_value,
            "max": self.max_value,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing RGB PNG GAF images.")
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        required=True,
        help="Autoencoder checkpoint directory, typically checkpoint-final.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_latent"), help="Latent dataset output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

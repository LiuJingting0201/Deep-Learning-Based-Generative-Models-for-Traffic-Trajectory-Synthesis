"""Evaluate autoencoder reconstructions and decoded latent-DDPM samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.image_dataset import DiffusionImageDataset
from cnr_trajectory.models.gaf_autoencoder import GAFAutoencoder


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class GeneratedImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int) -> None:
        self.paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No generated images found in {image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = GAFAutoencoder.from_pretrained(args.autoencoder_checkpoint / "autoencoder", map_location=device).to(device)
    autoencoder.eval()

    real_dataset = DiffusionImageDataset(args.real_dir, image_size=args.image_size)
    generated_dataset = GeneratedImageDataset(args.generated_dir, image_size=args.image_size)

    recon_metrics = compute_reconstruction_metrics(autoencoder, real_dataset, device, args)
    generated_pair_metrics = compute_pair_metrics(real_dataset, generated_dataset, args)
    save_comparison_grid(autoencoder, real_dataset, generated_dataset, output_dir / "side_by_side.png", device, args)

    metrics = {
        "real_dir": str(args.real_dir),
        "generated_dir": str(args.generated_dir),
        "num_real": len(real_dataset),
        "num_generated": len(generated_dataset),
        "reconstruction": recon_metrics,
        "generated_vs_real_sorted_pairs": generated_pair_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def compute_reconstruction_metrics(
    autoencoder: GAFAutoencoder,
    dataset: DiffusionImageDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    total_mse = 0.0
    total_mae = 0.0
    total_pixels = 0
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            recon = autoencoder(images)
            total_mse += F.mse_loss(recon, images, reduction="sum").item()
            total_mae += F.l1_loss(recon, images, reduction="sum").item()
            total_pixels += images.numel()
    return {"mse": total_mse / total_pixels, "mae": total_mae / total_pixels}


def compute_pair_metrics(
    real_dataset: DiffusionImageDataset,
    generated_dataset: GeneratedImageDataset,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    count = min(len(real_dataset), len(generated_dataset), args.max_metric_pairs)
    total_mse = 0.0
    total_mae = 0.0
    total_pixels = 0
    for index in range(count):
        real = real_dataset[index]
        generated = generated_dataset[index]
        total_mse += F.mse_loss(generated, real, reduction="sum").item()
        total_mae += F.l1_loss(generated, real, reduction="sum").item()
        total_pixels += real.numel()
    return {"mse": total_mse / total_pixels, "mae": total_mae / total_pixels, "num_pairs": count}


def save_comparison_grid(
    autoencoder: GAFAutoencoder,
    real_dataset: DiffusionImageDataset,
    generated_dataset: GeneratedImageDataset,
    output_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    count = min(args.num_comparison_images, len(real_dataset), len(generated_dataset))
    with torch.no_grad():
        real = torch.stack([real_dataset[index] for index in range(count)]).to(device)
        recon = autoencoder(real).cpu()
    generated = torch.stack([generated_dataset[index] for index in range(count)])
    rows = torch.stack([real.cpu(), recon, generated], dim=1).flatten(0, 1)
    save_image(make_grid(denormalize(rows), nrow=3), output_path)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory of real GAF images.")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory of decoded latent-DDPM samples.")
    parser.add_argument(
        "--autoencoder-checkpoint",
        type=Path,
        required=True,
        help="Autoencoder checkpoint-final directory containing autoencoder/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and comparison images.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--num-comparison-images", type=int, default=8, help="Rows in side-by-side comparison.")
    parser.add_argument("--max-metric-pairs", type=int, default=1000, help="Max sorted generated/real pairs for MSE/MAE.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

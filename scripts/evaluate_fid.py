"""Compute FID between real and generated image directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    def __init__(self, image_dir: Path, transform: transforms.Compose) -> None:
        self.image_dir = image_dir
        self.paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No images found in {image_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = build_inception()
        model.fc = nn.Identity()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)


def build_inception() -> nn.Module:
    try:
        from torchvision.models import Inception_V3_Weights

        return inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    except ImportError:
        return inception_v3(pretrained=True, transform_input=False)


def main() -> None:
    args = parse_args()
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    real_dataset = FlatImageDataset(args.real_dir, transform)
    generated_dataset = FlatImageDataset(args.generated_dir, transform)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    generated_loader = DataLoader(generated_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = InceptionFeatureExtractor().to(device)
    real_features = get_features(real_loader, model, device)
    generated_features = get_features(generated_loader, model, device)
    mu_real, sigma_real = compute_stats(real_features)
    mu_generated, sigma_generated = compute_stats(generated_features)
    fid = compute_fid(mu_real, sigma_real, mu_generated, sigma_generated)

    payload = {
        "fid": float(fid),
        "real_dir": str(args.real_dir),
        "generated_dir": str(args.generated_dir),
        "num_real": len(real_dataset),
        "num_generated": len(generated_dataset),
    }
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"FID: {fid:.6f}")


def get_features(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    features = []
    for images in loader:
        images = images.to(device)
        features.append(model(images).detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True, help="Directory of real images.")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory of generated images.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write FID metrics JSON.")
    parser.add_argument("--batch-size", type=int, default=32, help="FID feature extraction batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

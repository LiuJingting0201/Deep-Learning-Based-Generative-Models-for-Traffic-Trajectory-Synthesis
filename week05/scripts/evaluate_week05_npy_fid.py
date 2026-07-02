"""Evaluate Week 5 generated .npy samples with the repo FID method.

This mirrors scripts/evaluate_fid.py: Inception-v3 features, empirical
mean/covariance, and scipy.linalg.sqrtm. Inputs are float arrays instead of PNGs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK05_SCRIPTS_DIR = PROJECT_ROOT / "week05" / "scripts"
if str(WEEK05_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK05_SCRIPTS_DIR))

from train_week05_ddpm import Week05DeltaGASFDataset


class Week05RealFIDDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        image_size: int,
        normalization_file: Path,
        sigmoid_k: float,
        split: str,
    ) -> None:
        self.dataset = Week05DeltaGASFDataset(
            data_root=data_root,
            image_size=image_size,
            normalization_file=normalization_file,
            sigmoid_k=sigmoid_k,
            split=split,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.dataset[index]
        return resize_for_fid(image)


class NpyFIDDataset(Dataset):
    def __init__(self, npy_path: Path) -> None:
        self.npy_path = npy_path
        self.array = np.load(npy_path, mmap_mode="r")
        if self.array.ndim != 4:
            raise ValueError(f"Expected (N,H,W,C) or (N,C,H,W), got {self.array.shape} in {npy_path}")
        if 3 not in (self.array.shape[1], self.array.shape[-1]):
            raise ValueError(f"Expected a 3-channel array, got {self.array.shape} in {npy_path}")

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        image = torch.from_numpy(np.array(self.array[index], dtype=np.float32, copy=True))
        if image.ndim != 3:
            raise ValueError(f"Expected one generated sample to be 3D, got {tuple(image.shape)}")
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1).contiguous()
        image = torch.clamp(image, 0.0, 1.0) * 2.0 - 1.0
        return resize_for_fid(image)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = build_inception()
        model.fc = nn.Identity()
        self.model = model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    samples_root = args.samples_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = output_dir / "feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = InceptionFeatureExtractor().to(device)
    generated_paths = sorted(samples_root.glob("week05_*/*.npy"))
    if not generated_paths:
        raise ValueError(f"No generated .npy files found under {samples_root}")

    real_stats_by_sigmoid: dict[float, tuple[np.ndarray, np.ndarray, int, Path]] = {}
    rows = []
    for generated_path in generated_paths:
        config_name, step, num_samples = parse_generated_name(generated_path)
        sigmoid_k = infer_sigmoid_k(config_name)
        if sigmoid_k not in real_stats_by_sigmoid:
            real_stats_by_sigmoid[sigmoid_k] = compute_or_load_real_stats(
                data_root=data_root,
                image_size=args.image_size,
                split=args.real_split,
                sigmoid_k=sigmoid_k,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                model=model,
                feature_cache_dir=feature_cache_dir,
                overwrite=args.overwrite_features,
            )
        mu_real, sigma_real, num_real, real_stats_path = real_stats_by_sigmoid[sigmoid_k]

        started = time.perf_counter()
        generated_dataset = NpyFIDDataset(generated_path)
        generated_loader = DataLoader(
            generated_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        generated_features = get_features(generated_loader, model, device)
        mu_generated, sigma_generated = compute_stats(generated_features)
        fid = compute_fid(mu_real, sigma_real, mu_generated, sigma_generated)
        elapsed = time.perf_counter() - started

        payload = {
            "fid": float(fid),
            "config": config_name,
            "step": step,
            "sigmoid_k": sigmoid_k,
            "real_split": args.real_split,
            "real_data_root": str(data_root),
            "generated_path": str(generated_path),
            "num_real": num_real,
            "num_generated": len(generated_dataset),
            "generated_name_num_samples": num_samples,
            "fid_method_reference": "scripts/evaluate_fid.py",
            "inception_input_range": "[-1, 1]",
            "image_size_before_resize": args.image_size,
            "image_size_for_inception": 299,
            "real_stats_path": str(real_stats_path),
            "elapsed_seconds": elapsed,
        }
        per_file_json = output_dir / config_name / f"{generated_path.stem}_fid.json"
        per_file_json.parent.mkdir(parents=True, exist_ok=True)
        per_file_json.write_text(json.dumps(payload, indent=2) + "\n")
        rows.append(payload)
        print(f"{config_name} step={step}: FID={fid:.6f}", flush=True)

    rows = sorted(rows, key=lambda row: (row["config"], row["step"]))
    write_summary(rows, output_dir)


def compute_or_load_real_stats(
    data_root: Path,
    image_size: int,
    split: str,
    sigmoid_k: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: nn.Module,
    feature_cache_dir: Path,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, int, Path]:
    key = f"real_{split}_sigmoid_{sigmoid_k:g}_inception_stats.npz"
    stats_path = feature_cache_dir / key
    if stats_path.exists() and not overwrite:
        cached = np.load(stats_path)
        return cached["mu"], cached["sigma"], int(cached["num_real"]), stats_path

    normalization_file = data_root / "normalization_diagnostics.json"
    real_dataset = Week05RealFIDDataset(
        data_root=data_root,
        image_size=image_size,
        normalization_file=normalization_file,
        sigmoid_k=sigmoid_k,
        split=split,
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"extracting real features split={split} sigmoid_k={sigmoid_k:g}", flush=True)
    features = get_features(real_loader, model, device)
    mu, sigma = compute_stats(features)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma, num_real=np.array(len(real_dataset), dtype=np.int64))
    return mu, sigma, len(real_dataset), stats_path


def resize_for_fid(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    if image.shape[-2:] == (299, 299):
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def get_features(loader: DataLoader, model: nn.Module, device: torch.device) -> np.ndarray:
    features = []
    for images in loader:
        images = images.to(device, non_blocking=True)
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


def parse_generated_name(path: Path) -> tuple[str, int, int | None]:
    match = re.fullmatch(r"(.+)_step_(\d+)_samples_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Generated file name does not match expected pattern: {path.name}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def infer_sigmoid_k(config_name: str) -> float:
    if "sig10" in config_name:
        return 1.0
    if "sig15" in config_name:
        return 1.5
    raise ValueError(f"Cannot infer sigmoid_k from config name: {config_name}")


def build_inception() -> nn.Module:
    try:
        from torchvision.models import Inception_V3_Weights

        return inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    except ImportError:
        return inception_v3(pretrained=True, transform_input=False)


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary_json = output_dir / "week05_fid_summary.json"
    summary_csv = output_dir / "week05_fid_summary.csv"
    summary_json.write_text(json.dumps({"results": rows}, indent=2) + "\n")
    fieldnames = [
        "config",
        "step",
        "fid",
        "sigmoid_k",
        "real_split",
        "num_real",
        "num_generated",
        "generated_path",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"wrote {summary_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224",
    )
    parser.add_argument("--samples-root", type=Path, default=PROJECT_ROOT / "week05" / "results_npy_samples")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "week05" / "fid_results")
    parser.add_argument("--real-split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

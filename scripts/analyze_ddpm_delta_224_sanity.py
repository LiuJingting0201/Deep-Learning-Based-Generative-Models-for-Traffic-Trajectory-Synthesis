"""Analyze the 224x224 delta-displacement DDPM sanity experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_fid import FlatImageDataset, InceptionFeatureExtractor, compute_fid, compute_stats, get_features
from run_resnet18_decoder_ablation import LabelScaler, ResNet18TrajectoryDecoder


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CHANNELS = ("R", "G", "B")


class GeneratedImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous(), path.stem


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    diagnostics_dir = output_dir / "image_diagnostics"
    novelty_dir = output_dir / "novelty_diagnostics"
    grids_dir = output_dir / "visual_grids"
    decoded_dir = output_dir / "decoded_generated_sanity"
    for path in [diagnostics_dir, novelty_dir, novelty_dir / "nearest_neighbor_grids", grids_dir, decoded_dir]:
        path.mkdir(parents=True, exist_ok=True)

    real_paths = image_paths(args.real_dir)
    generated_paths = image_paths(args.generated_dir)
    generated_by_checkpoint = {
        name: image_paths(path)
        for name, path in sorted(args.generated_by_checkpoint_dir.glob("checkpoint-*").items())
    } if False else {}
    if args.generated_by_checkpoint_dir.exists():
        generated_by_checkpoint = {
            path.name: image_paths(path)
            for path in sorted(args.generated_by_checkpoint_dir.glob("checkpoint-*"))
            if path.is_dir()
        }

    image_validation = validate_images(generated_paths, expected_size=(224, 224))
    for name, paths in generated_by_checkpoint.items():
        image_validation[f"generated_by_checkpoint/{name}"] = validate_images(paths, expected_size=(224, 224))
    write_json(diagnostics_dir / "generated_image_validation.json", image_validation)

    training_summary = summarize_training(output_dir)
    write_json(output_dir / "training_run_summary.json", training_summary)

    make_image_grid(real_paths, grids_dir / "real_grid.png", title="Real delta-displacement images")
    make_image_grid(generated_paths, grids_dir / "generated_final_grid.png", title="Generated final checkpoint")
    for name, paths in generated_by_checkpoint.items():
        suffix = name.replace("checkpoint-", "")
        make_image_grid(paths, grids_dir / f"generated_checkpoint_{suffix}_grid.png", title=name)

    fid_payload, kid_payload, by_checkpoint_rows = compute_feature_metrics(args, real_paths, generated_paths, generated_by_checkpoint)
    write_json(output_dir / "fid_result.json", fid_payload)
    write_json(output_dir / "kid_result.json", kid_payload)
    pd.DataFrame(by_checkpoint_rows).to_csv(output_dir / "fid_by_checkpoint.csv", index=False)
    write_json(output_dir / "fid_by_checkpoint.json", by_checkpoint_rows)
    pd.DataFrame(by_checkpoint_rows).to_csv(output_dir / "kid_by_checkpoint.csv", index=False)
    write_json(output_dir / "kid_by_checkpoint.json", by_checkpoint_rows)

    channel_summary, histogram_rows, histogram_distances = channel_diagnostics(real_paths, generated_paths)
    write_json(diagnostics_dir / "channel_stats.json", channel_summary)
    pd.DataFrame(histogram_rows).to_csv(diagnostics_dir / "channel_histograms.csv", index=False)
    write_json(diagnostics_dir / "channel_histogram_distances.json", histogram_distances)

    diversity_summary = image_diversity_diagnostics(real_paths, generated_paths, args.seed)
    write_json(diagnostics_dir / "image_diversity_diagnostics.json", diversity_summary)
    structure_summary = structure_diagnostics(real_paths, generated_paths)
    write_json(diagnostics_dir / "structure_summary.json", structure_summary)

    novelty_summary = novelty_diagnostics(args, generated_paths, novelty_dir)
    decoded_summary = decode_and_analyze(args, generated_paths, decoded_dir)

    summary = collect_summary(training_summary, fid_payload, kid_payload, channel_summary, diversity_summary, structure_summary, novelty_summary, decoded_summary)
    write_json(output_dir / "summary_metrics.json", summary)
    pd.DataFrame([flatten_dict(summary)]).to_csv(output_dir / "summary_metrics.csv", index=False)
    write_report(output_dir / "ddpm_delta_displacement_224_sanity_report_zh.txt", args, summary, by_checkpoint_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def image_paths(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image01(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def validate_images(paths: list[Path], expected_size: tuple[int, int]) -> dict[str, Any]:
    bad: list[dict[str, str]] = []
    modes: dict[str, int] = {}
    sizes: dict[str, int] = {}
    min_value = 255
    max_value = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                modes[image.mode] = modes.get(image.mode, 0) + 1
                sizes[str(image.size)] = sizes.get(str(image.size), 0) + 1
                if image.mode != "RGB" or image.size != expected_size:
                    bad.append({"path": str(path), "reason": f"{image.mode}:{image.size}"})
                arr = np.asarray(image.convert("RGB"))
                min_value = min(min_value, int(arr.min()))
                max_value = max(max_value, int(arr.max()))
        except Exception as exc:  # noqa: BLE001
            bad.append({"path": str(path), "reason": repr(exc)})
    return {
        "count": len(paths),
        "modes": modes,
        "sizes": sizes,
        "min_pixel": int(min_value) if paths else None,
        "max_pixel": int(max_value) if paths else None,
        "bad_count": len(bad),
        "bad_examples": bad[:20],
    }


def summarize_training(output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / "train_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()] if log_path.exists() else []
    checkpoints = sorted(p.name for p in output_dir.glob("checkpoint-*") if p.is_dir())
    sample_count = len(image_paths(output_dir / "samples")) if (output_dir / "samples").exists() else 0
    final = rows[-1] if rows else {}
    return {
        "log_rows": len(rows),
        "final_step": final.get("step"),
        "final_loss": final.get("loss"),
        "runtime_seconds": final.get("elapsed_seconds"),
        "average_steps_per_second": final.get("steps_per_second"),
        "device": final.get("device"),
        "checkpoints": checkpoints,
        "checkpoint_status": {name: (output_dir / name / "unet").exists() and (output_dir / name / "scheduler").exists() for name in checkpoints},
        "periodic_sample_count": sample_count,
    }


def make_image_grid(paths: list[Path], output_path: Path, title: str, max_images: int = 36) -> None:
    selected = paths[:max_images]
    if not selected:
        return
    cols = 6
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, path in zip(axes_arr.ravel(), selected):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(path.stem, fontsize=6)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_feature_metrics(args: argparse.Namespace, real_paths: list[Path], generated_paths: list[Path], by_checkpoint: dict[str, list[Path]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        model = InceptionFeatureExtractor().to(device)
        real_features = features_for_dir(args.real_dir, transform, model, device, args.fid_batch_size)
        generated_features = features_for_dir(args.generated_dir, transform, model, device, args.fid_batch_size)
        fid = fid_from_features(real_features, generated_features)
        kid_mean, kid_std = compute_kid(real_features, generated_features, seed=args.seed)
        fid_payload = {
            "fid": fid,
            "real_dir": str(args.real_dir),
            "generated_dir": str(args.generated_dir),
            "num_real": len(real_paths),
            "num_generated": len(generated_paths),
            "preprocessing": "existing evaluate_fid.py convention: RGB, resize 299x299, normalize mean/std 0.5",
        }
        kid_payload = {"kid_mean": kid_mean, "kid_std": kid_std, "status": "computed", "num_real": len(real_paths), "num_generated": len(generated_paths)}
        rows = [{"checkpoint": "final", "fid": fid, "kid_mean": kid_mean, "kid_std": kid_std, "num_generated": len(generated_paths)}]
        for name, paths in by_checkpoint.items():
            if not paths:
                continue
            feature_dir = args.generated_by_checkpoint_dir / name
            gen_features = features_for_dir(feature_dir, transform, model, device, args.fid_batch_size)
            ckpt_fid = fid_from_features(real_features, gen_features)
            ckpt_kid_mean, ckpt_kid_std = compute_kid(real_features, gen_features, seed=args.seed)
            rows.append({"checkpoint": name, "fid": ckpt_fid, "kid_mean": ckpt_kid_mean, "kid_std": ckpt_kid_std, "num_generated": len(paths)})
        return fid_payload, kid_payload, rows
    except Exception as exc:  # noqa: BLE001
        skipped = {"status": "skipped", "reason": repr(exc)}
        return skipped, skipped, []


def features_for_dir(directory: Path, transform: transforms.Compose, model: torch.nn.Module, device: torch.device, batch_size: int) -> np.ndarray:
    dataset = FlatImageDataset(directory, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return get_features(loader, model, device)


def fid_from_features(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    mu_real, sigma_real = compute_stats(real_features)
    mu_generated, sigma_generated = compute_stats(generated_features)
    return compute_fid(mu_real, sigma_real, mu_generated, sigma_generated)


def compute_kid(real_features: np.ndarray, generated_features: np.ndarray, seed: int, subsets: int = 50, subset_size: int = 100) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(real_features), len(generated_features))
    values = []
    for _ in range(subsets):
        x = real_features[rng.choice(len(real_features), m, replace=False)]
        y = generated_features[rng.choice(len(generated_features), m, replace=False)]
        values.append(polynomial_mmd2(x, y))
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def polynomial_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    k_xx = (x @ x.T / d + 1.0) ** 3
    k_yy = (y @ y.T / d + 1.0) ** 3
    k_xy = (x @ y.T / d + 1.0) ** 3
    m = x.shape[0]
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    return float(k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean())


def channel_diagnostics(real_paths: list[Path], generated_paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    real_stats, real_hists = channel_stats(real_paths)
    gen_stats, gen_hists = channel_stats(generated_paths)
    rows = []
    distances = {}
    for ci, channel in enumerate(CHANNELS):
        for bin_index, (r, g) in enumerate(zip(real_hists[ci], gen_hists[ci])):
            rows.append({"channel": channel, "bin": bin_index, "real_density": float(r), "generated_density": float(g)})
        distances[channel] = {
            "l1": float(np.abs(real_hists[ci] - gen_hists[ci]).sum()),
            "l2": float(np.linalg.norm(real_hists[ci] - gen_hists[ci])),
        }
    summary = {"real": real_stats, "generated": gen_stats, "mean_abs_diff": float(np.mean(np.abs(np.array(real_stats["mean"]) - np.array(gen_stats["mean"]))))}
    return summary, rows, distances


def channel_stats(paths: list[Path]) -> tuple[dict[str, Any], np.ndarray]:
    sums = np.zeros(3)
    sums2 = np.zeros(3)
    mins = np.ones(3)
    maxs = np.zeros(3)
    hist = np.zeros((3, 256), dtype=np.float64)
    count = 0
    for path in paths:
        arr = read_image01(path)
        flat = arr.reshape(-1, 3)
        sums += flat.sum(axis=0)
        sums2 += (flat ** 2).sum(axis=0)
        mins = np.minimum(mins, flat.min(axis=0))
        maxs = np.maximum(maxs, flat.max(axis=0))
        pixels = np.asarray(Image.open(path).convert("RGB"))
        for ci in range(3):
            hist[ci] += np.bincount(pixels[:, :, ci].reshape(-1), minlength=256)
        count += flat.shape[0]
    mean = sums / max(1, count)
    std = np.sqrt(np.maximum(sums2 / max(1, count) - mean ** 2, 0))
    hist = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1)
    return {"mean": mean.tolist(), "std": std.tolist(), "min": mins.tolist(), "max": maxs.tolist(), "num_images": len(paths)}, hist


def image_diversity_diagnostics(real_paths: list[Path], generated_paths: list[Path], seed: int) -> dict[str, Any]:
    real = load_subset_flat(real_paths, seed)
    gen = load_subset_flat(generated_paths, seed + 1)
    real_l2 = average_pairwise_l2(real)
    gen_l2 = average_pairwise_l2(gen)
    real_var = float(np.var(real))
    gen_var = float(np.var(gen))
    return {
        "real_average_pairwise_l2": real_l2,
        "generated_average_pairwise_l2": gen_l2,
        "pairwise_l2_ratio_generated_over_real": gen_l2 / real_l2 if real_l2 else None,
        "real_pixel_variance": real_var,
        "generated_pixel_variance": gen_var,
        "pixel_variance_ratio_generated_over_real": gen_var / real_var if real_var else None,
    }


def load_subset_flat(paths: list[Path], seed: int, max_count: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = paths if len(paths) <= max_count else [paths[i] for i in rng.choice(len(paths), max_count, replace=False)]
    return np.stack([read_image01(p).reshape(-1) for p in selected])


def average_pairwise_l2(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    distances = []
    for i in range(len(values)):
        diff = values[i + 1:] - values[i]
        if len(diff):
            distances.extend(np.sqrt(np.mean(diff ** 2, axis=1)).tolist())
    return float(np.mean(distances))


def structure_diagnostics(real_paths: list[Path], generated_paths: list[Path]) -> dict[str, Any]:
    return {"real": structure_stats(real_paths), "generated": structure_stats(generated_paths)}


def structure_stats(paths: list[Path]) -> dict[str, Any]:
    symmetry = []
    diag_mean = []
    diag_std = []
    corr = []
    for path in paths:
        arr = read_image01(path)
        symmetry.append(np.mean(np.abs(arr - np.transpose(arr, (1, 0, 2))), axis=(0, 1)))
        diag = np.stack([np.diag(arr[:, :, ci]) for ci in range(3)], axis=1)
        diag_mean.append(diag.mean(axis=0))
        diag_std.append(diag.std(axis=0))
        flat = arr.reshape(-1, 3)
        corr.append(channel_corr(flat))
    return {
        "symmetry_error_channel_mean": np.mean(symmetry, axis=0).tolist(),
        "symmetry_error_overall_mean": float(np.mean(symmetry)),
        "diagonal_mean_channel_mean": np.mean(diag_mean, axis=0).tolist(),
        "diagonal_std_channel_mean": np.mean(diag_std, axis=0).tolist(),
        "channel_correlation_mean": dict(zip(["R-G", "R-B", "G-B"], np.nanmean(corr, axis=0).tolist())),
        "num_images": len(paths),
    }


def channel_corr(flat: np.ndarray) -> np.ndarray:
    pairs = [(0, 1), (0, 2), (1, 2)]
    values = []
    for a, b in pairs:
        if np.std(flat[:, a]) < 1e-8 or np.std(flat[:, b]) < 1e-8:
            values.append(np.nan)
        else:
            values.append(float(np.corrcoef(flat[:, a], flat[:, b])[0, 1]))
    return np.asarray(values)


def novelty_diagnostics(args: argparse.Namespace, generated_paths: list[Path], novelty_dir: Path) -> dict[str, Any]:
    split = pd.read_csv(args.split_metadata)
    train_paths = [args.data_root / row.image_path for row in split[split["split"] == "train"].itertuples(index=False)]
    test_paths = [args.data_root / row.image_path for row in split[split["split"] == "test"].itertuples(index=False)]
    train_matrix = load_flat_matrix(train_paths)
    test_matrix = load_flat_matrix(test_paths)
    generated_matrix = load_flat_matrix(generated_paths)
    gen_train = nearest_rows(generated_matrix, train_matrix, generated_paths, train_paths)
    gen_test = nearest_rows(generated_matrix, test_matrix, generated_paths, test_paths)
    test_train = nearest_rows(test_matrix, train_matrix, test_paths, train_paths)
    pd.DataFrame(gen_train).to_csv(novelty_dir / "generated_to_train_nn.csv", index=False)
    pd.DataFrame(gen_test).to_csv(novelty_dir / "generated_to_test_nn.csv", index=False)
    pd.DataFrame(test_train).to_csv(novelty_dir / "test_to_train_reference.csv", index=False)
    threshold = float(np.quantile([r["l2"] for r in test_train], 0.01))
    near_count = int(sum(r["l2"] <= threshold for r in gen_train))
    make_nn_grids(gen_train, gen_test, novelty_dir / "nearest_neighbor_grids")
    summary = {
        "train_count": len(train_paths),
        "test_count": len(test_paths),
        "generated_count": len(generated_paths),
        "test_to_train_l2_p01_threshold": threshold,
        "generated_to_train_l2_min": float(min(r["l2"] for r in gen_train)),
        "generated_to_train_l2_mean": float(np.mean([r["l2"] for r in gen_train])),
        "generated_near_duplicate_count": near_count,
        "generated_near_duplicate_rate": near_count / max(1, len(gen_train)),
        "interpretation_note": "Close nearest neighbors are warnings, not proof of memorization.",
    }
    write_json(novelty_dir / "nearest_neighbor_summary.json", summary)
    return summary


def load_flat_matrix(paths: list[Path], size: int = 56) -> np.ndarray:
    arrays = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arrays.append((np.asarray(image, dtype=np.float32) / 255.0).reshape(-1))
    return np.stack(arrays).astype(np.float32)


def nearest_rows(query: np.ndarray, reference: np.ndarray, query_paths: list[Path], ref_paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for i, item in enumerate(query):
        best_l2 = float("inf")
        best_l1 = float("inf")
        best_idx = 0
        for start in range(0, len(reference), 256):
            chunk = reference[start:start + 256]
            diff = chunk - item
            l2 = np.sqrt(np.mean(diff ** 2, axis=1))
            local = int(np.argmin(l2))
            if float(l2[local]) < best_l2:
                best_l2 = float(l2[local])
                best_l1 = float(np.mean(np.abs(diff[local])))
                best_idx = start + local
        rows.append({"query_path": str(query_paths[i]), "nearest_path": str(ref_paths[best_idx]), "l2": best_l2, "l1": best_l1})
    return rows


def make_nn_grids(gen_train: list[dict[str, Any]], gen_test: list[dict[str, Any]], output_dir: Path, max_count: int = 12) -> None:
    for index in range(min(max_count, len(gen_train), len(gen_test))):
        train_row = gen_train[index]
        test_row = gen_test[index]
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        panels = [
            (train_row["query_path"], "generated"),
            (train_row["nearest_path"], f"nearest train\nL2={train_row['l2']:.4f}"),
            (test_row["nearest_path"], f"nearest test\nL2={test_row['l2']:.4f}"),
        ]
        for ax, (path, title) in zip(axes, panels):
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"nn_grid_{index:03d}.png", dpi=160)
        plt.close(fig)


def decode_and_analyze(args: argparse.Namespace, generated_paths: list[Path], decoded_dir: Path) -> dict[str, Any]:
    pred_delta_dir = decoded_dir / "predictions_delta_displacement"
    pred_abs_dir = decoded_dir / "predictions_absolute_integrated"
    metrics_dir = decoded_dir / "metrics"
    plots_dir = decoded_dir / "plots"
    for directory in [pred_delta_dir, pred_abs_dir, metrics_dir, plots_dir, plots_dir / "individual_plots"]:
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_data = json.loads(args.decoder_scaler.read_text())
    scaler = LabelScaler(mean=scaler_data["mean"], std=scaler_data["std"])
    model = ResNet18TrajectoryDecoder(dropout=args.decoder_dropout).to(device)
    checkpoint = torch.load(args.decoder_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(GeneratedImageDataset(generated_paths), batch_size=args.decoder_batch_size, shuffle=False, num_workers=0)
    split = pd.read_csv(args.split_metadata)
    train_rows = split[split["split"] == "train"].copy().reset_index(drop=True)
    sampled = train_rows.sample(n=len(generated_paths), replace=True, random_state=args.seed).reset_index(drop=True)
    assignments = []
    pred_deltas = []
    pred_abs = []
    with torch.no_grad():
        offset = 0
        for images, stems in loader:
            pred_norm = model(images.to(device))
            batch_delta = scaler.inverse_transform_tensor(pred_norm).cpu().numpy().astype(np.float32)
            for j, stem in enumerate(stems):
                row = sampled.iloc[offset]
                start = np.load(args.data_root / row.start_path).astype(np.float32)
                delta = batch_delta[j]
                absolute = integrate_delta(start, delta)
                np.save(pred_delta_dir / f"{stem}_pred_delta.npy", delta)
                np.save(pred_abs_dir / f"{stem}_pred_absolute_integrated.npy", absolute)
                assignments.append({"generated_image": str(generated_paths[offset]), "sampled_start_path": str(args.data_root / row.start_path), "source_sample_id": str(row.sample_id), "start_x": float(start[0]), "start_y": float(start[1])})
                pred_deltas.append(delta)
                pred_abs.append(absolute)
                offset += 1
    pd.DataFrame(assignments).to_csv(decoded_dir / "generated_start_assignments.csv", index=False)
    pred_deltas_arr = np.stack(pred_deltas)
    pred_abs_arr = np.stack(pred_abs)
    real_delta = load_label_stack(args.data_root, split, "label_delta_displacement_path")
    real_abs = load_label_stack(args.data_root, split, "label_absolute_path")
    delta_summary = delta_distribution(pred_deltas_arr, real_delta)
    integrated_summary = integrated_distribution(pred_abs_arr, real_abs)
    jump_summary = jump_statistics(pred_abs_arr, real_abs)
    smoothness_summary = smoothness_statistics(pred_abs_arr, real_abs)
    diversity_summary = trajectory_diversity(pred_abs_arr, real_abs)
    write_json(metrics_dir / "delta_distribution_summary.json", delta_summary)
    write_json(metrics_dir / "integrated_trajectory_distribution.json", integrated_summary)
    write_json(metrics_dir / "jump_statistics.json", jump_summary)
    write_json(metrics_dir / "smoothness_statistics.json", smoothness_summary)
    write_json(metrics_dir / "trajectory_diversity_diagnostics.json", diversity_summary)
    plot_decoded(pred_deltas_arr, pred_abs_arr, assignments, generated_paths, plots_dir)
    return {
        "delta_distribution": delta_summary,
        "integrated_distribution": integrated_summary,
        "jump_statistics": jump_summary,
        "smoothness_statistics": smoothness_summary,
        "trajectory_diversity": diversity_summary,
        "decoded_count": len(pred_deltas),
    }


def integrate_delta(start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    absolute = np.empty_like(delta, dtype=np.float32)
    absolute[0] = start.astype(np.float32)
    absolute[1:] = start.astype(np.float32) + np.cumsum(delta[1:], axis=0)
    return absolute


def load_label_stack(data_root: Path, split: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack([np.load(data_root / getattr(row, column)).astype(np.float32) for row in split.itertuples(index=False)])


def delta_distribution(pred: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    pred_mag = np.linalg.norm(pred, axis=2).reshape(-1)
    real_mag = np.linalg.norm(real, axis=2).reshape(-1)
    return {"generated": describe_values(pred_mag), "real_gt": describe_values(real_mag), "generated_near_zero_delta_ratio": float(np.mean(pred_mag <= 1e-3)), "real_near_zero_delta_ratio": float(np.mean(real_mag <= 1e-3)), "generated_delta_variance": float(np.var(pred)), "real_delta_variance": float(np.var(real))}


def integrated_distribution(pred: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    bounds_min = real.reshape(-1, 2).min(axis=0)
    bounds_max = real.reshape(-1, 2).max(axis=0)
    out_points = np.any((pred < bounds_min) | (pred > bounds_max), axis=2)
    lengths_pred = trajectory_lengths(pred)
    lengths_real = trajectory_lengths(real)
    boxes = bounding_boxes(pred)
    return {"generated_trajectory_length": describe_values(lengths_pred), "real_trajectory_length": describe_values(lengths_real), "generated_start_end_displacement": describe_values(np.linalg.norm(pred[:, -1] - pred[:, 0], axis=1)), "bbox_width": describe_values(boxes[:, 0]), "bbox_height": describe_values(boxes[:, 1]), "bbox_area": describe_values(boxes[:, 2]), "coordinate_min": pred.reshape(-1, 2).min(axis=0).tolist(), "coordinate_max": pred.reshape(-1, 2).max(axis=0).tolist(), "training_coordinate_min": bounds_min.tolist(), "training_coordinate_max": bounds_max.tolist(), "out_of_range_point_ratio": float(out_points.mean()), "out_of_range_sample_ratio": float(out_points.any(axis=1).mean()), "trajectory_variance": float(np.var(pred))}


def jump_statistics(pred: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    pred_steps = step_sizes(pred)
    real_steps = step_sizes(real)
    p95 = float(np.quantile(real_steps, 0.95))
    p99 = float(np.quantile(real_steps, 0.99))
    return {"generated_step_size": describe_values(pred_steps), "real_step_size": describe_values(real_steps), "real_p95_threshold": p95, "real_p99_threshold": p99, "generated_jump_ratio_gt_real_p95": float(np.mean(pred_steps > p95)), "generated_jump_ratio_gt_real_p99": float(np.mean(pred_steps > p99))}


def smoothness_statistics(pred: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    pred_heading = heading_changes(pred)
    real_heading = heading_changes(real)
    threshold = float(np.quantile(real_heading, 0.95))
    return {"generated_heading_change": describe_values(pred_heading), "real_heading_change": describe_values(real_heading), "sharp_turn_threshold_real_p95": threshold, "generated_sharp_turn_ratio": float(np.mean(pred_heading > threshold))}


def trajectory_diversity(pred: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    pred_var = float(np.var(pred))
    real_var = float(np.var(real))
    pred_len_var = float(np.var(trajectory_lengths(pred)))
    real_len_var = float(np.var(trajectory_lengths(real)))
    return {"generated_coordinate_variance": pred_var, "real_coordinate_variance": real_var, "coordinate_variance_ratio": pred_var / real_var if real_var else None, "generated_start_end_variance": float(np.var(pred[:, -1] - pred[:, 0])), "real_start_end_variance": float(np.var(real[:, -1] - real[:, 0])), "generated_trajectory_length_variance": pred_len_var, "real_trajectory_length_variance": real_len_var, "trajectory_length_variance_ratio": pred_len_var / real_len_var if real_len_var else None}


def trajectory_lengths(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(values, axis=1), axis=2).sum(axis=1)


def step_sizes(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(values, axis=1), axis=2).reshape(-1)


def bounding_boxes(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=1)
    maxs = values.max(axis=1)
    wh = maxs - mins
    return np.column_stack([wh[:, 0], wh[:, 1], wh[:, 0] * wh[:, 1]])


def heading_changes(values: np.ndarray) -> np.ndarray:
    steps = np.diff(values, axis=1)
    angles = np.arctan2(steps[:, :, 1], steps[:, :, 0])
    diff = np.diff(angles, axis=1)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff).reshape(-1)


def plot_decoded(pred_delta: np.ndarray, pred_abs: np.ndarray, assignments: list[dict[str, Any]], generated_paths: list[Path], plots_dir: Path) -> None:
    count = min(30, len(pred_delta))
    cols = 5
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for i, ax in enumerate(axes_arr.ravel()[:count]):
        traj = pred_abs[i]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2)
        ax.scatter(traj[0, 0], traj[0, 1], s=10)
        ax.set_title(f"{Path(assignments[i]['source_sample_id']).name}\nstart=({assignments[i]['start_x']:.1f},{assignments[i]['start_y']:.1f})", fontsize=7)
        ax.axis("equal")
    fig.tight_layout()
    fig.savefig(plots_dir / "decoded_generated_trajectories_grid.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2))
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for i, ax in enumerate(axes_arr.ravel()[:count]):
        ax.plot(np.linalg.norm(pred_delta[i], axis=1), linewidth=1.0)
        ax.set_title(Path(generated_paths[i]).stem, fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "decoded_generated_delta_magnitude_grid.png", dpi=180)
    plt.close(fig)

    for i in range(count):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(Image.open(generated_paths[i]).convert("RGB"))
        axes[0].set_title("generated image")
        axes[1].plot(np.linalg.norm(pred_delta[i], axis=1))
        axes[1].set_title("delta magnitude")
        axes[2].plot(pred_abs[i, :, 0], pred_abs[i, :, 1])
        axes[2].scatter(pred_abs[i, 0, 0], pred_abs[i, 0, 1], s=12)
        axes[2].axis("equal")
        axes[2].set_title("integrated trajectory")
        for ax in axes:
            ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(plots_dir / "individual_plots" / f"decoded_{i:03d}.png", dpi=160)
        plt.close(fig)


def describe_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def collect_summary(training: dict[str, Any], fid: dict[str, Any], kid: dict[str, Any], channel: dict[str, Any], diversity: dict[str, Any], structure: dict[str, Any], novelty: dict[str, Any], decoded: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": training,
        "fid": fid,
        "kid": kid,
        "channel_mean_abs_diff": channel.get("mean_abs_diff"),
        "generated_image_diversity_ratio": diversity.get("pairwise_l2_ratio_generated_over_real"),
        "generated_pixel_variance_ratio": diversity.get("pixel_variance_ratio_generated_over_real"),
        "real_symmetry_error": structure["real"]["symmetry_error_overall_mean"],
        "generated_symmetry_error": structure["generated"]["symmetry_error_overall_mean"],
        "real_diagonal_mean": structure["real"]["diagonal_mean_channel_mean"],
        "generated_diagonal_mean": structure["generated"]["diagonal_mean_channel_mean"],
        "novelty": novelty,
        "decoded": decoded,
    }


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value)
        else:
            out[name] = value
    return out


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any], by_checkpoint_rows: list[dict[str, Any]]) -> None:
    training = summary["training"]
    fid = summary["fid"]
    kid = summary["kid"]
    decoded = summary["decoded"]
    lines = [
        "224x224 delta-displacement DDPM sanity report",
        "",
        "1. 训练设置",
        f"- 数据目录: {args.real_dir}",
        f"- 输出目录: {args.output_dir}",
        "- 图像尺寸: 224x224 RGB, 未使用128x128生成后缩放。",
        "- 条件输入: 未使用 speed、map 或 conditional input。",
        f"- 最终 step: {training.get('final_step')}, final loss: {training.get('final_loss')}, runtime seconds: {training.get('runtime_seconds')}",
        f"- checkpoint: {training.get('checkpoints')}",
        "",
        "2. 图像层面质量",
        f"- FID: {fid.get('fid', fid.get('status'))}",
        f"- KID: mean={kid.get('kid_mean')}, std={kid.get('kid_std')}, status={kid.get('status', 'computed')}",
        f"- checkpoint 趋势: {by_checkpoint_rows}",
        "- 这些指标是自然图像 Inception 特征上的 sanity 参考，不能直接等同于轨迹质量。",
        "",
        "3. 结构一致性",
        f"- real symmetry error: {summary.get('real_symmetry_error')}",
        f"- generated symmetry error: {summary.get('generated_symmetry_error')}",
        f"- real diagonal mean: {summary.get('real_diagonal_mean')}",
        f"- generated diagonal mean: {summary.get('generated_diagonal_mean')}",
        f"- channel mean abs diff: {summary.get('channel_mean_abs_diff')}",
        "- 这些诊断用于检查 GAF/MTF-like 结构是否被粗略保留，而不是最终轨迹指标。",
        "",
        "4. 记忆化/新颖性",
        f"- generated-to-train nearest L2 mean: {summary['novelty'].get('generated_to_train_l2_mean')}",
        f"- near duplicate rate: {summary['novelty'].get('generated_near_duplicate_rate')}",
        "- 最近邻很近只能作为风险信号，不能单独证明模型记忆训练集。",
        "",
        "5. 解码后轨迹有效性",
        f"- decoded generated count: {decoded.get('decoded_count')}",
        f"- generated delta magnitude mean: {decoded['delta_distribution']['generated']['mean']}",
        f"- real delta magnitude mean: {decoded['delta_distribution']['real_gt']['mean']}",
        f"- generated trajectory length mean: {decoded['integrated_distribution']['generated_trajectory_length']['mean']}",
        f"- real trajectory length mean: {decoded['integrated_distribution']['real_trajectory_length']['mean']}",
        f"- out-of-range point ratio: {decoded['integrated_distribution']['out_of_range_point_ratio']}",
        f"- jump ratio > real p95: {decoded['jump_statistics']['generated_jump_ratio_gt_real_p95']}",
        f"- trajectory coordinate variance ratio: {decoded['trajectory_diversity']['coordinate_variance_ratio']}",
        "- 这些是 validity/distribution checks，不是 ADE/FDE，因为生成图像没有真实轨迹标签。",
        "",
        "6. 总体 sanity 结论",
        "- 该报告只对应 3000-step sanity run。是否适合更长 10k run 应结合 FID/KID 趋势、结构诊断、最近邻风险和解码轨迹是否崩塌来判断。",
        "- 不包含 map matching 或道路有效性指标；积分轨迹依赖 sampled real start point，因此只适合做局部分布合理性检查。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ddpm_delta_displacement_224_sanity"))
    parser.add_argument("--real-dir", type=Path, default=Path("data_no_speed_delta_displacement_paired/images"))
    parser.add_argument("--generated-dir", type=Path, default=Path("results/ddpm_delta_displacement_224_sanity/generated_100"))
    parser.add_argument("--generated-by-checkpoint-dir", type=Path, default=Path("results/ddpm_delta_displacement_224_sanity/generated_by_checkpoint"))
    parser.add_argument("--data-root", type=Path, default=Path("data_no_speed_delta_displacement_paired"))
    parser.add_argument("--split-metadata", type=Path, default=Path("data_no_speed_delta_displacement_paired/splits/split_metadata.csv"))
    parser.add_argument("--decoder-checkpoint", type=Path, default=Path("experiments/decoder_no_speed_delta_displacement_resnet18_ablation/checkpoints/best_model.pt"))
    parser.add_argument("--decoder-scaler", type=Path, default=Path("experiments/decoder_no_speed_delta_displacement_resnet18_ablation/label_scaler.json"))
    parser.add_argument("--decoder-dropout", type=float, default=0.3)
    parser.add_argument("--decoder-batch-size", type=int, default=16)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Build the Week 4 constrained GASF dataset with [-1, 1] dx/dy scaling.

This is a representation-distribution ablation. It is not analytically
invertible from the GASF diagonal alone because the diagonal loses sign.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


WEEK04_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_OUTPUT_ROOT = WEEK04_ROOT / "data_constrained_gasf_dxdy_start_neg11"
ORIGINAL_01_ROOT = WEEK04_ROOT / "data_constrained_gasf_dxdy_start"
REPRESENTATION = "GASF_dx_GASF_dy_start_heatmap_neg11"


@dataclass(frozen=True)
class LoadedSample:
    sample_id: str
    abs_xy: np.ndarray
    delta_xy: np.ndarray
    initial_point: np.ndarray
    adjusted_delta: bool


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    validate_input_root(input_root)
    output_dirs = make_output_dirs(output_root)

    selected_ids = select_sample_ids(input_root, args.split, args.max_samples)
    train_ids, normalization_source = normalization_sample_ids(input_root)
    stats, stats_skips = compute_training_stats(input_root, train_ids, args.image_size)
    write_normalization(output_root, input_root, stats, args.image_size, normalization_source)

    copy_dataset_sidecars(input_root, output_root)
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = list(stats_skips)
    processed: list[LoadedSample] = []

    for sample_id in selected_ids:
        sample, skip_reason = load_sample(input_root, sample_id, args.image_size)
        if sample is None:
            skipped_rows.append(skipped_row(sample_id, skip_reason))
            continue

        image_float, image_uint8 = encode_sample(sample, stats, args.image_size, args.heatmap_sigma)
        save_sample(output_dirs, sample, image_float, image_uint8)
        processed.append(sample)
        rows.append(metadata_row(output_root, sample, stats))

    write_metadata(output_root / "metadata.csv", rows)
    write_skipped_report(output_root / "skipped_samples.csv", skipped_rows)
    sanity_rows = run_representation_sanity(
        output_root=output_root,
        samples=processed[: args.num_sanity_checks],
        stats=stats,
    )
    distribution_summary = run_distribution_analysis(
        output_root=output_root,
        input_root=input_root,
        train_ids=train_ids,
        stats=stats,
        image_size=args.image_size,
        original_01_root=args.original_01_root.resolve(),
    )
    write_readme(output_root, args.image_size, args.heatmap_sigma)
    write_summary(
        output_root / "dataset_summary.json",
        input_root,
        output_root,
        args,
        selected_ids,
        processed,
        skipped_rows,
        sanity_rows,
        distribution_summary,
        normalization_source,
    )

    print("Dataset created.")
    print(f"Output root: {output_root}")
    print(f"Total samples: {len(processed)}")
    print("Normalization range: [-1, 1]")
    print(f"dx_min={stats['dx_min']:.12g}, dx_max={stats['dx_max']:.12g}")
    print(f"dy_min={stats['dy_min']:.12g}, dy_max={stats['dy_max']:.12g}")
    print("Distribution statistics:")
    for key in [
        "dx_mean",
        "dx_std",
        "dy_mean",
        "dy_std",
        "fraction_dx_abs_001",
        "fraction_dx_abs_002",
        "fraction_dx_abs_005",
        "fraction_dy_abs_001",
        "fraction_dy_abs_002",
        "fraction_dy_abs_005",
    ]:
        print(f"  {key}: {distribution_summary[key]:.12g}")
    print("Histogram paths:")
    print(f"  {output_root / 'distribution_analysis' / 'hist_dx_norm_neg11.png'}")
    print(f"  {output_root / 'distribution_analysis' / 'hist_dy_norm_neg11.png'}")
    print(
        "This version is NOT analytically invertible from GASF diagonal alone "
        "because sign information is lost."
    )
    print("It is intended only as a representation-distribution ablation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--original-01-root", type=Path, default=ORIGINAL_01_ROOT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument("--split", default="train", help="Split to process: train, val, test, or all.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-sanity-checks", type=int, default=100)
    return parser.parse_args()


def validate_input_root(input_root: Path) -> None:
    required = ["labels_absolute", "labels_delta_displacement", "metadata.csv"]
    missing = [name for name in required if not (input_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{input_root} is missing required entries: {missing}")


def make_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "images": output_root / "images",
        "arrays": output_root / "arrays",
        "labels_absolute": output_root / "labels_absolute",
        "labels_delta_displacement": output_root / "labels_delta_displacement",
        "starts": output_root / "starts",
        "splits": output_root / "splits",
        "sanity_checks": output_root / "sanity_checks",
        "distribution_analysis": output_root / "distribution_analysis",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def select_sample_ids(input_root: Path, split: str, max_samples: int | None) -> list[str]:
    split = split.lower()
    if split == "all":
        sample_ids = sorted(path.stem for path in (input_root / "labels_absolute").glob("*.npy"))
    else:
        sample_ids = read_split_ids(input_root, split)
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    return sample_ids


def normalization_sample_ids(input_root: Path) -> tuple[list[str], str]:
    try:
        return read_split_ids(input_root, "train"), "train_split"
    except FileNotFoundError:
        warnings.warn("No train split file found; using all samples for normalization.")
        return sorted(path.stem for path in (input_root / "labels_absolute").glob("*.npy")), "all_samples_warning_no_train_split"


def read_split_ids(input_root: Path, split: str) -> list[str]:
    split_path = input_root / "splits" / f"{split}_ids.csv"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise ValueError(f"{split_path} must contain a sample_id column")
        return [row["sample_id"] for row in reader if row.get("sample_id")]


def compute_training_stats(
    input_root: Path, sample_ids: list[str], image_size: int
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    dx_min = np.inf
    dx_max = -np.inf
    dy_min = np.inf
    dy_max = -np.inf
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    skipped: list[dict[str, Any]] = []
    valid_count = 0

    for sample_id in sample_ids:
        sample, skip_reason = load_sample(input_root, sample_id, image_size)
        if sample is None:
            skipped.append(skipped_row(sample_id, f"normalization_skip: {skip_reason}"))
            continue
        dx = sample.delta_xy[:, 0]
        dy = sample.delta_xy[:, 1]
        dx_min = min(dx_min, float(dx.min()))
        dx_max = max(dx_max, float(dx.max()))
        dy_min = min(dy_min, float(dy.min()))
        dy_max = max(dy_max, float(dy.max()))
        x_min = min(x_min, float(sample.abs_xy[:, 0].min()))
        x_max = max(x_max, float(sample.abs_xy[:, 0].max()))
        y_min = min(y_min, float(sample.abs_xy[:, 1].min()))
        y_max = max(y_max, float(sample.abs_xy[:, 1].max()))
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid samples found for normalization.")

    stats = {
        "dx_min": dx_min,
        "dx_max": dx_max,
        "dy_min": dy_min,
        "dy_max": dy_max,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    for low_key, high_key in [("dx_min", "dx_max"), ("dy_min", "dy_max"), ("x_min", "x_max"), ("y_min", "y_max")]:
        if stats[low_key] == stats[high_key]:
            raise ValueError(f"Cannot normalize with zero range: {low_key} == {high_key}")
    return stats, skipped


def load_sample(input_root: Path, sample_id: str, image_size: int) -> tuple[LoadedSample | None, str]:
    abs_path = input_root / "labels_absolute" / f"{sample_id}.npy"
    delta_path = input_root / "labels_delta_displacement" / f"{sample_id}.npy"
    if not abs_path.exists():
        return None, f"missing absolute label: {abs_path}"
    if not delta_path.exists():
        return None, f"missing delta label: {delta_path}"

    abs_xy = np.load(abs_path).astype(np.float64)
    delta_xy = np.load(delta_path).astype(np.float64)
    if abs_xy.shape != (image_size, 2):
        return None, f"absolute shape {abs_xy.shape} != ({image_size}, 2)"
    if delta_xy.shape == (image_size - 1, 2):
        delta_xy = np.vstack([np.zeros((1, 2), dtype=np.float64), delta_xy])
        adjusted_delta = True
    elif delta_xy.shape == (image_size, 2):
        adjusted_delta = False
    else:
        return None, f"delta shape {delta_xy.shape} is incompatible with image_size={image_size}"

    return LoadedSample(sample_id, abs_xy, delta_xy, abs_xy[0].astype(np.float64), adjusted_delta), ""


def encode_sample(
    sample: LoadedSample,
    stats: dict[str, float],
    image_size: int,
    heatmap_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx_norm = normalize_to_neg11(sample.delta_xy[:, 0], stats["dx_min"], stats["dx_max"])
    dy_norm = normalize_to_neg11(sample.delta_xy[:, 1], stats["dy_min"], stats["dy_max"])
    gasf_dx = gasf_encode_neg11(dx_norm)
    gasf_dy = gasf_encode_neg11(dy_norm)
    heatmap = make_start_heatmap(sample.initial_point, stats, image_size, heatmap_sigma)
    image_float = np.stack([(gasf_dx + 1.0) / 2.0, (gasf_dy + 1.0) / 2.0, heatmap], axis=-1)
    image_float = np.clip(image_float, 0.0, 1.0).astype(np.float32)
    image_uint8 = np.round(image_float * 255.0).astype(np.uint8)
    return image_float, image_uint8


def normalize_to_neg11(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    scaled = 2.0 * (values - minimum) / (maximum - minimum) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def normalize_to_01(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def gasf_encode_neg11(values_neg11: np.ndarray) -> np.ndarray:
    phi = np.arccos(np.clip(values_neg11, -1.0, 1.0))
    return np.cos(phi[:, None] + phi[None, :])


def decode_gasf_diagonal_abs(channel_01: np.ndarray) -> np.ndarray:
    gasf = channel_01 * 2.0 - 1.0
    diag = np.diag(gasf)
    return np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, 1.0))


def make_start_heatmap(
    initial_point: np.ndarray,
    stats: dict[str, float],
    image_size: int,
    sigma: float,
) -> np.ndarray:
    x0, y0 = initial_point
    px = (x0 - stats["x_min"]) / (stats["x_max"] - stats["x_min"]) * (image_size - 1)
    py = (y0 - stats["y_min"]) / (stats["y_max"] - stats["y_min"]) * (image_size - 1)
    px = float(np.clip(px, 0.0, image_size - 1))
    py = float(np.clip(py, 0.0, image_size - 1))
    yy, xx = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    heatmap = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma**2))
    maximum = float(heatmap.max())
    if maximum > 0.0:
        heatmap = heatmap / maximum
    return heatmap.astype(np.float64)


def save_sample(
    output_dirs: dict[str, Path],
    sample: LoadedSample,
    image_float: np.ndarray,
    image_uint8: np.ndarray,
) -> None:
    filename = f"{sample.sample_id}.npy"
    Image.fromarray(image_uint8, mode="RGB").save(output_dirs["images"] / f"{sample.sample_id}.png")
    np.save(output_dirs["arrays"] / filename, image_float)
    np.save(output_dirs["labels_absolute"] / filename, sample.abs_xy)
    np.save(output_dirs["labels_delta_displacement"] / filename, sample.delta_xy)
    np.save(output_dirs["starts"] / filename, sample.initial_point)


def copy_dataset_sidecars(input_root: Path, output_root: Path) -> None:
    metadata_path = input_root / "metadata.csv"
    if metadata_path.exists():
        shutil.copy2(metadata_path, output_root / "source_metadata.csv")
    splits_dir = input_root / "splits"
    if splits_dir.exists():
        for path in splits_dir.glob("*"):
            if path.is_file():
                shutil.copy2(path, output_root / "splits" / path.name)


def metadata_row(output_root: Path, sample: LoadedSample, stats: dict[str, float]) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": (output_root / "images" / f"{sample.sample_id}.png").relative_to(output_root).as_posix(),
        "array_path": (output_root / "arrays" / f"{sample.sample_id}.npy").relative_to(output_root).as_posix(),
        "label_delta_displacement_path": (
            output_root / "labels_delta_displacement" / f"{sample.sample_id}.npy"
        ).relative_to(output_root).as_posix(),
        "label_absolute_path": (
            output_root / "labels_absolute" / f"{sample.sample_id}.npy"
        ).relative_to(output_root).as_posix(),
        "start_path": (output_root / "starts" / f"{sample.sample_id}.npy").relative_to(output_root).as_posix(),
        "encoding_method": REPRESENTATION,
        "coordinate_type": "delta_displacement_xy",
        "normalization_range": "[-1,1]",
        "sign_lost_from_gasf_diagonal": True,
        "initial_x": float(sample.initial_point[0]),
        "initial_y": float(sample.initial_point[1]),
        "delta_was_prepended_with_zero": sample.adjusted_delta,
        **stats,
        "valid_or_skipped": "valid",
        "skip_reason": "",
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "image_path",
        "array_path",
        "label_delta_displacement_path",
        "label_absolute_path",
        "start_path",
        "encoding_method",
        "coordinate_type",
        "normalization_range",
        "sign_lost_from_gasf_diagonal",
        "initial_x",
        "initial_y",
        "delta_was_prepended_with_zero",
        "dx_min",
        "dx_max",
        "dy_min",
        "dy_max",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "valid_or_skipped",
        "skip_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def skipped_row(sample_id: str, reason: str) -> dict[str, Any]:
    return {"sample_id": sample_id, "status": "skipped", "reason": reason}


def write_skipped_report(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def write_normalization(
    output_root: Path,
    input_root: Path,
    stats: dict[str, float],
    image_size: int,
    normalization_source: str,
) -> None:
    payload: dict[str, Any] = {
        "representation": REPRESENTATION,
        "image_size": image_size,
        **stats,
        "normalization_range": "[-1,1]",
        "normalization_formula": "2 * (value - min) / (max - min) - 1",
        "gasf_input_range": "[-1,1]",
        "sign_lost_from_gasf_diagonal": True,
        "analytical_inverse_note": (
            "GASF diagonal gives abs(x)=sqrt((diag+1)/2), not signed x. "
            "This dataset is not exactly invertible from the diagonal alone."
        ),
        "normalization_source": normalization_source,
        "input_root": str(input_root),
        "output_root": str(output_root),
    }
    with (output_root / "constrained_gasf_normalization.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_representation_sanity(
    output_root: Path,
    samples: list[LoadedSample],
    stats: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = output_root / "sanity_checks" / "neg11_representation_stats.csv"
    for sample in samples:
        dx_norm = normalize_to_neg11(sample.delta_xy[:, 0], stats["dx_min"], stats["dx_max"])
        dy_norm = normalize_to_neg11(sample.delta_xy[:, 1], stats["dy_min"], stats["dy_max"])
        array = np.load(output_root / "arrays" / f"{sample.sample_id}.npy").astype(np.float64)
        abs_dx_rec = decode_gasf_diagonal_abs(array[:, :, 0])
        abs_dy_rec = decode_gasf_diagonal_abs(array[:, :, 1])
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dx_norm_mean": float(np.mean(dx_norm)),
                "dx_norm_std": float(np.std(dx_norm)),
                "dy_norm_mean": float(np.mean(dy_norm)),
                "dy_norm_std": float(np.std(dy_norm)),
                "fraction_dx_abs_001": float(np.mean(np.abs(dx_norm) <= 0.01)),
                "fraction_dx_abs_002": float(np.mean(np.abs(dx_norm) <= 0.02)),
                "fraction_dx_abs_005": float(np.mean(np.abs(dx_norm) <= 0.05)),
                "fraction_dy_abs_001": float(np.mean(np.abs(dy_norm) <= 0.01)),
                "fraction_dy_abs_002": float(np.mean(np.abs(dy_norm) <= 0.02)),
                "fraction_dy_abs_005": float(np.mean(np.abs(dy_norm) <= 0.05)),
                "abs_dx_diag_mae": float(np.mean(np.abs(abs_dx_rec - np.abs(dx_norm)))),
                "abs_dy_diag_mae": float(np.mean(np.abs(abs_dy_rec - np.abs(dy_norm)))),
                "sign_lost": True,
                "status": "sign_lost_not_trajectory_reconstructable",
            }
        )

    columns = [
        "sample_id",
        "dx_norm_mean",
        "dx_norm_std",
        "dy_norm_mean",
        "dy_norm_std",
        "fraction_dx_abs_001",
        "fraction_dx_abs_002",
        "fraction_dx_abs_005",
        "fraction_dy_abs_001",
        "fraction_dy_abs_002",
        "fraction_dy_abs_005",
        "abs_dx_diag_mae",
        "abs_dy_diag_mae",
        "sign_lost",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_distribution_analysis(
    output_root: Path,
    input_root: Path,
    train_ids: list[str],
    stats: dict[str, float],
    image_size: int,
    original_01_root: Path,
) -> dict[str, Any]:
    dx_values, dy_values = load_delta_values(input_root, train_ids, image_size)
    dx_norm = normalize_to_neg11(dx_values, stats["dx_min"], stats["dx_max"])
    dy_norm = normalize_to_neg11(dy_values, stats["dy_min"], stats["dy_max"])
    dx_norm_01 = normalize_to_01(dx_values, stats["dx_min"], stats["dx_max"])
    dy_norm_01 = normalize_to_01(dy_values, stats["dy_min"], stats["dy_max"])

    summary = {
        "representation": REPRESENTATION,
        "normalization_range": "[-1,1]",
        "sign_lost_from_gasf_diagonal": True,
        "num_train_values": int(len(dx_values)),
        "dx_mean": float(np.mean(dx_norm)),
        "dx_std": float(np.std(dx_norm)),
        "dy_mean": float(np.mean(dy_norm)),
        "dy_std": float(np.std(dy_norm)),
        "fraction_dx_abs_001": float(np.mean(np.abs(dx_norm) <= 0.01)),
        "fraction_dx_abs_002": float(np.mean(np.abs(dx_norm) <= 0.02)),
        "fraction_dx_abs_005": float(np.mean(np.abs(dx_norm) <= 0.05)),
        "fraction_dy_abs_001": float(np.mean(np.abs(dy_norm) <= 0.01)),
        "fraction_dy_abs_002": float(np.mean(np.abs(dy_norm) <= 0.02)),
        "fraction_dy_abs_005": float(np.mean(np.abs(dy_norm) <= 0.05)),
        "original_01_comparison": {
            "original_01_root": str(original_01_root),
            "dx_norm_01_mean": float(np.mean(dx_norm_01)),
            "dx_norm_01_std": float(np.std(dx_norm_01)),
            "dy_norm_01_mean": float(np.mean(dy_norm_01)),
            "dy_norm_01_std": float(np.std(dy_norm_01)),
            "fraction_dx_norm_01_within_0p01_of_0p5": float(np.mean(np.abs(dx_norm_01 - 0.5) <= 0.01)),
            "fraction_dx_norm_01_within_0p02_of_0p5": float(np.mean(np.abs(dx_norm_01 - 0.5) <= 0.02)),
            "fraction_dx_norm_01_within_0p05_of_0p5": float(np.mean(np.abs(dx_norm_01 - 0.5) <= 0.05)),
            "fraction_dy_norm_01_within_0p01_of_0p5": float(np.mean(np.abs(dy_norm_01 - 0.5) <= 0.01)),
            "fraction_dy_norm_01_within_0p02_of_0p5": float(np.mean(np.abs(dy_norm_01 - 0.5) <= 0.02)),
            "fraction_dy_norm_01_within_0p05_of_0p5": float(np.mean(np.abs(dy_norm_01 - 0.5) <= 0.05)),
        },
    }

    analysis_dir = output_root / "distribution_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plot_histogram(dx_norm, analysis_dir / "hist_dx_norm_neg11.png", "dx_norm [-1,1]", "Train dx_norm [-1,1]")
    plot_histogram(dy_norm, analysis_dir / "hist_dy_norm_neg11.png", "dy_norm [-1,1]", "Train dy_norm [-1,1]")
    with (analysis_dir / "distribution_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def load_delta_values(input_root: Path, sample_ids: list[str], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    dx_chunks: list[np.ndarray] = []
    dy_chunks: list[np.ndarray] = []
    for sample_id in sample_ids:
        sample, skip_reason = load_sample(input_root, sample_id, image_size)
        if sample is None:
            raise ValueError(f"Cannot load train sample {sample_id}: {skip_reason}")
        dx_chunks.append(sample.delta_xy[:, 0])
        dy_chunks.append(sample.delta_xy[:, 1])
    return np.concatenate(dx_chunks), np.concatenate(dy_chunks)


def plot_histogram(values: np.ndarray, path: Path, xlabel: str, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=160, color="#287c71", alpha=0.88)
    axis.axvline(0.0, color="#d1495b", linewidth=1.5, label="zero")
    axis.axvspan(-0.05, 0.05, color="#edae49", alpha=0.24, label="|value| <= 0.05")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(output_root: Path, image_size: int, heatmap_sigma: float) -> None:
    text = f"""# Constrained GASF dx/dy/start Dataset, [-1,1] Ablation

This dataset uses:

R = GASF(dx) with global min-max scaling to [-1,1]
G = GASF(dy) with global min-max scaling to [-1,1]
B = initial-point heatmap

Representation name: `{REPRESENTATION}`
Image size: `{image_size}`
Start heatmap sigma: `{heatmap_sigma}`
Normalization file: `constrained_gasf_normalization.json`

PNG images in `images/` are DDPM-compatible training inputs.
Float32 arrays in `arrays/` preserve the same saved channels before uint8 quantization.

Important: this version is NOT analytically invertible from the GASF diagonal
alone because `diag = 2x^2 - 1` only recovers `abs(x)`. Sign information is
lost. This dataset is intended only as a representation-distribution ablation.
"""
    (output_root / "README.md").write_text(text)


def write_summary(
    path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    selected_ids: list[str],
    processed: list[LoadedSample],
    skipped_rows: list[dict[str, Any]],
    sanity_rows: list[dict[str, Any]],
    distribution_summary: dict[str, Any],
    normalization_source: str,
) -> None:
    payload: dict[str, Any] = {
        "representation": REPRESENTATION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_size": args.image_size,
        "heatmap_sigma": args.heatmap_sigma,
        "split": args.split,
        "max_samples": args.max_samples,
        "normalization_range": "[-1,1]",
        "normalization_source": normalization_source,
        "requested_samples": len(selected_ids),
        "processed_samples": len(processed),
        "skipped_samples": len(skipped_rows),
        "sanity_checked_samples": len(sanity_rows),
        "sign_lost_from_gasf_diagonal": True,
        "distribution_summary": distribution_summary,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

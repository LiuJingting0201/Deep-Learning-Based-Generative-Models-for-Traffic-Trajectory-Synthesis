"""Build the Week 4 constrained GASF dx/dy/start-heatmap dataset."""

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
DEFAULT_OUTPUT_ROOT = WEEK04_ROOT / "data_constrained_gasf_dxdy_start"
REPRESENTATION = "GASF_dx_GASF_dy_start_heatmap"
FLOAT_RECON_CLOSE_THRESHOLD = 1e-4


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

    if args.sanity_only:
        run_sanity_only(args, output_root)
        return

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
    sanity_rows = run_sanity_checks(
        output_root=output_root,
        samples=processed[: args.num_sanity_checks],
        stats=stats,
        image_size=args.image_size,
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
        normalization_source,
    )

    print(f"Created constrained GASF dataset at {output_root}")
    print(f"Requested samples: {len(selected_ids)}")
    print(f"Processed samples: {len(processed)}")
    print(f"Skipped samples: {len(skipped_rows)}")
    print(f"Images: {len(list((output_root / 'images').glob('*.png')))}")
    print(f"Float arrays: {len(list((output_root / 'arrays').glob('*.npy')))}")
    print(f"Normalization: {output_root / 'constrained_gasf_normalization.json'}")
    print(f"Sanity metrics: {output_root / 'sanity_checks' / 'sanity_check_metrics.csv'}")
    warn_if_sanity_is_not_close(sanity_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-sigma", type=float, default=3.0)
    parser.add_argument(
        "--split",
        default="train",
        help="Split to process: train, val, test, or all.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-sanity-checks", type=int, default=20)
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Refresh sanity_check_metrics.csv and plots from an existing output dataset.",
    )
    return parser.parse_args()


def run_sanity_only(args: argparse.Namespace, output_root: Path) -> None:
    stats = read_existing_normalization(output_root)
    (output_root / "sanity_checks" / "plots").mkdir(parents=True, exist_ok=True)

    sample_ids = sorted(path.stem for path in (output_root / "labels_absolute").glob("*.npy"))
    if not sample_ids:
        raise FileNotFoundError(f"No existing labels found in {output_root / 'labels_absolute'}")

    selected_ids = sample_ids[: args.num_sanity_checks]
    samples = [load_existing_output_sample(output_root, sample_id, args.image_size) for sample_id in selected_ids]
    sanity_rows = run_sanity_checks(
        output_root=output_root,
        samples=samples,
        stats=stats,
        image_size=args.image_size,
    )

    print(f"Refreshed sanity diagnostics for existing dataset at {output_root}")
    print(f"Sanity samples: {len(sanity_rows)}")
    print(f"Sanity metrics: {output_root / 'sanity_checks' / 'sanity_check_metrics.csv'}")
    warn_if_sanity_is_not_close(sanity_rows)


def read_existing_normalization(output_root: Path) -> dict[str, float]:
    path = output_root / "constrained_gasf_normalization.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    return {
        key: float(payload[key])
        for key in ["dx_min", "dx_max", "dy_min", "dy_max", "x_min", "x_max", "y_min", "y_max"]
    }


def load_existing_output_sample(output_root: Path, sample_id: str, image_size: int) -> LoadedSample:
    image_path = output_root / "images" / f"{sample_id}.png"
    array_path = output_root / "arrays" / f"{sample_id}.npy"
    abs_path = output_root / "labels_absolute" / f"{sample_id}.npy"
    delta_path = output_root / "labels_delta_displacement" / f"{sample_id}.npy"
    start_path = output_root / "starts" / f"{sample_id}.npy"
    for path in [image_path, array_path, abs_path, delta_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    abs_xy = np.load(abs_path).astype(np.float64)
    delta_xy = np.load(delta_path).astype(np.float64)
    if abs_xy.shape != (image_size, 2):
        raise ValueError(f"{abs_path} has shape {abs_xy.shape}; expected ({image_size}, 2)")
    if delta_xy.shape != (image_size, 2):
        raise ValueError(f"{delta_path} has shape {delta_xy.shape}; expected ({image_size}, 2)")

    initial_point = (
        np.load(start_path).astype(np.float64)
        if start_path.exists()
        else abs_xy[0].astype(np.float64)
    )
    return LoadedSample(
        sample_id=sample_id,
        abs_xy=abs_xy,
        delta_xy=delta_xy,
        initial_point=initial_point,
        adjusted_delta=False,
    )


def validate_input_root(input_root: Path) -> None:
    required = [
        "labels_absolute",
        "labels_delta_displacement",
        "metadata.csv",
    ]
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
        "sanity_plots": output_root / "sanity_checks" / "plots",
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


def load_sample(
    input_root: Path, sample_id: str, image_size: int
) -> tuple[LoadedSample | None, str]:
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
        return None, (
            f"delta shape {delta_xy.shape} is incompatible with absolute shape "
            f"{abs_xy.shape} and image_size={image_size}"
        )

    return LoadedSample(
        sample_id=sample_id,
        abs_xy=abs_xy,
        delta_xy=delta_xy,
        initial_point=abs_xy[0].astype(np.float64),
        adjusted_delta=adjusted_delta,
    ), ""


def encode_sample(
    sample: LoadedSample,
    stats: dict[str, float],
    image_size: int,
    heatmap_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx_norm = normalize_to_unit(sample.delta_xy[:, 0], stats["dx_min"], stats["dx_max"])
    dy_norm = normalize_to_unit(sample.delta_xy[:, 1], stats["dy_min"], stats["dy_max"])
    gasf_dx = gasf_encode(dx_norm)
    gasf_dy = gasf_encode(dy_norm)
    heatmap = make_start_heatmap(sample.initial_point, stats, image_size, heatmap_sigma)
    image_float = np.stack([(gasf_dx + 1.0) / 2.0, (gasf_dy + 1.0) / 2.0, heatmap], axis=-1)
    image_float = np.clip(image_float, 0.0, 1.0).astype(np.float32)
    image_uint8 = np.round(image_float * 255.0).astype(np.uint8)
    return image_float, image_uint8


def normalize_to_unit(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def gasf_encode(values_01: np.ndarray) -> np.ndarray:
    phi = np.arccos(np.clip(values_01, 0.0, 1.0))
    return np.cos(phi[:, None] + phi[None, :])


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


def metadata_row(
    output_root: Path, sample: LoadedSample, stats: dict[str, float]
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "image_path": (output_root / "images" / f"{sample.sample_id}.png").relative_to(output_root).as_posix(),
        "label_delta_displacement_path": (
            output_root / "labels_delta_displacement" / f"{sample.sample_id}.npy"
        ).relative_to(output_root).as_posix(),
        "label_absolute_path": (
            output_root / "labels_absolute" / f"{sample.sample_id}.npy"
        ).relative_to(output_root).as_posix(),
        "start_path": (output_root / "starts" / f"{sample.sample_id}.npy").relative_to(output_root).as_posix(),
        "encoding_method": REPRESENTATION,
        "coordinate_type": "delta_displacement_xy",
        "initial_x": float(sample.initial_point[0]),
        "initial_y": float(sample.initial_point[1]),
        "delta_was_prepended_with_zero": sample.adjusted_delta,
        "dx_min": stats["dx_min"],
        "dx_max": stats["dx_max"],
        "dy_min": stats["dy_min"],
        "dy_max": stats["dy_max"],
        "x_min": stats["x_min"],
        "x_max": stats["x_max"],
        "y_min": stats["y_min"],
        "y_max": stats["y_max"],
        "valid_or_skipped": "valid",
        "skip_reason": "",
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "image_path",
        "label_delta_displacement_path",
        "label_absolute_path",
        "start_path",
        "encoding_method",
        "coordinate_type",
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
        "normalization_source": normalization_source,
        "input_root": str(input_root),
        "output_root": str(output_root),
    }
    with (output_root / "constrained_gasf_normalization.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_sanity_checks(
    output_root: Path,
    samples: list[LoadedSample],
    stats: dict[str, float],
    image_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_path = output_root / "sanity_checks" / "sanity_check_metrics.csv"
    plots_dir = output_root / "sanity_checks" / "plots"

    for sample in samples:
        float_array = np.load(output_root / "arrays" / f"{sample.sample_id}.npy").astype(np.float64)
        png_array = (
            np.asarray(Image.open(output_root / "images" / f"{sample.sample_id}.png")).astype(np.float64)
            / 255.0
        )

        xy_float = reconstruct_from_encoded_array(float_array, sample.initial_point, stats, image_size)
        xy_png = reconstruct_from_encoded_array(png_array, sample.initial_point, stats, image_size)
        float_metrics = reconstruction_metrics(xy_float, sample.abs_xy)
        png_metrics = reconstruction_metrics(xy_png, sample.abs_xy)
        status = (
            "ok"
            if float_metrics["max_error"] <= FLOAT_RECON_CLOSE_THRESHOLD
            else "warning_float_array_reconstruction_error"
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "ade_float": float_metrics["ade"],
                "fde_float": float_metrics["fde"],
                "max_error_float": float_metrics["max_error"],
                "ade_png": png_metrics["ade"],
                "fde_png": png_metrics["fde"],
                "max_error_png": png_metrics["max_error"],
                "delta_shape": str(tuple(sample.delta_xy.shape)),
                "absolute_shape": str(tuple(sample.abs_xy.shape)),
                "status": status,
            }
        )
        plot_sanity(
            plots_dir / f"{sample.sample_id}.png",
            sample,
            xy_float,
            float_metrics["ade"],
            float_metrics["fde"],
            float_metrics["max_error"],
        )

    with metrics_path.open("w", newline="") as handle:
        fieldnames = [
            "sample_id",
            "ade_float",
            "fde_float",
            "max_error_float",
            "ade_png",
            "fde_png",
            "max_error_png",
            "delta_shape",
            "absolute_shape",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def decode_gasf_channel(channel_01: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    gasf = channel_01 * 2.0 - 1.0
    diag = np.diag(gasf)
    u_rec = np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, 1.0))
    return u_rec * (maximum - minimum) + minimum


def reconstruct_from_encoded_array(
    encoded_array: np.ndarray,
    initial_point: np.ndarray,
    stats: dict[str, float],
    image_size: int,
) -> np.ndarray:
    dx_rec = decode_gasf_channel(encoded_array[:, :, 0], stats["dx_min"], stats["dx_max"])
    dy_rec = decode_gasf_channel(encoded_array[:, :, 1], stats["dy_min"], stats["dy_max"])
    return integrate_delta_mode_a(initial_point, dx_rec, dy_rec, image_size)


def reconstruction_metrics(xy_rec: np.ndarray, abs_xy: np.ndarray) -> dict[str, float]:
    errors = np.linalg.norm(xy_rec - abs_xy, axis=1)
    return {
        "ade": float(errors.mean()),
        "fde": float(errors[-1]),
        "max_error": float(errors.max()),
    }


def integrate_delta_mode_a(
    initial_point: np.ndarray, dx: np.ndarray, dy: np.ndarray, image_size: int
) -> np.ndarray:
    xy_rec = np.empty((image_size, 2), dtype=np.float64)
    xy_rec[0] = initial_point
    for timestep in range(1, image_size):
        xy_rec[timestep] = xy_rec[timestep - 1] + np.array([dx[timestep], dy[timestep]])
    return xy_rec


def integrate_delta_mode_b(
    initial_point: np.ndarray, dx: np.ndarray, dy: np.ndarray, image_size: int
) -> np.ndarray:
    xy_rec = np.empty((image_size, 2), dtype=np.float64)
    xy_rec[0] = initial_point
    for timestep in range(1, image_size):
        xy_rec[timestep] = xy_rec[timestep - 1] + np.array([dx[timestep - 1], dy[timestep - 1]])
    return xy_rec


def integrate_delta_mode_c(initial_point: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    delta_xy = np.stack([dx, dy], axis=1)
    return initial_point + np.cumsum(delta_xy, axis=0)


def plot_sanity(
    path: Path,
    sample: LoadedSample,
    xy_rec: np.ndarray,
    ade: float,
    fde: float,
    max_error: float,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot(sample.abs_xy[:, 0], sample.abs_xy[:, 1], label="ground truth", linewidth=1.6)
    axis.plot(xy_rec[:, 0], xy_rec[:, 1], label="float-array analytical reconstruction", linewidth=1.2)
    axis.scatter(sample.abs_xy[0, 0], sample.abs_xy[0, 1], s=28, label="start")
    axis.set_title(f"{sample.sample_id} | float | ADE={ade:.4g}, FDE={fde:.4g}, max={max_error:.4g}")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best", fontsize=8)
    axis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def warn_if_sanity_is_not_close(rows: list[dict[str, Any]]) -> None:
    warning_rows = [row for row in rows if row["status"] != "ok"]
    if not warning_rows:
        return
    max_error = max(float(row["max_error_float"]) for row in warning_rows)
    print(
        "WARNING: analytical decoding from saved float arrays was not near-zero "
        f"for {len(warning_rows)} sanity samples; max_error={max_error:.6g}. "
        "This should be close to zero; check GASF decoding and timestep alignment."
    )


def write_readme(output_root: Path, image_size: int, heatmap_sigma: float) -> None:
    text = f"""# Constrained GASF dx/dy/start Dataset

This dataset uses:

R = GASF(dx)
G = GASF(dy)
B = initial-point heatmap

It replaces the previous GASF/GADF/MTF representation.
It is designed for analytical diagonal decoding.

Representation name: `{REPRESENTATION}`
Image size: `{image_size}`
Start heatmap sigma: `{heatmap_sigma}`
Normalization file: `constrained_gasf_normalization.json`

PNG images in `images/` are the DDPM training inputs.
Float32 arrays in `arrays/` preserve the same channels before uint8 quantization
and are used for exact analytical decoding verification.
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
        "normalization_source": normalization_source,
        "requested_samples": len(selected_ids),
        "processed_samples": len(processed),
        "skipped_samples": len(skipped_rows),
        "sanity_checked_samples": len(sanity_rows),
        "sanity_warnings": sum(1 for row in sanity_rows if row["status"] != "ok"),
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

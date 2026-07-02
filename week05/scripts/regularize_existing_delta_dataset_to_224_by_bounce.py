"""Regularize an existing variable-length delta dataset to length 224 by bounce padding.

The input dataset is expected to contain variable-length absolute coordinate
arrays. This script recomputes delta displacement from the output absolute
arrays and never depends on existing delta files. Short trajectories are
extended with reflect/bounce padding, while long trajectories are either
truncated or split into sliding windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIAGNOSTIC_PERCENTILES = [0, 0.5, 1, 2, 5, 25, 50, 75, 95, 98, 99, 99.5, 100]
SIGMOID_K_VALUES = [1.0, 1.5, 2.0]
NORMALIZED_RATIO_INTERVALS = [
    (0.0, 0.01),
    (0.0, 0.02),
    (0.0, 0.05),
    (0.45, 0.55),
    (0.4, 0.6),
    (0.25, 0.75),
    (0.95, 1.0),
    (0.98, 1.0),
    (0.99, 1.0),
]
MAX_RECONSTRUCTION_ERROR = 1e-8


@dataclass(frozen=True)
class RegularizedSample:
    sample_id: str
    source_sample_id: str
    vehicle_id: str
    source_absolute_path: Path
    original_length: int
    target_length: int
    regularization_method: str
    long_mode: str
    window_index: int
    window_start_index: int
    window_end_index: int
    absolute_path: Path
    delta_path: Path
    start_path: Path
    max_reconstruction_error: float


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    validate_args(args, input_root, output_root)

    labels_absolute_in = input_root / "labels_absolute"
    require_input_layout(input_root, labels_absolute_in)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_root)

    labels_absolute_out = output_root / "labels_absolute"
    labels_delta_out = output_root / "labels_delta_displacement"
    starts_out = output_root / "starts"
    splits_out = output_root / "splits"
    for directory in (labels_absolute_out, labels_delta_out, starts_out, splits_out):
        directory.mkdir(parents=True, exist_ok=True)

    source_metadata = read_source_metadata(input_root / "metadata.csv")
    source_paths = sorted(labels_absolute_in.glob("*.npy"), key=natural_sample_sort_key)
    if not source_paths:
        raise FileNotFoundError(f"No .npy files found in {labels_absolute_in}")

    samples: list[RegularizedSample] = []
    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    source_lengths: list[int] = []
    method_counts: Counter[str] = Counter()

    for source_absolute_path in source_paths:
        source_sample_id = source_absolute_path.stem
        absolute_xy = np.load(source_absolute_path).astype(np.float64)
        validate_absolute_array(source_absolute_path, absolute_xy)

        original_length = int(absolute_xy.shape[0])
        source_lengths.append(original_length)
        if original_length < args.min_length:
            skipped_rows.append(
                {
                    "old_sample_id": source_sample_id,
                    "original_length": original_length,
                    "skip_reason": "shorter_than_min_length",
                }
            )
            continue

        vehicle_id = source_metadata.get(source_sample_id, {}).get("vehicle_id", "")
        regularized_windows = build_regularized_windows(absolute_xy, args)
        for window_index, window in enumerate(regularized_windows):
            regularized_abs = window["absolute_xy"]
            regularization_method = window["regularization_method"]
            window_start_index = window["window_start_index"]
            window_end_index = window["window_end_index"]
            method_counts[regularization_method] += 1

            delta_xy = compute_delta_displacement(regularized_abs)
            start_xy = regularized_abs[0].astype(np.float64)
            reconstructed_xy = integrate_delta(start_xy, delta_xy)
            max_error = float(np.max(np.abs(reconstructed_xy - regularized_abs)))
            if max_error > MAX_RECONSTRUCTION_ERROR:
                raise AssertionError(
                    f"Delta integration failed for {source_sample_id} window "
                    f"{window_index}: max_error={max_error}"
                )

            sample_id = f"sample_{len(samples):06d}"
            absolute_path = labels_absolute_out / f"{sample_id}.npy"
            delta_path = labels_delta_out / f"{sample_id}.npy"
            start_path = starts_out / f"{sample_id}.npy"
            np.save(absolute_path, regularized_abs)
            np.save(delta_path, delta_xy)
            np.save(start_path, start_xy)

            sample = RegularizedSample(
                sample_id=sample_id,
                source_sample_id=source_sample_id,
                vehicle_id=vehicle_id,
                source_absolute_path=source_absolute_path,
                original_length=original_length,
                target_length=args.target_length,
                regularization_method=regularization_method,
                long_mode=args.long_mode,
                window_index=window_index,
                window_start_index=window_start_index,
                window_end_index=window_end_index,
                absolute_path=absolute_path,
                delta_path=delta_path,
                start_path=start_path,
                max_reconstruction_error=max_error,
            )
            samples.append(sample)
            metadata_rows.append(metadata_row(sample, output_root))

    write_metadata(output_root / "metadata.csv", metadata_rows)
    write_skipped_samples(output_root / "skipped_samples.csv", skipped_rows)
    split_to_ids = write_splits(splits_out, [sample.sample_id for sample in samples], args)
    split_counts = {split: len(ids) for split, ids in split_to_ids.items()}
    training_values = collect_training_values(labels_delta_out, split_to_ids["train"])
    diagnostics = write_normalization_diagnostics(
        output_root / "normalization_diagnostics.json", training_values
    )
    plot_normalization_histograms(output_root / "normalization_plots", training_values, diagnostics)

    max_reconstruction_error = (
        max(sample.max_reconstruction_error for sample in samples) if samples else 0.0
    )
    write_dataset_summary(
        output_root / "dataset_summary.json",
        args=args,
        input_root=input_root,
        output_root=output_root,
        source_lengths=source_lengths,
        num_source_samples=len(source_paths),
        num_output_samples=len(samples),
        num_skipped_samples=len(skipped_rows),
        method_counts=method_counts,
        split_counts=split_counts,
        max_reconstruction_error=max_reconstruction_error,
    )

    print(f"num_output_samples: {len(samples)}")
    print(f"regularization_method_counts: {dict(sorted(method_counts.items()))}")
    print(
        "dx_norm ratio [0.45,0.55]: "
        f"{diagnostics['dx_norm_interval_ratios']['[0.45,0.55]']:.10g}"
    )
    print(
        "dy_norm ratio [0.45,0.55]: "
        f"{diagnostics['dy_norm_interval_ratios']['[0.45,0.55]']:.10g}"
    )


def load_runtime_dependencies() -> None:
    try:
        import numpy as numpy_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs numpy. Run it in the project environment "
            "or install numpy before regularizing the dataset."
        ) from exc

    globals()["np"] = numpy_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-length", type=int, default=224)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--long-mode", choices=["truncate", "sliding"], default="truncate")
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, input_root: Path, output_root: Path) -> None:
    if args.target_length <= 0:
        raise ValueError("--target-length must be positive.")
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2 for bounce padding.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.train_ratio < 0.0 or args.val_ratio < 0.0 or args.train_ratio + args.val_ratio > 1.0:
        raise ValueError(
            "Require --train-ratio >= 0, --val-ratio >= 0, "
            "and train + val <= 1."
        )
    if input_root == output_root:
        raise ValueError("--output-root must be different from --input-root.")


def require_input_layout(input_root: Path, labels_absolute_dir: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if not labels_absolute_dir.is_dir():
        raise FileNotFoundError(f"Expected input directory: {labels_absolute_dir}")


def read_source_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            return {}
        return {str(row["sample_id"]): row for row in reader}


def natural_sample_sort_key(path: Path) -> tuple[int, Any]:
    stem = path.stem
    if stem.startswith("sample_") and stem.removeprefix("sample_").isdigit():
        return (0, int(stem.removeprefix("sample_")))
    return (1, stem)


def validate_absolute_array(path: Path, absolute_xy: Any) -> None:
    if absolute_xy.ndim != 2 or absolute_xy.shape[1] != 2:
        raise ValueError(f"{path} has shape {absolute_xy.shape}; expected (N, 2).")
    if not np.isfinite(absolute_xy).all():
        raise ValueError(f"{path} contains non-finite values.")


def build_regularized_windows(absolute_xy: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    original_length = int(absolute_xy.shape[0])
    if original_length < args.target_length:
        return [
            {
                "absolute_xy": bounce_to_length(absolute_xy, args.target_length),
                "regularization_method": "bounce_reflect_padding",
                "window_start_index": 0,
                "window_end_index": original_length,
            }
        ]
    if original_length == args.target_length:
        return [
            {
                "absolute_xy": absolute_xy.copy(),
                "regularization_method": "unchanged",
                "window_start_index": 0,
                "window_end_index": original_length,
            }
        ]
    if args.long_mode == "truncate":
        return [
            {
                "absolute_xy": absolute_xy[: args.target_length].copy(),
                "regularization_method": f"truncate_first_{args.target_length}",
                "window_start_index": 0,
                "window_end_index": args.target_length,
            }
        ]

    windows: list[dict[str, Any]] = []
    for start_index in sliding_start_indices(
        original_length, args.target_length, args.stride, args.include_tail
    ):
        end_index = start_index + args.target_length
        windows.append(
            {
                "absolute_xy": absolute_xy[start_index:end_index].copy(),
                "regularization_method": "sliding_window",
                "window_start_index": start_index,
                "window_end_index": end_index,
            }
        )
    return windows


def bounce_to_length(abs_xy: Any, target_length: int) -> Any:
    original_length = int(abs_xy.shape[0])
    if original_length < 2:
        raise ValueError("bounce_to_length requires at least two points.")
    if original_length >= target_length:
        return abs_xy[:target_length].copy()

    forward = list(range(original_length))
    backward = list(range(original_length - 2, 0, -1))
    base = forward + backward
    repeats = (target_length + len(base) - 1) // len(base)
    indices = (base * repeats)[:target_length]
    return abs_xy[indices].copy()


def sliding_start_indices(
    original_length: int,
    target_length: int,
    stride: int,
    include_tail: bool,
) -> list[int]:
    last_start = original_length - target_length
    starts = list(range(0, last_start + 1, stride))
    if include_tail and starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def compute_delta_displacement(abs_xy: Any) -> Any:
    delta_xy = np.zeros_like(abs_xy, dtype=np.float64)
    delta_xy[1:] = abs_xy[1:] - abs_xy[:-1]
    return delta_xy


def integrate_delta(start: Any, delta: Any) -> Any:
    integrated_xy = np.empty_like(delta, dtype=np.float64)
    integrated_xy[0] = start.astype(np.float64)
    integrated_xy[1:] = start.astype(np.float64) + np.cumsum(delta[1:], axis=0)
    return integrated_xy


def metadata_row(sample: RegularizedSample, output_root: Path) -> dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        "source_sample_id": sample.source_sample_id,
        "source_absolute_path": str(sample.source_absolute_path),
        "original_length": sample.original_length,
        "target_length": sample.target_length,
        "regularization_method": sample.regularization_method,
        "long_mode": sample.long_mode,
        "window_index": sample.window_index,
        "window_start_index": sample.window_start_index,
        "window_end_index": sample.window_end_index,
        "label_absolute_path": sample.absolute_path.relative_to(output_root).as_posix(),
        "label_delta_displacement_path": sample.delta_path.relative_to(output_root).as_posix(),
        "start_path": sample.start_path.relative_to(output_root).as_posix(),
        "max_reconstruction_error": sample.max_reconstruction_error,
        "coordinate_type": "delta_displacement_xy_regularized_224",
        "valid_or_skipped": "valid",
        "skip_reason": "",
    }
    if sample.vehicle_id:
        row["vehicle_id"] = sample.vehicle_id
    return row


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "source_sample_id",
        "source_absolute_path",
        "original_length",
        "target_length",
        "regularization_method",
        "long_mode",
        "window_index",
        "window_start_index",
        "window_end_index",
        "label_absolute_path",
        "label_delta_displacement_path",
        "start_path",
        "max_reconstruction_error",
        "coordinate_type",
        "valid_or_skipped",
        "skip_reason",
    ]
    if any("vehicle_id" in row for row in rows):
        columns.insert(2, "vehicle_id")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["old_sample_id", "original_length", "skip_reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_splits(
    splits_dir: Path,
    sample_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(args.seed)
    shuffled_ids = list(sample_ids)
    rng.shuffle(shuffled_ids)

    num_samples = len(shuffled_ids)
    num_train = int(num_samples * args.train_ratio)
    num_val = int(num_samples * args.val_ratio)
    split_to_ids = {
        "train": shuffled_ids[:num_train],
        "val": shuffled_ids[num_train : num_train + num_val],
        "test": shuffled_ids[num_train + num_val :],
    }

    for split, ids in split_to_ids.items():
        with (splits_dir / f"{split}_ids.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id"])
            writer.writeheader()
            for sample_id in ids:
                writer.writerow({"sample_id": sample_id})
    return split_to_ids


def collect_training_values(
    labels_delta_dir: Path,
    train_sample_ids: list[str],
) -> dict[str, Any]:
    if not train_sample_ids:
        raise ValueError("Train split is empty; cannot write normalization diagnostics.")

    dx_values = []
    dy_values = []
    for sample_id in train_sample_ids:
        delta_path = labels_delta_dir / f"{sample_id}.npy"
        delta_xy = np.load(delta_path).astype(np.float64)
        if delta_xy.ndim != 2 or delta_xy.shape[1] != 2:
            raise ValueError(f"{delta_path} has shape {delta_xy.shape}; expected (N, 2).")
        dx_values.append(delta_xy[:, 0])
        dy_values.append(delta_xy[:, 1])
    return {
        "dx": np.concatenate(dx_values, axis=0),
        "dy": np.concatenate(dy_values, axis=0),
    }


def write_normalization_diagnostics(path: Path, training_values: dict[str, Any]) -> dict[str, Any]:
    dx = training_values["dx"]
    dy = training_values["dy"]
    dx_min = float(dx.min())
    dx_max = float(dx.max())
    dy_min = float(dy.min())
    dy_max = float(dy.max())
    dx_mean = float(dx.mean())
    dx_std = float(dx.std())
    dy_mean = float(dy.mean())
    dy_std = float(dy.std())
    dx_norm = normalize_to_unit(dx, dx_min, dx_max)
    dy_norm = normalize_to_unit(dy, dy_min, dy_max)

    diagnostics: dict[str, Any] = {
        "normalization_source": "train_split",
        "num_train_values": int(dx.shape[0]),
        "percentiles": DIAGNOSTIC_PERCENTILES,
        "sigmoid_k_values": SIGMOID_K_VALUES,
        "sigmoid_definition": "sigmoid(k * z), where z = (value - train_mean) / train_std",
        "dx_min": dx_min,
        "dx_max": dx_max,
        "dy_min": dy_min,
        "dy_max": dy_max,
        "dx_mean": dx_mean,
        "dx_std": dx_std,
        "dy_mean": dy_mean,
        "dy_std": dy_std,
        "dx_percentiles": percentile_dict(dx),
        "dy_percentiles": percentile_dict(dy),
        "dx_norm_percentiles": percentile_dict(dx_norm),
        "dy_norm_percentiles": percentile_dict(dy_norm),
        "dx_norm_interval_ratios": interval_ratio_dict(dx_norm),
        "dy_norm_interval_ratios": interval_ratio_dict(dy_norm),
    }
    diagnostics.update(
        sigmoid_diagnostics(
            dx=dx,
            dy=dy,
            dx_mean=dx_mean,
            dx_std=dx_std,
            dy_mean=dy_mean,
            dy_std=dy_std,
        )
    )
    with path.open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    return diagnostics


def normalize_to_unit(values: Any, minimum: float, maximum: float) -> Any:
    if maximum == minimum:
        raise ValueError(f"Cannot normalize values with zero range: {minimum} == {maximum}")
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def sigmoid_diagnostics(
    dx: Any,
    dy: Any,
    dx_mean: float,
    dx_std: float,
    dy_mean: float,
    dy_std: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for k_value in SIGMOID_K_VALUES:
        dx_sigmoid = sigmoid_zscore_normalize(dx, dx_mean, dx_std, k_value)
        dy_sigmoid = sigmoid_zscore_normalize(dy, dy_mean, dy_std, k_value)
        diagnostics[f"sigmoid_k_{k_value:.1f}"] = {
            "dx_norm_percentiles": percentile_dict(dx_sigmoid),
            "dy_norm_percentiles": percentile_dict(dy_sigmoid),
            "dx_norm_interval_ratios": interval_ratio_dict(dx_sigmoid),
            "dy_norm_interval_ratios": interval_ratio_dict(dy_sigmoid),
        }
    return diagnostics


def sigmoid_zscore_normalize(values: Any, mean: float, std: float, k_value: float) -> Any:
    if std == 0.0:
        raise ValueError("Cannot apply z-score sigmoid normalization with zero standard deviation.")
    logits = np.clip(k_value * ((values - mean) / std), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def percentile_dict(values: Any) -> dict[str, float]:
    percentile_values = np.percentile(values, DIAGNOSTIC_PERCENTILES)
    return {
        format_percentile(percentile): float(value)
        for percentile, value in zip(DIAGNOSTIC_PERCENTILES, percentile_values)
    }


def interval_ratio_dict(values: Any) -> dict[str, float]:
    return {
        interval_label(low, high): normalized_interval_ratio(values, low, high)
        for low, high in NORMALIZED_RATIO_INTERVALS
    }


def normalized_interval_ratio(values: Any, low: float, high: float) -> float:
    return float(np.mean((values >= low) & (values <= high)))


def format_percentile(percentile: float) -> str:
    return str(int(percentile)) if float(percentile).is_integer() else str(percentile)


def interval_label(low: float, high: float) -> str:
    return f"[{format_interval_endpoint(low)},{format_interval_endpoint(high)}]"


def format_interval_endpoint(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == 1.0:
        return "1.0"
    return f"{value:g}"


def plot_normalization_histograms(
    plots_dir: Path,
    training_values: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_one_axis_distribution(
        plots_dir / "dx_distribution.png",
        raw_values=training_values["dx"],
        linear_values=normalize_to_unit(
            training_values["dx"], diagnostics["dx_min"], diagnostics["dx_max"]
        ),
        sigmoid_values=sigmoid_zscore_normalize(
            training_values["dx"], diagnostics["dx_mean"], diagnostics["dx_std"], 1.5
        ),
        axis_name="dx",
        plt=plt,
    )
    plot_one_axis_distribution(
        plots_dir / "dy_distribution.png",
        raw_values=training_values["dy"],
        linear_values=normalize_to_unit(
            training_values["dy"], diagnostics["dy_min"], diagnostics["dy_max"]
        ),
        sigmoid_values=sigmoid_zscore_normalize(
            training_values["dy"], diagnostics["dy_mean"], diagnostics["dy_std"], 1.5
        ),
        axis_name="dy",
        plt=plt,
    )


def plot_one_axis_distribution(
    path: Path,
    raw_values: Any,
    linear_values: Any,
    sigmoid_values: Any,
    axis_name: str,
    plt: Any,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(raw_values, bins=100)
    axes[0].set_title(f"raw {axis_name}")
    axes[0].set_xlabel(axis_name)
    axes[0].set_ylabel("count")
    axes[1].hist(linear_values, bins=100, range=(0.0, 1.0))
    axes[1].set_title(f"linear norm {axis_name}")
    axes[1].set_xlabel(f"{axis_name}_norm")
    axes[1].set_ylabel("count")
    axes[2].hist(sigmoid_values, bins=100, range=(0.0, 1.0))
    axes[2].set_title(f"sigmoid k=1.5 {axis_name}")
    axes[2].set_xlabel(f"{axis_name}_sigmoid")
    axes[2].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_dataset_summary(
    path: Path,
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    source_lengths: list[int],
    num_source_samples: int,
    num_output_samples: int,
    num_skipped_samples: int,
    method_counts: Counter[str],
    split_counts: dict[str, int],
    max_reconstruction_error: float,
) -> None:
    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_source_samples": num_source_samples,
        "num_output_samples": num_output_samples,
        "num_skipped_samples": num_skipped_samples,
        "target_length": args.target_length,
        "min_length": args.min_length,
        "long_mode": args.long_mode,
        "stride": args.stride,
        "include_tail": args.include_tail,
        "source_length": summary_stats(source_lengths),
        "regularization_method_counts": dict(sorted(method_counts.items())),
        "split_counts": split_counts,
        "max_reconstruction_error": max_reconstruction_error,
        "outputs": {
            "metadata": "metadata.csv",
            "skipped_samples": "skipped_samples.csv",
            "splits": "splits/{train,val,test}_ids.csv",
            "normalization_diagnostics": "normalization_diagnostics.json",
            "normalization_plots": "normalization_plots/{dx_distribution,dy_distribution}.png",
            "absolute_labels": "labels_absolute/*.npy",
            "delta_displacement_labels": "labels_delta_displacement/*.npy",
            "starts": "starts/*.npy",
        },
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def summary_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }


if __name__ == "__main__":
    main()

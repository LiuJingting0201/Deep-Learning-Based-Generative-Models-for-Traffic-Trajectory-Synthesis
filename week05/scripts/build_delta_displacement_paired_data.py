"""Build id-paired delta-displacement data from absolute vehicle coordinates.

This week05 variant intentionally does not build trajectory images. It reads a
CSV of absolute vehicle positions, groups rows by vehicle id, sorts each group
by time when a time column is available, and writes paired absolute/delta/start
arrays plus metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "week05" / "data" / "vehicle_positions_TS_New.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired"

ID_CANDIDATES = ("vehicle_id", "id", "track_id", "trajectory_id", "veh_id")
TIME_CANDIDATES = ("time", "timestamp", "frame", "frame_id", "t")
X_CANDIDATES = ("x", "pos_x", "position_x", "global_x", "local_x")
Y_CANDIDATES = ("y", "pos_y", "position_y", "global_y", "local_y")
DIAGNOSTIC_PERCENTILES = [0, 0.5, 1, 2, 5, 25, 50, 75, 95, 98, 99, 99.5, 100]
SIGMOID_K_VALUES = [0.5, 1.0, 1.5, 2.0]
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


@dataclass(frozen=True)
class PairedSample:
    sample_id: str
    vehicle_id: str
    original_num_points: int
    absolute_path: Path
    delta_path: Path
    start_path: Path
    first_time: Any
    last_time: Any
    max_reconstruction_error: float


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    labels_absolute_dir = output_dir / "labels_absolute"
    labels_delta_dir = output_dir / "labels_delta_displacement"
    starts_dir = output_dir / "starts"
    splits_dir = output_dir / "splits"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_dir)

    for directory in (labels_absolute_dir, labels_delta_dir, starts_dir, splits_dir):
        directory.mkdir(parents=True)

    frame = read_csv(input_path)
    id_column = resolve_column(args.id_column, frame, ID_CANDIDATES, "id")
    x_column = resolve_column(args.x_column, frame, X_CANDIDATES, "x")
    y_column = resolve_column(args.y_column, frame, Y_CANDIDATES, "y")
    time_column = resolve_optional_column(args.time_column, frame, TIME_CANDIDATES)

    validate_required_values(frame, id_column, x_column, y_column, time_column)
    frame = frame.copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[x_column] = pd.to_numeric(frame[x_column], errors="raise")
    frame[y_column] = pd.to_numeric(frame[y_column], errors="raise")

    samples: list[PairedSample] = []
    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    ordered_ids = sort_vehicle_ids(frame[id_column].unique())
    for sample_index, vehicle_id in enumerate(ordered_ids):
        group = frame.loc[frame[id_column] == vehicle_id]
        if time_column is not None:
            group = group.sort_values(time_column, kind="mergesort")
        else:
            group = group.sort_index(kind="mergesort")

        if len(group) < args.min_points:
            skipped_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "num_points": len(group),
                    "skip_reason": f"fewer_than_min_points_{args.min_points}",
                }
            )
            continue

        absolute_xy = group[[x_column, y_column]].to_numpy(dtype=np.float64)
        delta_xy = compute_delta_displacement(absolute_xy)
        start_xy = absolute_xy[0].astype(np.float64)
        reconstructed_xy = integrate_delta(start_xy, delta_xy)
        max_error = float(np.max(np.abs(reconstructed_xy - absolute_xy)))
        if max_error > args.max_reconstruction_error:
            raise AssertionError(
                f"Delta integration failed for id={vehicle_id}: max_error={max_error}"
            )

        sample_id = f"sample_{len(samples):06d}"
        absolute_path = labels_absolute_dir / f"{sample_id}.npy"
        delta_path = labels_delta_dir / f"{sample_id}.npy"
        start_path = starts_dir / f"{sample_id}.npy"
        np.save(absolute_path, absolute_xy)
        np.save(delta_path, delta_xy)
        np.save(start_path, start_xy)

        first_time, last_time = get_time_bounds(group, time_column)
        sample = PairedSample(
            sample_id=sample_id,
            vehicle_id=vehicle_id,
            original_num_points=len(group),
            absolute_path=absolute_path,
            delta_path=delta_path,
            start_path=start_path,
            first_time=first_time,
            last_time=last_time,
            max_reconstruction_error=max_error,
        )
        samples.append(sample)
        metadata_rows.append(metadata_row(sample, output_dir))

    write_metadata(output_dir / "metadata.csv", metadata_rows)
    write_skipped(output_dir / "skipped_ids.csv", skipped_rows)
    write_unique_ids(output_dir / "unique_vehicle_ids.csv", samples)
    split_to_ids = write_splits(splits_dir, samples, args.train_ratio, args.val_ratio, args.seed)
    split_counts = {split: len(ids) for split, ids in split_to_ids.items()}
    training_values = collect_training_values(labels_delta_dir, split_to_ids["train"])
    diagnostics_path = output_dir / "normalization_diagnostics.json"
    plots_dir = output_dir / "normalization_plots"
    diagnostics = write_normalization_diagnostics(diagnostics_path, training_values)
    plot_normalization_histograms(plots_dir, training_values, diagnostics)
    write_summary(
        output_dir / "dataset_summary.json",
        frame,
        samples,
        skipped_rows,
        split_counts,
        input_path,
        id_column,
        x_column,
        y_column,
        time_column,
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Raw rows: {len(frame)}")
    print(f"Valid paired ids: {len(samples)}")
    print(f"Skipped ids: {len(skipped_rows)}")
    print(f"Splits: {split_counts}")
    print(f"Normalization diagnostics: {diagnostics_path}")
    print(f"Normalization histogram plots: {plots_dir}")
    print(
        "dx_norm ratio in [0.45,0.55]: "
        f"{diagnostics['dx_norm_interval_ratios']['[0.45,0.55]']:.10g}"
    )
    print(
        "dy_norm ratio in [0.45,0.55]: "
        f"{diagnostics['dy_norm_interval_ratios']['[0.45,0.55]']:.10g}"
    )
    if samples:
        max_error = max(sample.max_reconstruction_error for sample in samples)
        print(f"Max delta integration reconstruction error: {max_error:.10g}")


def load_runtime_dependencies() -> None:
    try:
        import numpy as numpy_module
        import pandas as pandas_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs numpy and pandas. Run it in the project environment "
            "or install the requirements before building the dataset."
        ) from exc

    globals()["np"] = numpy_module
    globals()["pd"] = pandas_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--x-column", default=None)
    parser.add_argument("--y-column", default=None)
    parser.add_argument("--min-points", type=int, default=2)
    parser.add_argument("--max-reconstruction-error", type=float, default=1e-8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Place vehicle_positions_TS_New.csv in week05/data "
            "or pass --input /path/to/file.csv."
        )
    return pd.read_csv(path, low_memory=False)


def resolve_column(
    requested: str | None,
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Requested {role} column {requested!r} not found.")
        return requested

    lower_to_original = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(
        f"Could not infer {role} column. Columns are {list(frame.columns)}. "
        f"Pass --{role}-column explicitly."
    )


def resolve_optional_column(
    requested: str | None,
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Requested time column {requested!r} not found.")
        return requested

    lower_to_original = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def validate_required_values(
    frame: pd.DataFrame,
    id_column: str,
    x_column: str,
    y_column: str,
    time_column: str | None,
) -> None:
    required_columns = [id_column, x_column, y_column]
    if time_column is not None:
        required_columns.append(time_column)
    null_counts = frame[required_columns].isna().sum()
    if int(null_counts.sum()) != 0:
        raise ValueError(f"Null values found in required columns: {null_counts.to_dict()}")


def sort_vehicle_ids(vehicle_ids: Any) -> list[str]:
    def key(vehicle_id: Any) -> tuple[int, Any]:
        text = str(vehicle_id)
        if text.startswith("car") and text[3:].isdigit():
            return (0, int(text[3:]))
        if text.isdigit():
            return (1, int(text))
        return (2, text)

    return [str(vehicle_id) for vehicle_id in sorted(vehicle_ids, key=key)]


def compute_delta_displacement(absolute_xy: np.ndarray) -> np.ndarray:
    delta_xy = np.zeros_like(absolute_xy, dtype=np.float64)
    delta_xy[1:] = absolute_xy[1:] - absolute_xy[:-1]
    return delta_xy


def normalize_to_unit(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum == minimum:
        raise ValueError(f"Cannot normalize values with zero range: {minimum} == {maximum}")
    return np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)


def integrate_delta(start_xy: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    integrated_xy = np.empty_like(delta_xy, dtype=np.float64)
    integrated_xy[0] = start_xy.astype(np.float64)
    integrated_xy[1:] = start_xy.astype(np.float64) + np.cumsum(delta_xy[1:], axis=0)
    return integrated_xy


def get_time_bounds(group: pd.DataFrame, time_column: str | None) -> tuple[Any, Any]:
    if time_column is None:
        return "", ""
    return group[time_column].iloc[0], group[time_column].iloc[-1]


def metadata_row(sample: PairedSample, output_dir: Path) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "vehicle_id": sample.vehicle_id,
        "original_num_points": sample.original_num_points,
        "label_delta_displacement_path": sample.delta_path.relative_to(output_dir).as_posix(),
        "label_absolute_path": sample.absolute_path.relative_to(output_dir).as_posix(),
        "start_path": sample.start_path.relative_to(output_dir).as_posix(),
        "first_time": sample.first_time,
        "last_time": sample.last_time,
        "coordinate_type": "delta_displacement_xy",
        "max_reconstruction_error": sample.max_reconstruction_error,
        "valid_or_skipped": "valid",
        "skip_reason": "",
    }


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "vehicle_id",
        "original_num_points",
        "label_delta_displacement_path",
        "label_absolute_path",
        "start_path",
        "first_time",
        "last_time",
        "coordinate_type",
        "max_reconstruction_error",
        "valid_or_skipped",
        "skip_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["vehicle_id", "num_points", "skip_reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_unique_ids(path: Path, samples: list[PairedSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "vehicle_id"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"sample_id": sample.sample_id, "vehicle_id": sample.vehicle_id})


def write_splits(
    splits_dir: Path,
    samples: list[PairedSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if train_ratio < 0.0 or val_ratio < 0.0 or train_ratio + val_ratio > 1.0:
        raise ValueError("Require train_ratio >= 0, val_ratio >= 0, and train_ratio + val_ratio <= 1.")

    sample_ids = [sample.sample_id for sample in samples]
    rng = np.random.default_rng(seed)
    shuffled_ids = list(sample_ids)
    rng.shuffle(shuffled_ids)

    num_samples = len(shuffled_ids)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
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
    labels_delta_dir: Path, train_sample_ids: list[str]
) -> dict[str, np.ndarray]:
    dx_values: list[np.ndarray] = []
    dy_values: list[np.ndarray] = []
    for sample_id in train_sample_ids:
        delta_path = labels_delta_dir / f"{sample_id}.npy"
        if not delta_path.exists():
            raise FileNotFoundError(delta_path)
        delta_xy = np.load(delta_path).astype(np.float64)
        if delta_xy.ndim != 2 or delta_xy.shape[1] != 2:
            raise ValueError(f"{delta_path} has shape {delta_xy.shape}; expected (N, 2)")
        dx_values.append(delta_xy[:, 0])
        dy_values.append(delta_xy[:, 1])

    if not dx_values:
        raise ValueError("Train split is empty; cannot write normalization diagnostics.")
    return {
        "dx": np.concatenate(dx_values, axis=0),
        "dy": np.concatenate(dy_values, axis=0),
    }


def write_normalization_diagnostics(
    path: Path, training_values: dict[str, np.ndarray]
) -> dict[str, Any]:
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


def sigmoid_diagnostics(
    dx: np.ndarray,
    dy: np.ndarray,
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


def sigmoid_zscore_normalize(
    values: np.ndarray, mean: float, std: float, k_value: float
) -> np.ndarray:
    if std == 0.0:
        raise ValueError("Cannot apply z-score sigmoid normalization with zero standard deviation.")
    logits = np.clip(k_value * ((values - mean) / std), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    percentile_values = np.percentile(values, DIAGNOSTIC_PERCENTILES)
    return {
        format_percentile(percentile): float(value)
        for percentile, value in zip(DIAGNOSTIC_PERCENTILES, percentile_values)
    }


def interval_ratio_dict(values: np.ndarray) -> dict[str, float]:
    return {
        interval_label(low, high): normalized_interval_ratio(values, low, high)
        for low, high in NORMALIZED_RATIO_INTERVALS
    }


def normalized_interval_ratio(values: np.ndarray, low: float, high: float) -> float:
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
    training_values: dict[str, np.ndarray],
    diagnostics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_one_axis_distribution(
        plots_dir / "dx_distribution.png",
        raw_values=training_values["dx"],
        normalized_values=normalize_to_unit(
            training_values["dx"], diagnostics["dx_min"], diagnostics["dx_max"]
        ),
        axis_name="dx",
    )
    plot_one_axis_distribution(
        plots_dir / "dy_distribution.png",
        raw_values=training_values["dy"],
        normalized_values=normalize_to_unit(
            training_values["dy"], diagnostics["dy_min"], diagnostics["dy_max"]
        ),
        axis_name="dy",
    )


def plot_one_axis_distribution(
    path: Path,
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    axis_name: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(raw_values, bins=100)
    axes[0].set_title(f"raw {axis_name}")
    axes[0].set_xlabel(axis_name)
    axes[0].set_ylabel("count")
    axes[1].hist(normalized_values, bins=100, range=(0.0, 1.0))
    axes[1].set_title(f"normalized {axis_name}")
    axes[1].set_xlabel(f"{axis_name}_norm")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(
    path: Path,
    frame: pd.DataFrame,
    samples: list[PairedSample],
    skipped_rows: list[dict[str, Any]],
    split_counts: dict[str, int],
    input_path: Path,
    id_column: str,
    x_column: str,
    y_column: str,
    time_column: str | None,
) -> None:
    row_counts = [sample.original_num_points for sample in samples]
    payload = {
        "input_path": str(input_path),
        "raw_shape": list(frame.shape),
        "raw_columns": list(frame.columns),
        "id_column": id_column,
        "x_column": x_column,
        "y_column": y_column,
        "time_column": time_column,
        "num_valid_ids": len(samples),
        "num_skipped_ids": len(skipped_rows),
        "num_output_samples": len(samples),
        "split_counts": split_counts,
        "points_per_valid_id": {
            "min": min(row_counts) if row_counts else 0,
            "max": max(row_counts) if row_counts else 0,
            "mean": float(np.mean(row_counts)) if row_counts else 0.0,
        },
        "max_reconstruction_error": (
            max(sample.max_reconstruction_error for sample in samples) if samples else 0.0
        ),
        "outputs": {
            "metadata": "metadata.csv",
            "unique_ids": "unique_vehicle_ids.csv",
            "skipped_ids": "skipped_ids.csv",
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


if __name__ == "__main__":
    main()

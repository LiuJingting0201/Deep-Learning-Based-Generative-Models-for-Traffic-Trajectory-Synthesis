"""Week3 Experiment A: channel structure statistics for pseudo-modal GAF images.

This script does not train any decoder. It only analyzes real RGB GAF PNG
images, where R=GASF, G=GADF, and B=MTF, to verify channel structure before
building structure-aware decoder models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/home/jliu/Thesis/week03/logs/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_DATA_DIR = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_OUTPUT_DIR = Path("/home/jliu/Thesis/week03/results/channel_structure_stats")
EXPECTED_SHAPE = (224, 224, 3)
KEY_SUMMARY_METRICS = [
    "gasf_symmetry_error_mean",
    "gasf_symmetry_error_rel_std",
    "gadf_antisymmetry_error_mean",
    "gadf_antisymmetry_error_rel_std",
    "mtf_symmetry_error_mean",
    "mtf_symmetry_error_rel_std",
    "gasf_residual_mean_abs",
    "gadf_residual_mean_abs",
    "gasf_residual_to_error_ratio",
    "gadf_residual_to_error_ratio",
    "gasf_diag_mean",
    "gadf_diag_mean",
    "mtf_diag_mean",
]
JSON_SUMMARY_METRICS = [
    "gasf_symmetry_error_mean",
    "gasf_symmetry_error_rel_std",
    "gadf_antisymmetry_error_mean",
    "gadf_antisymmetry_error_rel_std",
    "mtf_symmetry_error_mean",
    "mtf_symmetry_error_rel_std",
    "gasf_residual_mean_abs",
    "gadf_residual_mean_abs",
]


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir = output_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_image_paths(data_dir, args.max_samples)
    vehicle_ids = load_vehicle_ids(data_dir / "metadata.csv")
    visualization_paths = select_visualization_paths(
        image_paths, args.num_visualizations, args.seed
    )

    rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        channels = load_theory_channels(image_path)
        sample_id = image_path.stem
        row = compute_sample_metrics(
            sample_id=sample_id,
            image_path=image_path,
            data_dir=data_dir,
            channels=channels,
            vehicle_id=vehicle_ids.get(sample_id, ""),
        )
        rows.append(row)
        if image_path in visualization_paths:
            save_structure_visualization(
                visualizations_dir / f"{sample_id}_structure_components.png",
                sample_id,
                channels,
            )

    csv_path = output_dir / "channel_structure_stats.csv"
    write_csv(csv_path, rows)

    aggregates = summarize_rows(rows)
    write_markdown_summary(
        output_dir / "channel_structure_stats_summary.md",
        data_dir=data_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        num_samples=len(rows),
        aggregates=aggregates,
    )
    write_json_summary(
        output_dir / "channel_structure_stats_summary.json",
        data_dir=data_dir,
        output_dir=output_dir,
        num_samples=len(rows),
        aggregates=aggregates,
    )

    print(f"Processed {len(rows)} images")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {output_dir / 'channel_structure_stats_summary.md'}")
    print(f"JSON: {output_dir / 'channel_structure_stats_summary.json'}")
    print(f"Visualizations: {visualizations_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_image_paths(data_dir: Path, max_samples: int | None) -> list[Path]:
    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    image_paths = sorted(images_dir.glob("*.png"), key=lambda path: path.name)
    if not image_paths:
        raise ValueError(f"No PNG images found in {images_dir}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided.")
        image_paths = image_paths[:max_samples]
    return image_paths


def load_vehicle_ids(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.exists():
        return {}
    metadata = pd.read_csv(metadata_path)
    if not {"sample_id", "vehicle_id"}.issubset(metadata.columns):
        return {}
    return {
        str(row.sample_id): str(row.vehicle_id)
        for row in metadata[["sample_id", "vehicle_id"]].itertuples(index=False)
    }


def select_visualization_paths(
    image_paths: list[Path], num_visualizations: int, seed: int
) -> set[Path]:
    if num_visualizations <= 0:
        return set()
    rng = random.Random(seed)
    selected_count = min(num_visualizations, len(image_paths))
    return set(rng.sample(image_paths, selected_count))


def load_theory_channels(image_path: Path) -> dict[str, np.ndarray]:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        array = np.asarray(image_rgb)
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{image_path} has shape {array.shape}; expected {EXPECTED_SHAPE}."
        )
    png = array.astype(np.float64) / 255.0
    return {
        "R": png[:, :, 0] * 2.0 - 1.0,
        "G": png[:, :, 1] * 2.0 - 1.0,
        "B": png[:, :, 2],
    }


def compute_sample_metrics(
    sample_id: str,
    image_path: Path,
    data_dir: Path,
    channels: dict[str, np.ndarray],
    vehicle_id: str,
) -> dict[str, Any]:
    r = channels["R"]
    g = channels["G"]
    b = channels["B"]

    r_sym = symmetric_component(r)
    g_antisym = antisymmetric_component(g)
    r_sym_error = np.abs(r - r.T)
    g_antisym_error = np.abs(g + g.T)
    b_sym_error = np.abs(b - b.T)
    r_std = float(r.std())
    g_std = float(g.std())
    b_std = float(b.std())
    gasf_symmetry_error_mean = float(r_sym_error.mean())
    gadf_antisymmetry_error_mean = float(g_antisym_error.mean())
    mtf_symmetry_error_mean = float(b_sym_error.mean())
    gasf_residual_mean_abs = float(np.abs(r - r_sym).mean())
    gadf_residual_mean_abs = float(np.abs(g - g_antisym).mean())

    row: dict[str, Any] = {
        "sample_id": sample_id,
        "vehicle_id": vehicle_id,
        "image_path": relative_or_absolute(image_path, data_dir),
        "image_shape": "224x224x3",
        "gasf_symmetry_error_mean": gasf_symmetry_error_mean,
        "gasf_symmetry_error_max": float(r_sym_error.max()),
        "gasf_symmetry_error_rel_std": gasf_symmetry_error_mean / (r_std + 1e-12),
        "gasf_sym_component_mean": float(r_sym.mean()),
        "gasf_residual_mean_abs": gasf_residual_mean_abs,
        "gasf_residual_to_error_ratio": gasf_residual_mean_abs
        / (gasf_symmetry_error_mean + 1e-12),
        "gasf_diag_mean": float(np.diag(r).mean()),
        "gasf_diag_std": float(np.diag(r).std()),
        "gasf_mean": float(r.mean()),
        "gasf_std": r_std,
        "gasf_min": float(r.min()),
        "gasf_max": float(r.max()),
        "gadf_antisymmetry_error_mean": gadf_antisymmetry_error_mean,
        "gadf_antisymmetry_error_max": float(g_antisym_error.max()),
        "gadf_antisymmetry_error_rel_std": gadf_antisymmetry_error_mean
        / (g_std + 1e-12),
        "gadf_antisym_component_mean": float(g_antisym.mean()),
        "gadf_residual_mean_abs": gadf_residual_mean_abs,
        "gadf_residual_to_error_ratio": gadf_residual_mean_abs
        / (gadf_antisymmetry_error_mean + 1e-12),
        "gadf_diag_mean": float(np.diag(g).mean()),
        "gadf_diag_std": float(np.diag(g).std()),
        "gadf_mean": float(g.mean()),
        "gadf_std": g_std,
        "gadf_min": float(g.min()),
        "gadf_max": float(g.max()),
        "mtf_symmetry_error_mean": mtf_symmetry_error_mean,
        "mtf_symmetry_error_max": float(b_sym_error.max()),
        "mtf_symmetry_error_rel_std": mtf_symmetry_error_mean / (b_std + 1e-12),
        "mtf_diag_mean": float(np.diag(b).mean()),
        "mtf_diag_std": float(np.diag(b).std()),
        "mtf_mean": float(b.mean()),
        "mtf_std": b_std,
        "mtf_min": float(b.min()),
        "mtf_max": float(b.max()),
    }
    return row


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def symmetric_component(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values + values.T)


def antisymmetric_component(values: np.ndarray) -> np.ndarray:
    return 0.5 * (values - values.T)


def local_average_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, pad_width=1, mode="edge")
    total = np.zeros_like(values, dtype=np.float64)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded[
                row_offset : row_offset + values.shape[0],
                col_offset : col_offset + values.shape[1],
            ]
    return total / 9.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"sample_id", "vehicle_id", "image_path", "image_shape"}
    ]
    summary: dict[str, dict[str, float]] = {}
    for column in metric_columns:
        values = frame[column].to_numpy(dtype=np.float64)
        summary[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def write_markdown_summary(
    path: Path,
    data_dir: Path,
    output_dir: Path,
    max_samples: int | None,
    num_samples: int,
    aggregates: dict[str, dict[str, float]],
) -> None:
    lines = [
        "# Week3 Experiment A: Channel Structure Statistics",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Output path: `{output_dir}`",
        f"- Processed samples: `{num_samples}`",
        f"- max_samples setting: `{max_samples}`",
        "",
        "## Image Scale Convention",
        "",
        "PNG channels are stored as uint8 values in `[0, 255]`. For mathematical structure checks, GASF and GADF are mapped back to their theoretical `[-1, 1]` scale, while MTF remains in `[0, 1]`:",
        "",
        "- `R_theory = R_png / 255 * 2 - 1`",
        "- `G_theory = G_png / 255 * 2 - 1`",
        "- `B_theory = B_png / 255`",
        "",
        "R/G are converted to `[-1, 1]` because GASF symmetry and GADF anti-symmetry are defined around zero. Computing GADF anti-symmetry directly in PNG `[0, 1]` space would be mathematically wrong. B remains in `[0, 1]` because MTF is a transition/statistical field rather than a zero-centered GAF matrix.",
        "",
        "## Key Metrics Over Samples",
        "",
        "| metric | mean | std | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in KEY_SUMMARY_METRICS:
        stats = aggregates[metric]
        lines.append(
            f"| `{metric}` | {stats['mean']:.8g} | {stats['std']:.8g} | "
            f"{stats['min']:.8g} | {stats['max']:.8g} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If GASF symmetry error is very small, hard or decomposed symmetric projection is justified.",
            "",
            "If GADF anti-symmetry error is very small, hard or decomposed anti-symmetric projection is justified.",
            "",
            "If errors are not small, prefer decomposition rather than hard projection, because residual information may be useful.",
            "",
            "MTF symmetry error should be treated only as a diagnostic, not as a reason to hard-enforce symmetry.",
            "",
            "`gasf_residual_to_error_ratio` and `gadf_residual_to_error_ratio` should be close to `0.5` by construction. They are included as a sanity check that the residual definitions match the corresponding symmetry and anti-symmetry error definitions.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_json_summary(
    path: Path,
    data_dir: Path,
    output_dir: Path,
    num_samples: int,
    aggregates: dict[str, dict[str, float]],
) -> None:
    payload = {
        "num_samples": num_samples,
        "data_dir": data_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "metrics": {metric: aggregates[metric] for metric in JSON_SUMMARY_METRICS},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_structure_visualization(
    path: Path, sample_id: str, channels: dict[str, np.ndarray]
) -> None:
    r = channels["R"]
    g = channels["G"]
    b = channels["B"]
    r_sym = symmetric_component(r)
    g_antisym = antisymmetric_component(g)
    b_local_avg = local_average_3x3(b)

    panels = [
        [
            ("GASF raw", r, "coolwarm", -1.0, 1.0),
            ("GASF sym", r_sym, "coolwarm", -1.0, 1.0),
            ("GASF residual", r - r_sym, "coolwarm", None, None),
            ("GASF symmetry error", np.abs(r - r.T), "magma", 0.0, None),
        ],
        [
            ("GADF raw", g, "coolwarm", -1.0, 1.0),
            ("GADF antisym", g_antisym, "coolwarm", -1.0, 1.0),
            ("GADF residual", g - g_antisym, "coolwarm", None, None),
            ("GADF anti-sym error", np.abs(g + g.T), "magma", 0.0, None),
        ],
        [
            ("MTF raw", b, "viridis", 0.0, 1.0),
            ("MTF local avg 3x3", b_local_avg, "viridis", 0.0, 1.0),
            ("MTF minus local avg", b - b_local_avg, "coolwarm", None, None),
            ("MTF symmetry error diagnostic", np.abs(b - b.T), "magma", 0.0, None),
        ],
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"{sample_id} structure components", fontsize=14)
    for row_index, row in enumerate(panels):
        for col_index, (title, values, cmap, vmin, vmax) in enumerate(row):
            axis = axes[row_index, col_index]
            image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

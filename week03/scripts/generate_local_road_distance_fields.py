"""Generate local road-distance fields from rasterized OSM map crops."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK03_ROOT = PROJECT_ROOT / "week03"
DEFAULT_DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_REPORT_PATH = WEEK03_ROOT / "docs" / "local_road_distance_fields.md"
DEFAULT_DEBUG_DIR = WEEK03_ROOT / "results" / "distance_field_debug"
IMAGE_SIZE = 224


@dataclass(frozen=True)
class DistanceFieldJob:
    crop_mode: str
    map_dir: Path
    extent_csv: Path
    output_dir: Path
    metadata_csv: Path


def main() -> None:
    args = parse_args()
    np, pd, Image, distance_transform_edt = import_runtime_dependencies()
    jobs = resolve_jobs(args)

    summaries = []
    for job in jobs:
        summaries.append(
            generate_for_job(
                np=np,
                pd=pd,
                image_class=Image,
                distance_transform_edt=distance_transform_edt,
                job=job,
                debug_dir=args.debug_dir.resolve(),
                debug_samples=args.debug_samples,
                debug_clip_m=args.debug_clip_m,
                max_samples=args.max_samples,
            )
        )

    write_report(args.report_path.resolve(), summaries)
    for summary in summaries:
        print(
            f"Wrote {summary['num_fields']} {summary['crop_mode']} distance fields "
            f"to {summary['output_dir']}"
        )
        print(f"Wrote metadata: {summary['metadata_csv']}")
    print(f"Wrote report: {args.report_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--crop-mode",
        choices=["oracle_bbox", "start_center", "all"],
        default="all",
        help="Standard crop mode to process when --map-dir/--extent-csv are not both provided.",
    )
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--extent-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--debug-samples", type=int, default=20)
    parser.add_argument("--debug-clip-m", type=float, default=50.0)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def import_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        pd = None
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
        Image = None
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        missing.append("scipy")
        distance_transform_edt = None

    if missing:
        raise SystemExit(
            "Missing runtime package(s): "
            + ", ".join(sorted(set(missing)))
            + ". scipy.ndimage.distance_transform_edt is required for this preprocessing step."
        )
    return np, pd, Image, distance_transform_edt


def resolve_jobs(args: argparse.Namespace) -> list[DistanceFieldJob]:
    data_root = args.data_root.resolve()
    if args.map_dir is not None or args.extent_csv is not None:
        if args.map_dir is None or args.extent_csv is None:
            raise SystemExit("--map-dir and --extent-csv must be provided together.")
        crop_mode = infer_crop_mode(args.map_dir, args.extent_csv, args.crop_mode)
        return [
            DistanceFieldJob(
                crop_mode=crop_mode,
                map_dir=args.map_dir.resolve(),
                extent_csv=args.extent_csv.resolve(),
                output_dir=resolve_output_dir(args.output_dir, data_root, crop_mode),
                metadata_csv=resolve_metadata_csv(args.metadata_csv, data_root, crop_mode),
            )
        ]

    crop_modes = ["oracle_bbox", "start_center"] if args.crop_mode == "all" else [args.crop_mode]
    return [
        DistanceFieldJob(
            crop_mode=crop_mode,
            map_dir=data_root / f"maps_{crop_mode}",
            extent_csv=data_root / f"maps_{crop_mode}_extents.csv",
            output_dir=resolve_output_dir(args.output_dir, data_root, crop_mode),
            metadata_csv=resolve_metadata_csv(args.metadata_csv, data_root, crop_mode),
        )
        for crop_mode in crop_modes
    ]


def resolve_output_dir(output_dir: Path | None, data_root: Path, crop_mode: str) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return data_root / f"maps_{crop_mode}_distance_fields"


def resolve_metadata_csv(metadata_csv: Path | None, data_root: Path, crop_mode: str) -> Path:
    if metadata_csv is not None:
        return metadata_csv.resolve()
    return data_root / f"maps_{crop_mode}_distance_fields_metadata.csv"


def infer_crop_mode(map_dir: Path | None, extent_csv: Path | None, fallback: str) -> str:
    for path in [map_dir, extent_csv]:
        if path is None:
            continue
        name = path.name
        if "oracle_bbox" in name:
            return "oracle_bbox"
        if "start_center" in name:
            return "start_center"
    if fallback == "all":
        raise SystemExit("--crop-mode must be oracle_bbox or start_center when custom paths are used.")
    return fallback


def generate_for_job(
    np: Any,
    pd: Any,
    image_class: Any,
    distance_transform_edt: Any,
    job: DistanceFieldJob,
    debug_dir: Path,
    debug_samples: int,
    debug_clip_m: float,
    max_samples: int | None,
) -> dict[str, Any]:
    if not job.map_dir.is_dir():
        raise FileNotFoundError(f"Map raster directory not found: {job.map_dir}")
    if not job.extent_csv.exists():
        raise FileNotFoundError(f"Extent metadata CSV not found: {job.extent_csv}")

    extents = pd.read_csv(job.extent_csv)
    required = {"sample_id", "min_x", "max_x", "min_y", "max_y"}
    missing = required.difference(extents.columns)
    if missing:
        raise ValueError(f"{job.extent_csv} is missing required columns: {sorted(missing)}")
    if "crop_mode" in extents.columns:
        modes = sorted(set(extents["crop_mode"].astype(str)))
        if len(modes) == 1 and modes[0] != job.crop_mode:
            print(
                f"WARNING: extent crop_mode is {modes[0]!r}, but output crop_mode is {job.crop_mode!r}.",
                file=sys.stderr,
            )

    job.output_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / job.crop_mode).mkdir(parents=True, exist_ok=True)

    rows = []
    extent_rows = extents.reset_index(drop=True)
    if max_samples is not None:
        extent_rows = extent_rows.head(max_samples)

    missing_maps = 0
    empty_road_maps = 0
    debug_written = 0
    for row in extent_rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        map_path = job.map_dir / f"{sample_id}_map.png"
        if not map_path.exists():
            missing_maps += 1
            continue

        image = image_class.open(map_path).convert("L")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} map for {sample_id}, got {image.size}")
        road = np.asarray(image, dtype=np.uint8) > 0
        meters_per_pixel_x = (float(row.max_x) - float(row.min_x)) / IMAGE_SIZE
        meters_per_pixel_y = (float(row.max_y) - float(row.min_y)) / IMAGE_SIZE
        meters_per_pixel_used = 0.5 * (meters_per_pixel_x + meters_per_pixel_y)

        if road.any():
            distance_m = distance_transform_edt(~road).astype(np.float32) * np.float32(meters_per_pixel_used)
        else:
            empty_road_maps += 1
            distance_m = np.full((IMAGE_SIZE, IMAGE_SIZE), np.nan, dtype=np.float32)

        out_path = job.output_dir / f"{sample_id}_distance.npy"
        np.save(out_path, distance_m.astype(np.float32))
        max_distance = float(np.nanmax(distance_m)) if np.isfinite(distance_m).any() else float("nan")
        mean_distance = float(np.nanmean(distance_m)) if np.isfinite(distance_m).any() else float("nan")
        rows.append(
            {
                "sample_id": sample_id,
                "distance_path": str(out_path),
                "min_x": float(row.min_x),
                "max_x": float(row.max_x),
                "min_y": float(row.min_y),
                "max_y": float(row.max_y),
                "meters_per_pixel_x": float(meters_per_pixel_x),
                "meters_per_pixel_y": float(meters_per_pixel_y),
                "meters_per_pixel_used": float(meters_per_pixel_used),
                "max_distance": max_distance,
                "mean_distance": mean_distance,
            }
        )

        if debug_written < debug_samples:
            debug_path = debug_dir / job.crop_mode / f"{sample_id}_distance_debug.png"
            save_debug_png(np, image_class, distance_m, road, debug_path, debug_clip_m)
            debug_written += 1

    write_metadata_csv(job.metadata_csv, rows)
    return {
        "crop_mode": job.crop_mode,
        "source_map_dir": str(job.map_dir),
        "extent_csv": str(job.extent_csv),
        "output_dir": str(job.output_dir),
        "metadata_csv": str(job.metadata_csv),
        "num_extent_rows": int(len(extent_rows)),
        "num_fields": int(len(rows)),
        "num_missing_maps": int(missing_maps),
        "num_empty_road_maps": int(empty_road_maps),
        "debug_dir": str(debug_dir / job.crop_mode),
        "debug_samples": int(debug_written),
        "debug_clip_m": float(debug_clip_m),
        "distance_unit": "meters",
    }


def save_debug_png(np: Any, image_class: Any, distance_m: Any, road: Any, path: Path, clip_m: float) -> None:
    clipped = np.nan_to_num(distance_m, nan=clip_m, posinf=clip_m, neginf=0.0)
    normalized = np.clip(clipped / clip_m, 0.0, 1.0)
    pixels = np.round(normalized * 255.0).astype(np.uint8)
    pixels[road] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    image_class.fromarray(pixels, mode="L").save(path)


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "distance_path",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "meters_per_pixel_x",
        "meters_per_pixel_y",
        "meters_per_pixel_used",
        "max_distance",
        "mean_distance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    sections = []
    for summary in summaries:
        sections.append(
            f"""## {summary["crop_mode"]}

- Source map dir: `{summary["source_map_dir"]}`
- Extent metadata CSV: `{summary["extent_csv"]}`
- Output distance field dir: `{summary["output_dir"]}`
- Output metadata CSV: `{summary["metadata_csv"]}`
- Fields generated: {summary["num_fields"]}
- Extent rows considered: {summary["num_extent_rows"]}
- Missing map rasters skipped: {summary["num_missing_maps"]}
- Empty-road rasters: {summary["num_empty_road_maps"]}
- Distance unit: {summary["distance_unit"]}
- Debug visualizations: `{summary["debug_dir"]}` ({summary["debug_samples"]} sample(s), clipped at {summary["debug_clip_m"]} m)
"""
        )

    text = f"""# Local Road-Distance Fields

This preprocessing converts local binary road rasters into float32 Euclidean distance fields. Road pixels are map pixels with value greater than zero; every non-road pixel stores approximate distance in meters to the nearest road pixel.

{"".join(sections)}
## Intended Use

These fields are intended for a differentiable pointwise road-distance loss by sampling the local distance image at predicted trajectory coordinates with `grid_sample`. This commit only prepares the fields; it does not modify training code or add a loss term.

## Limitations

- Distances are computed in raster space, so they inherit the 224x224 crop resolution and road line width used during map rasterization.
- When `meters_per_pixel_x` and `meters_per_pixel_y` differ, the saved distance uses the approximate isotropic scale `0.5 * (meters_per_pixel_x + meters_per_pixel_y)`.
- Oracle-bbox fields depend on ground-truth trajectory extent and should be treated differently from start-center fields in any deployable setting.
- Empty-road crops are saved as `NaN` fields and should be filtered or handled explicitly before use.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    main()

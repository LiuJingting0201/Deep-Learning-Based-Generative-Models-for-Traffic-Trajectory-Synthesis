"""Run parameter sweeps for map-aware trajectory refinement post-processing."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "week03" / "results" / "baseline_raw_resnet18_bs16" / "evaluation"
DEFAULT_DATA_ROOT = Path("/home/jliu/data_no_speed_delta_displacement_paired")
DEFAULT_AFFINE_JSON = PROJECT_ROOT / "week03" / "data" / "maps" / "raw_xy_to_osm_affine.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "week03" / "results" / "refinement_sweep_oracle"
REFINER_SCRIPT = PROJECT_ROOT / "week03" / "scripts" / "refine_trajectory_map_aware_sliding_window.py"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    max_samples = None if args.full_run else args.max_samples

    rows = []
    configs = list(
        itertools.product(
            args.search_radius_values,
            args.w_road_values,
            args.w_ref_values,
            args.w_curvature_values,
        )
    )
    for index, (radius, w_road, w_ref, w_curvature) in enumerate(configs, start=1):
        run_name = config_name(radius, w_road, w_ref, w_curvature)
        run_dir = output_root / run_name
        print(f"===== sweep {index}/{len(configs)}: {run_name} =====", flush=True)
        command = [
            sys.executable,
            str(REFINER_SCRIPT),
            "--evaluation-dir",
            str(args.evaluation_dir.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
            "--crop-mode",
            args.crop_mode,
            "--distance-field-dir",
            str(args.distance_field_dir.resolve()),
            "--distance-field-metadata",
            str(args.distance_field_metadata.resolve()),
            "--raw-to-osm-affine-json",
            str(args.raw_to_osm_affine_json.resolve()),
            "--output-dir",
            str(run_dir),
            "--method",
            "pointwise_grid",
            "--search-radius-m",
            str(radius),
            "--search-step-m",
            str(args.search_step_m),
            "--w-road",
            str(w_road),
            "--w-ref",
            str(w_ref),
            "--w-curvature",
            str(w_curvature),
            "--w-step",
            str(args.w_step),
            "--w-heading",
            str(args.w_heading),
            "--w-terminal",
            str(args.w_terminal),
            "--terminal-power",
            str(args.terminal_power),
            "--debug-samples",
            str(args.debug_samples),
        ]
        if max_samples is not None:
            command.extend(["--max-samples", str(max_samples)])
        if args.plot_distance_background:
            command.extend(["--plot-distance-background", "--background-clip-m", str(args.background_clip_m)])
        subprocess.run(command, check=True)
        rows.append(sweep_row(run_name, run_dir, radius, w_road, w_ref, w_curvature))

    ranked = rank_rows(rows)
    write_csv(output_root / "sweep_summary.csv", ranked)
    write_markdown(output_root / "sweep_summary.md", ranked, args, max_samples)
    print(f"Wrote sweep summary: {output_root / 'sweep_summary.csv'}")
    print(f"Wrote sweep report: {output_root / 'sweep_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--crop-mode", choices=["oracle_bbox", "start_center"], required=True)
    parser.add_argument("--distance-field-dir", type=Path, required=True)
    parser.add_argument("--distance-field-metadata", type=Path, required=True)
    parser.add_argument("--raw-to-osm-affine-json", type=Path, default=DEFAULT_AFFINE_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--full-run", action="store_true", help="Ignore --max-samples and process all samples.")
    parser.add_argument("--debug-samples", type=int, default=10)
    parser.add_argument("--search-radius-values", type=float, nargs="+", default=[10.0, 15.0, 25.0])
    parser.add_argument("--w-road-values", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--w-ref-values", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--w-curvature-values", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--search-step-m", type=float, default=2.0)
    parser.add_argument("--w-step", type=float, default=0.1)
    parser.add_argument("--w-heading", type=float, default=0.05)
    parser.add_argument("--w-terminal", type=float, default=0.2)
    parser.add_argument("--terminal-power", type=float, default=2.0)
    parser.add_argument("--plot-distance-background", action="store_true")
    parser.add_argument("--background-clip-m", type=float, default=50.0)
    return parser.parse_args()


def config_name(radius: float, w_road: float, w_ref: float, w_curvature: float) -> str:
    return f"r{fmt(radius)}_wr{fmt(w_road)}_wref{fmt(w_ref)}_wc{fmt(w_curvature)}"


def fmt(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def sweep_row(
    run_name: str,
    run_dir: Path,
    radius: float,
    w_road: float,
    w_ref: float,
    w_curvature: float,
) -> dict[str, Any]:
    summary = json.loads((run_dir / "refinement_summary.json").read_text())
    raw = summary["means"]["raw"]
    refined = summary["means"]["refined"]
    comp = summary["comparisons"]
    ade_change = comp["ADE_change_refined_minus_raw"]
    row = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "search_radius_m": radius,
        "w_road": w_road,
        "w_ref": w_ref,
        "w_curvature": w_curvature,
        "w_terminal": summary["parameters"].get("w_terminal"),
        "terminal_power": summary["parameters"].get("terminal_power"),
        "num_samples": summary["num_samples"],
        "raw_integrated_ADE": raw["integrated_ADE"],
        "refined_integrated_ADE": refined["integrated_ADE"],
        "ADE_change_refined_minus_raw": ade_change,
        "raw_integrated_FDE": raw["integrated_FDE"],
        "refined_integrated_FDE": refined["integrated_FDE"],
        "FDE_change_refined_minus_raw": comp["FDE_change_refined_minus_raw"],
        "raw_mean_road_distance": raw["mean_road_distance"],
        "refined_mean_road_distance": refined["mean_road_distance"],
        "road_distance_improvement": comp["road_distance_improvement_raw_minus_refined"],
        "raw_offroad_ratio_10m": raw["offroad_ratio_10m"],
        "refined_offroad_ratio_10m": refined["offroad_ratio_10m"],
        "offroad_ratio_10m_improvement": comp["offroad_ratio_10m_improvement_raw_minus_refined"],
        "trajectory_length_ratio_change": comp["trajectory_length_ratio_change_refined_minus_raw"],
        "ADE_degradation": max(0.0, ade_change),
    }
    row["acceptable"] = row["offroad_ratio_10m_improvement"] > 0.10 and row["ADE_degradation"] < 5.0
    return row


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["offroad_ratio_10m_improvement"]),
            float(row["ADE_degradation"]),
            float(row["refined_mean_road_distance"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = ["rank"] + [key for key in rows[0].keys() if key != "rank"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, max_samples: int | None) -> None:
    lines = [
        "# Refinement Parameter Sweep",
        "",
        "- task: map-aware trajectory refinement post-processing",
        f"- crop_mode: {args.crop_mode}",
        f"- max_samples: {max_samples if max_samples is not None else 'all'}",
        "- ranking: maximize offroad_ratio_10m improvement, then minimize ADE degradation, then minimize refined mean road distance",
        "- acceptable: offroad_ratio_10m improvement > 0.10 and ADE degradation < 5.0",
        "",
        "| rank | acceptable | run | offroad10 improvement | ADE change | ADE degradation | refined road mean | road improvement | FDE change | length-ratio change |",
        "|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {'yes' if row['acceptable'] else 'no'} | {row['run_name']} | "
            f"{row['offroad_ratio_10m_improvement']:.6f} | "
            f"{row['ADE_change_refined_minus_raw']:.6f} | "
            f"{row['ADE_degradation']:.6f} | "
            f"{row['refined_mean_road_distance']:.6f} | "
            f"{row['road_distance_improvement']:.6f} | "
            f"{row['FDE_change_refined_minus_raw']:.6f} | "
            f"{row['trajectory_length_ratio_change']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

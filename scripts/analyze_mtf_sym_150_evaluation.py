"""Six-way 150-step MTFSym structural-prior evaluation.

This report intentionally reuses existing raw and ChannelNorm baseline samples
and FID JSONs. It only expects new MTFSym sample directories/FIDs to be present.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_ddpm_50k_comparison_fast import TARGET_RGB, analyze_set, list_pngs
from analyze_gaf_channel_diagnostics import CHANNELS, per_channel_symmetry, streaming_channel_stats, write_csv, write_heatmaps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = Path("/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images")
MAIN_OUT = PROJECT_ROOT / "results_hpc/mtfSym_50k_sampling_steps_150_analysis"
CHANNEL_OUT = PROJECT_ROOT / "results_hpc/mtfSym_channel_diagnostics_150"

SAMPLE_DIRS = {
    "raw_pureMSE": PROJECT_ROOT / "results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_150steps",
    "raw_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_150steps",
    "channelNorm_pureMSE": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_channelNorm_pureMSE/samples_150steps_matched",
    "channelNorm_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed/samples_150steps_matched",
    "mtfSym_pureMSE": PROJECT_ROOT / "results_hpc/ddpm_scratch_50000step_bs8_mtfSym_pureMSE/samples_150steps",
    "mtfSym_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed/samples_150steps",
}

FID_JSONS = {
    "raw_pureMSE": PROJECT_ROOT / "results_hpc/ddpm_50k_sampling_steps_150_analysis/fid_pureMSE_150_vs_real.json",
    "raw_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_50k_sampling_steps_150_analysis/fid_dataDrivenDiag_150_vs_real.json",
    "channelNorm_pureMSE": PROJECT_ROOT / "results_hpc/channelNorm_50k_sampling_steps_150_analysis/fid_pureMSE.json",
    "channelNorm_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/channelNorm_50k_sampling_steps_150_analysis/fid_dataDrivenDiag.json",
    "mtfSym_pureMSE": MAIN_OUT / "fid_mtfSym_pureMSE.json",
    "mtfSym_dataDrivenDiag": MAIN_OUT / "fid_mtfSym_dataDrivenDiag.json",
}


def main() -> None:
    args = parse_args()
    main_out = args.output_dir.resolve()
    channel_out = args.channel_output_dir.resolve()
    real_dir = args.real_dir.resolve()
    sample_dirs = SAMPLE_DIRS.copy()
    fid_jsons = FID_JSONS.copy()

    main_out.mkdir(parents=True, exist_ok=True)
    channel_out.mkdir(parents=True, exist_ok=True)

    paths = {"real": list_pngs(real_dir)}
    paths.update({name: list_pngs(path) for name, path in sample_dirs.items()})
    validate_counts(paths)

    image_metrics = {"real": analyze_set("real", paths["real"], diagnostic_limit=args.real_diagnostic_limit)}
    image_metrics.update({name: analyze_set(name, path_list) for name, path_list in paths.items() if name != "real"})

    channel_stats = {name: streaming_channel_stats(path_list) for name, path_list in paths.items()}
    channel_symmetry = {name: per_channel_symmetry(path_list) for name, path_list in paths.items()}
    fids = {name: load_fid(path) for name, path in fid_jsons.items()}

    real_mean = np.array(channel_stats["real"]["mean"], dtype=np.float64)
    rows = build_rows(image_metrics, channel_stats, channel_symmetry, fids, real_mean)
    write_main_outputs(main_out, rows, image_metrics, channel_stats, channel_symmetry, fids, sample_dirs, real_dir)
    write_channel_outputs(channel_out, channel_stats, channel_symmetry, sample_dirs, real_dir)
    make_grid(main_out / "representative_comparison_grid.png", sample_dirs)
    write_markdown_summary(main_out / "summary.md", rows)

    print(json.dumps({"main_output": str(main_out), "channel_diagnostics": str(channel_out)}, indent=2))


def validate_counts(paths: dict[str, list[Path]]) -> None:
    missing = [name for name, path_list in paths.items() if not path_list]
    if missing:
        raise ValueError(f"No PNG images found for: {missing}")
    for name, path_list in paths.items():
        if name != "real" and len(path_list) != 80:
            raise ValueError(f"{name} expected 80 generated PNGs, found {len(path_list)}")


def load_fid(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    value = payload.get("fid")
    return None if value is None else float(value)


def build_rows(
    image_metrics: dict[str, dict[str, object]],
    channel_stats: dict[str, dict[str, object]],
    channel_symmetry: dict[str, dict[str, list[float]]],
    fids: dict[str, float | None],
    real_mean: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for name in ["real", *SAMPLE_DIRS.keys()]:
        img = image_metrics[name]
        stats = channel_stats[name]
        sym = channel_symmetry[name]
        corr = np.array(stats["correlation"], dtype=np.float64)
        rgb_mean = np.array(stats["mean"], dtype=np.float64)
        diag_mean = np.array(img["diag_rgb_mean"], dtype=np.float64)
        row = {
            "set": name,
            "num_images": img["num_images"],
            "fid": "" if name == "real" else fids.get(name),
            "rgb_mean": format_vector(rgb_mean),
            "rgb_mean_l2_to_real": "" if name == "real" else float(np.linalg.norm(rgb_mean - real_mean)),
            "diag_rgb_mean": format_vector(diag_mean),
            "diag_target_l2": "" if name == "real" else float(np.linalg.norm(diag_mean - TARGET_RGB)),
            "global_symmetry": img["symmetry_error"]["mean"],
            "gasf_symmetry": sym["mean"][0],
            "gadf_symmetry": sym["mean"][1],
            "mtf_b_symmetry": sym["mean"][2],
            "corr_gasf_mtf": float(corr[0, 2]),
            "edge_strength": img["edge_strength"]["mean"],
            "saturation_rgb": format_vector(stats["saturation_ratio_abs_gt_0_98"]),
        }
        rows.append(row)
    return rows


def format_vector(values: np.ndarray | list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    return "[" + ", ".join(f"{x:.4f}" for x in arr.tolist()) + "]"


def write_main_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    image_metrics: dict[str, dict[str, object]],
    channel_stats: dict[str, dict[str, object]],
    channel_symmetry: dict[str, dict[str, list[float]]],
    fids: dict[str, float | None],
    sample_dirs: dict[str, Path],
    real_dir: Path,
) -> None:
    write_csv(out_dir / "metrics_summary.csv", rows)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "analysis_note": "150-step DDPM comparison. Baseline samples/FIDs are reused; only MTFSym FIDs are new.",
                "real_dir": str(real_dir),
                "sample_dirs": {name: str(path) for name, path in sample_dirs.items()},
                "fid_jsons": {name: str(path) for name, path in FID_JSONS.items()},
                "target_diag_rgb": TARGET_RGB.astype(float).tolist(),
                "targets": {"mtf_b_symmetry": 0.0673, "corr_gasf_mtf": -0.1559},
                "rows": rows,
                "image_metrics": image_metrics,
                "channel_stats": channel_stats,
                "channel_symmetry": channel_symmetry,
                "fids": fids,
            },
            indent=2,
        )
    )


def write_channel_outputs(
    out_dir: Path,
    stats: dict[str, dict[str, object]],
    symmetry: dict[str, dict[str, list[float]]],
    sample_dirs: dict[str, Path],
    real_dir: Path,
) -> None:
    channel_rows = []
    saturation_rows = []
    symmetry_rows = []
    for set_name, set_stats in stats.items():
        for i, channel in enumerate(CHANNELS):
            channel_rows.append(
                {
                    "set": set_name,
                    "channel": channel,
                    "mean": set_stats["mean"][i],
                    "std": set_stats["std"][i],
                    "q01": set_stats["q01"][i],
                    "q50": set_stats["q50"][i],
                    "q99": set_stats["q99"][i],
                }
            )
            saturation_rows.append(
                {
                    "set": set_name,
                    "channel": channel,
                    "saturation_ratio_abs_gt_0_98": set_stats["saturation_ratio_abs_gt_0_98"][i],
                }
            )
            symmetry_rows.append(
                {
                    "set": set_name,
                    "channel": channel,
                    "symmetry_error_mean": symmetry[set_name]["mean"][i],
                    "symmetry_error_std": symmetry[set_name]["std"][i],
                }
            )
    write_csv(out_dir / "channel_stats.csv", channel_rows)
    write_csv(out_dir / "channel_saturation.csv", saturation_rows)
    write_csv(out_dir / "channel_symmetry.csv", symmetry_rows)

    corr_rows = []
    pairs = [("GASF_R", "GADF_G", 0, 1), ("GASF_R", "MTF_B", 0, 2), ("GADF_G", "MTF_B", 1, 2)]
    for set_name, set_stats in stats.items():
        corr = np.array(set_stats["correlation"], dtype=np.float64)
        for a, b, i, j in pairs:
            corr_rows.append({"set": set_name, "channel_a": a, "channel_b": b, "correlation": float(corr[i, j])})
    write_csv(out_dir / "channel_correlation.csv", corr_rows)
    write_heatmaps(out_dir, stats)

    (out_dir / "channel_stats.json").write_text(
        json.dumps(
            {
                "channel_order": CHANNELS,
                "real_dir": str(real_dir),
                "sample_dirs": {name: str(path) for name, path in sample_dirs.items()},
                "stats": stats,
                "symmetry": symmetry,
            },
            indent=2,
        )
    )


def make_grid(out_path: Path, sample_dirs: dict[str, Path]) -> None:
    selected: list[tuple[str, Path]] = []
    for label, root in sample_dirs.items():
        choices = [
            root / "checkpoint-25000" / "checkpoint25000_sample000.png",
            root / "checkpoint-50000" / "checkpoint50000_sample000.png",
        ]
        if choices[0].exists():
            selected.append((f"{label} 25k", choices[0]))
        if choices[1].exists():
            selected.append((f"{label} 50k", choices[1]))
    if not selected:
        return

    tile_w, tile_h, label_h = 224, 224, 24
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    canvas = Image.new("RGB", (tile_w * cols, (tile_h + label_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (label, path) in enumerate(selected):
        x = (i % cols) * tile_w
        y = (i // cols) * (tile_h + label_h)
        draw.text((x + 6, y + 6), label[:34], fill=(0, 0, 0), font=font)
        canvas.paste(Image.open(path).convert("RGB").resize((tile_w, tile_h)), (x, y + label_h))
    canvas.save(out_path)


def write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    by_name = {row["set"]: row for row in rows}
    mtf_pure = by_name["mtfSym_pureMSE"]
    mtf_diag = by_name["mtfSym_dataDrivenDiag"]
    raw_pure = by_name["raw_pureMSE"]
    raw_diag = by_name["raw_dataDrivenDiag"]
    lines = [
        "# MTFSym 150-step evaluation",
        "",
        "Targets: real MTF/B symmetry about `0.0673`; real `corr(GASF,MTF)` about `-0.1559`.",
        "",
        "| Method | FID | RGB mean L2 | MTF/B sym | corr(GASF,MTF) | Diag target L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["set"] == "real":
            continue
        lines.append(
            f"| {row['set']} | {fmt(row['fid'])} | {fmt(row['rgb_mean_l2_to_real'])} | "
            f"{fmt(row['mtf_b_symmetry'])} | {fmt(row['corr_gasf_mtf'])} | {fmt(row['diag_target_l2'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation prompts",
            "",
            f"- MTFSym PureMSE MTF/B symmetry: `{fmt(mtf_pure['mtf_b_symmetry'])}` vs raw PureMSE `{fmt(raw_pure['mtf_b_symmetry'])}`.",
            f"- MTFSym DataDrivenDiag MTF/B symmetry: `{fmt(mtf_diag['mtf_b_symmetry'])}` vs raw DataDrivenDiag `{fmt(raw_diag['mtf_b_symmetry'])}`.",
            f"- MTFSym PureMSE FID: `{fmt(mtf_pure['fid'])}` vs raw PureMSE `{fmt(raw_pure['fid'])}`.",
            f"- MTFSym DataDrivenDiag FID: `{fmt(mtf_diag['fid'])}` vs raw DataDrivenDiag `{fmt(raw_diag['fid'])}`.",
            f"- DataDrivenDiag + MTFSym diagonal target distance: `{fmt(mtf_diag['diag_target_l2'])}` vs raw DataDrivenDiag `{fmt(raw_diag['diag_target_l2'])}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=REAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=MAIN_OUT)
    parser.add_argument("--channel-output-dir", type=Path, default=CHANNEL_OUT)
    parser.add_argument("--real-diagnostic-limit", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    main()

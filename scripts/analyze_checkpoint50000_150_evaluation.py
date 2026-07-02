"""Checkpoint-50000-only 150-step image statistics for six DDPM variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_ddpm_50k_comparison_fast import TARGET_RGB, analyze_set, list_pngs  # noqa: E402
from analyze_gaf_channel_diagnostics import (  # noqa: E402
    CHANNELS,
    per_channel_symmetry,
    streaming_channel_stats,
    write_csv,
    write_heatmaps,
)
from evaluate_fid import (  # noqa: E402
    FlatImageDataset,
    InceptionFeatureExtractor,
    compute_fid,
    compute_stats,
    get_features,
)


REAL_DIR = Path("/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images")
MAIN_OUT = PROJECT_ROOT / "results_hpc/checkpoint50000_sampling_steps_150_analysis"
CHANNEL_OUT = PROJECT_ROOT / "results_hpc/checkpoint50000_channel_diagnostics_150"

SAMPLE_DIRS = {
    "raw_pureMSE": PROJECT_ROOT / "results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_150steps/checkpoint-50000",
    "raw_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_150steps/checkpoint-50000",
    "channelNorm_pureMSE": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_channelNorm_pureMSE/samples_150steps_matched/checkpoint-50000",
    "channelNorm_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed/samples_150steps_matched/checkpoint-50000",
    "mtfSym_pureMSE": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_mtfSym_pureMSE/samples_150steps/checkpoint-50000",
    "mtfSym_dataDrivenDiag": PROJECT_ROOT
    / "results_hpc/ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed/samples_150steps/checkpoint-50000",
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    channel_output_dir = args.channel_output_dir.resolve()
    real_dir = args.real_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_output_dir.mkdir(parents=True, exist_ok=True)

    paths = {"real": list_pngs(real_dir)}
    paths.update({name: list_pngs(path) for name, path in SAMPLE_DIRS.items()})
    validate_counts(paths)

    image_metrics = {"real": analyze_set("real", paths["real"], diagnostic_limit=args.real_diagnostic_limit)}
    for name, path_list in paths.items():
        if name != "real":
            image_metrics[name] = analyze_set(name, path_list)

    channel_stats = {name: streaming_channel_stats(path_list) for name, path_list in paths.items()}
    channel_symmetry = {name: per_channel_symmetry(path_list) for name, path_list in paths.items()}
    fids = compute_fids(args, real_dir, SAMPLE_DIRS, output_dir)

    rows = build_rows(image_metrics, channel_stats, channel_symmetry, fids)
    write_outputs(output_dir, channel_output_dir, rows, image_metrics, channel_stats, channel_symmetry, fids, real_dir)
    make_grid(output_dir / "representative_comparison_grid.png")
    write_markdown_summary(output_dir / "summary.md", rows)

    print(json.dumps({"output_dir": str(output_dir), "channel_output_dir": str(channel_output_dir)}, indent=2))


def validate_counts(paths: dict[str, list[Path]]) -> None:
    if not paths["real"]:
        raise ValueError("No real images found.")
    for name, path_list in paths.items():
        if name == "real":
            continue
        if len(path_list) != 40:
            raise ValueError(f"{name} expected 40 checkpoint-50000 PNGs, found {len(path_list)}")


def compute_fids(
    args: argparse.Namespace,
    real_dir: Path,
    sample_dirs: dict[str, Path],
    output_dir: Path,
) -> dict[str, float]:
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = InceptionFeatureExtractor().to(device)

    real_dataset = FlatImageDataset(real_dir, transform)
    real_loader = DataLoader(real_dataset, batch_size=args.fid_batch_size, shuffle=False, num_workers=args.num_workers)
    real_features = get_features(real_loader, model, device)
    mu_real, sigma_real = compute_stats(real_features)

    fids: dict[str, float] = {}
    for name, generated_dir in sample_dirs.items():
        dataset = FlatImageDataset(generated_dir, transform)
        loader = DataLoader(dataset, batch_size=args.fid_batch_size, shuffle=False, num_workers=args.num_workers)
        generated_features = get_features(loader, model, device)
        mu_generated, sigma_generated = compute_stats(generated_features)
        fid = compute_fid(mu_real, sigma_real, mu_generated, sigma_generated)
        fids[name] = float(fid)
        (output_dir / f"fid_{name}.json").write_text(
            json.dumps(
                {
                    "fid": float(fid),
                    "real_dir": str(real_dir),
                    "generated_dir": str(generated_dir),
                    "checkpoint": "checkpoint-50000",
                    "sampling_steps": 150,
                    "num_real": len(real_dataset),
                    "num_generated": len(dataset),
                },
                indent=2,
            )
        )
        print(f"{name}: FID={fid:.6f}")
    return fids


def build_rows(
    image_metrics: dict[str, dict[str, object]],
    channel_stats: dict[str, dict[str, object]],
    channel_symmetry: dict[str, dict[str, list[float]]],
    fids: dict[str, float],
) -> list[dict[str, object]]:
    real_mean = np.array(channel_stats["real"]["mean"], dtype=np.float64)
    rows = []
    for name in ["real", *SAMPLE_DIRS.keys()]:
        img = image_metrics[name]
        stats = channel_stats[name]
        sym = channel_symmetry[name]
        corr = np.array(stats["correlation"], dtype=np.float64)
        rgb_mean = np.array(stats["mean"], dtype=np.float64)
        diag_mean = np.array(img["diag_rgb_mean"], dtype=np.float64)
        rows.append(
            {
                "set": name,
                "num_images": img["num_images"],
                "fid": "" if name == "real" else fids[name],
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
        )
    return rows


def write_outputs(
    output_dir: Path,
    channel_output_dir: Path,
    rows: list[dict[str, object]],
    image_metrics: dict[str, dict[str, object]],
    channel_stats: dict[str, dict[str, object]],
    channel_symmetry: dict[str, dict[str, list[float]]],
    fids: dict[str, float],
    real_dir: Path,
) -> None:
    write_csv(output_dir / "metrics_summary.csv", rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "analysis_note": "checkpoint-50000 only, 150-step DDPM samples, 40 generated PNGs per method.",
                "real_dir": str(real_dir),
                "sample_dirs": {name: str(path) for name, path in SAMPLE_DIRS.items()},
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
    write_channel_outputs(channel_output_dir, channel_stats, channel_symmetry, real_dir)


def write_channel_outputs(
    out_dir: Path,
    stats: dict[str, dict[str, object]],
    symmetry: dict[str, dict[str, list[float]]],
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
                "analysis_note": "checkpoint-50000 only, 150-step DDPM samples, 40 generated PNGs per method.",
                "channel_order": CHANNELS,
                "real_dir": str(real_dir),
                "sample_dirs": {name: str(path) for name, path in SAMPLE_DIRS.items()},
                "stats": stats,
                "symmetry": symmetry,
            },
            indent=2,
        )
    )


def make_grid(out_path: Path) -> None:
    selected = []
    for label, root in SAMPLE_DIRS.items():
        paths = list_pngs(root)
        if paths:
            selected.append((label, paths[0]))
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
    lines = [
        "# checkpoint-50000 150-step evaluation",
        "",
        "All generated sets use only `checkpoint-50000` samples, `n=40` per method.",
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
    path.write_text("\n".join(lines) + "\n")


def format_vector(values: np.ndarray | list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    return "[" + ", ".join(f"{x:.4f}" for x in arr.tolist()) + "]"


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=REAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=MAIN_OUT)
    parser.add_argument("--channel-output-dir", type=Path, default=CHANNEL_OUT)
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--real-diagnostic-limit", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

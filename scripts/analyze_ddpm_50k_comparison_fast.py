"""Fast NumPy/Pillow analysis for the 50k DDPM GAF comparison."""

from __future__ import annotations

import csv
import json
import math
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PURE_DIR = Path("/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples")
DIAG_DIR = Path("/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples")
REAL_DIR = Path("/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images")
OUT_DIR = Path("/home/jliu/Thesis/results_hpc/ddpm_50k_comparison_analysis")
TARGET_RGB = np.array([-0.3689744770526886, 0.003921627998352051, 0.7749511003494263], dtype=np.float64)
REAL_DIAGNOSTIC_LIMIT = 80
STEP_RE = re.compile(r"step_(\d+)_")


def list_pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.png") if p.is_file())


def load_norm(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0


def step_groups(paths: list[Path]) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for path in paths:
        match = STEP_RE.search(path.name)
        groups[match.group(1) if match else "unknown"] += 1
    return dict(sorted(groups.items()))


def image_diagnostics(img: np.ndarray) -> dict[str, object]:
    diag = np.diagonal(img, axis1=0, axis2=1).T
    symmetry = float(np.mean(np.abs(img - np.transpose(img, (1, 0, 2))))) if img.shape[0] == img.shape[1] else None
    dx = float(np.mean(np.abs(np.diff(img, axis=1))))
    dy = float(np.mean(np.abs(np.diff(img, axis=0))))
    return {
        "diag_rgb": np.mean(diag, axis=0).astype(float).tolist(),
        "symmetry_error": symmetry,
        "edge_strength": dx + dy,
    }


def analyze_set(label: str, paths: list[Path], diagnostic_limit: int | None = None) -> dict[str, object]:
    hist_rgb = np.zeros((3, 256), dtype=np.int64)
    hist_all = np.zeros(256, dtype=np.int64)
    sizes = Counter()
    diag_values: list[list[float]] = []
    symmetry_values: list[float] = []
    edge_values: list[float] = []
    metric_paths = set(paths if diagnostic_limit is None else paths[:diagnostic_limit])

    for path in paths:
        raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        sizes[f"{raw.shape[1]}x{raw.shape[0]}"] += 1
        for c in range(3):
            hist_rgb[c] += np.bincount(raw[:, :, c].ravel(), minlength=256)
        hist_all += np.bincount(raw.ravel(), minlength=256)

        if path in metric_paths:
            diag = image_diagnostics(raw.astype(np.float32) / 127.5 - 1.0)
            diag_values.append(diag["diag_rgb"])
            if diag["symmetry_error"] is not None:
                symmetry_values.append(float(diag["symmetry_error"]))
            edge_values.append(float(diag["edge_strength"]))

    values = np.arange(256, dtype=np.float64) / 127.5 - 1.0

    def mean_std(hist: np.ndarray) -> tuple[float, float]:
        n = int(hist.sum())
        mean = float((values * hist).sum() / n)
        var = float((((values - mean) ** 2) * hist).sum() / (n - 1)) if n > 1 else 0.0
        return mean, math.sqrt(var)

    def quantile(hist: np.ndarray, q: float) -> float:
        idx = int(np.searchsorted(np.cumsum(hist), math.ceil(float(hist.sum()) * q), side="left"))
        return float(values[min(idx, 255)])

    rgb_mean_std = [mean_std(hist_rgb[c]) for c in range(3)]
    overall_mean, overall_std = mean_std(hist_all)
    diag_array = np.array(diag_values, dtype=np.float64) if diag_values else np.zeros((0, 3), dtype=np.float64)

    return {
        "label": label,
        "num_images": len(paths),
        "num_diagnostic_images": len(metric_paths),
        "sizes": dict(sizes),
        "step_groups": step_groups(paths),
        "rgb_mean": [x[0] for x in rgb_mean_std],
        "rgb_std": [x[1] for x in rgb_mean_std],
        "rgb_quantiles": {
            "q01": [quantile(hist_rgb[c], 0.01) for c in range(3)],
            "q50": [quantile(hist_rgb[c], 0.50) for c in range(3)],
            "q99": [quantile(hist_rgb[c], 0.99) for c in range(3)],
        },
        "overall": {
            "mean": overall_mean,
            "std": overall_std,
            "q01": quantile(hist_all, 0.01),
            "q50": quantile(hist_all, 0.50),
            "q99": quantile(hist_all, 0.99),
        },
        "diag_rgb_mean": diag_array.mean(axis=0).astype(float).tolist(),
        "diag_rgb_std": diag_array.std(axis=0, ddof=1).astype(float).tolist(),
        "symmetry_error": {
            "mean": float(np.mean(symmetry_values)),
            "std": float(np.std(symmetry_values, ddof=1)),
        },
        "edge_strength": {
            "mean": float(np.mean(edge_values)),
            "std": float(np.std(edge_values, ddof=1)),
        },
        "paths": [str(p) for p in paths],
    }


def pick_representatives(paths: list[Path], root: Path, prefix: str) -> list[tuple[str, Path]]:
    if not paths:
        return []
    preferred = [
        ("2.5k", root / "step_002500_000.png"),
        ("25k", root / "step_025000_000.png"),
        ("50k", root / "step_050000_000.png"),
        ("25k", root / "checkpoint-25000" / "sample_000000.png"),
        ("50k", root / "checkpoint-50000" / "sample_000000.png"),
    ]
    selected: list[tuple[str, Path]] = []
    for label, path in preferred:
        if path.exists():
            selected.append((f"{prefix} {label}", path))
    if selected:
        return selected[:3]
    indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    return [(f"{prefix} {i + 1}", paths[i]) for i in indices]


def make_grid(out_path: Path, pure_paths: list[Path], diag_paths: list[Path], pure_root: Path, diag_root: Path) -> None:
    selected = []
    pure_reps = pick_representatives(pure_paths, pure_root, "pure")
    diag_reps = pick_representatives(diag_paths, diag_root, "diag")
    for i in range(max(len(pure_reps), len(diag_reps))):
        if i < len(pure_reps):
            selected.append(pure_reps[i])
        if i < len(diag_reps):
            selected.append(diag_reps[i])
    selected = selected[:6]
    tile_w, tile_h, label_h = 224, 224, 24
    canvas = Image.new("RGB", (tile_w * len(selected), tile_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (label, path) in enumerate(selected):
        canvas.paste(Image.open(path).convert("RGB").resize((tile_w, tile_h)), (i * tile_w, label_h))
        draw.text((i * tile_w + 6, 6), label, fill=(0, 0, 0), font=font)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    pure_dir = args.pure_dir.resolve()
    diag_dir = args.diag_dir.resolve()
    real_dir = args.real_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    pure_paths = list_pngs(pure_dir)
    diag_paths = list_pngs(diag_dir)
    real_paths = list_pngs(real_dir)
    results = {
        "analysis_note": "Pixel values are normalized from uint8 PNG values to [-1, 1] as value / 127.5 - 1.",
        "target_rgb": TARGET_RGB.astype(float).tolist(),
        "sets": {
            "real": analyze_set("real", real_paths, diagnostic_limit=REAL_DIAGNOSTIC_LIMIT),
            "pureMSE": analyze_set("pureMSE", pure_paths),
            "dataDrivenDiag": analyze_set("dataDrivenDiag", diag_paths),
        },
        "file_structure": {
            "pureMSE_dir": str(pure_dir),
            "dataDrivenDiag_dir": str(diag_dir),
            "real_dir": str(real_dir),
            "individual_sample_images": True,
            "grid_images_detected": False,
        },
    }
    diag_mean = np.array(results["sets"]["dataDrivenDiag"]["diag_rgb_mean"], dtype=np.float64)
    results["dataDrivenDiag_distance_to_target_l2"] = float(np.linalg.norm(diag_mean - TARGET_RGB))

    json_path = out_dir / "metrics_summary.json"
    csv_path = out_dir / "metrics_summary.csv"
    json_path.write_text(json.dumps(results, indent=2))

    rows = []
    for key, payload in results["sets"].items():
        rows.append({
            "set": key,
            "num_images": payload["num_images"],
            "num_diagnostic_images": payload["num_diagnostic_images"],
            "rgb_mean_r": payload["rgb_mean"][0],
            "rgb_mean_g": payload["rgb_mean"][1],
            "rgb_mean_b": payload["rgb_mean"][2],
            "rgb_std_r": payload["rgb_std"][0],
            "rgb_std_g": payload["rgb_std"][1],
            "rgb_std_b": payload["rgb_std"][2],
            "overall_mean": payload["overall"]["mean"],
            "overall_std": payload["overall"]["std"],
            "diag_mean_r": payload["diag_rgb_mean"][0],
            "diag_mean_g": payload["diag_rgb_mean"][1],
            "diag_mean_b": payload["diag_rgb_mean"][2],
            "symmetry_error_mean": payload["symmetry_error"]["mean"],
            "symmetry_error_std": payload["symmetry_error"]["std"],
            "edge_strength_mean": payload["edge_strength"]["mean"],
            "edge_strength_std": payload["edge_strength"]["std"],
        })
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grid_path = out_dir / "representative_comparison_grid.png"
    make_grid(grid_path, pure_paths, diag_paths, pure_dir, diag_dir)
    print(json.dumps({
        "num_real": len(real_paths),
        "num_pureMSE": len(pure_paths),
        "num_dataDrivenDiag": len(diag_paths),
        "metrics_summary_json": str(json_path),
        "metrics_summary_csv": str(csv_path),
        "representative_grid": str(grid_path),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-dir", type=Path, default=PURE_DIR, help="Pure MSE sample directory.")
    parser.add_argument("--diag-dir", type=Path, default=DIAG_DIR, help="Data-driven diagonal sample directory.")
    parser.add_argument("--real-dir", type=Path, default=REAL_DIR, help="Real image directory.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Output analysis directory.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

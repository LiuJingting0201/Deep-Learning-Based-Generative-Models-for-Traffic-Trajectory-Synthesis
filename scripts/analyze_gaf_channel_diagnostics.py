"""Channel-wise diagnostics and post-hoc calibration for generated GAF images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


CHANNELS = ("GASF_R", "GADF_G", "MTF_B")
DEFAULT_REAL_DIR = Path("/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images")
DEFAULT_PURE_DIR = Path("/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_150steps")
DEFAULT_DIAG_DIR = Path(
    "/home/jliu/Thesis/results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_150steps"
)
DEFAULT_OUT_DIR = Path("/home/jliu/Thesis/results_hpc/channel_diagnostics")


def list_pngs(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.png") if p.is_file())


def load_norm(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0


def save_norm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip((np.clip(image, -1.0, 1.0) + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def streaming_channel_stats(paths: list[Path]) -> dict[str, object]:
    hist = np.zeros((3, 256), dtype=np.int64)
    sum_channels = np.zeros(3, dtype=np.float64)
    sumsq_channels = np.zeros(3, dtype=np.float64)
    count = 0
    saturation_count = np.zeros(3, dtype=np.int64)
    corr_sum = np.zeros(3, dtype=np.float64)
    corr_sumsq = np.zeros(3, dtype=np.float64)
    corr_cross = np.zeros((3, 3), dtype=np.float64)
    sizes: dict[str, int] = {}

    for path in paths:
        raw = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        sizes[f"{raw.shape[1]}x{raw.shape[0]}"] = sizes.get(f"{raw.shape[1]}x{raw.shape[0]}", 0) + 1
        norm = raw.astype(np.float32) / 127.5 - 1.0
        flat = norm.reshape(-1, 3).astype(np.float64)
        n = flat.shape[0]
        count += n
        sum_channels += flat.sum(axis=0)
        sumsq_channels += (flat * flat).sum(axis=0)
        saturation_count += (np.abs(flat) > 0.98).sum(axis=0)
        corr_sum += flat.sum(axis=0)
        corr_sumsq += (flat * flat).sum(axis=0)
        corr_cross += flat.T @ flat
        for c in range(3):
            hist[c] += np.bincount(raw[:, :, c].ravel(), minlength=256)

    values = np.arange(256, dtype=np.float64) / 127.5 - 1.0
    means = sum_channels / count
    variances = (sumsq_channels - count * means * means) / max(count - 1, 1)
    stds = np.sqrt(np.maximum(variances, 0.0))

    corr_means = corr_sum / count
    cov = (corr_cross - count * np.outer(corr_means, corr_means)) / max(count - 1, 1)
    corr_stds = np.sqrt(np.maximum((corr_sumsq - count * corr_means * corr_means) / max(count - 1, 1), 0.0))
    denom = np.outer(corr_stds, corr_stds)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)

    def quantile(channel_hist: np.ndarray, q: float) -> float:
        idx = int(np.searchsorted(np.cumsum(channel_hist), int(np.ceil(channel_hist.sum() * q)), side="left"))
        return float(values[min(idx, 255)])

    return {
        "num_images": len(paths),
        "num_pixels": int(count),
        "sizes": sizes,
        "mean": means.astype(float).tolist(),
        "std": stds.astype(float).tolist(),
        "q01": [quantile(hist[c], 0.01) for c in range(3)],
        "q50": [quantile(hist[c], 0.50) for c in range(3)],
        "q99": [quantile(hist[c], 0.99) for c in range(3)],
        "saturation_ratio_abs_gt_0_98": (saturation_count / count).astype(float).tolist(),
        "correlation": corr.astype(float).tolist(),
    }


def per_channel_symmetry(paths: list[Path]) -> dict[str, list[float]]:
    values = []
    for path in paths:
        image = load_norm(path)
        if image.shape[0] != image.shape[1]:
            continue
        diff = np.abs(image - np.transpose(image, (1, 0, 2)))
        values.append(diff.mean(axis=(0, 1)))
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0, ddof=1).astype(float).tolist(),
    }


def calibrate_images(paths: list[Path], input_root: Path, output_root: Path, gen_stats: dict[str, object], real_stats: dict[str, object]) -> None:
    gen_mean = np.array(gen_stats["mean"], dtype=np.float32).reshape(1, 1, 3)
    gen_std = np.array(gen_stats["std"], dtype=np.float32).reshape(1, 1, 3)
    real_mean = np.array(real_stats["mean"], dtype=np.float32).reshape(1, 1, 3)
    real_std = np.array(real_stats["std"], dtype=np.float32).reshape(1, 1, 3)
    gen_std = np.maximum(gen_std, 1e-6)
    for path in paths:
        image = load_norm(path)
        calibrated = ((image - gen_mean) / gen_std) * real_std + real_mean
        save_norm(output_root / path.relative_to(input_root), calibrated)


def write_csvs(out_dir: Path, stats: dict[str, dict[str, object]], symmetry: dict[str, dict[str, list[float]]]) -> None:
    channel_stats_rows = []
    saturation_rows = []
    symmetry_rows = []
    for set_name, set_stats in stats.items():
        for i, channel in enumerate(CHANNELS):
            channel_stats_rows.append({
                "set": set_name,
                "channel": channel,
                "mean": set_stats["mean"][i],
                "std": set_stats["std"][i],
                "q01": set_stats["q01"][i],
                "q50": set_stats["q50"][i],
                "q99": set_stats["q99"][i],
            })
            saturation_rows.append({
                "set": set_name,
                "channel": channel,
                "saturation_ratio_abs_gt_0_98": set_stats["saturation_ratio_abs_gt_0_98"][i],
            })
            symmetry_rows.append({
                "set": set_name,
                "channel": channel,
                "symmetry_error_mean": symmetry[set_name]["mean"][i],
                "symmetry_error_std": symmetry[set_name]["std"][i],
            })

    write_csv(out_dir / "channel_stats.csv", channel_stats_rows)
    write_csv(out_dir / "channel_saturation.csv", saturation_rows)
    write_csv(out_dir / "channel_symmetry.csv", symmetry_rows)

    corr_rows = []
    pairs = [("GASF_R", "GADF_G", 0, 1), ("GASF_R", "MTF_B", 0, 2), ("GADF_G", "MTF_B", 1, 2)]
    for set_name, set_stats in stats.items():
        corr = np.array(set_stats["correlation"], dtype=np.float64)
        for a, b, i, j in pairs:
            corr_rows.append({"set": set_name, "channel_a": a, "channel_b": b, "correlation": float(corr[i, j])})
    write_csv(out_dir / "channel_correlation.csv", corr_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_heatmaps(out_dir: Path, stats: dict[str, dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for set_name, set_stats in stats.items():
        corr = np.array(set_stats["correlation"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(3), CHANNELS, rotation=35, ha="right")
        ax.set_yticks(range(3), CHANNELS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")
        ax.set_title(f"{set_name} channel correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"channel_correlation_heatmap_{set_name}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "real": args.real_dir.resolve(),
        "pureMSE": args.pure_dir.resolve(),
        "dataDrivenDiag": args.diag_dir.resolve(),
    }
    paths = {name: list_pngs(path) for name, path in dirs.items()}
    stats = {name: streaming_channel_stats(path_list) for name, path_list in paths.items()}
    symmetry = {name: per_channel_symmetry(path_list) for name, path_list in paths.items()}

    write_csvs(out_dir, stats, symmetry)
    write_heatmaps(out_dir, stats)
    (out_dir / "channel_stats.json").write_text(
        json.dumps(
            {
                "channel_order": CHANNELS,
                "dirs": {name: str(path) for name, path in dirs.items()},
                "stats": stats,
                "symmetry": symmetry,
            },
            indent=2,
        )
    )

    pure_calibrated = out_dir / "pureMSE_calibrated"
    diag_calibrated = out_dir / "dataDrivenDiag_calibrated"
    calibrate_images(paths["pureMSE"], dirs["pureMSE"], pure_calibrated, stats["pureMSE"], stats["real"])
    calibrate_images(paths["dataDrivenDiag"], dirs["dataDrivenDiag"], diag_calibrated, stats["dataDrivenDiag"], stats["real"])

    print(
        json.dumps(
            {
                "num_real": len(paths["real"]),
                "num_pureMSE": len(paths["pureMSE"]),
                "num_dataDrivenDiag": len(paths["dataDrivenDiag"]),
                "output_dir": str(out_dir),
                "pureMSE_calibrated": str(pure_calibrated),
                "dataDrivenDiag_calibrated": str(diag_calibrated),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--pure-dir", type=Path, default=DEFAULT_PURE_DIR)
    parser.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()

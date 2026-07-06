"""Package Week06 raw diffusion batch samples into decoded-trajectory layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "week06" / "results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "week06" / "decoded"
DEFAULT_RUNS = (
    "raw_absolute_temporal_resnet",
    "raw_absolute_temporal_resnet_dilated",
    "raw_absolute_unet1d",
    "raw_delta_temporal_resnet",
    "raw_delta_temporal_resnet_dilated",
    "raw_delta_unet1d",
)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run_name in args.runs:
        run_dir = args.results_root / run_name
        summary_path = run_dir / "train_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        mode = summary["mode"]
        step = int(summary["global_step"])
        decoded_dir = args.output_root / f"{run_name}_step_{step:06d}"
        record = package_run(run_dir, decoded_dir, run_name, mode, step, args.max_samples)
        manifest.append(record)
        print(json.dumps(record, indent=2), flush=True)
    (args.output_root / "raw_diffusion_sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def package_run(run_dir: Path, decoded_dir: Path, run_name: str, mode: str, step: int, max_samples: int | None) -> dict:
    samples_dir = run_dir / "samples"
    trajectories_path = resolve_trajectories_path(samples_dir, mode, step)
    trajectories = load_batch(trajectories_path, max_samples=max_samples)
    if trajectories.ndim != 3 or trajectories.shape[1:] != (224, 2):
        raise ValueError(f"Expected {trajectories_path} to have shape N,224,2; got {trajectories.shape}")

    if decoded_dir.exists():
        shutil.rmtree(decoded_dir)
    trajectory_dir = decoded_dir / "trajectories_raw_xy"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for index, trajectory in enumerate(trajectories):
        np.save(trajectory_dir / f"generated_{index:06d}.npy", trajectory.astype(np.float32))
    np.save(decoded_dir / "trajectories_raw_xy.npy", trajectories.astype(np.float32))

    copied = {
        "run_name": run_name,
        "mode": mode,
        "step": step,
        "decoded_dir": str(decoded_dir.resolve()),
        "num_samples": int(len(trajectories)),
        "trajectory_shape": list(trajectories.shape[1:]),
        "source_trajectories": str(trajectories_path.resolve()),
    }
    if mode == "delta":
        delta_path = samples_dir / f"step_{step:06d}_delta_raw.npy"
        starts_path = samples_dir / f"step_{step:06d}_sampled_starts.npy"
        if delta_path.exists():
            delta = load_batch(delta_path, max_samples=max_samples)
            np.save(decoded_dir / "delta_raw_xy.npy", delta.astype(np.float32))
            copied["source_delta"] = str(delta_path.resolve())
        if starts_path.exists():
            starts = load_starts(starts_path, max_samples=max_samples)
            np.save(decoded_dir / "starts_raw_xy.npy", starts.astype(np.float32))
            copied["source_starts"] = str(starts_path.resolve())

    for filename in ("startup_config.json", "train_summary.json", "normalization_stats.json"):
        source = run_dir / filename
        if source.exists():
            shutil.copy2(source, decoded_dir / filename)

    metric_summary = trajectory_summary(trajectories)
    payload = {**copied, "metrics": metric_summary}
    (decoded_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def resolve_trajectories_path(samples_dir: Path, mode: str, step: int) -> Path:
    if mode == "absolute":
        path = samples_dir / f"step_{step:06d}_absolute_raw.npy"
    elif mode == "delta":
        path = samples_dir / f"step_{step:06d}_delta_integrated_trajectories.npy"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_batch(path: Path, max_samples: int | None) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if max_samples is not None:
        array = array[:max_samples]
    return array


def load_starts(path: Path, max_samples: int | None) -> np.ndarray:
    array = np.load(path).astype(np.float32)
    if max_samples is not None:
        array = array[:max_samples]
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected {path} to have shape N,2; got {array.shape}")
    return array


def trajectory_summary(trajectories: np.ndarray) -> dict:
    steps = np.diff(trajectories.astype(np.float64), axis=1)
    step_lengths = np.linalg.norm(steps, axis=2)
    displacement = np.linalg.norm(trajectories[:, -1, :] - trajectories[:, 0, :], axis=1)
    path_length = np.sum(step_lengths, axis=1)
    return {
        "displacement": summarize(displacement),
        "path_length": summarize(path_length),
        "mean_step_length": summarize(np.mean(step_lengths, axis=1)),
        "max_step_length": summarize(np.max(step_lengths, axis=1)),
        "bbox_width": summarize(np.ptp(trajectories[:, :, 0], axis=1)),
        "bbox_height": summarize(np.ptp(trajectories[:, :, 1], axis=1)),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

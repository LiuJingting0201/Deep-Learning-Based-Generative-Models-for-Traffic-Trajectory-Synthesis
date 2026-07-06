"""Train raw-sequence DDPM directly on Week05 delta-displacement trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from trajectory_diffusion_common import train_raw_sequence_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "week05" / "data" / "delta_displacement_paired_regularized_bounce224"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "week06" / "results" / "raw_delta_displacement_diffusion"


def main() -> None:
    train_raw_sequence_diffusion(parse_args(), mode="delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normalization-file", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--sequence-length", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument("--num-sample-sequences", type=int, default=64)
    parser.add_argument("--sample-inference-steps", type=int, default=100)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--model-type",
        choices=("temporal_resnet", "temporal_resnet_dilated", "unet1d"),
        default="temporal_resnet",
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

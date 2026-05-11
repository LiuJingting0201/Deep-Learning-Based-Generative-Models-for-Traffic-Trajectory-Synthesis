"""Encode raw trajectories into GAF/MTF RGB images.

This is a minimal, reviewable CLI for the original notebook pipeline. It avoids
hard-coded paths and writes only to the requested output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.data.loader import load_trajectory_csv
from cnr_trajectory.encoding.gaf import encode_trajectory_frame


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    image_dir = output_dir / "all"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = load_trajectory_csv(args.input, nrows=args.nrows)
    encoded = encode_trajectory_frame(
        frame,
        sequence_length=args.sequence_length,
        eps=args.hilbert_eps,
        max_samples=args.max_samples,
    )

    for index, image in enumerate(encoded.images):
        label = str(encoded.labels[index])
        vehicle_id = str(encoded.vehicle_ids[index])
        filename = f"{index:04d}_{vehicle_id}_{label}.png"
        Image.fromarray(image.astype(np.uint8)).save(image_dir / filename)

    np.save(output_dir / "images.npy", encoded.images)
    np.save(output_dir / "sequences.npy", encoded.sequences)

    metadata_path = output_dir / "metadata.csv"
    metadata_path.write_text(
        "filename,vehicle_id,sequence_length,image_shape,source_file\n"
        + "\n".join(
            f"{index:04d}_{encoded.vehicle_ids[index]}_{encoded.labels[index]}.png,"
            f"{encoded.vehicle_ids[index]},"
            f"{encoded.sequences.shape[1]},"
            f"\"{tuple(encoded.images[index].shape)}\","
            f"{args.input}"
            for index in range(len(encoded.images))
        )
        + "\n"
    )

    print(
        f"Encoded {len(encoded.images)} trajectories to {output_dir} "
        f"with shape {encoded.images.shape}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw trajectory CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where encoded outputs will be written. Images go into OUTPUT/all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="Maximum number of vehicle trajectories to encode.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=20000,
        help="Maximum raw rows to read for this minimal pipeline.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=224,
        help="Target sequence length used by the original notebook.",
    )
    parser.add_argument(
        "--hilbert-eps",
        type=float,
        default=0.0037,
        help="Hilbert precision used by the original notebook.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

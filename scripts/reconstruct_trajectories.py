"""Convert predicted normalized coordinate sequences to trace CSV files.

This script integrates the clear part of the original `Map_Matching` notebook.
It does not decode generated images. It starts from an existing prediction file
such as `trajectories_prediction.pkl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cnr_trajectory.reconstruction.map_matching import (
    load_predicted_trajectories,
    match_trace_csv,
    rescale_prediction,
    write_trace_csv,
)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predicted_trajectories(args.predictions)
    max_count = min(args.max_samples, predictions.shape[0])
    metadata = []

    for trajectory_index in range(max_count):
        trace_frame = rescale_prediction(
            predictions,
            args.reference_csv,
            trajectory_index=trajectory_index,
            sequence_length=args.sequence_length,
        )
        trace_path = write_trace_csv(
            trace_frame,
            trace_dir / f"trace_{trajectory_index:04d}.csv",
        )
        row = {
            "trajectory_index": trajectory_index,
            "trace_csv": str(trace_path.relative_to(output_dir)),
            "num_points": len(trace_frame),
            "source_predictions": str(args.predictions),
            "reference_csv": str(args.reference_csv),
        }

        if args.run_map_matching:
            # This may require mappymatch and online road network access.
            match_result = match_trace_csv(trace_path)
            row["matched_path_length"] = len(match_result.path)

        metadata.append(row)

    metadata_path = output_dir / "reconstruction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(
        f"Wrote {len(metadata)} trace CSV file(s) to {trace_dir}. "
        f"metadata={metadata_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Pickle file with normalized coordinate predictions.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="Raw trajectory CSV used to rescale normalized coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for trace CSV files and metadata.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Maximum prediction rows to convert.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=224,
        help="Number of lon and lat points per prediction row.",
    )
    parser.add_argument(
        "--run-map-matching",
        action="store_true",
        help="Also run mappymatch. This may download road network data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

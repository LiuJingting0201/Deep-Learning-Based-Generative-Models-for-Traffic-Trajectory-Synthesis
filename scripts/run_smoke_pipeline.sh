#!/usr/bin/env bash
set -euo pipefail

RAW_FILE="${RAW_FILE:-Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls}"
ENCODE_OUTPUT="${ENCODE_OUTPUT:-data/processed/gaf_rgb}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-results/logs/diffusion_smoke}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
NROWS="${NROWS:-1000}"
IMAGE_SIZE="${IMAGE_SIZE:-32}"
TRAIN_STEPS="${TRAIN_STEPS:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

export PYTHONPATH="${PYTHONPATH:-src}"

"${PYTHON_BIN}" scripts/encode_trajectories.py \
  --input "${RAW_FILE}" \
  --output "${ENCODE_OUTPUT}" \
  --max-samples "${MAX_SAMPLES}" \
  --nrows "${NROWS}"

"${PYTHON_BIN}" scripts/train_diffusion_smoke.py \
  --input-dir "${ENCODE_OUTPUT}/all" \
  --output-dir "${SMOKE_OUTPUT}" \
  --image-size "${IMAGE_SIZE}" \
  --max-samples "${MAX_SAMPLES}" \
  --train-steps "${TRAIN_STEPS}"

echo "Smoke pipeline completed."

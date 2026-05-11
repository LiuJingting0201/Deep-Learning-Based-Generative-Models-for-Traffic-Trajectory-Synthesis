# CNR Trajectory Generation

Research codebase for a master's thesis project on trajectory-to-image encoding, image generation, and image-to-trajectory reconstruction.

The repository was refactored from a notebook-heavy workspace. Original notebooks are preserved, while the currently verified minimal pipeline is exposed through small Python modules, scripts, and tests.

## Current Status

- The original codebase has been refactored into reproducible Python modules, CLI scripts, tests, and documentation.
- DDPM training, DDPM/DDIM sample generation, and FID evaluation are available through `scripts/train_ddpm.py`, `scripts/generate_ddpm_samples.py`, and `scripts/evaluate_fid.py`.
- CUDA GPU sanity training has passed in WSL.
- A 3000-step preliminary 128x128 DDPM baseline completed with EMA enabled.
- Preliminary FID decreases with training steps, but the 100-sample FID values are sanity checks only, not final scientific metrics.
- The image-to-trajectory decoder is still not integrated.
- The next step is a longer DDPM baseline and more stable FID using 500 or 1000 generated samples.

## Current Minimal Pipeline

The following real pipeline is currently runnable on a tiny subset:

```text
raw trajectory CSV
-> Hilbert trajectory sequence extraction
-> GASF/GADF/MTF RGB image encoding
-> PNG image dataset loader
-> tiny DDPM/UNet forward + backward optimizer step
-> smoke_log.json
```

This is a smoke pipeline only. It does not run full training and does not implement image-to-trajectory reconstruction.

## Repository Structure

```text
configs/                 YAML configuration files
data/
  raw/                   Local raw data, ignored by default
  processed/             Generated encoded data, ignored by default
notebooks/
  original/              Copies of original notebooks
  cleaned/               Future cleaned notebooks
src/cnr_trajectory/
  data/                  CSV and PNG dataset loaders
  encoding/              Hilbert, sequence, GAF/GADF/MTF encoding
  models/                Model placeholders
  reconstruction/        Reconstruction placeholders
  evaluation/            Evaluation placeholders
  visualization/         Plotting placeholders
scripts/                 Runnable command-line smoke tools
results/                 Generated figures/logs/metrics, ignored by default
tests/                   Unit and smoke tests
docs/                    Project status documentation
```

## Environment Setup

Create and activate a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install core preprocessing and test dependencies:

```bash
pip install -r requirements-core.txt
```

Install training smoke dependencies:

```bash
pip install -r requirements-train.txt
```

Optional map matching dependencies:

```bash
pip install -r requirements-mapmatching.txt
```

Compatibility install for everything:

```bash
pip install -r requirements.txt
```

Optional aliases:

```bash
source aliases.sh
```

## Run Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Expected current status:

```text
18 passed, 1 skipped
```

## Run Encoding Smoke

The raw trajectory file currently used by the original notebook is:

```text
Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls
```

Despite the `.xls` suffix, it is read as CSV, preserving the original notebook behavior.

```bash
PYTHONPATH=src .venv/bin/python scripts/encode_trajectories.py \
  --input Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls \
  --output data/processed/gaf_rgb \
  --max-samples 2 \
  --nrows 1000
```

Output layout:

```text
data/processed/gaf_rgb/
├── all/
│   ├── 0000_<vehicle_id>_<label>.png
│   └── ...
├── metadata.csv
├── images.npy
└── sequences.npy
```

`metadata.csv` records:

```text
filename, vehicle_id, sequence_length, image_shape, source_file
```

## Run Diffusion Smoke

Run one tiny real DDPM optimizer step on the encoded PNG images:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_diffusion_smoke.py \
  --input-dir data/processed/gaf_rgb/all \
  --output-dir results/logs/diffusion_smoke \
  --image-size 32 \
  --max-samples 2 \
  --train-steps 1
```

This saves only:

```text
results/logs/diffusion_smoke/smoke_log.json
```

No model checkpoint is created.

## Convert Predicted Coordinates To Trace CSV

The `Map_Matching` notebook starts from an existing prediction tensor, not from generated images. To reproduce the clear part of that notebook:

```bash
PYTHONPATH=src .venv/bin/python scripts/reconstruct_trajectories.py \
  --predictions Map_Matching/trajectories_prediction.pkl \
  --reference-csv Map_Matching/gps_with_speed_224_UPDATED.xls \
  --output-dir results/logs/reconstruction_trace_smoke \
  --max-samples 1
```

This writes trace CSV files under `OUTPUT/traces/` and a `reconstruction_metadata.json` file. It does not run map matching unless `--run-map-matching` is passed.

## One-Command Smoke Pipeline

```bash
bash scripts/run_smoke_pipeline.sh
```

Optional overrides:

```bash
RAW_FILE=Image_Generation/Image_Generation/gps_with_speed_224_UPDATED.xls \
ENCODE_OUTPUT=data/processed/gaf_rgb \
SMOKE_OUTPUT=results/logs/diffusion_smoke \
MAX_SAMPLES=2 \
NROWS=1000 \
IMAGE_SIZE=32 \
TRAIN_STEPS=1 \
bash scripts/run_smoke_pipeline.sh
```


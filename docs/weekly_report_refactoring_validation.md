# Weekly Report: Refactoring and Validation of the Notebook-Based Trajectory Diffusion Pipeline

## 1. Original State of the Codebase

The original project was primarily implemented through Jupyter notebooks. The core scientific workflow and experimental logic came from these notebooks rather than from reusable Python modules or command-line scripts.

The main notebook-based components were:

- `Image_Generation/Image_Generation/23_Speed-1_folder.ipynb`: trajectory preprocessing, sequence handling, Hilbert ordering, and image encoding using GAF/MTF-style representations.
- `Diffusion_Models/Diffusion_Models/01_Diffusion_GAF_128-Loss_Function.ipynb`: DDPM training logic for trajectory images, including the UNet model, DDPM scheduler, training loop, notebook-specific loss terms, and sample generation.
- `Diffusion_Models/Diffusion_Models/02_FID_Images_Comparison.ipynb`: image-level FID evaluation using Inception features.
- `Diffusion_Models/Diffusion_Models/00_Diffusion_Tutorial.ipynb`: tutorial/reference material for diffusion modeling.
- `Diffusion_Models/Diffusion_Models/03_Reduce_ImSize.ipynb`: image resizing and post-processing utilities.
- `Map_Matching/01_Sumo_data_MM.ipynb`: reconstruction/map-matching logic starting from predicted coordinate artifacts, especially `trajectories_prediction.pkl`.

The original structure had several engineering limitations:

- Notebook-heavy implementation made reproducible reruns difficult.
- Several paths were hardcoded, including absolute paths from the original environment.
- Pipeline order was implicit rather than documented as reusable stages.
- Automated tests were missing or incomplete.


## 2. Repository Structure Added or Reorganized

The repository has been reorganized into a more reproducible Python project structure.

### `src/`

The `src/` directory contains reusable Python modules. It separates core logic from notebooks and scripts. It includes trajectory loading, image encoding, image dataset handling, reconstruction helpers, and map-matching wrappers.

### `scripts/`

The `scripts/` directory contains executable CLI entry points. These scripts turn notebook cells into reproducible commands for encoding, training, sampling, FID evaluation, reconstruction, and smoke pipeline execution.

### `tests/`

The `tests/` directory contains lightweight validation tests. These tests verify imports, minimal pipeline behavior, diffusion training smoke checks, DDPM sampling smoke checks, FID formula behavior, and reconstruction/map-matching helper logic.

### `docs/`

The `docs/` directory contains project status documentation. It records what has been extracted, what remains unresolved, and where notebook boundaries still exist.

### `configs/`

The `configs/` directory contains reusable configuration defaults, including dataset paths, image size, diffusion hyperparameters, and evaluation settings.

### `requirements*.txt`

The project separates dependencies by use case:

- `requirements-core.txt` for core preprocessing, testing, and utilities.
- `requirements-train.txt` for diffusion training dependencies such as PyTorch and Diffusers.
- `requirements-mapmatching.txt` for optional map-matching dependencies.
- `requirements.txt` as the broader project dependency file.

### `README.md`

The README documents the project purpose, installation flow, expected commands, and current pipeline status.

### `.gitignore`

The `.gitignore` excludes virtual environments, Python caches, notebook checkpoints, generated data folders, model checkpoint files, generated results, and other local artifacts that should not be uploaded to GitHub.

## 3. Functionality Extracted from Notebooks into Reusable Code

Several notebook components have been extracted into reusable modules and scripts.

Trajectory encoding logic was moved into reusable code under `src/cnr_trajectory/encoding/`. This includes sequence normalization, sampling, Hilbert indexing, GAF/GADF/MTF-style image construction, and RGB image generation.

Image dataset loading was extracted into reusable dataset logic. The DDPM scripts load image folders, convert images to RGB, resize them consistently, and normalize tensors into the expected diffusion range.

DDPM training was extracted into `scripts/train_ddpm.py`. The script supports command-line configuration, from-scratch training, checkpoint saving, optional EMA, resume support, periodic sample generation, and JSONL training logs.

Sample generation was extracted into `scripts/generate_ddpm_samples.py`. It loads saved checkpoints, optionally uses EMA weights, supports DDPM and DDIM schedulers, denormalizes samples from `[-1, 1]` to `[0, 1]`, and saves RGB PNG files.

FID evaluation was extracted into `scripts/evaluate_fid.py`. It loads real and generated image folders, applies consistent RGB conversion and resizing, extracts Inception features, computes FID, and writes the result to JSON.

Predicted-coordinate-to-trace reconstruction was extracted into `scripts/reconstruct_trajectories.py` and reconstruction modules. This stage starts from predicted coordinate artifacts such as `trajectories_prediction.pkl`; it does not currently implement a learned generated-image-to-trajectory decoder.

An optional map-matching wrapper was added around reconstructed traces. This keeps map matching separate from image generation and coordinate reconstruction.

Smoke pipeline scripts were added so that the project can be validated without full training or large output generation.

## 4. Scripts Added and Their Purpose

### `scripts/encode_trajectories.py`

Encodes raw trajectory CSV-style data into trajectory images. It produces RGB image outputs, NumPy arrays, sequence artifacts, and metadata. This script corresponds to the trajectory-to-image logic originally implemented in the image generation notebook.

### `scripts/train_diffusion_smoke.py`

Runs a very small diffusion training smoke test. It is intended to validate that encoded images can be loaded and passed through a minimal diffusion training loop without running a full experiment.

### `scripts/reconstruct_trajectories.py`

Converts predicted normalized coordinate sequences into trace CSV files and metadata. It optionally calls map matching if dependencies and map data are available. It does not decode generated images into trajectories.

### `scripts/train_ddpm.py`

Runs the extracted DDPM training pipeline. It supports configurable data directory, output directory, image size, batch size, learning rate, maximum train steps, gradient accumulation, checkpoint interval, sample interval, EMA, and resume-from-checkpoint behavior.

### `scripts/generate_ddpm_samples.py`

Generates image samples from a trained DDPM checkpoint. It supports checkpoint loading, optional EMA loading, DDPM/DDIM scheduler selection, inference step control, image-size validation, and PNG output.

### `scripts/evaluate_fid.py`

Computes image-level FID between a real image directory and a generated image directory. It saves a JSON result containing the FID value and sample counts.

### `scripts/run_smoke_pipeline.sh`

Runs a lightweight end-to-end smoke pipeline. It is intended for quick validation rather than full scientific training.

## 5. Tests Added and What They Validate

The test suite validates the refactored pipeline using small synthetic or temporary data where possible.

Import and compile tests check that the package and scripts can be imported or compiled without syntax errors.

The original pipeline minimal test checks that the original notebooks remain present/readable, that minimal raw trajectory data can be loaded, and that core extracted pipeline components behave as expected.

The diffusion smoke test validates that a small diffusion model can perform a minimal forward/training step.

The map-matching/reconstruction smoke test validates reconstruction helper behavior without requiring full online map matching.

The DDPM pipeline smoke test validates one-step DDPM training, sample generation, and FID formula behavior on tiny data.

Path and metric-related tests verify that key pipeline assumptions, outputs, and minimal metrics are stable enough for automated validation.

## 6. Changes Compared with Original Notebook Behavior

### A. Pure Refactoring / Engineering Changes

The main scientific logic was not replaced. It was reorganized from notebook cells into reusable modules and scripts.

The following changes are primarily engineering refactors:

- Notebook cell logic was moved into CLI scripts.
- Reusable functions were placed under `src/`.
- Hardcoded notebook workflows were converted into explicit commands.
- Tests were added around minimal reproducible behavior.
- Training, sampling, FID, and reconstruction steps were separated into distinct entry points.
- Outputs are now written into structured directories.

### B. Reproducibility Fixes

Several changes improve reproducibility:

- CLI arguments replace hardcoded paths.
- Training logs are written to `train_log.jsonl`.
- Training summaries are saved as JSON.
- Checkpoints are saved in structured directories.
- Scheduler configuration is saved alongside the model.
- Resume-from-checkpoint is supported.
- Tests use tiny data and `tmp_path` where possible.
- The DDPM pipeline can train from scratch when no checkpoint is available.

### C. Minimal Correctness Fixes

The DDPM script was adjusted to better match the original notebook behavior and avoid reproducibility bugs:

- The notebook-consistent loss was restored: `MSE + 0.01 * symmetry_loss + 0.01 * diag_loss`.
- Scheduler loading during resume was fixed so that the saved scheduler configuration is actually reloaded.
- Default batch size and learning rate were aligned with the notebook-style baseline: `batch_size = 4`, `lr = 2e-4`.
- `train_log.jsonl` logging was added to make training behavior auditable over time.

These are correctness and reproducibility fixes, not changes to the scientific method.

### D. Intentional Differences from Notebooks

Some differences are intentional:

- CLI paths are used instead of notebook-specific absolute paths.
- Reusable scripts replace manually executed notebook cells.
- Checkpoints are stored as structured checkpoint directories rather than one-off notebook `state_dict` files.
- EMA support is optional and explicit.
- Full training is not run by default.
- Tests use tiny temporary datasets rather than the full experimental dataset.
- FID is run through a script with explicit real/generated directories and JSON output.

## 7. Validation Run

The refactored pipeline has been validated with lightweight and medium sanity checks.

The full pytest suite passed:

```text
18 passed, 1 skipped
```

CUDA availability was checked after the WSL CUDA environment was fixed. PyTorch detected CUDA successfully, using an NVIDIA GeForce RTX 4060 Laptop GPU.

A one-step DDPM GPU training smoke test passed. This confirmed that the extracted training script can run on GPU and complete a minimal optimization step.

A 128x128 DDPM sanity baseline was run with the following settings:

```text
image_size: 128
batch_size: 4
learning_rate: 2e-4
max_train_steps: 3000
EMA: enabled
```

The sanity baseline completed with:

```text
elapsed_seconds: 884.49
average_speed: 3.39 steps/s
final_loss: 0.004508
```

The expected checkpoints were saved:

```text
results/ddpm_sanity_128/checkpoint-1000/
results/ddpm_sanity_128/checkpoint-2000/
results/ddpm_sanity_128/checkpoint-3000/
results/ddpm_sanity_128/checkpoint-final/
```

Periodic samples and logs were also saved:

```text
results/ddpm_sanity_128/samples/
results/ddpm_sanity_128/train_log.jsonl
results/ddpm_sanity_128/train_summary.json
```

Then 100 preliminary samples were generated from the final checkpoint:

```text
results/ddpm_sanity_128/generated_100/
```

The generated sample set was verified to contain 100 PNG files with RGB mode and 128x128 image size.

Preliminary FID was computed with 100 generated samples. The FID values by checkpoint were:

```text
checkpoint-1000: 288.81
checkpoint-2000: 254.43
checkpoint-3000 / final: 226.14


## 8. What These Validation Results Prove

These results prove that the refactored pipeline can load data, encode trajectories, train a small diffusion/DDPM model, save checkpoints, generate samples, and compute image-level FID.

The decreasing FID trend from checkpoint 1000 to checkpoint 3000 suggests that the DDPM/FID pipeline behaves consistently as training progresses.

However, these results do not prove final model quality.

The 100-sample FID values are preliminary sanity checks only. They are useful for detecting obvious pipeline problems, but they should not be treated as final scientific metrics.

## 9. Remaining Unresolved Items

Update after the decoder validation pass: a controlled real-image decoder validation stage has now been added using clean paired no-speed images and x/y trajectory labels. This reduces the earlier decoder gap for real encoded images, but it does not yet validate the generated-image -> decoder -> trajectory -> map-matching pipeline.

`trajectories_prediction.pkl` is treated as an external artifact in the current repository. The current reconstruction pipeline starts from this kind of coordinate prediction file rather than from generated DDPM images.

The full pipeline from generated image to decoded path to map-matched path has not yet been validated.

FID is an image-level metric and does not guarantee trajectory validity, geographic plausibility, or map-match quality.

Larger training runs and more stable FID evaluation with more generated samples are still needed.

## 10. Recommended Next Steps

Before pushing to GitHub, check `.gitignore` carefully and make sure generated results, checkpoints, virtual environments, and large data outputs are excluded.

Run a 10000-step preliminary DDPM baseline to get a stronger sanity result before considering full 50000-step training.

Generate 500 or 1000 samples for a more stable FID estimate.

If the original checkpoint becomes available, reproduce its FID using the new scripted evaluation pipeline.

Continue improving and validating the image-to-trajectory decoder, and then integrate it into the generated-image evaluation pipeline.

After that, validate the full generated-image to decoded-trajectory to map-matched-trajectory pipeline.

## Short Summary for Group Meeting

- The original notebook-based trajectory diffusion workflow has been reorganized into reusable modules, CLI scripts, tests, and documentation.
- The scientific logic mainly comes from the existing notebooks; the contribution here is reproducibility, auditing, refactoring, and validation.
- DDPM training, sample generation, and FID evaluation now run from scripts without relying on unavailable checkpoints.
- A 3000-step 128x128 GPU sanity baseline completed successfully, and FID decreased from 288.81 to 226.14 across checkpoints.
- Remaining work includes larger DDPM training, more stable FID, strengthening the image-to-trajectory decoder, and validating generated-image-to-trajectory path-level quality.
- A ResNet18 decoder ablation improved test ADE from 160.30 to 120.92 compared with the original CNN-FC decoder, suggesting that decoder capacity is one bottleneck, although residual reconstruction errors remain.


## Do Not Upload to GitHub Checklist

Do not commit:

```text
.venv/
results/
checkpoints/
generated images
GeneratedImages*/
RGB_Images_PNG*/
Data_*/
data/raw/*
data/processed/*
*.pt
*.pth
*.ckpt
*.safetensors
large NumPy outputs
external prediction artifacts such as trajectories_prediction.pkl
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
*:Zone.Identifier
```

The DDPM/FID refactoring validates that trajectory images can be generated and compared at the image-distribution level. However, image-level generation quality does not guarantee trajectory-space reconstruction quality: a generated RGB image may have acceptable visual or FID behavior while still decoding into a poor coordinate sequence, or into a trajectory that is difficult to map match. For that reason, the next validation step is to test whether real encoded images can be decoded back into trajectory coordinates under a controlled paired-dataset setting before using the decoder on diffusion-generated images.

## 2026-05-12 — Decoder Refactoring and Representation Invertibility Validation

### What Was Refactored

This validation step adds a controlled image-to-trajectory decoder evaluation before using diffusion-generated images for downstream trajectory reconstruction. The current clean pipeline rebuilds a no-speed paired dataset from raw GPS-derived trajectories and stores each sample with explicit image-label pairing, stable `sample_id`, `vehicle_id`, and deterministic train/validation/test assignment.

The main reusable scripts for this stage are:

- `scripts/build_no_speed_paired_dataset.py`
- `scripts/create_no_speed_splits.py`
- `scripts/train_decoder_no_speed_baseline.py`
- `scripts/evaluate_decoder_no_speed_baseline.py`
- `scripts/run_original_config_decoder_experiment.py`
- `scripts/analyze_decoder_errors.py`
- `scripts/run_relative_target_decoder_experiment.py`

The refactored experiments separate dataset construction, split creation, training, evaluation, error diagnostics, and the relative-target ablation into auditable command-line stages. Labels are normalized using train-only statistics, while evaluation is reported back in the original x/y coordinate space.

The quantitative and diagnostic interpretation below is based on the current outputs in:

- `experiments/decoder_no_speed_original_config/`: original-style absolute decoder configuration, training log, best/last checkpoints, evaluation metrics, naive mean-trajectory baseline, prediction variance diagnostics, per-sample metrics, qualitative grids, and error-analysis summaries.
- `experiments/decoder_no_speed_relative_target/`: relative-target experiment configuration, best/last checkpoints, oracle-start evaluation metrics, per-sample metrics, qualitative grids, and comparison outputs against the absolute decoder.

### Difference From the Original Notebook-Based Decoder Workflow

The original decoder work was notebook-heavy and exploratory. Several duplicated experimental copies mixed related decoder variants across files, including the original CNN-FC regression decoder and variants involving map-channel inputs, invalid-point penalties, binary trajectory outputs, BCE/false-positive penalties, and LSTM-style correction. The original workflow also relied on pre-existing `Data_23` image and label folders, while the pairing between raw GPS rows, image files, and label arrays was not fully represented as a reusable metadata table.

The current validation pipeline reconstructs the paired no-speed dataset from raw GPS-derived samples and makes the image-label relationship explicit in `metadata.csv` and `splits/split_metadata.csv`. It preserves sample identity, uses deterministic splits, packages configurations and metrics in experiment folders, and evaluates an original-style CNN-FC decoder as a baseline under a fixed split. This is an original-style reproduction under a clean paired dataset, not a bitwise rerun of the original notebook. The current report therefore gives a cleaner paired-dataset reconstruction evaluation than the original qualitative notebook plots, although it should still be interpreted as one controlled validation setting rather than a final claim about all possible decoders.

### Dataset and Split

| Item | Count |
| --- | ---: |
| Total valid paired samples | 3159 |
| Train | 2527 |
| Validation | 315 |
| Test | 317 |

Each sample uses a no-speed Hilbert/GASF/GADF/MTF RGB image and a corresponding 224-point x/y trajectory label.

This validation uses x/y coordinate labels for coordinate-space reconstruction evaluation. The original presentation described lon/lat-style outputs in later reconstruction/map-matching stages, so a direct x/y-versus-lon/lat comparison remains a possible follow-up if coordinate-system effects need to be isolated.

### Original-Style Absolute CNN-FC Decoder Results

The original-style absolute decoder was trained with `OriginalStyleCNNFCDecoder`, batch size 8, learning rate `1e-4`, dropout 0.3, Adam, train-only mean/std label normalization, and early stopping patience 80. The configured maximum was 1000 epochs. Training stopped at epoch 717, which indicates early stopping was triggered. The best validation-loss epoch in `train_log.csv` was epoch 637.

| Split | ADE mean | FDE mean | ADE median | FDE median |
| --- | ---: | ---: | ---: | ---: |
| Train | 61.16 | 92.37 | 59.46 | 78.71 |
| Validation | 138.76 | 181.86 | 94.82 | 131.97 |
| Test | 160.30 | 201.58 | 91.02 | 116.95 |

ADE and FDE are reported in the original x/y coordinate space after inverse label normalization. They should therefore be interpreted as coordinate-space reconstruction errors unless the underlying map coordinate system is confirmed to be metric. The train-test gap suggests that the decoder learns meaningful structure from the paired representation, but does not yet provide uniformly precise reconstruction under the current experimental setting.

### Naive Mean-Trajectory Baseline

| Model | Test ADE mean | Test FDE mean |
| --- | ---: | ---: |
| Naive train-mean trajectory | 640.89 | 730.67 |
| Original-style CNN-FC decoder | 160.30 | 201.58 |

Relative to the naive mean-trajectory baseline, the CNN-FC decoder improves test ADE by approximately 74.99% and test FDE by approximately 72.41%. This suggests that the no-speed Hilbert/GAF/MTF RGB image representation contains useful decodable information under the current decoder, rather than only supporting a mean-route prediction.

### Prediction Variance Diagnostics

| Split | Ground-truth variance | Predicted variance | Pred/GT variance ratio | Collapse flag |
| --- | ---: | ---: | ---: | --- |
| Train | 239449.61 | 227565.58 | 0.950 | false |
| Validation | 238351.50 | 201957.72 | 0.847 | false |
| Test | 251839.36 | 214496.28 | 0.852 | false |

The predicted-to-ground-truth variance ratios do not indicate severe regression-to-mean collapse in this diagnostic. The model predictions retain substantial output variance, even though the remaining ADE/FDE values and worst-case behavior indicate that reconstruction accuracy is still limited.

### Error Analysis

The test-set correlations between ADE/FDE and simple geometric complexity features are small:

| Feature correlation on test split | Value |
| --- | ---: |
| ADE vs trajectory length | 0.0149 |
| ADE vs tortuosity | 0.0300 |
| ADE vs bounding-box area | 0.0412 |
| ADE vs sharp-turn count | 0.0849 |
| FDE vs trajectory length | 0.0833 |
| FDE vs tortuosity | 0.0486 |
| FDE vs bounding-box area | 0.0885 |
| FDE vs sharp-turn count | 0.0460 |

Additional test-set correlations with localization and aligned-shape diagnostics are stronger:

| ADE correlation on test split | Value |
| --- | ---: |
| ADE vs start point error | 0.8556 |
| ADE vs end point error | 0.8121 |
| ADE vs centroid error | 0.9825 |
| ADE vs aligned-start shape error | 0.8046 |
| ADE vs aligned-centroid shape error | 0.8714 |

These diagnostics suggest that the largest errors are not simply explained by longer trajectories, higher tortuosity, larger spatial extent, or more sharp turns. The stronger correlations with start-point, endpoint, centroid, and aligned-shape errors indicate that both spatial localization and shape/topology reconstruction contribute to the observed failures. Median qualitative cases appear to recover reasonable route shape, while worst cases show route-level mismatches including spatial shift, endpoint error, and shape/topology mismatch.

### Relative-Target and Oracle-Start Ablation

| Model / target setting | Test ADE mean | Test FDE mean |
| --- | ---: | ---: |
| Absolute original-style decoder | 160.30 | 201.58 |
| Relative target, oracle start | 212.17 | 316.28 |

The relative-target + oracle-start ablation did not improve the overall test distribution. Compared with the absolute decoder, oracle-start relative decoding changed test ADE by -32.36% and test FDE by -56.90% when expressed as improvement, meaning that the aggregate metrics were worse in this ablation. However, the per-sample comparison table indicates that some previous worst cases improved, for example several high-ADE absolute cases had lower relative/oracle-start ADE. This mixed result suggests heterogeneous failure modes: oracle-start relative decoding may help some localization-dominated cases, while the overall distribution still worsens, indicating that shape/topology reconstruction under the current CNN-FC decoder may also be a limiting factor.

### Decoder Validation Diagnostic Summary

The original-style CNN-FC decoder, when trained on the clean paired no-speed dataset, substantially outperforms a naive mean-trajectory baseline. This suggests that the no-speed Hilbert/GAF/MTF RGB representation contains useful trajectory information that is decodable under the current experimental setting.

At the same time, the remaining test ADE/FDE and the train-test gap indicate that the current decoder does not yet support a strong claim of high-precision or uniformly reliable reconstruction. The representation is not shown to be non-decodable; rather, it appears partially decodable and meaningfully informative, while the current original-style decoder remains insufficient for a robust reconstruction claim.

The error analysis further suggests that the worst cases are not simply a function of trajectory length, tortuosity, bounding-box size, or sharp-turn count. Route-level failures appear to include spatial mismatch, endpoint error, and shape/topology mismatch. The relative-target + oracle-start ablation also did not improve aggregate test performance, which may point to reconstruction limitations beyond absolute spatial localization alone.

Overall, this should be framed as a controlled decoder validation step before evaluating diffusion-generated images. Before using the trained decoder as the main judge of generated-image quality, decoder reliability should be further validated or strengthened.

### Limitations

- The current conclusions apply to the original-style CNN-FC decoder and the present no-speed paired dataset split.
- The results do not yet establish robust reconstruction for diffusion-generated images.
- Performance on real encoded images is a controlled baseline and should not be treated as a guarantee for diffusion-generated images, which may contain artifacts or structural inconsistencies outside the real-image decoder training distribution.
- The relative-target result uses oracle start and should be treated as an ablation, not as a deployable reconstruction setting.
- The current diagnostics are based on simple geometric features and qualitative grouping; further inspection of individual failure modes is still needed.
- Map-space validity and map-matching quality are not evaluated in this decoder experiment.

```

### Stronger Decoder Ablation: ResNet18 No-Speed Decoder

A stronger-decoder ablation was run to distinguish representation limitation from the original CNN-FC decoder limitation.

The experiment used the same paired no-speed dataset and the same fixed split:

```text
data_no_speed_paired/
data_no_speed_paired/splits/split_metadata.csv
```

The dataset was not regenerated and the split was not changed. The experiment did not use speed, map inputs, diffusion, or GANs. It used the same absolute x/y labels as the original decoder experiment, the same train-only mean/std label normalization strategy, and all metrics were evaluated after converting predictions back into the original x/y coordinate space.

The new experiment was saved under:

```text
experiments/decoder_no_speed_resnet18_ablation/
```

Model and training configuration:

```text
model: scratch ResNet18 backbone
pretrained: false
head: feature_dim -> 1024 -> 448, reshaped to [224, 2]
optimizer: Adam
learning_rate: 1e-4
batch_size: 8
epochs: 1000
early_stopping_patience: 80
dropout: 0.3
weight_decay: 0.0
seed: 42
scheduler: none
```

The run completed successfully. The best checkpoint was selected at epoch 712, and early stopping triggered at epoch 792.

Comparison against `experiments/decoder_no_speed_original_config/`:

```text
Original CNN-FC test mean ADE: 160.3001
Original CNN-FC test mean FDE: 201.5806
ResNet18 test mean ADE:        120.9169
ResNet18 test mean FDE:        186.5896
ADE change:                    -24.57%
FDE change:                    -7.44%
```

Worst-case behavior also improved for ADE:

```text
Original test ADE p95: 538.3825
ResNet18 test ADE p95: 423.8822
Original test ADE max: 1624.0846
ResNet18 test ADE max: 1562.4547
```

FDE tail behavior improved at p95 but not at the single worst sample:

```text
Original test FDE p95: 715.2005
ResNet18 test FDE p95: 559.6885
Original test FDE max: 1725.3634
ResNet18 test FDE max: 1772.3219
```

Train/validation/test ADE for the ResNet18 model:

```text
train mean ADE: 34.2425
val mean ADE:   100.9557
test mean ADE:  120.9169
```

Train-test gap comparison:

```text
Original CNN-FC ADE train-test gap: 99.1354
ResNet18 ADE train-test gap:        86.6744
```

The ResNet18 decoder therefore improved test ADE/FDE and reduced the ADE worst-case tail. It also reduced the ADE train-test gap relative to the original CNN-FC decoder, although the absolute train/val/test separation remains substantial. This indicates that the original CNN-FC decoder was a bottleneck. At the same time, the remaining test ADE and the persistent validation/test gap suggest that the no-speed image representation may still limit precise trajectory reconstruction.




Saved artifacts include:

```text
experiments/decoder_no_speed_resnet18_ablation/evaluation/metrics_summary.json
experiments/decoder_no_speed_resnet18_ablation/evaluation/per_sample_metrics.csv
experiments/decoder_no_speed_resnet18_ablation/evaluation/per_timestep_ade.csv
experiments/decoder_no_speed_resnet18_ablation/evaluation/predictions/
experiments/decoder_no_speed_resnet18_ablation/evaluation/plots/
experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/comparison_summary.json
experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/comparison_table.csv
experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/test_ADE_distribution_comparison.png
experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/test_FDE_distribution_comparison.png
experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/best_median_worst_side_by_side_grid.png
experiments/decoder_no_speed_resnet18_ablation/resnet18_ablation_report_zh.txt
```

### Next Steps

- Inspect worst-case plots and categorize failure modes into localization, endpoint, shape, topology, and possible label/pairing issues.
- Compare the same dataset and split with a stronger decoder, such as a ResNet18-style backbone or a more spatial CNN architecture.
- Compare x/y labels with lon/lat labels if coordinate scaling or projection effects remain a concern.
- Evaluate diffusion-generated images through a trained decoder only after decoder reliability is better understood.
- Add map-space metrics after the image-to-trajectory reconstruction baseline is stable.

# Experiment inventory

## Scope and evidence policy

This inventory was prepared by content-searching the complete workspace on 2026-08-03. The workspace contains about 581,000 files, including 132 Python files, 216 Markdown files, 193 pairs of scheduler `.out`/`.err` logs, 561 CSV files, 554,323 NumPy arrays, 22,115 PNG figures, 659 PyTorch `.pt` files, 539 `safetensors` files, four `.npz` archives, and OSM data in GeoPackage and GraphML form. No Jupyter notebook (`.ipynb`) is present. An older Markdown report names original notebooks and an `experiments/` result tree that are no longer in the workspace; those experiments are retained below as **report-only evidence** rather than being silently dropped. Repeated per-sample arrays and figures are treated as artifact families rather than as separate scientific experiments.

Evidence labels used below:

- **Measured:** found in a retained JSON, CSV, log, or generated artifact.
- **Reconstructed:** derived by joining configs, source code, logs, and output directories.
- **Missing:** the experiment exists, but the requested evaluation was not retained or was never run.
- **Not comparable:** populations, coordinate systems, representations, or metrics differ.

## Repository and pipeline map

| Research component | Primary implementation | What it does |
|---|---|---|
| Legacy data loading | `src/cnr_trajectory/data/loader.py` | Reads the legacy file with `pandas.read_csv`; the `.xls` suffix is misleading because the source is CSV-formatted text. |
| Fixed-length preprocessing | `src/cnr_trajectory/encoding/sequence.py` | Squeezes long sequences and reflect/duplicates short sequences to length 224. |
| Hilbert representation | `src/cnr_trajectory/encoding/hilbert.py` | Recursively maps normalized 2D points to a scalar Hilbert index. |
| Legacy GAF/GASF/GADF/MTF | `src/cnr_trajectory/encoding/gaf.py`, `scripts/encode_trajectories.py` | Applies pyts GASF, GADF, and an 8-bin MTF to the Hilbert sequence and stores them as RGB images. |
| Delta-paired data | `scripts/build_no_speed_delta_displacement_paired_dataset.py` | Builds 3,159 paired 224-step delta targets and legacy GAF images. |
| Constrained GASF | `week04/scripts/build_constrained_gasf_dataset*.py`, `week05/scripts/build_delta_displacement_paired_data.py` | Encodes normalized `dx` and `dy` separately as GASF channels and a start-location heatmap as the third channel. |
| Pixel diffusion | `scripts/train_ddpm.py`, `scripts/train_ddpm_scratch.py`, `week04/scripts/train_week04_ddpm.py`, `week05/scripts/train_week05_ddpm.py` | Trains 2D U-Net DDPMs on 3-channel GAF/GASF images. |
| Latent diffusion | `src/cnr_trajectory/models/gaf_autoencoder.py`, `scripts/train_autoencoder.py`, `scripts/train_latent_ddpm.py` | Compresses GAF images and trains a 2D DDPM in the learned latent space. |
| Learned GAF inverse | `src/cnr_trajectory/reconstruction/delta_displacement.py`, decoder scripts in `scripts/` and `week03/scripts/` | Predicts displacement sequences and/or global trajectory attributes from GAF images. The nominal `decoder.py` is a placeholder. |
| Raw trajectory DDPM | `week06/scripts/trajectory_diffusion_common.py`, `train_absolute_trajectory_diffusion.py`, `train_delta_displacement_diffusion.py` | Trains temporal ResNet, dilated temporal ResNet, and 1D U-Net models directly on `[2,224]` coordinates or deltas. |
| Evaluation | `scripts/evaluate_fid.py`, `scripts/analyze_ddpm_delta_224_sanity.py`, Week 03/05/06 evaluation scripts | Computes image FID/KID/channel diagnostics, reconstruction ADE/FDE, geometric statistics, road distance, empirical support, and rejection rates. |
| Visualization | `scripts/plot_results.py`, decoder evaluation scripts, `week06/scripts/visualize_*` | Produces image grids, reconstruction plots, map overlays, contact sheets, and model comparisons. The package-level `trajectory_plots.py` is a placeholder. |
| OSM and map constraints | `src/cnr_trajectory/reconstruction/map_matching.py`, `src/cnr_trajectory/guidance/road_distance_field.py`, Week 03/06 scripts | Downloads/rasterizes OSM, fits coordinate alignment, evaluates nearest-road distance, filters trajectories, and performs gradient guidance/refinement. |

## Mathematical definitions used across experiments

For a normalized scalar sequence \(x_t\in[-1,1]\), the angular encoding is \(\phi_t=\arccos(x_t)\). The Gramian Angular Summation Field is

\[
G^{\mathrm{S}}_{ij}=\cos(\phi_i+\phi_j),
\]

and the Gramian Angular Difference Field is

\[
G^{\mathrm{D}}_{ij}=\sin(\phi_i-\phi_j).
\]

GASF is symmetric and GADF is antisymmetric. The Markov Transition Field used by the legacy pipeline discretizes the scalar sequence into eight bins and spreads transition probabilities over time indices.

The diffusion experiments use the standard forward process

\[
q(x_t\mid x_0)=\mathcal N\!\left(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right)
\]

and train a noise predictor with

\[
\mathcal L_{\mathrm{simple}}=\mathbb E_{x_0,\epsilon,t}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].
\]

For a predicted delta sequence \(\hat d_t\) and starting point \(p_0\), the reconstructed trajectory is \(\hat p_t=p_0+\sum_{k=1}^{t}\hat d_k\). Reconstruction is evaluated with average and final displacement error,

\[
\mathrm{ADE}=T^{-1}\sum_t\|\hat p_t-p_t\|_2,\qquad
\mathrm{FDE}=\|\hat p_T-p_T\|_2.
\]

## Phase 0 — preprocessing and representation validation

### Experiment P0.1 — Legacy Hilbert–GAF dataset construction

**Experiment name:** Legacy 224-step Hilbert–GASF/GADF/MTF RGB encoding.

**Motivation:** Convert variable-length 2D vehicle trajectories into fixed-size images suitable for off-the-shelf 2D CNNs and image diffusion.

**Research question:** Can spatial paths be represented as texture-like images without directly modeling coordinate sequences?

**Input representation:** Globally normalized longitude/latitude, recursively mapped to one Hilbert scalar per time step, fixed to 224 samples, then encoded as GASF/GADF/MTF RGB.

**Mathematical formulation:** Hilbert mapping \(h_t=H(x_t,y_t)\); red channel \(G^S(h)\), green channel \(G^D(h)\), blue channel \(\mathrm{MTF}_8(h)\).

**Implementation:** `loader.py`, `sequence.py`, `hilbert.py`, `gaf.py`, `scripts/encode_trajectories.py`, and the paired builders.

**Model architecture:** None; deterministic preprocessing.

**Training setting:** Not applicable.

**Dataset:** Legacy no-speed dataset, 3,159 vehicle trajectories, 224 points per retained trajectory.

**Hyperparameters:** Hilbert recursion tolerance `eps=0.0037`; sequence length 224; MTF bins 8; uint8 PNG storage in the original image path.

**Evaluation metrics:** File counts, shape checks, pairing checks, and later decoder reconstruction metrics.

**Results:** The dataset was successfully created and used throughout the root/Week 02 and Week 03 studies. It is compact for CNN input but does not retain an analytic 2D inverse.

**Figures available:** Thousands of encoded RGB images and later channel-structure visualizations; see `figure_inventory.md`.

**Checkpoint available:** Not applicable; paired CSV/NumPy/PNG artifacts are available.

**Problems encountered:** Variable-length sequences are squeezed or reflect-duplicated; Hilbert scalarization entangles the two spatial dimensions; uint8 quantization adds error; global normalization and later coordinate-source changes make provenance fragile.

**Current conclusion:** Useful as an exploratory image representation, but unsuitable as the clean final basis for trajectory synthesis because recovery of physical coordinates is learned, not guaranteed.

### Experiment P0.2 — Week 04 constrained GASF invertibility study

**Experiment name:** Separate-axis GASF with start heatmap, `[0,1]` versus `[-1,1]` normalization.

**Motivation:** Repair the missing inverse of the Hilbert representation by encoding `dx` and `dy` independently and making the diagonal analytically decodable.

**Research question:** Under which normalization is GASF invertible enough for generation?

**Input representation:** Red = GASF of normalized `dx`; green = GASF of normalized `dy`; blue = Gaussian start-location heatmap.

**Mathematical formulation:** Since \(G^S_{tt}=\cos(2\phi_t)=2x_t^2-1\), `[0,1]` normalization permits \(x_t=\sqrt{(G^S_{tt}+1)/2}\). With `[-1,1]`, only \(|x_t|\) is recoverable from the diagonal, producing sign ambiguity.

**Implementation:** `week04/scripts/build_constrained_gasf_dataset.py`, `build_constrained_gasf_dataset_neg11.py`, `check_alignment_modes.py`, `check_delta_consistency.py`, and `check_neg11_invertibility.py`.

**Model architecture:** None; deterministic reconstruction tests.

**Training setting:** Full 3,159-trajectory audit plus PNG and float round trips.

**Dataset:** Legacy delta-displacement dataset, 224 steps.

**Hyperparameters:** Per-axis min/max or symmetric normalization; float arrays and uint8 PNGs tested separately.

**Evaluation metrics:** ADE under alternative index alignments, maximum delta reconstruction error, sign fractions, diagonal magnitude and signed MAE.

**Results:** Correct alignment modes A/C gave mean ADE \(3.37\times10^{-17}\) and \(5.16\times10^{-14}\); the off-by-one mode B gave ADE 9.1718. Float reconstruction was accurate to about `1e-5`–`1e-4`. PNG round trips ranged from roughly 0.5 to more than 10 ADE, with maximum errors around 23. In `[-1,1]`, magnitude recovery was near exact, but assuming positive sign gave MAE 0.1540 (`dx`) and 0.1326 (`dy`); 32.10% and 30.75% of the corresponding normalized values were negative. Oracle signs restored exact reconstruction.

**Figures available:** GASF construction sanity images in both Week 04 data directories.

**Checkpoint available:** Not applicable.

**Problems encountered:** A first `[-1,1]` diagnostic crashed with `KeyError: fraction_negative_dx` and was rerun successfully. Quantized PNG storage is not a faithful inverse. The diagonal alone cannot resolve sign in `[-1,1]`.

**Current conclusion:** Float `[0,1]` GASF is almost lossless for real encoded data. This does not imply generated images are valid GASF matrices; generative manifold validity is the remaining difficulty.

### Experiment P0.3 — Week 05 fixed-length policy comparison

**Experiment name:** Variable-length, sliding-window, and bounce/truncate preparation.

**Motivation:** Adapt the newer 12,061-vehicle dataset to a 224-step fixed model input.

**Research question:** Which length policy retains the largest representative training set?

**Input representation:** Raw planar positions and derived deltas before GASF encoding.

**Mathematical formulation:** Windowing extracts length-224 subsequences with stride 112; bounce extension reflects a short trajectory repeatedly until length 224; long paths are truncated.

**Implementation:** `week05/scripts/build_delta_displacement_paired_data.py`, `window_existing_delta_dataset.py`, and `regularize_existing_delta_dataset_to_224_by_bounce.py`.

**Model architecture:** None.

**Training setting:** Deterministic preprocessing with reproducible train/validation/test split.

**Dataset:** `week05/data/vehicle_positions_TS_New.csv`, 2,189,591 rows, 12,061 IDs; lengths 13–657, mean 181.543, median 173; only 40 trajectories naturally have 224 points.

**Hyperparameters:** Target length 224; sliding stride 112; no tail window in the windowing run.

**Evaluation metrics:** Retained examples, skipped short trajectories, reconstruction error, and policy counts.

**Results:** Variable-length preparation retained all 12,061 and reconstructed within `3.18e-12`. Sliding windows produced 3,877 examples and skipped 8,738 short trajectories (72.45%). Bounce/truncate retained all 12,061: 8,738 bounced, 3,283 truncated, 40 unchanged, with maximum reconstruction error `3.41e-12`.

**Figures available:** Dataset samples under `week05/data`; no single summary plot was retained.

**Checkpoint available:** Not applicable; CSV/NumPy artifacts available.

**Problems encountered:** Bounce creates artificial reversal motifs; truncation discards endpoints; windowing excludes most vehicles and changes the sampling distribution.

**Current conclusion:** Bounce/truncate enabled the largest training set, but the fixed-length policy is a substantive modeling assumption and should be analyzed as a limitation.

## Phase 1 — legacy pixel-space diffusion

### Experiment P1.0 — Historical 128×128 DDPM refactoring sanity run

**Experiment name:** 3,000-step 128×128 DDPM pipeline validation.

**Motivation:** Verify that the notebook-derived workflow could be executed through the refactored loader, trainer, sampler, checkpoint, and FID scripts.

**Research question:** Does the extracted pipeline train end to end, and does its preliminary FID improve with training?

**Input representation:** Legacy RGB Hilbert–GASF/GADF/MTF images resized to 128×128.

**Mathematical formulation:** Standard DDPM noise prediction with the notebook-consistent auxiliary formulation described in the historical report.

**Implementation:** `scripts/train_ddpm.py`, `generate_ddpm_samples.py`, and `evaluate_fid.py`; details are preserved in `docs/weekly_report_refactoring_validation.md`.

**Model architecture:** 2D U-Net DDPM at 128×128.

**Training setting:** 3,000 steps on an RTX 4060 Laptop GPU; 100 generated samples; checkpoints at 1k, 2k, 3k/final.

**Dataset:** Legacy encoded trajectory-image set.

**Hyperparameters:** Batch 4, learning rate `2e-4`, EMA; elapsed 884.49 s at 3.39 steps/s.

**Evaluation metrics:** Final training loss and 100-sample image FID by checkpoint.

**Results:** Final loss 0.004508. FID fell from 288.81 at 1k to 254.43 at 2k and 226.14 at 3k/final.

**Figures available:** The historical report names `results/ddpm_sanity_128/samples/` and `generated_100/`, but those directories are absent from the current workspace.

**Checkpoint available:** Not in the current workspace; checkpoint paths are report-only.

**Problems encountered:** Small sample count and short training; only proves pipeline operation. Original notebooks referenced by the report are also absent now.

**Current conclusion:** Successful engineering validation, not a final scientific baseline.

### Experiment P1.0b — Historical 224×224 delta-GAF DDPM sanity matrix

**Experiment name:** Notebook auxiliary loss versus MSE-only versus data-driven diagonal, 3,000 steps.

**Motivation:** Validate generation at the decoder's native 224×224 resolution and test whether old representation-specific auxiliary terms transfer to delta-displacement images.

**Research question:** Which short-run objective best preserves image statistics and decoded-trajectory validity?

**Input representation:** Legacy Hilbert–GASF/GADF/MTF RGB constructed from delta-displacement trajectories.

**Mathematical formulation:** Noise MSE alone; notebook auxiliary \(\mathcal L=\mathcal L_{MSE}+0.01\mathcal L_{R\text{-sym}}+0.01\mathcal L_{G\text{-diag-fixed}}\); or data-driven diagonal \(\mathcal L=\mathcal L_{MSE}+0.01\mathcal L_{diag-data}\).

**Implementation:** Historical runs used `scripts/train_ddpm.py` and `scripts/analyze_ddpm_delta_224_sanity.py`; full tables remain in `docs/weekly_report_refactoring_validation.md`.

**Model architecture:** 224×224 2D U-Net DDPM.

**Training setting:** Three matched 3,000-step runs; batch 2, accumulation 2 (effective 4), learning rate `1e-4`, EMA; 100 final samples each.

**Dataset:** Legacy 3,159-image delta-displacement paired dataset.

**Hyperparameters:** Auxiliary weights 0.01 where used; data-driven diagonal target `[-0.3690, 0.0039, 0.7750]` from 2,527 train images.

**Evaluation metrics:** Final loss, FID, KID, channel/structure/diversity statistics, decoded delta/path length, out-of-range and jump ratios.

**Results:** Notebook auxiliary: loss 0.03244, FID 197.456, KID 0.2276, generated path length 1442.13, out-of-range sample ratio 0.69. MSE-only: loss 0.0138568, FID 137.419, KID 0.1199, path length 1639.67, out-of-range samples 0.57. Data-driven diagonal: loss 0.01529 (MSE 0.01415, diagonal 0.11389), FID 134.957, KID 0.1167, path length 1679.43, out-of-range samples 0.61. Real mean path length was 2054.49. The diagonal run improved B diagonal mean to 0.7379 versus real 0.8871, compared with 0.5042/0.4688 for auxiliary/MSE.

**Figures available:** Visual comparison is described in the historical report, but `results/ddpm_delta_displacement_224_*` is absent from the current workspace.

**Checkpoint available:** Not currently present; report-only paths.

**Problems encountered:** All three are short sanity runs; old fixed auxiliary assumptions were channel-specific and transferred poorly; trajectory-length variance remained only 0.1223–0.1460 of real; 57–69% of samples had an out-of-range point.

**Current conclusion:** MSE-only is the cleaner default despite the diagonal run's slightly lower FID/KID. These results first exposed the image-metric/trajectory-validity mismatch.

### Experiment P1.1 — Pure-MSE and data-driven-diagonal DDPM

**Experiment name:** 224×224 legacy GAF DDPM, PureMSE versus DataDrivenDiag.

**Motivation:** Establish whether a standard image DDPM can reproduce trajectory-encoded textures and whether an explicit diagonal constraint helps.

**Research question:** Does adding diagonal supervision improve GAF image fidelity?

**Input representation:** Legacy Hilbert GASF/GADF/MTF RGB, scaled to `[-1,1]`.

**Mathematical formulation:** PureMSE uses \(\mathcal L_{simple}\). DataDrivenDiag adds MSE between predicted and precomputed real diagonal targets; optional symmetry terms penalize disagreement with transpose.

**Implementation:** `scripts/train_ddpm.py`, `train_ddpm_scratch.py`, `precompute_diag_targets.py`, `generate_ddpm_samples.py`, and evaluation scripts.

**Model architecture:** Diffusers `UNet2DModel`; for 224/128-scale inputs, two layers per block, widths `(128,128,256,256,512,512)`, one attention down/up stage, three input/output channels.

**Training setting:** 50,000 optimizer steps, batch 8, AdamW, cosine schedule with warmup, EMA, 1,000 diffusion timesteps, seed 42; checkpoints every 2,500. Earlier 10-step smoke and 10,000-step baselines used the same family.

**Dataset:** 3,159 legacy GAF images.

**Hyperparameters:** Learning rate `2e-4`; the primary evaluation used 80 matched samples formed by 40 samples from checkpoint 25k plus 40 from checkpoint 50k, at 50 or 150 reverse steps. A separate checkpoint-50k-only audit used 40 samples.

**Evaluation metrics:** Image FID, channel means/std/correlation/saturation, GASF/GADF/MTF symmetry, and visual grids.

**Results:** In the 80-image 25k+50k aggregate, at 150 DDPM steps PureMSE FID was 74.62 and DataDrivenDiag 81.03; at 50 steps they were 78.03 and 93.93. In the 40-image checkpoint-50k-only 150-step audit, FID was 87.996 and 99.054. An earlier unmatched/mixed periodic-sample-directory diagnostic gave 86.337 and 84.553 and should not be used as the primary comparison. PureMSE won both matched step-count comparisons; extra sampling steps helped both. The real channel means were approximately `[-0.383, 0.0004, -0.1905]`; aggregate PureMSE generated `[0.0137, 0.0033, -0.0719]`, showing remaining distribution mismatch.

**Figures available:** Training samples in each run; `representative_comparison_grid.png` in the 50- and 150-step analysis directories; channel-correlation heatmaps.

**Checkpoint available:** Yes, including 25k, 50k, and final checkpoints under `results_hpc/ddpm_scratch_50000step_bs8_*`.

**Problems encountered:** Small dataset; image FID uses Inception features trained on natural images; only 80 matched aggregate samples and 40 checkpoint-only samples; FID changes substantially with sample provenance; diagonal loss worsened the matched FID; no reliable trajectory inverse for this representation.

**Current conclusion:** Standard pixel DDPM can learn visible GAF texture, but diagonal regularization did not improve the best image metric and image quality alone cannot establish valid trajectory generation.

### Experiment P1.2 — Channel normalization and MTF symmetry ablations

**Experiment name:** ChannelNorm and MTFSym, each with PureMSE and DataDrivenDiag.

**Motivation:** Correct severe inter-channel distribution imbalance and enforce known MTF symmetry.

**Research question:** Do representation-aware channel statistics or symmetry constraints improve generation?

**Input representation:** Same legacy RGB GAF images as P1.1.

**Mathematical formulation:** ChannelNorm standardizes each channel with empirical moments. MTFSym adds \(\|B-B^\top\|^2\); diagonal variants additionally use the diagonal target loss.

**Implementation:** Options inside `scripts/train_ddpm.py`/`train_ddpm_scratch.py`; diagnostics in `analyze_gaf_channel_diagnostics.py` and `analyze_mtf_sym_150_evaluation.py`.

**Model architecture:** Same 2D U-Net as P1.1.

**Training setting:** Four 50,000-step runs, otherwise matched to P1.1.

**Dataset:** 3,159 legacy GAF images.

**Hyperparameters:** 150-step, 80-sample primary comparison; empirical channel normalization; MTF symmetry loss configured per run.

**Evaluation metrics:** FID, channel statistics/saturation/correlation, and symmetry errors.

**Results:** ChannelNorm FID was 130.62 (PureMSE) and 137.00 (diagonal): means matched better but FID became much worse. MTFSym FID was 84.98 and 93.96. MTF symmetry improved to 0.1207/0.1124 from raw generated 0.2715/0.2280, toward the real 0.0673, but blue-channel saturation increased to 0.4423/0.3866.

**Figures available:** Representative comparison grids and correlation heatmaps under the corresponding `results_hpc/*analysis` and diagnostics directories.

**Checkpoint available:** Yes, complete 50k checkpoint series for all four runs.

**Problems encountered:** Optimizing one structural statistic displaced other image statistics. Better mean or symmetry agreement did not imply better FID.

**Current conclusion:** Hand-added pixel constraints trade among incompatible image statistics and do not solve trajectory validity.

### Experiment P1.3 — DDPM step-count and DDIM sampler ablation

**Experiment name:** 50 versus 150 DDPM steps and 150-step DDIM.

**Motivation:** Separate model quality from reverse-sampler choice and computational budget.

**Research question:** Can a faster/deterministic sampler match the retained DDPM checkpoints?

**Input representation:** Legacy RGB GAF images.

**Mathematical formulation:** Same trained score/noise model; DDPM ancestral versus DDIM implicit reverse updates.

**Implementation:** `scripts/generate_ddpm_samples.py` and analysis scripts.

**Model architecture:** Frozen P1.1 models.

**Training setting:** No retraining; matched 50k checkpoints and 80 generated images.

**Dataset:** Same 3,159-image training population and real reference set.

**Hyperparameters:** DDPM 50/150 inference steps; DDIM 150 steps.

**Evaluation metrics:** FID and channel diagnostics.

**Results:** DDPM improved from 78.03 to 74.62 (PureMSE) and 93.93 to 81.03 (diagonal) when increasing 50 to 150 steps. DDIM at 150 steps scored 93.41 and 98.71, worse than ancestral DDPM.

**Figures available:** DDIM/diagnostic samples and the two DDPM representative grids.

**Checkpoint available:** Uses P1.1 checkpoints; generated samples retained.

**Problems encountered:** Small evaluation sample and no uncertainty intervals.

**Current conclusion:** The 150-step ancestral DDPM is the strongest legacy sampler tested, but sampler tuning does not address representation inversion.

### Experiment P1.4 — Post-hoc channel affine calibration

**Experiment name:** Channel-wise mean/std calibration of generated legacy-GAF images.

**Motivation:** Determine how much of the FID gap is caused by channel marginal shift rather than spatial structure.

**Research question:** Does matching generated RGB channel moments to real moments improve image FID without retraining?

**Input representation:** Eighty 150-step samples each from the 50k PureMSE and DataDrivenDiag models.

**Mathematical formulation:** Per channel, \(x'_c=((x_c-\mu^{gen}_c)/\sigma^{gen}_c)\sigma^{real}_c+\mu^{real}_c\), followed by image-range clipping.

**Implementation:** Channel calibration/diagnostic path in `scripts/analyze_gaf_channel_diagnostics.py`; outputs in `results_hpc/channel_diagnostics/*_calibrated` and `calibrated_image_quality_analysis`.

**Model architecture:** No new model; deterministic post-processing of P1.1 samples.

**Training setting:** No training; matched 80-image evaluation.

**Dataset:** Same real 3,159-image reference and generated subsets as P1.1.

**Hyperparameters:** Empirical real/generated mean and standard deviation independently per RGB channel.

**Evaluation metrics:** FID, RGB mean/std, diagonal statistics, global symmetry, and edge strength.

**Results:** PureMSE FID improved 74.6189→63.2486; DataDrivenDiag 81.0313→67.6198. PureMSE RGB mean moved to `[-0.3641,0.0011,-0.1623]` and diagonal to `[-0.3633,0.0028,0.6077]`; diagonal-model RGB mean moved to `[-0.3631,0.0006,-0.2037]` and diagonal to `[-0.3859,0.0044,0.7701]`. Global symmetry slightly worsened 0.2738→0.2940 and 0.2892→0.2907.

**Figures available:** Eighty calibrated images per checkpoint/method family, three correlation heatmaps, and `calibrated_image_quality_analysis/representative_comparison_grid.png`.

**Checkpoint available:** Uses existing P1.1 checkpoints; calibrated images and metrics retained.

**Problems encountered:** This uses the real test/reference moments and is post-hoc distribution fitting; improved FID does not demonstrate a better generator or better trajectories.

**Current conclusion:** Channel marginal mismatch is a large component of legacy image FID, but structural and physical validity remain unresolved. Calibration is diagnostic, not a deployable generative result.

## Phase 2 — latent diffusion and learned inversion

### Experiment P2.0 — Historical absolute/relative decoder reproductions

**Experiment name:** Original-style CNN-FC, naive mean, relative-target oracle-start, and scratch ResNet18 absolute decoders.

**Motivation:** Establish whether real legacy GAF images contain decodable coordinate information and whether decoder capacity or target parameterization is the bottleneck.

**Research question:** How much of reconstruction failure comes from the representation versus a weak decoder?

**Input representation:** Legacy no-speed Hilbert–GASF/GADF/MTF RGB; absolute 224×2 target or relative target with true start.

**Mathematical formulation:** Train-normalized coordinate MSE; ADE/FDE after inverse normalization. The naive baseline predicts the train mean trajectory.

**Implementation:** `run_original_config_decoder_experiment.py`, `run_relative_target_decoder_experiment.py`, `train_decoder_no_speed_baseline.py`, and evaluation/error-analysis scripts. Quantitative evidence is preserved in `docs/weekly_report_refactoring_validation.md`.

**Model architecture:** Original-style CNN-FC and scratch ResNet18 with a `feature_dim→1024→448` head.

**Training setting:** Split 2,527/315/317. CNN-FC: batch 8, lr `1e-4`, dropout 0.3, early stopping at 717, best epoch 637. ResNet18: same main settings, best epoch 712, stopped 792.

**Dataset:** 3,159 legacy paired absolute trajectories.

**Hyperparameters:** Adam, maximum 1,000 epochs, patience 80, seed 42; true start for the relative ablation.

**Evaluation metrics:** ADE/FDE, naive improvement, prediction variance, error correlations, p95/max tails.

**Results:** Naive mean 640.89/730.67 ADE/FDE. CNN-FC 160.30/201.58, a 74.99%/72.41% improvement; test predicted/GT variance ratio 0.852. Relative + oracle start worsened to 212.17/316.28. Absolute ResNet18 improved to 120.9169/186.5896, with ADE p95 423.88 versus 538.38 for CNN-FC. CNN-FC ADE correlated strongly with centroid error (0.9825), start error (0.8556), and endpoint error (0.8121), but weakly with basic complexity features.

**Figures available:** The historical report names qualitative grids and distribution plots under `experiments/decoder_no_speed_*`, but the entire `experiments/` tree is absent now.

**Checkpoint available:** Not in the current workspace; paths are historical report references.

**Problems encountered:** Report-only provenance; original notebook workflow and its additional map-channel, invalid-point, BCE, binary-output, and LSTM-style variants are described but neither notebooks nor quantitative artifacts survive. Relative targets helped some worst cases but worsened aggregates.

**Current conclusion:** GAF images are informative, and decoder capacity matters, but reconstruction remains too inaccurate to use an early decoder as a neutral judge of generated images.

### Experiment P2.1 — GAF autoencoder and latent DDPM

**Experiment name:** 14×14 and 28×28 learned-latent diffusion.

**Motivation:** Reduce the quadratic cost of 224×224 image diffusion.

**Research question:** Can a compressed GAF latent retain reconstruction quality while supporting diffusion?

**Input representation:** Legacy RGB GAF image compressed by `GAFAutoencoder` to 14×14 or 28×28 latent maps.

**Mathematical formulation:** Autoencoder loss combines L1 and MSE; latent DDPM uses the standard noise-prediction MSE.

**Implementation:** `src/cnr_trajectory/models/gaf_autoencoder.py`, `scripts/train_autoencoder.py`, `encode_dataset_to_latent.py`, `train_latent_ddpm.py`, `generate_latent_samples.py`, and `evaluate_latent_generation.py`.

**Model architecture:** Convolutional encoder/decoder; latent 2D U-Net with base widths 64/128. Runs include a known bad baseline, 14×14 v2, and 28×28.

**Training setting:** Autoencoder 50 epochs; both principal latent DDPMs 50,000 steps.

**Dataset:** 3,159 legacy images and their encoded latents.

**Hyperparameters:** Latent spatial sizes 14 and 28; other optimizer details are retained in run summaries.

**Evaluation metrics:** Autoencoder MAE/MSE, rolling training loss, and visual sample grids.

**Results:** The 28×28 autoencoder reached MAE 0.039219 and MSE 0.006421. Best rolling latent-DDPM loss was 0.075972 at 14×14 and 0.061122 at 28×28. Thirty-two best and thirty-two final samples are retained per principal run.

**Figures available:** 84 PNGs in each principal latent result directory, including training/sample grids.

**Checkpoint available:** Yes: autoencoder and latent DDPM best/final checkpoints. The bad-baseline directory is also retained.

**Problems encountered:** No FID/KID or decoded trajectory evaluation was completed, so lower latent loss cannot be interpreted as better generation.

**Current conclusion:** Computationally feasible but scientifically incomplete; it should be documented as an explored branch, not a comparative thesis result.

### Experiment P2.2 — Delta decoder architecture/channel/loss ablation

**Experiment name:** CNN, multibranch, mid-fusion, normalization, channel-only, and integrated-loss decoder studies.

**Motivation:** Recover physical 2D deltas from legacy GAF images and identify which channels carry reconstructive information.

**Research question:** Can a learned inverse overcome the non-analytic Hilbert/GAF representation?

**Input representation:** R=GASF, G=GADF, B=MTF; targets are 224×2 deltas and their integrated coordinates.

**Mathematical formulation:** \(\mathcal L=\mathrm{MSE}_{delta}+\beta\,\mathrm{MSE}_{integrated}\), optionally on normalized targets.

**Implementation:** `train_delta_displacement_decoder.py`, `train_multi_branch_delta_decoder.py`, decoder runner scripts, and `collect_delta_decoder_results.py`.

**Model architecture:** Simple CNN, three-branch CNN, RGB mid-fusion, ResNet18 variants, and single-channel variants.

**Training setting:** Common split 2,527/315/317, 100 epochs, batch 16, learning rate `1e-3`.

**Dataset:** 3,159 paired legacy images/delta trajectories.

**Hyperparameters:** Integrated loss \(\beta\in\{0,0.05,0.2\}\); raw or normalized targets; R/G/B-only ablations.

**Evaluation metrics:** Delta ADE and integrated coordinate ADE/FDE on the test split.

**Results:** Mid-fusion RGB with \(\beta=0.05\) was best in this family at 139.43 ADE / 217.99 FDE. Mid-fusion \(\beta=0.2\): 141.37/234.06; normalized mid-fusion: 145.58/237.31; simple CNN: 149.34/232.32; normalized simple CNN: 158.56/249.70. Multibranch \(\beta=0\): 208.27/356.35 despite delta ADE 6.59, demonstrating cumulative drift; \(\beta=0.05\): 162.92/257.63; the HPC \(\beta=0.2\) repeat: 164.93/272.77. A second retained local \(\beta=0.2\) run in `results/decoder_multibranch_delta` reached 153.43/249.62 (best epoch 96); it should be reported as a separate stochastic/run-provenance result, not silently substituted. Single channels R/G/B gave 205.26/335.30, 183.81/296.22, and 397.77/720.23 respectively. A delta-displacement ResNet18 integrated from the true start reached 135.29/224.34, versus 120.92/186.59 for its absolute-target comparator; it improved aligned shape and worst-case tails despite worse raw means.

**Figures available:** Per-sample reconstruction comparison plots and error analyses under each decoder run.

**Checkpoint available:** Yes, best checkpoints and summaries for all listed runs.

**Problems encountered:** Locally small delta errors accumulate into large path drift. MTF alone is especially weak. Metrics from older original/relative-target scripts use adjacent targets and should not be merged without checking their split definitions.

**Current conclusion:** A learned inverse is possible but remains a major source of error; the representation has shifted the generative problem into a second supervised model.

### Experiment P2.3 — Hybrid global heads, geometry loss, and centroid correction

**Experiment name:** Hybrid multitask V1/V2/V3 and post-hoc centroid correction.

**Motivation:** Reduce cumulative drift by predicting start/centroid/global geometry alongside local deltas.

**Research question:** Do global attributes or explicit heading/turn losses stabilize integration?

**Input representation:** Legacy RGB GAF image with delta, start, centroid, and geometry targets depending on version.

**Mathematical formulation:** Weighted sums of delta, integrated-path, centroid/start, endpoint, heading, and turn losses. Centroid runs predict a correction added to the integrated path.

**Implementation:** `run_hybrid_multitask_delta_decoder*.py` and `run_delta_centroid_correction_decoder_ablation.py`.

**Model architecture:** Shared CNN/ResNet image encoder with sequence and global regression heads.

**Training setting:** Full train/validation/test runs with best checkpoint selection; version-specific weights retained in each config.

**Dataset:** Same 3,159 paired legacy dataset and split.

**Hyperparameters:** V3 mild/strong geometry weights; centroid correction weights 0.05, 0.10, 0.20.

**Evaluation metrics:** Delta ADE; integrated ADE/FDE; oracle versus learned-start integration; geometry metrics for V3; learned versus ground-truth correction.

**Results:** V1 reached 125.01/170.00 integrated ADE/FDE. V2 was best at 107.85/147.20; oracle-start ADE was unexpectedly worse at 158.46. V3 mild reached 124.95/164.06 and strong 132.31/184.92, so geometry losses did not improve V2. Centroid weights 0.05/0.10/0.20 produced corrected ADE 226.84/215.63/209.84, while ground-truth correction gave roughly 99–105, revealing a learnability gap.

**Figures available:** Evaluation plots in each hybrid/centroid directory.

**Checkpoint available:** Yes, best checkpoints for all V1–V3 and centroid runs.

**Problems encountered:** Oracle-start behavior indicates that some heads/errors compensate one another; geometry terms improved selected local quantities but not the main path metric; learned centroid corrections were poor.

**Current conclusion:** V2 is the strongest root-era learned inverse, but added loss engineering has diminishing returns and unstable interpretation.

### Experiment P2.4 — Translation and similarity-transform upper bounds

**Experiment name:** Oracle centroid translation and similarity alignment for delta ResNet18 and Hybrid V1 predictions.

**Motivation:** Quantify how much reconstruction error is global placement/scale/rotation rather than local path topology.

**Research question:** What error remains after an oracle rigid similarity transform to ground truth?

**Input representation:** Retained test predictions and ground-truth trajectories; no image input is reprocessed.

**Mathematical formulation:** Compare raw paths, centroid translation, and the least-squares similarity transform consisting of translation, uniform scale, and rotation, without shear/nonuniform scale.

**Implementation:** `scripts/evaluate_affine_upper_bound.py`; outputs under each run's `evaluation/affine_upper_bound`.

**Model architecture:** Post-hoc oracle analysis; models are delta ResNet18 and Hybrid V1.

**Training setting:** No new training; all 317 test samples.

**Dataset:** Legacy decoder test split.

**Hyperparameters:** Similarity transform only; no shear.

**Evaluation metrics:** Raw/translated/aligned ADE/FDE, rotation, scale, translation norm, and turn-sharpness preservation.

**Results:** Delta ResNet18: raw 173.41/297.28, translation 97.48/169.45, similarity 55.47/94.46; mean rotation 6.47°, scale 1.114, translation norm 376.92. Hybrid V1: raw 125.01/170.00, translation 73.77/120.72, similarity 57.01/92.20; rotation 3.18°, scale 1.062, translation norm 187.12. Similarity improved ADE over raw for 317/317 delta samples and 316/317 Hybrid samples; turn-sharpness ratio was unchanged by construction.

**Figures available:** Metrics CSV/JSON and text reports are retained; the `plots` paths contain no PNGs.

**Checkpoint available:** Uses retained decoder checkpoints/predictions; no new checkpoint.

**Problems encountered:** Oracle ground-truth alignment is not deployable and raw delta numbers differ from the primary decoder summary because this analysis uses its own retained prediction/evaluation path. It is an upper bound only.

**Current conclusion:** A large share of inverse error is global translation/scale/orientation, but substantial local error remains; this motivated learned global heads and correction experiments.

## Phase 3 — structure-aware inversion, temporal latent alignment, and maps

### Experiment P3.1 — GAF structural projection and decomposition

**Experiment name:** Raw, hard-projected, decomposed, and decomposed-plus-local-MTF ResNet18 decoders.

**Motivation:** Give the decoder the exact symmetry/antisymmetry structure of the image channels.

**Research question:** Does separating mathematically valid structure from residual error improve reconstruction?

**Input representation:** Raw RGB; hard `[R_sym,G_antisym,B]`; decomposed `[R_sym,R_res,G_antisym,G_res,B]`; and decomposed plus local MTF features.

**Mathematical formulation:** \(R_{sym}=(R+R^\top)/2\), \(R_{res}=(R-R^\top)/2\); \(G_{anti}=(G-G^\top)/2\), \(G_{res}=(G+G^\top)/2\).

**Implementation:** `compute_channel_structure_stats.py`, `structure_aware_resnet18_models.py`, and `train_structure_resnet18_decoder.py`.

**Model architecture:** ResNet18 sequence decoder with variant-specific input stems/local MTF branch.

**Training setting:** Batch 16 and long epoch-based training with best validation checkpoint; best epochs 711, 599, 798, and 611.

**Dataset:** 3,159 legacy images, same split family.

**Hyperparameters:** Variant-specific channel decomposition; otherwise matched decoder setup.

**Evaluation metrics:** Structural residuals, delta ADE/FDE, integrated ADE/FDE.

**Results:** Real channel structure was nearly exact: GASF symmetry error 0, GADF antisymmetry mean 0.000884, MTF symmetry 0.0336. Raw decoder: delta 4.289/6.981, integrated 151.56/255.95. Hard: 4.156/6.963 and 143.85/247.90. Decomposed: 3.945/6.492 and 140.16/235.19. Decomposed+MTF-local: 4.012/6.789 and 138.10/230.69, best of this card.

**Figures available:** Eight channel-component visualizations, loss curves, and ADE/FDE histograms.

**Checkpoint available:** Yes, best checkpoints in all four result directories.

**Problems encountered:** Improvements are modest relative to remaining integration error; training is long for a small dataset.

**Current conclusion:** Known representation structure helps learned inversion, but cannot eliminate cumulative drift.

### Experiment P3.2 — Drift-aware correction heads

**Experiment name:** E1 raw correction, E1b correction target, E2 decomposed local-MTF correction, and stronger E2.

**Motivation:** Explicitly predict accumulated endpoint/trajectory correction.

**Research question:** Can a learned global drift field correct locally plausible deltas?

**Input representation:** Raw or decomposed/local-MTF GAF features.

**Mathematical formulation:** A correction head predicts a low-dimensional/global correction distributed over the integrated path; E1b/E2 supervise the required correction directly.

**Implementation:** `drift_aware_*models.py` and `train_drift_aware_*decoder*.py`.

**Model architecture:** ResNet18 sequence backbone plus drift/correction head.

**Training setting:** Full decoder training; E2 correction weights 0.1 and 0.2 variants.

**Dataset:** Same legacy paired split.

**Hyperparameters:** E2 base correction 0.1; strong version 0.2 plus magnitude scaling retained in directory name.

**Evaluation metrics:** Raw and corrected integrated ADE/FDE, correction norm ratio, train/validation behavior.

**Results:** E1: 154.17/268.30 to 150.90/262.78 (~2%) and hurt train performance. E1b: 146.87/249.56 to 142.23/241.82 (~3%), norm ratio 0.1465. E2: 135.37/230.69 to 131.15/224.09 (~3%), norm ratio 0.1687, the best Week 03 reconstruction. Strong E2: 140.26/234.44 to 137.20/229.19, worse than base E2.

**Figures available:** Component/total loss curves and raw/corrected error histograms; drift diagnostic examples.

**Checkpoint available:** Yes for all four runs.

**Problems encountered:** Correction heads yield only small gains and stronger weighting degrades the base predictor.

**Current conclusion:** Drift is real and partially correctable, but the inverse remains the bottleneck.

### Experiment P3.3 — Sequence-latent alignment

**Experiment name:** Trajectory autoencoder plus rowwise, temporal-convolution, Transformer, and MTF-only image-to-sequence latent mapping.

**Motivation:** Map image features into a latent space explicitly trained to represent trajectory sequences.

**Research question:** Does temporal inductive bias improve GAF-to-trajectory inversion?

**Input representation:** Legacy RGB GAF image; target is a trajectory-sequence-autoencoder latent.

**Mathematical formulation:** Stage 1 minimizes trajectory reconstruction; Stage 2 minimizes latent alignment and decoded trajectory error.

**Implementation:** `sequence_latent_models.py`, `sequence_latent_utils.py`, `train_trajectory_sequence_autoencoder.py`, and `train_image_sequence_latent_alignment.py`.

**Model architecture:** Sequence autoencoder latent 64/128; rowwise MLP/CNN, temporal convolution, deeper kernel-7 temporal convolution, Transformer, and MTF-only Transformer.

**Training setting:** Two-stage training and test evaluation; configs stored with each result.

**Dataset:** Same 3,159 trajectories/images.

**Hyperparameters:** Latents 64 and 128; temporal-conv depth/kernel variants; Transformer and modality ablation.

**Evaluation metrics:** Integrated ADE/FDE after latent decoding.

**Results:** Stage-1 latent64 achieved 2.341/9.342 and latent128 2.517/11.960, showing the trajectory AE itself is accurate. Stage-2 rowwise failed at 239.08/431.80; temporal conv reached 178.65/319.65; deeper kernel-7 temporal conv reached 137.795/239.577; Transformer 146.600/254.415; MTF-only Transformer failed at 542.484/988.853.

**Figures available:** Loss/evaluation figures in sequence-latent run directories.

**Checkpoint available:** Yes for autoencoders and alignment models.

**Problems encountered:** Good trajectory latent reconstruction does not imply that GAF images can predict those latents. MTF-only carries insufficient information.

**Current conclusion:** Temporal bias helps, but the best aligned model remains worse than E2 and V2; this branch is a valuable negative result.

### Experiment P3.4 — Coordinate-to-OSM alignment and local map construction

**Experiment name:** Affine calibration, OSM download, local map raster, and road-distance-field generation.

**Motivation:** Measure and enforce road plausibility.

**Research question:** Can raw coordinates be placed in a metric OSM frame with differentiable road-distance information?

**Input representation:** Legacy raw XY/longitude-latitude and OSM road geometries.

**Mathematical formulation:** Least-squares affine map \([E,N,1]^\top=A[x,y,1]^\top\); local distance field stores nearest-road distance for raster pixels.

**Implementation:** `download_osm_map.py`, `fit_raw_xy_to_osm_affine.py`, `rasterize_local_maps.py`, and `generate_local_road_distance_fields.py`.

**Model architecture:** None.

**Training setting:** Full-dataset calibration and 3,159 local crops for oracle-bbox and start-centered modes.

**Dataset:** Legacy source at the recorded absolute path plus OSM; initial graph 1,465 nodes/2,672 edges, EPSG:32633.

**Hyperparameters:** 224×224 local rasters; crop definitions vary by oracle/start mode.

**Evaluation metrics:** Affine residual, visual alignment, graph/map counts.

**Results:** 707,616 points across 3,159 trajectories were fit; the affine matrix was effectively identity plus approximately `[506600, 4150650]`, with residual below `1e-6`. All local rasters/distance fields were generated.

**Figures available:** Twelve vehicle alignment figures and thousands of local road maps/distance fields.

**Checkpoint available:** No model; affine/GraphML/GeoPackage/raster artifacts retained.

**Problems encountered:** Week 05 uses a different coordinate source; reusing this calibration later produced severe coverage error.

**Current conclusion:** The legacy map pipeline is internally aligned, but its transform is dataset-specific and must not be reused without validation.

### Experiment P3.5 — Map-aware decoder fusion

**Experiment name:** Early/late fusion with oracle-bbox and start-centered maps.

**Motivation:** Test whether road context improves reconstruction from GAF images.

**Research question:** Can local road geometry regularize the learned inverse?

**Input representation:** GAF image plus a local road raster, with either future-trajectory oracle bbox or only start-centered context.

**Mathematical formulation:** Early fusion concatenates map/image inputs; late fusion embeds each separately before feature fusion.

**Implementation:** Map-aware branches inside `train_structure_resnet18_decoder.py` and associated run configurations/evaluation scripts.

**Model architecture:** ResNet18 early-concat or late-fusion sequence decoder.

**Training setting:** Partial early-fusion runs, full early-fusion runs, and full late-fusion runs; smoke run retained separately.

**Dataset:** Legacy paired data plus local OSM rasters.

**Hyperparameters:** Oracle-bbox versus start-centered crops; partial/full raster sets.

**Evaluation metrics:** Integrated ADE/FDE and nearest-road distance.

**Results:** Early partial oracle 159.86/266.72; early partial start 168.20/293.44. Full early oracle 155.69/258.72; full start again 168.20/293.44. Late oracle achieved 123.25/185.39, but uses future bbox; late start-centered achieved 141.27/250.29. For the oracle late model, predicted road distance was 12.55 m versus ground-truth 1.82 m and off-road-over-10 m was 0.462 versus 0.00112.

**Figures available:** Loss curves and error histograms for each run; local-map examples.

**Checkpoint available:** Yes for full runs.

**Problems encountered:** The strongest number depends on an oracle crop defined using the future path and is therefore an upper bound, not a deployable result. Better ADE did not ensure road adherence.

**Current conclusion:** Map context can help, but fair conditioning and direct road metrics are essential.

### Experiment P3.6 — Differentiable map refinement sweep

**Experiment name:** Sliding-window road/reference/curvature refinement, 108 configurations.

**Motivation:** Project reconstructed paths closer to roads without retraining the decoder.

**Research question:** Can differentiable post-processing improve road validity without excessive trajectory distortion?

**Input representation:** Predicted trajectories, oracle local map distance fields, and reference predictions.

**Mathematical formulation:** Minimize weighted road-distance, reference-deviation, and curvature energies over path points.

**Implementation:** `refine_trajectory_map_aware_sliding_window.py` and `run_refinement_parameter_sweep.py`.

**Model architecture:** Gradient-based optimizer, not a learned model.

**Training setting:** 200 samples per sweep configuration; selected configuration then applied more broadly.

**Dataset:** Predictions and oracle map crops from the Week 03 decoder/map pipeline.

**Hyperparameters:** Radius `{10,15,25}`, road weight `{0.5,1,2,5}`, reference weight `{0.02,0.05,0.1}`, curvature `{0.05,0.1,0.2}`. Selected `r25_wr5_wref0p02_wc0p05`.

**Evaluation metrics:** Road mean/off-road fraction and ADE/FDE change.

**Results:** The selected sweep reduced off-road-over-10 m by 0.290 on the 200-sample sweep with ADE change `-2.49`. A retained full application reports raw ADE 68.230 to 66.424, FDE 121.55 to 119.40, road mean 30.25 to 15.59, and off-road-over-10 m 0.4566 to 0.1706.

**Figures available:** Large best-10 raw/refined/ground-truth families with and without map backgrounds.

**Checkpoint available:** No learned checkpoint; sweep CSVs and refined arrays retained.

**Problems encountered:** Uses oracle future crop; reported population/scale differs from the decoder test tables and is not directly comparable. Thousands of diagnostic figures duplicate nearby configurations.

**Current conclusion:** Post-processing improves road proximity, but this is an oracle-conditioned upper bound rather than a final generative solution.

## Phase 4 — constrained GASF diffusion

### Experiment P4.1 — Week 04 six-run constrained-GASF DDPM matrix

**Experiment name:** `[0,1]` and `[-1,1]` GASF DDPM with MSE, symmetry, or GASF-structure loss.

**Motivation:** Train diffusion directly on an analytically decodable image representation.

**Research question:** Which normalization and structural loss generate valid invertible GASF channels?

**Input representation:** R/G = GASF(`dx`/`dy`), B = start heatmap, 224×224.

**Mathematical formulation:** Noise MSE plus optional symmetry or GASF-structure consistency. The latter attempts to reconstruct off-diagonal structure implied by the decoded diagonal.

**Implementation:** `week04/scripts/train_week04_ddpm.py` and `compute_gasf_diag_targets.py`.

**Model architecture:** 2D U-Net DDPM matching the root image architecture.

**Training setting:** Target 50k; batch 16, gradient accumulation 2 (effective 32), learning rate `2e-4`, 1,000 diffusion steps, EMA; save/sample every 2k.

**Dataset:** 3,159 constrained-GASF images, both normalization variants.

**Hyperparameters:** Loss variants MSE, symmetry, and GASF-structure.

**Evaluation metrics:** Training loss, numerical stability, checkpoint/sample availability; no retained trajectory evaluation.

**Results:** Healthy runs were manually stopped near 40.6k: `[0,1]` MSE last loss 0.000578; symmetry 0.000609; `[-1,1]` MSE 0.000615; symmetry 0.000640. Both original GASF-structure runs became NaN, first observed around steps 490 and 620, and have no valid scientific checkpoint even though an output directory exists.

**Figures available:** Periodic samples and 32 resampled images for the `[0,1]` MSE 40k checkpoint at 100/150 reverse steps.

**Checkpoint available:** Valid through 40k for four healthy runs. Structure-loss checkpoints must be treated as invalid/NaN.

**Problems encountered:** Structural loss numerical instability; healthy jobs ended before 50k; no FID or decoded trajectory metrics.

**Current conclusion:** Basic training is stable, but the principal scientific question—whether generated images decode into valid trajectories—was not answered in Week 04.

### Experiment P4.2 — Float-data and safe-structure debug runs

**Experiment name:** Float `[0,1]` MSE run and short safe-structure diagnostics.

**Motivation:** Remove PNG quantization and debug structure-loss NaNs.

**Research question:** Are failures caused by uint8 storage or unsafe structural arithmetic?

**Input representation:** Float constrained-GASF arrays.

**Mathematical formulation:** Same DDPM; safe variants clamp/protect inverse operations.

**Implementation:** Week 04 trainer/debug options and `week04/debug_runs`, `smoke_gasf_structure` artifacts.

**Model architecture:** Same 2D U-Net.

**Training setting:** Float MSE targeted a full run but hit wall time at step 33,390. Safe structure jobs targeted 700 steps and were stopped at 170/180.

**Dataset:** Same 3,159 constrained float images.

**Hyperparameters:** Debug-specific safe operations; otherwise matched.

**Evaluation metrics:** Loss finiteness and progress.

**Results:** Float MSE remained numerically trainable through 33,390 but has no complete final comparison. Short safe runs did not produce NaN, but ended before the 490/620 failure region of the original jobs.

**Figures available:** Debug/smoke samples.

**Checkpoint available:** Partial artifacts only.

**Problems encountered:** Scheduler time limit and insufficient debug duration.

**Current conclusion:** Neither hypothesis was conclusively tested; these are operationally informative failed/incomplete experiments.

## Phase 5 — larger-data GASF diffusion

### Experiment P5.1 — Week 05 six-run sigmoid-GASF DDPM matrix

**Experiment name:** Sigmoid scales 1.0/1.5 crossed with MSE, diagonal, and symmetry losses.

**Motivation:** Use all 12,061 newer trajectories, a smooth bounded normalization, and systematic checkpoint FID evaluation.

**Research question:** Which normalization steepness and image constraint best models GASF images?

**Input representation:** Bounce/truncated 224-step deltas. \(u=\sigma(k(d-\mu)/\sigma_d)\); R/G are GASF(`u`) mapped for image training; B is start heatmap.

**Mathematical formulation:** Standard noise MSE; diagonal runs add diagonal consistency; symmetry runs add transpose consistency.

**Implementation:** `week05/scripts/train_week05_ddpm.py`, `sample_week05_checkpoints_to_npy.py`, and `evaluate_week05_npy_fid.py`.

**Model architecture:** 224×224 2D U-Net DDPM.

**Training setting:** Six complete 50k runs, batch 8, accumulation 2, learning rate `2e-4`, 1,000 diffusion steps, EMA; evaluation of checkpoints 40k–50k every 2k with 1,000 samples and 50 reverse steps.

**Dataset:** 12,061 fixed-length trajectories; split 9,648/1,206/1,207.

**Hyperparameters:** `k=1.0` or `1.5`; MSE/diagonal/symmetry losses.

**Evaluation metrics:** Image FID and final component losses.

**Results:** Best was sigmoid 1.5 + diagonal at 40k, FID 44.5496. Its later FIDs rose to 44.8652, 47.6087, 48.0252, 48.8892, 52.4143. Sigmoid 1.0 + diagonal best was 52.8173 at 40k. MSE improved toward roughly 68–70 at 48–50k. Symmetry was worst, about 129–144. Final 50k losses: sig1.0 MSE 0.000807; diag total/MSE/diag 0.004644/0.001147/0.069941; sym 0.000937/0.000888/0.000986. Sig1.5: MSE 0.000912; diag 0.006084/0.001494/0.091788; sym 0.001033/0.000993/0.000807.

**Figures available:** Periodic training samples, 1,000-sample NumPy sets, and representative sample PNGs in all six run trees.

**Checkpoint available:** Yes, checkpoints 2k–50k plus final for all six; the 40k sig1.5 diagonal model is the selected GASF generator.

**Problems encountered:** FID is image-space only and favors an earlier checkpoint; no FID uncertainty intervals; fixed-length bounce artifacts; diagonal loss improves FID but does not guarantee valid decoded paths.

**Current conclusion:** This is the strongest image-space result, but Week 06 shows that its decoded physical trajectories are often invalid.

### Experiment P5.2 — GASF consistency guidance

**Experiment name:** Sampling-time consistency guidance, scales 0–0.5.

**Motivation:** Push samples toward the GASF matrix manifold without retraining.

**Research question:** Can small gradient guidance improve diagonal/off-diagonal consistency and FID?

**Input representation:** Selected Week 05 sig1.5 diagonal checkpoint.

**Mathematical formulation:** Reverse samples receive a gradient correction from a differentiable GASF consistency energy.

**Implementation:** `week05/scripts/sample_week05_gasf_consistency_guidance.py`.

**Model architecture:** Frozen Week 05 2D U-Net.

**Training setting:** Sampling-only ablation; scale 0 through 0.5.

**Dataset:** Same real FID reference and generated sample count.

**Hyperparameters:** Guidance scales up to 0.5, including a gradient-clipped variant.

**Evaluation metrics:** FID and mean consistency error.

**Results:** FID was effectively unchanged at 44.548–44.551; mean consistency changed only from 0.03634191 to 0.03633951.

**Figures available:** Guided sample directories.

**Checkpoint available:** Uses the selected 40k checkpoint; generated outputs retained.

**Problems encountered:** Gradient signal was too weak or poorly aligned with the generative error.

**Current conclusion:** Neutral/failed intervention; consistency guidance did not materially improve the image distribution.

### Experiment P5.3 — Absolute-position GASF DDPM

**Experiment name:** `week05_abspos_sig15_mse_diag_20k`.

**Motivation:** Avoid integration drift by encoding absolute `x` and `y` rather than deltas.

**Research question:** Can absolute-position GASF learn an easier image distribution while retaining analytic decoding?

**Input representation:** GASF of sigmoid-normalized absolute x/y with the start heatmap channel.

**Mathematical formulation:** Same sigmoid/GASF construction and MSE + diagonal loss.

**Implementation:** Week 05 trainer with `representation=absolute`; normalization generated by `compute_absolute_position_normalization.py`.

**Model architecture:** Same 2D U-Net.

**Training setting:** Complete 20k run, 9,648 training examples, batch 8, sigmoid 1.5, diagonal weight 0.05.

**Dataset:** Week 05 split; x mean/std 1626.710/641.207, y mean/std 2951.006/419.397.

**Hyperparameters:** 20,000 steps, checkpoints every 2,000.

**Evaluation metrics:** Training component losses and sample images only.

**Results:** Final total loss 0.00280264, MSE 0.00038791, diagonal 0.0482946; 160 PNG samples retained.

**Figures available:** Periodic samples in the run directory.

**Checkpoint available:** Yes, 11 checkpoint directories including final.

**Problems encountered:** No NumPy generation set, FID, decoded trajectory statistics, road evaluation, or direct comparison to delta GASF.

**Current conclusion:** Completed training but scientifically unevaluated; an important missing branch for a fair representation comparison.

## Phase 6 — trajectory-space evaluation and raw diffusion

### Experiment P6.1 — Decode selected GASF samples and validate coordinate/map alignment

**Experiment name:** Analytic GASF decode, trajectory statistics, OSM coverage diagnosis, and offset calibration.

**Motivation:** Move evaluation from pixels into physical trajectory space.

**Research question:** Does the best-FID GASF model generate plausible paths after inversion?

**Input representation:** Generated `[0,1]` GASF diagonals and start heatmap. Decode \(u=\sqrt{\operatorname{diag}(G)}\), then \(d=\mu+\sigma_d\operatorname{logit}(u)/k\), soft-argmax start, and integrate.

**Mathematical formulation:** For each axis, recover the bounded scalar from the GASF diagonal, invert the sigmoid normalization, and integrate displacements. For raw \(G\in[-1,1]\), \(u_t=\sqrt{(G_{tt}+1)/2}\); because Week 05 stores \(C=(G+1)/2\), the code equivalently uses \(u_t=\sqrt{C_{tt}}\). Then \(d_t=\mu_d+\sigma_d\operatorname{logit}(u_t)/k\) and \(p_t=p_0+\sum_{j=1}^{t}d_j\).

**Implementation:** `decode_week05_samples_to_trajectories.py`, `diagnose_week05_real_map_coverage.py`, `download_week05_osm_map.py`, `search_week05_map_offset.py`, and evaluation scripts.

**Model architecture:** Frozen Week 05 selected generator; deterministic decoder.

**Training setting:** 1,000 generated samples from sig1.5 diagonal 40k.

**Dataset:** Week 05 real reference and newly downloaded OSM graph.

**Hyperparameters:** Empirical support map; selected OSM offset `dx=+220`, `dy=-720`.

**Evaluation metrics:** Step/path/displacement/bbox statistics, decoded-range violation, road mean/p95/off-road fraction, support distance.

**Results:** Generated mean step 12.47 versus real 11.18 and path length 2781 versus 2492, but net displacement 2350 versus 876.86 and bbox 1665.61×1413.89 versus 972.61×681.24. Mean decoded out-of-range fraction was 0.3505. Empirical support mean/p95 was 417.55/1059.47 with 0.6869 beyond 20. The first OSM graph covered only 0.0395% of points and produced generated/real road means 1537/962.56; a new graph had 1,848 nodes/3,580 edges. Offset search improved real road mean 25.51 to 15.49 and p95 70.16 to 41.15. Under that alignment, generated road mean/p95 was 220.28/644.65 versus real 15.24/40.40.

**Figures available:** Raw/OSM coverage, offset heatmaps, decoded-vs-real overlays, and empirical-map contact sheets.

**Checkpoint available:** Generator checkpoint and decoded `.npy`/metrics artifacts retained.

**Problems encountered:** Dataset-specific map transform was initially reused incorrectly; generated diagonal values near bounds amplify through square root/logit; many images are visually plausible but physically expansive/off-support.

**Current conclusion:** This is decisive evidence that good image FID does not imply good trajectory generation.

### Experiment P6.2 — Rejection filters for GASF trajectories

**Experiment name:** Strict OSM, strict empirical-support, and loose p99 rejection.

**Motivation:** Test whether invalid generations can be removed after sampling.

**Research question:** What fraction of GASF samples satisfy physical/map plausibility thresholds?

**Input representation:** The 1,000 decoded GASF trajectories and road/support metrics.

**Mathematical formulation:** Accept a sample only if all configured thresholds on distance, extent, steps, and decoded range pass.

**Implementation:** `week06/scripts/filter_generated_trajectories.py`.

**Model architecture:** Rule-based filter.

**Training setting:** No training; three threshold configurations.

**Dataset:** Same 1,000 generated samples and real-derived thresholds.

**Hyperparameters:** Strict p95 OSM; strict empirical mean 25/p95 75; loose p99.

**Evaluation metrics:** Acceptance rate and failed-rule counts.

**Results:** Strict OSM accepted 128/1,000 (12.8%); top failures included road mean 763, road p95 740, and out-of-range 701. Strict empirical support accepted 92/1,000 (9.2%), with accepted support mean/p95 16.16/41.56. Loose p99 accepted 228/1,000 (22.8%).

**Figures available:** `filter_metric_comparison.png` for each policy and accepted trajectory overlays/contact sheets.

**Checkpoint available:** No new model; accepted masks/arrays retained.

**Problems encountered:** Filtering improves conditional quality by discarding 77–91% of samples and does not repair coverage/diversity.

**Current conclusion:** Rejection is useful diagnostically but the low acceptance rate is evidence of a weak base generator.

### Experiment P6.3 — Six raw-trajectory diffusion models

**Experiment name:** Absolute versus delta representation crossed with temporal ResNet, dilated temporal ResNet, and 1D U-Net.

**Motivation:** Establish the direct trajectory-space alternative to GASF image diffusion.

**Research question:** Does direct sequence diffusion produce more valid traffic trajectories, and which representation/architecture works best?

**Input representation:** Absolute z-scored `[2,224]` coordinates, or z-scored `[2,224]` deltas integrated from a sampled real start point.

**Mathematical formulation:** Standard DDPM noise MSE directly in coordinate/delta space.

**Implementation:** `trajectory_diffusion_common.py`, `train_absolute_trajectory_diffusion.py`, `train_delta_displacement_diffusion.py`, and `package_raw_diffusion_samples.py`.

**Model architecture:** Temporal residual network, dilated temporal residual network, and 1D U-Net; hidden width 128, eight residual blocks, time embedding 256.

**Training setting:** Six complete 50k runs, batch 64, AdamW, learning rate `2e-4`, weight decay `1e-4`, cosine schedule, EMA, 1,000 diffusion steps, 100 inference steps; 1,000 samples per model.

**Dataset:** Week 05 fixed-length 12,061 trajectories and split.

**Hyperparameters:** Representation `{absolute, delta}` × architecture `{resnet, dilated, unet1d}`.

**Evaluation metrics:** Final/min training loss, empirical-support acceptance, support mean/p95, displacement, bbox, maximum step.

**Results:** Final/min losses: absolute ResNet 0.01545/0.01374; absolute dilated 0.01412/0.01250; absolute U-Net 0.01126/0.01010; delta ResNet 0.05899/0.05180; delta dilated 0.05455/0.04801; delta U-Net 0.03866/0.03347. Acceptance was 92.7%, 95.9%, and 98.5% for absolute models versus 61.0%, 59.2%, and 60.5% for delta models. Absolute U-Net had support mean/p95 8.4/25.6, but displacement 549.7 and bbox 613.5×448.5 versus real 876.9 and 972.6×681.2, indicating conservative/mode-shrunk paths.

**Figures available:** Five empirical-map figures per model plus `raw_diffusion_best_median_worst_support.png`.

**Checkpoint available:** Yes, checkpoint series every 2k plus final for all six models.

**Problems encountered:** First scheduler submissions (jobs 1795817–1795822) disappeared because the log directory did not exist; resubmissions succeeded. Absolute models are valid but under-dispersed; delta models drift off support. No matched FID-like trajectory distribution metric or multi-seed uncertainty was run.

**Current conclusion:** Direct absolute-coordinate diffusion is the strongest current approach for physical validity, with 1D U-Net best on support/acceptance, but diversity and coverage remain unproven.

### Experiment P6.4 — Road guidance for raw diffusion

**Experiment name:** Six raw models with empirical support guidance strength 0.1.

**Motivation:** Determine whether a differentiable map prior further improves direct trajectory samples.

**Research question:** Does weak sampling-time guidance improve road/support validity without retraining?

**Input representation:** Each raw model's sequence state and differentiable empirical support field.

**Mathematical formulation:** At reverse steps, apply a negative gradient of squared sampled distance-field energy \(E=T^{-1}\sum_t D(p_t)^2\).

**Implementation:** `week06/scripts/sample_raw_diffusion_guidance.py` using `road_distance_field.py`.

**Model architecture:** Frozen six raw models.

**Training setting:** Sampling-only, strength 0.1.

**Dataset:** Same 1,000-sample evaluation framework.

**Hyperparameters:** Guidance strength 0.1.

**Evaluation metrics:** Empirical support mean/p95 and the same geometry statistics.

**Results:** Absolute ResNet/dilated/U-Net support became approximately 10.61/32.95, 10.15/31.05, and 8.319/25.406; delta variants remained roughly 44–46 mean and 105–109 p95. There was no consistent improvement over unguided sampling.

**Figures available:** Five empirical-map figures for each guided model.

**Checkpoint available:** Uses existing raw checkpoints; guided arrays retained.

**Problems encountered:** Weak/no benefit and possible geometry distortion; only one guidance strength was systematically packaged.

**Current conclusion:** Neutral negative result; raw models already learn the support better than this external correction can enforce.

### Experiment P6.5 — Road guidance for GASF diffusion

**Experiment name:** Full-channel guidance, fixed-diagonal guidance, and endpoint-capped guidance.

**Motivation:** Repair low-validity GASF generations during the reverse process.

**Research question:** Can map gradients act through GASF decoding without destroying the image diagonal?

**Input representation:** Selected sig1.5 diagonal GASF model decoded differentiably to trajectories.

**Mathematical formulation:** Distance-field energy is differentiated through heatmap start decoding, diagonal square-root/logit inverse, and cumulative sum. Later variants blend diagonals toward unguided values, clip decoded deltas, and cap endpoint displacement.

**Implementation:** `week06/scripts/sample_week05_gasf_road_guidance.py` and `src/cnr_trajectory/guidance/road_distance_field.py`.

**Model architecture:** Frozen 2D U-Net plus differentiable guidance.

**Training setting:** Sampling studies at full-channel strengths 0.1, 1, 10, 1000; fixed-diagonal blend 0.1/clip 37.5 at strengths 0.1, 1, 10; endpoint cap 1,916 with blend 0.25, clip 37.5, strength 1.

**Dataset:** 1,000-sample Week 06 evaluation framework.

**Hyperparameters:** As listed above; a GPU smoke test checked energy descent.

**Evaluation metrics:** Energy, decoded diagonal saturation, step/extent, support mean/p95/off-road fraction.

**Results:** Distance-field smoke energy fell from 16 to 11.6476 (mean distance 4 to 3.4129), and four tests passed. Full-channel guidance worsened support to roughly 551–600 versus unguided 417.55 and saturated diagonals (p99 near 1, deltas up to 85). Fixed-diagonal variants removed saturation; best strength 1 reached support 381.93/986.77, still poor. Endpoint-capped guidance improved to step 10.85, displacement 1960.04, bbox 1391.63×1215.49, support 270.25/706.73, off-road-over-20 0.645. The fractions of generated samples whose mean support distance was below 50/100/200/500 were 31.4%/42.2%/55.2%/80.0%.

**Figures available:** Five empirical-map figures for each major guidance variant.

**Checkpoint available:** Uses the selected GASF checkpoint; guided arrays and metrics retained.

**Problems encountered:** The inverse is ill-conditioned near diagonal bounds; map gradients can improve the road objective while breaking GASF/trajectory structure. The final rerun log reports `loaded_existing=1` and zero applied steps, so it validates the saved set but is not evidence of a fresh rerun.

**Current conclusion:** Partial improvement remains far from real/raw baselines; guidance cannot compensate for an invalid image-space sample manifold.

## Map matching status

The repository contains a true map-matching wrapper in `src/cnr_trajectory/reconstruction/map_matching.py`. It rescales normalized predictions to longitude/latitude, writes a trace CSV, optionally downloads an `NxMap`, and calls mappymatch LCSS. However, no retained LCSS match result, acceptance score, or map-matching error table was found. Week 03 and Week 06 results instead measure nearest-road distance, raster distance-field energy, OSM coverage, or empirical support distance. These must be described as **map-alignment/road-proximity evaluation**, not as completed quantitative map matching.

## Smoke, aborted, and operational experiments that must remain in the record

| Attempt | Purpose | Outcome | Scientific status |
|---|---|---|---|
| Root 10-step and channel-normalization smoke runs | Validate training/sampling plumbing | Checkpoints/samples exist; one duplicate 10-step scheduler job has empty logs | Engineering validation only |
| Historical 128×128 3k DDPM | Refactoring validation | FID 288.81→226.14 documented, but result tree is absent | Report-only sanity result |
| Historical 224×224 3k DDPM matrix | Compare notebook auxiliary/MSE/diagonal | MSE and diagonal improve image/validity metrics, but all remain weak; directories absent | Report-only preliminary result |
| Historical notebook decoder variants | Map channels, invalid-point/BCE/binary outputs, LSTM correction | Described without retained notebooks or quantitative artifacts | Unverifiable exploratory attempts |
| Root 10k PureMSE/diagonal runs | Early baseline | Completed, superseded by matched 50k studies | Historical baseline |
| Latent DDPM bad baseline | Debug latent pipeline | Retained directory explicitly marked bad | Failed; do not use quantitatively |
| Week 03 structure/drift/map/latent smoke runs | Shape, loss, and checkpoint validation | Completed short runs | Not scientific comparisons |
| Week 04 original structure losses | Enforce exact GASF manifold | NaN near steps 490/620 | Failed optimization |
| Week 04 safe-structure debug | Prevent NaN | Stopped at 170/180, too early to reproduce failure window | Inconclusive |
| Week 04 float MSE | Avoid quantization | Time limit at 33,390 | Incomplete |
| Week 04 neg11 diagnostic first submission | Test sign ambiguity | `KeyError`, later corrected | Operational failure, final test valid |
| Week 05 consistency guidance | Improve GASF structure | FID/consistency essentially unchanged | Negative result |
| Week 05 absolute-position GASF | Avoid integration drift | Training complete, no scientific evaluation | Incomplete |
| Week 06 raw first submissions | Launch six raw models | Missing log directory; jobs vanished/no results | Operational failure; resubmissions complete |
| Week 06 full-channel GASF road guidance | Enforce road validity | Worsened support and saturated diagonal | Failed intervention |
| Week 06 raw road guidance | Improve already valid raw samples | No consistent benefit | Negative result |

## Cross-experiment scientific conclusion

The experiments support a representation-centered narrative. GAF/GASF turns a temporal sequence into a global pairwise texture: this is attractive for CNN classification because discriminative features can exploit repeated motifs, symmetry, and broad correlations without reconstructing every coordinate. Generation is stricter. A model must create an entire matrix on a low-dimensional valid-GASF manifold, preserve a decodable diagonal, keep the off-diagonal entries mutually consistent with that diagonal, recover signs or use a sign-safe normalization, preserve start position, survive an ill-conditioned inverse near bounds, and satisfy cumulative geometric/map constraints.

The legacy Hilbert representation loses a clean 2D inverse outright. The constrained float `[0,1]` representation is almost lossless for **real encoded samples**, but generated pixel images depart from the mathematical manifold. This distinction explains why “GASF can classify well” and “GASF is difficult to generate from” are compatible claims. Pixel-space noise MSE and natural-image FID do not directly penalize trajectory-space endpoint, support, or road violations. Direct absolute-coordinate diffusion optimizes in the space where those constraints live and currently achieves far higher empirical validity, though its smaller extents expose a likely coverage/mode-shrinkage weakness.

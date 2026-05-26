# Week3 Code Reuse Audit

This audit covers reusable Week2 decoder, dataset, metric, and evaluation code for the Week3 structure-aware pseudo-modal GAF decoder experiments. Week3 work should live under `week03/` and should not write into existing Week1/Week2 result directories.

## Week3 Workspace

Created:

- `week03/scripts/`
- `week03/configs/`
- `week03/results/`
- `week03/logs/`
- `week03/docs/`

Target dataset:

- `/home/jliu/data_no_speed_delta_displacement_paired/`
- Input image: RGB GAF image, with `R=GASF`, `G=GADF`, `B=MTF`
- Target delta: `labels_delta_displacement/*.npy`, physical unnormalized delta displacement, shape `[224, 2]`
- True start: `starts/*.npy`, shape `[2]`
- Absolute target: `labels_absolute/*.npy`, shape `[224, 2]`
- Absolute reconstruction convention: `pred_abs[0] = true_start`; `pred_abs[1:] = true_start + cumsum(pred_delta[1:], axis=0)`

## Relevant Existing Files

### Core reusable module

`src/cnr_trajectory/reconstruction/delta_displacement.py`

- `DeltaDisplacementPairedDataset`: directly reusable for Week3. It already loads RGB pseudo-modal images, physical delta labels, absolute labels, starts, sample IDs, and vehicle IDs.
- `SingleChannelEncoder`: reusable as a small channel stem or baseline block.
- `MultiBranchDeltaDecoder`: reusable as the late-fusion baseline.
- `MidFusionDeltaDecoder`: directly relevant Week2 mid-fusion baseline. It has separate shallow stems for GASF/GADF/MTF, concatenates spatial feature maps, and decodes deltas.
- `SimpleCNNDeltaDecoder`: reusable as the early-fusion baseline.
- `build_delta_decoder`: reusable for baseline construction, but Week3 should probably add new model builders in `week03/scripts` or a Week3 module instead of expanding this shared module immediately.
- `integrate_delta_torch` and `integrate_delta_numpy`: directly reusable and match the required Week3 absolute reconstruction semantics.
- `compute_basic_metrics`, `compute_alignment_metrics`, `summarize_metric_rows`: directly reusable for delta and integrated absolute metrics.
- `load_or_create_split_metadata`, `write_split_json`: reusable for consistent train/val/test split handling.

### Current training/evaluation script

`scripts/train_delta_displacement_decoder.py`

- Best source for the Week3 training loop shape.
- Reusable pieces:
  - `TargetDeltaNormalizer`
  - `fit_target_delta_normalizer`
  - `run_epoch`
  - `evaluate`
  - `predict_split`
  - `plot_comparison`
  - `make_loader`
  - `make_config`
  - `resolve_device`
  - `set_seed`
  - JSON/JSONL writing helpers
- Should be copied/adapted into `week03/scripts/` rather than edited in place, so Week3 outputs default to `week03/results/...` and logs default to `week03/logs/...`.
- The existing script already supports `multi_branch_delta`, `simple_cnn_delta`, and `mid_fusion_delta`, so it is a good baseline runner.

### Dataset construction and split checks

`scripts/build_no_speed_delta_displacement_paired_dataset.py`

- Useful for semantic reference only. The dataset is already built, so Week3 should not rerun this by default.
- Reusable reference functions:
  - `compute_delta_displacement`
  - `normalize_delta_for_encoding`
  - `integrate_delta`
  - `write_normalization_json`
  - `run_sanity_checks`
- Do not copy the full builder into Week3 unless new data generation is needed.

`scripts/create_no_speed_delta_displacement_splits.py`

- Useful for validation logic around metadata columns, image shape, label shape, start shape, and split consistency.
- Week3 should reuse the existing dataset `splits/split_metadata.csv` if present, or use `load_or_create_split_metadata` for deterministic fallback.
- Do not create new splits unless the Week3 comparison requires a frozen split file in `week03/configs/`.

### Result collection

`scripts/collect_delta_decoder_results.py`

- Reusable as a template for Week3 result aggregation.
- Should be adapted to discover `week03/results/experiment_*` directories and include Week3-specific fields such as structure-aware model type, branch masks, fusion stage, and loss weights.

### Diagnostics and advanced evaluation

`scripts/compute_decoder_alignment_diagnostics.py`

- Reusable ideas: richer alignment diagnostics including centroid, scale ratio, and shape scale.
- Prefer the centralized `compute_alignment_metrics` from `src/cnr_trajectory/reconstruction/delta_displacement.py` for initial Week3, then copy only extra diagnostic columns if needed.

`scripts/evaluate_affine_upper_bound.py`

- Useful as an optional post-hoc analysis template.
- Do not include in initial Week3 training unless Experiment D explicitly evaluates affine/translation limits.

`scripts/run_delta_centroid_correction_decoder_ablation.py`

- Useful for centroid correction ideas and plotting/evaluation patterns.
- It duplicates dataset loading and metrics. Do not import it directly; copy only specific centroid diagnostic logic if Experiment D needs it.

`scripts/run_hybrid_multitask_delta_decoder_v3_geometry.py`

- Useful for geometry-aware loss references:
  - heading/turn supervision
  - turn-sharpness weighting
  - geometry diagnostics
  - oracle-start vs learned-start comparison
- It uses a ResNet18 multi-head architecture and many experiment-specific assumptions. Do not reuse wholesale for Week3 pseudo-modal GAF models.

### Tests and documentation

`tests/test_delta_displacement_decoder_smoke.py`

- Reusable as a template for Week3 smoke tests.
- Important checks to preserve:
  - dataset batch shapes
  - model output shape `[B, 224, 2]`
  - `integrate_delta_torch` ignores `delta[:, 0, :]` as motion
  - `mid_fusion_delta` accepts only RGB input

`docs/delta_multibranch_decoder.md`

- Useful documentation of Week2 assumptions, mid-fusion behavior, loss formulas, and recommended ablations.
- Week3 docs should reference this, but keep new experiment notes in `week03/docs/`.

## Parts To Copy Or Adapt Into Week3

- Copy/adapt `scripts/train_delta_displacement_decoder.py` into a Week3 runner, for example `week03/scripts/train_structure_aware_decoder.py`.
- Keep imports from `src/cnr_trajectory/reconstruction/delta_displacement.py` for stable shared utilities:
  - dataset loading
  - baseline builders
  - integration
  - basic/alignment metrics
  - split handling
- Add Week3 model definitions in a Week3-local module or script first, for example `week03/scripts/structure_aware_models.py`, to avoid destabilizing existing Week2 code.
- Adapt `collect_delta_decoder_results.py` into `week03/scripts/collect_week03_results.py`.
- Add Week3 smoke tests only after the first Week3 model is implemented; mirror the shape/integration tests from `tests/test_delta_displacement_decoder_smoke.py`.

## Parts Not To Reuse Directly

- Do not modify or write into existing `results/`, `results_hpc/`, `experiments/`, `Thesis_eval/`, or Week2 result directories.
- Do not train via existing scripts with their default output directories unless `--output-dir week03/results/...` is explicitly supplied.
- Do not import whole experiment scripts such as `run_delta_centroid_correction_decoder_ablation.py` or `run_hybrid_multitask_delta_decoder_v3_geometry.py`; they duplicate logic and carry unrelated assumptions.
- Do not rebuild `/home/jliu/data_no_speed_delta_displacement_paired/` for Week3 unless the dataset is later found invalid.
- Do not reuse absolute-coordinate decoder code for the main target. Week3 target semantics are physical unnormalized delta displacement plus true-start integration.
- Do not use ImageNet/natural-image assumptions as the default interpretation of RGB. The channels are pseudo-modal GASF/GADF/MTF.

## Proposed Week3 Experiment Plan

### Baseline check: Week2 mid-fusion reference

Goal: establish a clean reference result inside `week03/` without touching old outputs.

- Model: existing `mid_fusion_delta`
- Data: `/home/jliu/data_no_speed_delta_displacement_paired/`
- Target: physical unnormalized delta displacement
- Loss: `delta MSE + beta * integrated trajectory MSE`
- Default beta: `0.05`
- Output: `week03/results/baseline_midfusion/`
- Purpose: Week3 reference baseline only; this is not a new research experiment.

### Experiment A: Channel structure statistics

Goal: verify whether the expected mathematical structures are actually present in the current real GAF PNG dataset.

Compute per sample and summarize:

- GASF symmetry error: `mean(abs(R_theory - R_theory.T))`
- GADF anti-symmetry error: `mean(abs(G_theory + G_theory.T))`
- MTF symmetry error for diagnostic only: `mean(abs(B_theory - B_theory.T))`
- Channel mean/std/min/max
- Diagonal mean/std

Scale convention:

- `R_theory = R_png / 255 * 2 - 1`
- `G_theory = G_png / 255 * 2 - 1`
- `B_theory = B_png / 255`

Output:

- `week03/results/channel_structure_stats/channel_structure_stats.csv`
- `week03/results/channel_structure_stats/channel_structure_stats_summary.md`
- Visualizations of raw/projected/residual components

### Experiment B: Hard structure projection mid-fusion

Goal: directly test whether hard architectural projection helps.

Input transformation:

- GASF branch: `sym(R) = 0.5 * (R + R.T)`
- GADF branch: `antisym(G) = 0.5 * (G - G.T)`
- MTF branch: raw `B`

Architecture:

- Start from Week2 `MidFusionDeltaDecoder`
- Keep three branch stems
- Keep mid-level spatial fusion
- Keep target and loss unchanged

Output:

- `week03/results/hard_structure_midfusion/`

### Experiment C: Structure decomposition mid-fusion

Goal: test whether preserving both theoretical structure and empirical residual improves decoding.

Input transformation:

- GASF branch: `[R_sym, R_residual]`
- GADF branch: `[G_antisym, G_residual]`
- MTF branch: `[B_raw]`

Definitions:

- `R_sym = 0.5 * (R + R.T)`
- `R_residual = R - R_sym`
- `G_antisym = 0.5 * (G - G.T)`
- `G_residual = G - G_antisym`

Architecture:

- Main Week3 proposed model
- Keep pseudo-modal branch identities explicit
- Fuse after branch-specific stems, preserving the Week2 mid-fusion comparison point
- Keep target and loss unchanged from the baseline unless a later ablation explicitly changes them

Output:

- `week03/results/decomp_structure_midfusion/`

### Experiment D: Structure decomposition + MTF local texture

Goal: test whether MTF contributes as a local transition-texture cue.

Input transformation:

- GASF branch: `[R_sym, R_residual]`
- GADF branch: `[G_antisym, G_residual]`
- MTF branch: `[B_raw, B_local_avg]`

Definition:

- `B_local_avg = avg_pool2d(B, kernel_size=3, stride=1, padding=1)`

Output:

- `week03/results/decomp_mtf_local_midfusion/`

### Optional Experiment E: Soft branch gate

Goal: analyze pseudo-modal branch contribution.

- Only run this after Experiment C or D shows stable results.
- Keep it diagnostic-first; it should not block the main Week3 comparison.

## Immediate Next Implementation Step

Implement only after this audit is accepted:

1. Create a Week3 baseline runner for `mid_fusion_delta` that writes to `week03/results/baseline_midfusion/`.
2. Create `week03/scripts/analyze_channel_structure.py` for Experiment A.
3. Create a Week3-local model module for hard projection and decomposition mid-fusion variants.
4. Adapt the delta training/evaluation loop into `week03/scripts/train_structure_aware_decoder.py`.
5. Ensure all defaults point to `/home/jliu/data_no_speed_delta_displacement_paired/` and `week03/results/...`.
6. Add smoke modes that write only inside `week03/results/smoke_*`.





# Week3 Decoder Analysis: Structure-Aware Pseudo-Modal GAF Decoding and Drift Diagnostics

## 1. Background and Motivation

The first-stage decoder validation showed that real GAF images are partially decodable, but the reconstruction error remains too high for directly evaluating generated images. In Week1, the original CNN-FC decoder achieved a test ADE/FDE of 160.30/201.58, while the stronger ResNet18 decoder improved the absolute x/y reconstruction to 120.92/186.59. This confirmed that decoder capacity matters, but also that the image representation itself may still limit precise coordinate-level reconstruction. The delta-displacement ResNet18 experiment further showed that the delta representation improves aligned-shape metrics but suffers from integrated drift, with raw integrated ADE/FDE of 135.29/224.34. :contentReference[oaicite:0]{index=0}

Based on the Week2 discussion, the three RGB channels should not be treated as natural RGB. Instead, they are pseudo-modal mathematical representations:

- R channel: GASF
- G channel: GADF
- B channel: MTF

Therefore, Week3 focuses on whether channel-specific mathematical structure can be explicitly encoded into the decoder architecture or input representation.

---

## 2. Experiment A: Channel Structure Statistics

### Motivation

Before enforcing any channel structure, we first needed to verify whether the expected mathematical structures are actually preserved in the current real GAF PNG dataset. If GASF/GADF symmetry properties were broken by preprocessing, Hilbert mapping, scaling, or PNG quantization, then hard structure projection would not be justified.

### Method

For each real RGB GAF image, the PNG channels were mapped back to their theoretical scales:

- `R_theory = R_png / 255 * 2 - 1`
- `G_theory = G_png / 255 * 2 - 1`
- `B_theory = B_png / 255`

Then we computed:

- GASF symmetry error: `mean(abs(R - R.T))`
- GADF anti-symmetry error: `mean(abs(G + G.T))`
- MTF symmetry error: `mean(abs(B - B.T))`, used only as a diagnostic

### Result

The statistics over all 3159 samples showed:

- GASF symmetry error: `0`
- GADF anti-symmetry error mean: `0.000884`
- MTF symmetry error mean: `0.0336`

GASF is exactly symmetric after PNG loading. GADF is nearly anti-symmetric, with a very small residual mostly attributable to uint8 quantization around zero. MTF shows much larger symmetry error and should not be treated as a symmetric GAF-like channel.

### Finding

Experiment A confirms that the mathematical structure of GASF/GADF is preserved in the current dataset. Therefore, structure-aware decoding is justified for GASF and GADF. However, MTF should be treated as transition/statistical texture rather than forced into a symmetry constraint.

### Implication

This motivates the following controlled experiments:

- Hard projection: keep only theoretical structure
- Structure decomposition: keep theoretical structure plus residual
- MTF local texture: add a local texture view for MTF instead of enforcing symmetry

---

## 3. Experiment B: Hard Structure Projection ResNet18

### Motivation

Experiment B tests the most direct form of structural enforcement. Since GASF is symmetric and GADF is anti-symmetric, we project the input into these theoretical components before feeding it to ResNet18.

### Configuration

Input image: raw RGB GAF image in `[0,1]`

Internal conversion:

```python
R = x[:, 0:1] * 2 - 1
G = x[:, 1:2] * 2 - 1
B = x[:, 2:3]

Hard projection:

R_sym = 0.5 * (R + R.T)
G_antisym = 0.5 * (G - G.T)

Final model input:

X_B = [R_sym, G_antisym, B_raw]

Backbone:

Scratch ResNet18
conv1 input channels: 3
MLP head: 512 -> 1024 -> 448
Output: normalized delta displacement [224, 2]
Result

Experiment B achieved:

Split	Delta ADE	Delta FDE	Integrated ADE	Integrated FDE	Trajectory Length Ratio
Train	1.5866	2.3424	44.20	77.07	1.0157
Val	4.0900	6.7116	150.05	259.19	0.9345
Test	4.1557	6.9632	143.85	247.90	0.9208
Finding

The hard projection model performs worse than the historical raw delta ResNet18 baseline from Week1, whose integrated ADE/FDE was 135.29/224.34. It also shows a clear trajectory length underestimation on the test set, with a mean trajectory length ratio of 0.9208.

This suggests that hard projection is too restrictive. Although the theoretical structures are valid, directly discarding the residual components removes information that is still useful for trajectory reconstruction.

Implication

Hard structural enforcement alone is not sufficient. The residual components should not be discarded without testing whether they contain useful empirical information.

4. Experiment C: Structure-Decomposed ResNet18
Motivation

Experiment C tests whether a softer structure-aware representation is more effective than hard projection. Instead of discarding all residuals, we explicitly decompose each GAF channel into:

theoretical structural component
empirical residual component

This allows the model to use the mathematical prior while still preserving information introduced by quantization, Hilbert mapping, discretization, or other preprocessing effects.

Configuration

Input image: raw RGB GAF image in [0,1]

Internal conversion:

R = x[:, 0:1] * 2 - 1
G = x[:, 1:2] * 2 - 1
B = x[:, 2:3]

Structure decomposition:

R_sym = 0.5 * (R + R.T)
R_res = R - R_sym

G_antisym = 0.5 * (G - G.T)
G_res = G - G_antisym

Final model input:

X_C = [R_sym, R_res, G_antisym, G_res, B_raw]

Backbone:

Scratch ResNet18
conv1 input channels: 5
MLP head: 512 -> 1024 -> 448
Output: normalized delta displacement [224, 2]
Result

Experiment C achieved:

Split	Delta ADE	Delta FDE	Integrated ADE	Integrated FDE	Trajectory Length Ratio
Train	1.4778	1.9793	41.81	75.19	1.0016
Val	3.8679	6.7194	151.02	251.31	0.9171
Test	3.9450	6.4920	140.16	235.19	0.9122

Compared with Experiment B:

Model	Test Delta ADE	Test Delta FDE	Test Integrated ADE	Test Integrated FDE
B: Hard projection	4.1557	6.9632	143.85	247.90
C: Decomposition	3.9450	6.4920	140.16	235.19
Finding

Experiment C improves over the hard projection model in both delta-space and integrated trajectory metrics. This supports the hypothesis that the residual components are useful and should not be discarded.

However, the improvement over B is moderate, and C still does not clearly outperform the historical raw delta ResNet18 baseline from Week1. Therefore, explicit structure decomposition helps recover part of the information lost by hard projection, but it does not solve the main reconstruction bottleneck.

Implication

The result suggests that channel-level structure is not the dominant remaining bottleneck. A strong ResNet18 decoder may already learn much of the useful structure from the raw pseudo-modal RGB input. The persistent high integrated ADE/FDE indicates that the larger issue is likely accumulated delta integration drift and trajectory-level geometric mismatch.

5. Experiment D: Decomposition + MTF Local Texture
Motivation

MTF is not symmetric or anti-symmetric like GASF/GADF. Experiment A showed that MTF has a much larger symmetry error, so it should not be hard-projected into a GAF-like structure. Instead, MTF may be more useful as a transition-statistical texture channel.

Experiment D extends Experiment C by adding a local texture view of MTF.

Configuration

Experiment D input:

X_D = [R_sym, R_res, G_antisym, G_res, B_raw, B_local]

where:

B_local = avg_pool_3x3(B_raw)

with replicate padding.

Backbone:

Scratch ResNet18
conv1 input channels: 6
Same delta prediction head as B/C
Expected Finding

Experiment D tests whether adding a local MTF texture view improves reconstruction. If D improves over C, it would suggest that MTF contributes complementary transition-statistical information. If D is similar to or worse than C, it would suggest that ResNet18 already extracts sufficient local texture from raw MTF, or that the added MTF local channel is redundant for this small dataset.

6. E0 Diagnostic: Post-Hoc Drift and Length Correction
Motivation

B and C both show large integrated ADE/FDE even when delta-space errors are moderate. This suggests that errors accumulate through open-loop delta integration. To test whether the dominant problem is trajectory length underestimation, endpoint drift, or local shape mismatch, we performed post-hoc drift diagnostics without retraining any model.

Methods

For each trained model, we tested several post-hoc corrections:

Validation-estimated global length scaling
Estimate a global scale factor from validation trajectories and multiply all predicted deltas by this scale.
Oracle per-sample length scaling
Use the true trajectory length of each test sample to rescale predicted deltas. This is not deployable but provides an upper bound for length correction.
Oracle endpoint linear correction
Compute the final endpoint error and distribute it linearly over the trajectory.
Piecewise oracle correction
Use 4, 8, or 16 anchor corrections along the trajectory and linearly interpolate them over time.
Result
Model	Original ADE/FDE	Val-Scaled ADE/FDE	Oracle Length ADE/FDE	Endpoint-Corrected ADE	Piecewise16 ADE
Hard ResNet18	143.85 / 247.90	141.88 / 243.39	130.50 / 206.94	73.95	9.67
Decomp ResNet18	140.16 / 235.19	138.54 / 234.41	124.81 / 193.88	73.38	8.99
Finding

The validation-estimated global length scaling only gives minor improvement. This means the error is not simply caused by a global delta magnitude bias.

Oracle length scaling gives moderate improvement, so trajectory length mismatch is part of the problem, but it is not sufficient to explain the full error.

Endpoint correction reduces ADE by nearly half, indicating strong low-frequency accumulated drift. Even more importantly, piecewise16 correction reduces ADE below 10 for both hard and decomposed models. This suggests that the predicted trajectories preserve much of the local geometric structure, but open-loop integration accumulates low-frequency drift over time.

Implication

The main remaining bottleneck is not channel-level GAF structure. It is open-loop delta integration drift. Therefore, the next decoder direction should be drift-aware decoding, such as:

low-frequency correction head
error-feedback decoder
endpoint or anchor-based correction
trainable correction field over integrated trajectories

A direct integrated loss may be too coarse and may oversmooth trajectories. A better solution is to separate local motion prediction from low-frequency drift correction.

7. Overall Findings

The Week3 experiments lead to the following conclusions:

GAF channel structure is real.
GASF is symmetric, GADF is nearly anti-symmetric, and MTF behaves differently as a transition/statistical field.
Hard projection is too restrictive.
Removing residual components hurts reconstruction and leads to worse integrated ADE/FDE.
Structure decomposition is better than hard projection.
Keeping residual components improves over hard projection, confirming that empirical residual information is useful.
Structure decomposition does not solve the main bottleneck.
Compared with the historical raw delta ResNet18 baseline, structure decomposition does not clearly improve reconstruction. This suggests that a strong decoder may already learn much of the useful pseudo-modal structure from raw RGB input.
The dominant error is accumulated drift.
Post-hoc correction diagnostics show that endpoint and piecewise correction can dramatically reduce ADE. This indicates that predicted trajectories are locally meaningful but globally drift during open-loop integration.
Next direction: drift-aware decoding.
Instead of further enforcing GAF symmetry, future experiments should focus on correcting accumulated integration error while preserving local trajectory shape.
8. Proposed Next Experiment: Drift-Aware Low-Frequency Correction Decoder

A promising next step is to add a trainable low-frequency correction head:

ResNet18 feature
├── delta head -> pred_delta [224, 2]
└── correction head -> corr_anchor [16, 2]

Then:

pred_abs_raw = true_start + cumsum(pred_delta)
corr_full = interpolate(corr_anchor, 224)
pred_abs_corrected = pred_abs_raw + corr_full

The delta head preserves local motion, while the correction head learns low-frequency drift compensation. This design is more targeted than a full integrated loss because it avoids forcing every timestep to match the ground truth directly, which could oversmooth sharp turns and local dynamics.

A possible loss is:

L = L_delta_norm
  + λ_abs * L_corrected_abs_norm
  + λ_smooth * L_correction_smoothness
  + λ_mag * L_correction_magnitude

This experiment would directly test whether the large gap between raw integrated ADE and piecewise-corrected ADE can be learned from the GAF representation.
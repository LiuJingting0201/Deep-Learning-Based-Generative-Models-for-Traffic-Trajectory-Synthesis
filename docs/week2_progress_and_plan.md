# Week 2 Progress and Plan

Generated on 2026-05-16 from the current `week2` working tree.

## Repository State

- Current branch: `week2`.
- Recent HEAD commit: `cf2494d Add week2 work`.
- Earlier visible history includes `week1` / `origin/week1` at `6ff2e31 hpc scratch test`, followed by decoder, delta-displacement, and data-driven DDPM ablation work.
- Current working tree status: only `.gitignore` is modified.
- `.gitignore` has been updated locally to ignore `logs/`, `results_hpc/`, `*.out`, and `*.err`, in addition to existing Python caches, generated data folders, checkpoint formats (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`), and local result folders.

Important repository hygiene finding: many generated outputs are already tracked by Git, so the new ignore rules will not remove them from the index by themselves. Tracked generated artifacts include:

- SLURM logs under `logs/` and root-level `ddpm_scratch_10step_*.out/.err`.
- HPC copied result metadata, JSONL logs, scheduler/model config files, and periodic sample PNGs under `results_hpc/`.
- Generated checkpoint-final sample PNGs under `Thesis_eval/`.
- `results_hpc/` is approximately 31 GB locally. The largest binary checkpoint files (`training_state.pt`, `diffusion_pytorch_model.safetensors`) appear ignored, but the directory still contains many tracked generated files.

The current repository should therefore be treated as mixed: it contains reusable source code and useful experiment scripts, but also tracked logs/samples/result artifacts that should usually be excluded before pushing or sharing.

## Implemented Progress

Stable or reusable code now includes:

- Core package code under `src/cnr_trajectory/` for data loading, encoding, models, reconstruction, and visualization.
- DDPM training logic in `scripts/train_ddpm.py`, including configurable loss modes, checkpoint saving, resume support, periodic sampling, JSONL logs, and summaries.
- HPC-oriented DDPM entrypoint in `scripts/train_ddpm_scratch.py`, with scratch-aware paths, CUDA fail-fast checks, startup diagnostics, optional copy-back to `$HOME/Thesis/results_hpc/<run_name>`, and support for precomputed diagonal targets.
- Sample generation in `scripts/generate_ddpm_samples.py`, loading `checkpoint-final` style directories with `unet/`, `scheduler/`, and `training_state.pt`.
- Image-level FID evaluation in `scripts/evaluate_fid.py`.
- Precomputation of train-split RGB diagonal targets in `scripts/precompute_diag_targets.py`.

Experiment and orchestration scripts include:

- SLURM jobs for 10-step smoke testing, 20k-step DDPM training, 10k-step pure MSE training, data-driven diagonal training, precomputed diagonal-target training, target precomputation, and checkpoint-final sample generation.
- Decoder and dataset diagnostic scripts for no-speed and delta-displacement representations.
- Analysis scripts for DDPM delta-displacement sanity checks and decoder alignment diagnostics.

Generated outputs include:

- `logs/`: SLURM `.out` / `.err` files.
- `results_hpc/`: copied HPC run outputs, train logs, summaries, checkpoint metadata/configs, and periodic sample PNGs.
- `Thesis_eval/`: generated sample PNGs from checkpoint-final comparisons.
- Ignored but present checkpoint binaries under `results_hpc/**/training_state.pt` and `results_hpc/**/diffusion_pytorch_model.safetensors`.

These generated outputs are useful for local audit, but they should not usually be tracked as source files.

## Experimental Progress

The Week 2 direction has shifted from local/sanity DDPM experiments toward reproducible HPC-based DDPM training on 224x224 delta-displacement trajectory images.

Completed or visible runs include:

- A 10-step HPC smoke run using the scratch-oriented training script.
- A 10,000-step pure MSE DDPM run:
  - run name: `ddpm_scratch_10000step_bs8_pureMSE`
  - image size: 224
  - number of images: 3159
  - auxiliary loss mode: `none`
  - final global step: 10000
  - copied back to `results_hpc/ddpm_scratch_10000step_bs8_pureMSE`
- A data-driven diagonal-target DDPM attempt:
  - run name: `ddpm_scratch_10000step_bs8_dataDrivenDiag`
  - this job appears to have been cancelled before completion; exact reason is to be verified.
- A precomputed data-driven diagonal-target DDPM run:
  - run name: `ddpm_scratch_10000step_bs8_dataDrivenDiag_precomputed`
  - image size: 224
  - number of images: 3159
  - auxiliary loss mode: `data_driven_diag`
  - diagonal targets were precomputed from 2527 train images
  - final global step: 10000
  - copied back to `results_hpc/ddpm_scratch_10000step_bs8_dataDrivenDiag_precomputed`

Checkpoint-final samples were generated for comparison:

- pure MSE, 50 reverse steps: 16 samples
- pure MSE, 100 reverse steps: 16 samples
- data-driven diagonal precomputed, 50 reverse steps: 16 samples
- data-driven diagonal precomputed, 100 reverse steps: 16 samples

At this stage, these samples are evidence that checkpoint loading and generation work. They are not yet enough to support strong image-quality or trajectory-quality claims.

## Issues / Risks

- Generated artifacts are already tracked. Updating `.gitignore` is necessary but not sufficient; tracked logs/results/samples need to be removed from the Git index if the goal is a clean source repository.
- `Thesis_eval/` is not currently covered by the new `.gitignore` additions and contains tracked generated PNG samples.
- Some result directories contain only config JSON files tracked, while the corresponding checkpoint weights are ignored. This makes the tracked checkpoint folders incomplete for reproducibility and still noisy for Git history.
- The cancelled non-precomputed `dataDrivenDiag` run should be documented as incomplete unless rerun successfully.
- Current checkpoint-final sample sets are very small (`n=16` per condition), so visual inspection or FID/KID on these samples would be unstable.
- FID/KID and visual image metrics do not prove trajectory validity. The generated GAF images should eventually be decoded back to trajectories and evaluated in trajectory space.
- Generated-image-to-trajectory decoding remains to be verified for current DDPM outputs. Existing decoder work is valuable, but using it as a judge for generated samples requires careful validation.

## Tentative Week 2 Plan

1. Clean repository tracking.
   - Keep reusable source code, tests, docs, requirements, and SLURM scripts.
   - Remove generated logs, copied HPC outputs, checkpoint metadata, sample PNGs, and large result artifacts from the Git index while preserving local files if needed.
   - Add ignore coverage for `Thesis_eval/` or move generated evaluation samples under an already ignored output directory.

2. Preserve reproducible HPC workflows.
   - Keep `scripts/train_ddpm_scratch.py`, `scripts/precompute_diag_targets.py`, `scripts/generate_ddpm_samples.py`, and the SLURM files.
   - Consider adding a short command table documenting pure MSE vs data-driven diagonal precomputed runs.

3. Evaluate image generation more systematically.
   - Generate larger sample sets from `checkpoint-final` for pure MSE and precomputed data-driven diagonal runs.
   - Compute FID/KID or related image-level metrics against the real 224x224 delta-displacement image set.
   - Compare 50, 100, and possibly 1000 inference steps if compute budget allows.

4. Compare pure MSE vs data-driven diagonal target runs.
   - Compare final training losses, loss curves, sample grids, and image metrics.
   - Report the cancelled non-precomputed diagonal run separately from the completed precomputed run.
   - Avoid overclaiming unless the comparison uses consistent sample counts and seeds.

5. Move toward trajectory-level evaluation.
   - Verify whether generated GAF/delta-displacement images can be decoded back into trajectories with the existing decoder pipeline.
   - Compare decoded trajectories using ADE/FDE, aligned-shape metrics, start/end errors, and validity diagnostics.
   - If possible, test map-matching quality after decoding, but keep this separate from image-only metrics.

6. Prepare a concise meeting summary.
   - State that Week 2 established HPC training and checkpoint-final generation.
   - Present pure MSE as the clean baseline.
   - Present precomputed data-driven diagonal loss as a structural ablation.
   - Clearly mark metric results and trajectory decoding as still to be verified.

## Immediate Next Actions

- Decide whether to remove tracked generated artifacts from Git with `git rm --cached` while keeping the files locally.
- Add or confirm `.gitignore` coverage for `Thesis_eval/`, root-level SLURM `.out/.err`, `logs/`, `results_hpc/`, checkpoint binaries, and cache folders.
- Rerun `git status --ignored --short` after cleanup to verify that generated outputs are ignored rather than tracked.
- Generate larger, matched sample sets for pure MSE and data-driven diagonal precomputed checkpoints.
- Run image-level metrics, then start decoder-based trajectory-level evaluation.
- Write a short meeting note once image metrics and/or trajectory decoding results are available.

## Pixel-space DDPM 50k comparison: pure MSE vs data-driven diagonal constraint

This analysis compares generated GAF image quality for two 50k pixel-space DDPM runs. The scope is image-level GAF validity only. These models are not map-conditioned, so road-network validity, off-road behavior, and decoded trajectory/map compliance are intentionally not judged here.

### Experiment setup

| Field | Pure MSE baseline | Data-driven diagonal constraint |
|---|---:|---:|
| Run directory | `results_hpc/ddpm_scratch_50000step_bs8_pureMSE` | `results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed` |
| Sample directory | `results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples` | `results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples` |
| Image size | 224 | 224 |
| Batch size | 8 | 8 |
| Gradient accumulation | 2 | 2 |
| Max train steps | 50,000 | 50,000 |
| Sampling interval | every 2,500 steps | every 2,500 steps |
| Samples per interval | 4 | 4 |
| Total generated PNGs analyzed | 80 | 80 |

Real images were read from `/mnt/beegfs-compat/jliu/Thesis_data/data_no_speed_delta_displacement_paired/images` (`n=3159`). Both generated sample folders contain individual `224x224` PNG images named by training step, not stitched grids. No generated grid images were detected. There are 20 sampled checkpoints per method (`002500` through `050000`), with 4 PNGs per checkpoint.

Analysis outputs were saved under `results_hpc/ddpm_50k_comparison_analysis/`:

- `metrics_summary.json`
- `metrics_summary.csv`
- `fid_pureMSE_vs_real.json`
- `fid_dataDrivenDiag_vs_real.json`
- `representative_comparison_grid.png`

### Image-level metrics

Pixel values are normalized from PNG uint8 to `[-1, 1]` as `value / 127.5 - 1`. RGB distribution statistics use all real images and all generated samples. Diagonal, symmetry, and edge diagnostics use a comparable real subset of 80 images and the 80 generated images per method.

| Set | n | RGB mean | RGB std | Overall mean/std | RGB q01 | RGB q50 | RGB q99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real | 3159 | `[-0.3830, 0.0004, -0.1905]` | `[0.4845, 0.4361, 0.8526]` | `-0.1910 / 0.6391` | `[-1.0000, -0.9608, -1.0000]` | `[-0.5137, 0.0039, -0.7882]` | `[0.9216, 0.9608, 0.9608]` |
| Pure MSE | 80 | `[-0.1555, -0.0399, -0.0403]` | `[0.6331, 0.3371, 0.8020]` | `-0.0785 / 0.6236` | `[-1.0000, -0.9137, -1.0000]` | `[-0.2392, -0.0275, -0.2627]` | `[0.9922, 0.8431, 1.0000]` |
| Data-driven diag | 80 | `[-0.0032, -0.0580, -0.0457]` | `[0.6545, 0.4759, 0.8166]` | `-0.0356 / 0.6642` | `[-0.9922, -0.9922, -1.0000]` | `[0.0510, -0.0980, -0.3490]` | `[1.0000, 0.9922, 1.0000]` |

The generated sets are both shifted away from the real RGB distribution. The shift is strongest in the red channel: real images have mean red around `-0.3830`, while pure MSE is `-0.1555` and dataDrivenDiag is near neutral at `-0.0032`. The data-driven diagonal run therefore does not simply match the real image color distribution; it produces a more saturated/high-contrast color balance with a visible magenta/red bias in some middle and late samples.

### RGB statistics: color/channel-distribution validity

RGB/channel-distribution validity is used here as a simple image-space check for whether generated GAF images occupy the same color range as real GAF images. This is not a semantic trajectory metric, but it is important because the three image channels encode structured trajectory-derived quantities; a model can produce plausible-looking grids while still drifting into an unrealistic channel distribution.

The real image distribution has RGB mean `[-0.3830, 0.0004, -0.1905]`, with a strongly negative red mean, near-zero green mean, and broad blue-channel spread. Both generated 50k runs deviate from this reference:

- Pure MSE RGB mean is `[-0.1555, -0.0399, -0.0403]`, so red and blue are both shifted upward relative to real images.
- DataDrivenDiag RGB mean is `[-0.0032, -0.0580, -0.0457]`, which is even farther from the real red-channel mean and visually corresponds to stronger red/magenta bias.
- Pure MSE has RGB std `[0.6331, 0.3371, 0.8020]`, while DataDrivenDiag has `[0.6545, 0.4759, 0.8166]`; both have red-channel variance higher than real (`0.4845`), and DataDrivenDiag also increases green-channel spread.
- The generated q99 values frequently reach `1.0`, especially for red and blue, while real images have q99 `[0.9216, 0.9608, 0.9608]`. This indicates more saturation/clipping-like behavior in generated samples.

Interpretation: neither model fully matches the real channel distribution. Pure MSE is closer than DataDrivenDiag in red-channel mean, while DataDrivenDiag can look more grid-regular but introduces a stronger color-distribution shift. This supports treating diagonal/color constraints as only a partial structural prior: improving one GAF statistic can still leave global channel validity mismatched.

### GAF-structure diagnostics

| Set | Diagonal RGB mean | Diagonal RGB std | Symmetry error mean/std | Edge strength mean/std |
|---|---:|---:|---:|---:|
| Real | `[-0.3495, 0.0039, 0.7589]` | `[0.2241, 0.0000, 0.0702]` | `0.2274 / 0.0558` | `0.1501 / 0.0279` |
| Pure MSE | `[-0.1636, -0.0353, 0.5441]` | `[0.5461, 0.0958, 0.2755]` | `0.2674 / 0.0947` | `0.1352 / 0.0379` |
| Data-driven diag | `[-0.0202, -0.0516, 0.6186]` | `[0.5729, 0.3524, 0.1676]` | `0.2598 / 0.0897` | `0.1300 / 0.0343` |

The known precomputed diagonal target for the dataDrivenDiag run is:

`target_rgb = [-0.3689744770526886, 0.003921627998352051, 0.7749511003494263]`

The dataDrivenDiag generated diagonal mean has L2 distance `0.3862` from this target. Relative to pure MSE, it moves the blue diagonal component closer to the target (`0.6186` vs `0.5441`, target `0.7750`), but the red diagonal component moves farther from the target (`-0.0202` vs `-0.1636`, target `-0.3690`) and the green channel is also shifted. The auxiliary loss therefore visibly affects diagonal/color behavior, but this 50k sample set does not show clean convergence to the precomputed diagonal RGB target.

Symmetry error is lower for dataDrivenDiag than pure MSE (`0.2598` vs `0.2674`), but both are worse than the real subset (`0.2274`). This suggests a modest improvement in GAF-like structural consistency, not a decisive match to the real distribution. Edge strength is lower for both generated sets than real images, with dataDrivenDiag lowest (`0.1300`), consistent with smoother or broader stripe regions and less fine real-image edge content.

### Exploratory FID

The existing `scripts/evaluate_fid.py` supports these folders directly and was run with Inception weights cached under `/tmp/torch-codex`.

| Comparison | Real n | Generated n | FID |
|---|---:|---:|---:|
| Pure MSE vs real | 3159 | 80 | 86.3372 |
| Data-driven diag vs real | 3159 | 80 | 84.5526 |

These FID values should be treated as exploratory only. Each generated set has only 80 images, which is small for stable Inception covariance estimation. The slight FID advantage for dataDrivenDiag is consistent with a mild image-level improvement, but it is not strong evidence on its own.

### Qualitative observations

Early samples from both methods already show GAF-like horizontal and vertical stripe/grid structure. The pure MSE early sample has many green/blue grid bands and some blurred blended areas. The dataDrivenDiag early sample is visually similar but appears slightly more organized in large stripe blocks.

Middle-step samples diverge more clearly. Pure MSE at 25k shows broad pastel bands with many thin vertical/horizontal stripe artifacts. DataDrivenDiag at 25k shows a denser and more regular lattice-like structure, but with a strong red/magenta/yellow color bias and many tiny fragmented grid cells near the center.

Late/final samples from both runs can become overly simple. The pure MSE 50k representative is dominated by large magenta fields with a few strong stripe boundaries. The dataDrivenDiag 50k representative is even more collapsed toward a smooth magenta field with only a small number of boundary stripes. This supports the metric finding that dataDrivenDiag can increase regularity while also shifting color distribution and reducing edge strength.

### Preliminary conclusion

Pure MSE remains the unconstrained pixel-space DDPM baseline. It learns recognizable GAF stripe/grid structure, but its samples have higher symmetry error than real images and can show blurred or fragmented stripe artifacts.

The data-driven diagonal constraint appears to provide a small structural benefit: symmetry error is modestly lower than pure MSE, FID is slightly lower, and qualitative middle-step samples can look more regularly grid-like. However, the same run shows a substantial color-distribution shift, especially in the red channel, and its generated diagonal RGB mean is still far from the precomputed target. Diagonal constraints alone are therefore not sufficient to guarantee realistic GAF image statistics.

No road-network or map-compliance conclusion should be drawn from this comparison. These runs have no map prior and no trajectory-level conditioning, so map inconsistency is expected and should be reserved for later map-conditioned experiments.

### Limitations and next steps

- Generated sample count is only 80 images per method, with 4 samples per checkpoint. FID is therefore weak and should not be overinterpreted.
- The diagnostics are image-level only. They do not prove that decoded trajectories are kinematically plausible or map-compliant.
- The dataDrivenDiag comparison suggests a tradeoff between structural regularity and color bias, so future losses should track full-channel distribution and not only diagonal means.
- Next step: compute image-level GAF validity metrics on larger matched final-checkpoint sample sets, then decode generated images and evaluate trajectory-space metrics separately.
- Later step: introduce map-conditioned generation or map-aware guidance before judging road-network validity.

### Sampling-step ablation: 50 vs 150 reverse diffusion steps

A follow-up sampling-only ablation was run to test whether increasing reverse diffusion steps from 50 to 150 improves GAF image quality without retraining. To keep the comparison matched, both 50-step and 150-step sample sets reuse the same existing checkpoints: 40 samples from `checkpoint-25000` and 40 samples from `checkpoint-50000` for each method, giving 80 generated images per method and per sampling setting.

- Pure MSE, 50-step matched: `results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_50steps_matched`
- Pure MSE, 150-step: `results_hpc/ddpm_scratch_50000step_bs8_pureMSE/samples_150steps`
- DataDrivenDiag, 50-step matched: `results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_50steps_matched`
- DataDrivenDiag, 150-step: `results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/samples_150steps`
- 50-step matched analysis: `results_hpc/ddpm_50k_sampling_steps_50_matched_analysis`
- 150-step analysis: `results_hpc/ddpm_50k_sampling_steps_150_analysis`

| Method | Steps | n | RGB mean | Diagonal RGB mean | Symmetry error | Edge strength | FID |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure MSE | 50 | 80 | `[0.0349, 0.0260, -0.0538]` | `[0.0174, 0.0270, 0.6346]` | `0.2685` | `0.1436` | `78.0283` |
| Pure MSE | 150 | 80 | `[0.0137, 0.0033, -0.0719]` | `[0.0000, 0.0048, 0.6732]` | `0.2738` | `0.1465` | `74.6189` |
| DataDrivenDiag | 50 | 80 | `[0.2007, -0.0896, -0.2588]` | `[0.1616, -0.0854, 0.5792]` | `0.2504` | `0.1362` | `93.9261` |
| DataDrivenDiag | 150 | 80 | `[0.1939, -0.0365, -0.2593]` | `[0.1569, -0.0326, 0.6313]` | `0.2892` | `0.1519` | `81.0313` |

Increasing sampling steps improves FID for both methods under the matched checkpoint/sample-count protocol. Pure MSE FID drops from `78.03` to `74.62`, and DataDrivenDiag drops from `93.93` to `81.03`. These values remain exploratory because each generated set has only 80 samples, but the matched comparison supports a real sampling-step benefit for Inception-level image similarity.

Edge strength also improves with 150 steps. Pure MSE rises from `0.1436` to `0.1465`, closer to the real-image reference of `0.1501`. DataDrivenDiag rises from `0.1362` to `0.1519`, essentially matching or slightly exceeding the real reference. Qualitatively, the 150-step representative grid shows sharper stripe boundaries and less globally blurred texture.

Symmetry does not improve. Pure MSE symmetry error increases slightly from `0.2685` to `0.2738`, and DataDrivenDiag worsens more clearly from `0.2504` to `0.2892`. Since lower symmetry error is more GAF-like, extra reverse steps do not fix structural consistency and may amplify asymmetric stripe artifacts for the diagonal-loss run.

Diagonal statistics are mixed. Pure MSE moves the blue diagonal channel from `0.6346` to `0.6732`, closer to the real/target blue range, but red remains too high relative to the precomputed target red value (`-0.3690`). DataDrivenDiag improves blue diagonal value from `0.5792` to `0.6313` and green from `-0.0854` to `-0.0326`, but red remains strongly shifted positive (`0.1569`). The diagonal constraint therefore does not become correctly calibrated simply by using more inference steps.

Color shift remains the main concern. Pure MSE at 150 steps is slightly less red-positive than the matched 50-step set (`0.0137` vs `0.0349`), but both are far from the real red mean of `-0.3830`. DataDrivenDiag remains strongly red-biased under both settings (`0.2007` at 50 steps and `0.1939` at 150 steps). Qualitatively, 150-step samples have crisper grids, but DataDrivenDiag still shows strong red/yellow/magenta regions and the final-checkpoint examples remain color-shifted.

Recommendation: use 150 reverse diffusion steps when the goal is best image-level sample quality for reporting or offline evaluation, because matched FID and edge sharpness improve. Keep 50 steps for fast periodic monitoring during training, because the 150-step setting is more expensive and does not improve symmetry or diagonal-target calibration. As before, this ablation is image-only and does not evaluate decoded trajectory quality or map/road-network validity.

### Channel-wise GAF diagnostics: color shift as channel-distribution mismatch

Although the generated samples are saved as RGB PNG images, the three channels should not be interpreted as natural-image colors. They correspond to trajectory-derived representations: `R = GASF`, `G = GADF`, and `B = MTF`. Therefore abnormal magenta/yellow/red regions in generated images are better interpreted as distribution mismatch among the GASF/GADF/MTF channels, not as ordinary visual color artifacts.

The current channel diagnostics suggest that the major issue is not severe cross-channel coupling collapse. The pixel-level channel correlations are close in structure to the real data:

| Set | corr(GASF,GADF) | corr(GADF,MTF) | corr(GASF,MTF) |
|---|---:|---:|---:|
| Real | `-0.0003` | `-0.0089` | `-0.1559` |
| Pure MSE generated | `0.0044` | `-0.0151` | `-0.0682` |
| DataDrivenDiag generated | `-0.0446` | `-0.0225` | `-0.0942` |

The generated samples preserve the near-independent relationship between GASF/GADF and GADF/MTF, and partially preserve the weak negative correlation between GASF and MTF. The GASF-MTF negative correlation is weaker than in real data, especially for pure MSE, but the correlation structure is not completely destroyed.

The more likely dominant failure mode is channel-wise marginal distribution shift. Real images have RGB/channel mean `[-0.3830, 0.0004, -0.1905]`. The generated sample means are shifted upward in the GASF/R channel:

| Set | RGB/channel mean |
|---|---:|
| Real | `[-0.3830, 0.0004, -0.1905]` |
| Pure MSE | `[-0.1555, -0.0399, -0.0403]` |
| DataDrivenDiag | `[-0.0032, -0.0580, -0.0457]` |

The red/GASF channel shifts upward strongly, especially in DataDrivenDiag. This explains the visual appearance of abnormal magenta/yellow/red regions: the model tends to over-activate the GASF channel relative to the real distribution.

There is also saturation-like behavior. Generated samples often have q99 values close to `1.0`, especially in red/blue-related channels, while real images have q99 values below `1.0` (`[0.9216, 0.9608, 0.9608]`). This suggests that the issue is not merely a small mean drift; some generated channel values are pushed toward normalization boundaries.

Overall, the DDPM has learned coarse GAF-like grid structure and has not completely destroyed the channel relationship. However, it has not correctly matched the marginal distribution of each mathematical channel. This means the diagonal constraint improves only a local statistic; it is insufficient to enforce global channel validity. The apparent color artifacts are therefore better understood as GASF/GADF/MTF distribution mismatch.

Supporting per-channel symmetry diagnostics show that the structural errors are channel-specific, not just global blur. Real GASF/R is exactly symmetric, while generated GASF/R has nonzero symmetry error (`0.0410` for pure MSE and `0.0617` for DataDrivenDiag). Real GADF/G is naturally asymmetric (`0.6070`), so high asymmetry in that channel should not be interpreted as a failure by itself. The largest structural mismatch is in MTF/B: real MTF/B symmetry error is only `0.0673`, while pure MSE is `0.2715` and DataDrivenDiag is `0.2280`. Thus, MTF/B appears to be a major contributor to fragmented or structurally inconsistent GAF patterns.

Post-hoc affine calibration was also tested as a diagnostic, not as a final solution. Calibrated images were saved under `results_hpc/channel_diagnostics/pureMSE_calibrated` and `results_hpc/channel_diagnostics/dataDrivenDiag_calibrated`, with calibrated metrics under `results_hpc/channel_diagnostics/calibrated_image_quality_analysis`. Calibration improved exploratory FID substantially: pure MSE improved from `74.6189` to `63.2486`, and DataDrivenDiag improved from `81.0313` to `67.6198`. It also moved the RGB means much closer to the real distribution: pure MSE from `[0.0137, 0.0033, -0.0719]` to `[-0.3641, 0.0011, -0.1623]`, and DataDrivenDiag from `[0.1939, -0.0365, -0.2593]` to `[-0.3631, 0.0006, -0.2037]`.

However, calibration does not solve all structure. Global symmetry slightly worsened after calibration (`0.2738` to `0.2940` for pure MSE; `0.2892` to `0.2907` for DataDrivenDiag), because channel rescaling changes the relative weight of each channel in the global metric. This supports a nuanced interpretation: channel-wise mean/std mismatch is a major part of the image-level distribution problem, but cross-channel consistency and channel-specific GAF structure still need to be preserved.

The next step should be lightweight before changing model architecture. First, perform post-hoc channel-wise calibration:

`x'_c = ((x_c - mean_gen_c) / std_gen_c) * std_real_c + mean_real_c`

Then reevaluate FID, channel statistics, symmetry, edge strength, and diagonal statistics after calibration. If calibration helps, train a channel-wise normalized DDPM where each channel is standardized independently during training and inverse-transformed after generation. A cross-channel correlation loss should be considered later, because the current mismatch appears weaker than the marginal channel shift.

This remains an image-space diagnostic only. It does not evaluate decoded trajectories, road-network validity, or map compliance.

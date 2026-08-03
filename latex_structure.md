# Proposed LaTeX thesis structure

## Working title and central argument

**Generative Models for Traffic Trajectory Synthesis: A Comparative Study**

The thesis should be organized around one main question:

> How should traffic trajectories be represented for generative modeling?

The defensible central claim from the retained evidence is:

> A 2D GASF representation exposes useful global pairwise structure and can be modeled successfully in image space, but image fidelity is not sufficient for trajectory synthesis. Direct diffusion of absolute coordinate sequences currently yields far higher physical/support validity because it avoids image-manifold and inverse-transform errors. Its remaining weakness is under-coverage of real trajectory extent.

This wording avoids two overclaims. First, the experiments do not establish that GASF is intrinsically lossy: the Week 04 float `[0,1]` encoding reconstructs real trajectories almost exactly. Second, the raw model has not yet demonstrated full distributional fidelity: its high acceptance is accompanied by shorter displacement and smaller bounding boxes.

## Front matter

Include:

- Title page, declaration, acknowledgements.
- Abstract stating the representation question, the two main pipelines, the best image FID (44.55), the decoded GASF acceptance problem (9.2% strict empirical), the best raw acceptance (98.5%), and the raw-model coverage caveat.
- Table of contents, list of figures, list of tables, notation/acronyms.
- Reproducibility statement describing repository phases, data provenance, compute/scheduler setup, retained checkpoints, and the fact that failed experiments are documented.

Suggested acronyms: ADE, FDE, DDPM, DDIM, GAF, GASF, GADF, MTF, FID, KID, OSM, U-Net, EMA, CRS, UTM.

# Chapter 1 — Introduction

## 1.1 Problem context

**Content to write:** Introduce traffic trajectory synthesis, its uses in simulation, safety testing, mobility analysis, and data augmentation. Explain that a trajectory is a structured time series with geometric, kinematic, and map constraints; it is not simply an arbitrary vector or image.

**Experiments:** None in detail; preview the raw and GASF approaches.

**Figures/tables:** One conceptual figure showing the two branches:

`trajectory → raw [2×T] DDPM → trajectory` versus `trajectory → GASF [3×T×T] → image DDPM → inverse → trajectory`.

## 1.2 Research gap

**Content to write:** Representation choices common in discriminative time-series work are often carried into generation without testing invertibility or physical validity. Explain why classification can tolerate nuisance invariances and partial information while generation must reproduce every coordinate and obey support constraints.

**Experiments:** Cite the legacy Hilbert/GAF branch as the motivating case and the later direct raw baseline as the missing comparison that the thesis supplies.

**Figures/tables:** A compact requirements table: discriminative sufficiency, invertibility, dimensionality, manifold constraints, physical constraints, map conditioning.

## 1.3 Research questions

Use explicit questions:

1. How much information is preserved by raw, legacy Hilbert–GAF, and constrained separate-axis GASF representations?
2. Can image-space DDPM quality predict trajectory-space quality?
3. How do absolute-coordinate and displacement diffusion compare in physical validity and coverage?
4. Can structural losses, learned decoders, rejection, or map guidance close the GASF validity gap?

## 1.4 Contributions

**Content to write:** Present contributions conservatively:

1. A repository-scale comparative study spanning raw coordinate diffusion and multiple GAF/GASF formulations.
2. A controlled invertibility analysis distinguishing transform loss from generated-manifold violations.
3. A trajectory-space evaluation pipeline with empirical support, OSM distance, rejection rate, extent, displacement, and step statistics.
4. Evidence that the best GASF image FID does not translate to physical validity, while direct absolute 1D U-Net diffusion achieves 98.5% empirical acceptance but exhibits conservative extent.
5. A documented failure analysis of sign ambiguity, cumulative drift, structural-loss NaNs, map-frame mismatch, and ineffective guidance.

## 1.5 Thesis organization

**Content to write:** One paragraph per remaining chapter.

# Chapter 2 — Background

## 2.1 Traffic trajectory modeling

**Content to write:** Define trajectory \(p_{1:T}\in\mathbb R^{T\times2}\), absolute positions, displacements \(d_t=p_t-p_{t-1}\), length normalization, coordinate frames, map constraints, and cumulative error.

**Experiments:** Use the delta decoder's low local error/high integrated error as an illustrative observation, not yet a result claim.

**Figures/tables:** Small schematic of local delta error accumulating after integration.

## 2.2 Generative modeling of sequences

**Content to write:** Briefly position autoregressive, VAE/GAN, and diffusion approaches. Focus the technical review on denoising diffusion.

**Experiments:** None.

**Figures/tables:** Optional taxonomy table; keep short.

## 2.3 Denoising diffusion probabilistic models

**Content to write:** Derive the forward process, \(\beta_t\), \(\alpha_t\), \(\bar\alpha_t\), noise prediction loss, reverse mean, ancestral DDPM and DDIM sampling, and EMA. Explain how the same objective is applied to 2D images and 1D trajectory tensors.

**Experiments:** Root sampler ablation and all DDPM models belong later, but cite this section for their common formulation.

**Figures/tables:** Diffusion forward/reverse diagram; notation table.

## 2.4 Convolutional model families

**Content to write:** Explain 2D U-Net for GASF images, temporal residual networks, dilated convolutions, and 1D U-Net for raw sequences. State receptive-field implications.

**Experiments:** Root/Week 05 2D U-Net and Week 06 three raw architectures.

**Figures/tables:** Architecture comparison table with input/output shapes and principal widths.

## 2.5 Gramian Angular Fields

**Content to write:** Define angular encoding, GASF and GADF equations, diagonal identity, symmetry/antisymmetry, and MTF. Distinguish the legacy Hilbert scalar input from the later separate-axis input.

**Experiments:** Week 03 structure statistics and Week 04 invertibility diagnostics.

**Figures/tables:** One real RGB GAF example split into R/G/B, plus the Week 03 structure-component figure.

## 2.6 Evaluation of generated trajectories

**Content to write:** Separate reconstruction metrics (ADE/FDE) from unconditional distribution metrics. Discuss step/path/displacement/extent statistics, support/road distance, acceptance rate, and diversity/coverage. Explain FID and why Inception features on scientific GASF images are only a proxy.

**Experiments:** All result chapters.

**Figures/tables:** Metric-definition table with unit, direction, and limitations.

## 2.7 Digital maps and map constraints

**Content to write:** OSM graphs, CRS/UTM, affine alignment, raster distance fields, nearest-road distance, differentiable grid sampling, and map matching. Explicitly distinguish LCSS map matching from nearest-road/support evaluation.

**Experiments:** Week 03 map construction/refinement and Week 06 map diagnosis/guidance.

**Figures/tables:** OSM alignment example and distance-field schematic.

# Chapter 3 — Dataset and preprocessing

## 3.1 Data sources and provenance

**Content to write:** Document two datasets/configurations separately:

- Legacy `gps_with_speed_224_UPDATED.xls`, which is read as CSV-formatted text and yields 3,159 fixed trajectories.
- Week 05 `vehicle_positions_TS_New.csv`, containing 2,189,591 rows and 12,061 vehicle IDs.

Do not silently merge them. Explain coordinate fields, vehicle grouping, sorting by time, omitted speed where applicable, and dataset-specific coordinate/map transforms.

Also add an artifact-provenance note: `docs/weekly_report_refactoring_validation.md` records original notebooks and an `experiments/` tree that are absent at thesis-archaeology time. Results recoverable only from that report must be labeled historical/report-only.

**Experiments:** P0.1 and P0.3 in `experiment_inventory.md`.

**Figures/tables:** Dataset comparison table with source, IDs, length distribution, coordinate frame, split, and phase usage.

## 3.2 Cleaning and grouping

**Content to write:** CSV loading, grouping by vehicle ID, temporal sorting, coordinate extraction, missing/invalid-value handling observable in code, global normalization in the legacy branch, and z-score/sigmoid statistics in the newer branch.

**Experiments:** Deterministic builders and validation checks.

**Figures/tables:** Data-flow diagram; no need to include every raw-file screenshot.

## 3.3 Fixed-length policies

**Content to write:** Explain squeeze, reflect-duplicate, sliding windows, bounce extension, and truncation. Discuss how each changes the empirical distribution.

**Experiments:** Week 05 policy comparison.

**Figures/tables:** Table with 3,877 windows, 8,738 skipped short paths, and bounce/truncate counts. A length histogram should be generated if not already available.

## 3.4 Delta and absolute representations

**Content to write:** Define delta construction including the first zero delta in the legacy paired dataset, start-point storage, cumulative integration, and absolute z-scoring.

**Experiments:** Root paired builder and Week 06 raw models.

**Figures/tables:** One trajectory shown in absolute and delta coordinates; normalization-statistics table.

## 3.5 Train/validation/test splits and leakage risks

**Content to write:** Legacy split 2,527/315/317 and newer split 9,648/1,206/1,207. Verify grouping is by vehicle before final thesis submission. Discuss overlapping windows and the need to split by vehicle before window extraction. Note that Week 06 delta generation samples a real start point, which introduces a conditioning/data-access assumption.

**Experiments:** All supervised decoder and generation studies.

**Figures/tables:** Split table and leakage checklist.

## 3.6 Preprocessing validation

**Content to write:** Report maximum reconstruction errors, alignment-mode test, and data counts. Explain why these deterministic checks are prerequisites for interpreting model error.

**Experiments:** Week 04 alignment/invertibility and Week 05 reconstruction checks.

**Figures/tables:** Alignment mode table and PNG-versus-float reconstruction table.

# Chapter 4 — Trajectory representations

## 4.1 Raw trajectory

### 4.1.1 Absolute coordinates

**Content to write:** Define normalized absolute sequence \(z_t=(p_t-\mu)/\sigma\). Benefits: direct decode, no integration drift, physical constraints act directly. Costs: translation/location distribution must be learned; maps/coordinate frames matter.

**Experiments:** Week 06 absolute raw diffusion and Week 05 unevaluated absolute GASF as a contrast.

**Figures/tables:** Real absolute trajectory contact sheet and normalization table.

### 4.1.2 Displacements

**Content to write:** Define \(d_t\), starting point, integration, local stationarity advantages, and cumulative drift. Discuss real-start conditioning in Week 06.

**Experiments:** Decoder ablations and delta raw diffusion.

**Figures/tables:** Delta-error accumulation schematic and raw absolute/delta model table.

## 4.2 Gramian Angular Fields

### 4.2.1 Legacy Hilbert–GASF/GADF/MTF image

**Content to write:** Detail global lon/lat normalization, Hilbert scalar mapping, length fixing, pyts GASF/GADF/MTF, RGB scaling, and absence of a direct 2D inverse.

**Experiments:** Root pixel DDPM, learned decoder, Week 03 structure and latent alignment.

**Figures/tables:** Real encoded image, split channels, Hilbert/GAF processing diagram.

### 4.2.2 Separate-axis constrained GASF

**Content to write:** Explain R=GASF(dx), G=GASF(dy), B=start heatmap; `[0,1]` min-max and sigmoid normalization; analytic diagonal inverse.

**Experiments:** Week 04 and Week 05 builders/DDPMs.

**Figures/tables:** Constrained GASF sanity figure and formula-to-diagonal illustration.

### 4.2.3 Information preservation and inverse conditioning

**Content to write:** Present three different phenomena clearly:

1. **Representational loss:** legacy Hilbert scalarization, fixed-length modification, and uint8 quantization.
2. **Mathematical ambiguity:** `[-1,1]` GASF diagonal loses sign.
3. **Generative invalidity:** even a lossless real-data transform fails when generated matrices are inconsistent with any valid input sequence.

Discuss square-root/logit sensitivity near diagonal bounds and cumulative integration.

**Experiments:** Week 04 alignment/sign tests and Week 06 decoded-range/saturation analyses.

**Figures/tables:** `[0,1]`/`[-1,1]` invertibility table; float/PNG error plot; generated diagonal distribution versus real.

### 4.2.4 Why GASF can help classification but hinder generation

**Content to write:** This is a central argumentative section.

For classification, GASF provides:

- global pairwise temporal correlations as spatial textures;
- exact/near-exact symmetry patterns easily exploited by CNN filters;
- redundant motifs that can remain discriminative despite quantization or partial inverse loss;
- no requirement to recover every point.

For generation, GASF imposes:

- \(O(T^2)\) pixels for an \(O(T)\) sequence;
- a low-dimensional nonlinear valid-matrix manifold;
- diagonal/off-diagonal consistency and normalization bounds;
- sign/start/inverse requirements;
- physical constraints that are nonlocal in pixel space;
- an evaluation mismatch when FID rewards image texture rather than path geometry.

The correct conclusion is not “GASF loses all information.” It is “GASF moves the difficulty from sequence modeling into constrained image generation and inversion.”

**Experiments:** Week 05 FID versus Week 06 trajectory validity; structure-loss and guidance failures; learned decoder studies.

**Figures/tables:** A two-column classification-versus-generation requirements table. This should be one of the thesis's core explanatory visuals.

## 4.3 Representation comparison and hypotheses

**Content to write:** Summarize dimensionality, invertibility, physical locality, architecture compatibility, computational cost, and conditioning needs for raw absolute, raw delta, legacy GAF, and constrained GASF.

**Experiments:** All subsequent generative comparisons.

**Figures/tables:** Full representation comparison table.

# Chapter 5 — Generative models

## 5.1 Raw trajectory DDPM

### 5.1.1 Common formulation

**Content to write:** Tensor layout `[B,2,224]`, z-score parameters, noise sampling, timestep embedding, MSE, EMA, and reverse sampling. Explain how absolute output is denormalized directly and delta output is integrated from a sampled start.

**Experiments:** Six Week 06 raw models.

**Figures/tables:** Raw DDPM pipeline diagram.

### 5.1.2 Temporal ResNet

**Content to write:** Residual temporal blocks, time conditioning, hidden width 128, eight blocks.

**Experiments:** Absolute/delta temporal ResNet.

**Figures/tables:** Compact block diagram.

### 5.1.3 Dilated temporal ResNet

**Content to write:** Dilation and receptive-field rationale.

**Experiments:** Absolute/delta dilated runs.

### 5.1.4 One-dimensional U-Net

**Content to write:** Down/up sampling, skip connections, multiscale temporal structure.

**Experiments:** Absolute/delta 1D U-Net.

### 5.1.5 Raw sampling guidance

**Content to write:** Differentiable support-field energy and why it may be redundant or misaligned once the model already learns the support.

**Experiments:** Six strength-0.1 guided runs.

**Figures/tables:** Unguided/guided support comparison.

## 5.2 GASF diffusion

### 5.2.1 Pixel-space 2D U-Net

**Content to write:** Three channels, 224×224 input, block widths, attention stage, 1,000 timesteps, loss and EMA. Separate legacy and constrained inputs even though the network family is shared.

**Experiments:** Root 50k, Week 04, and Week 05 runs.

**Figures/tables:** Architecture diagram and phase-specific hyperparameter table.

### 5.2.2 Representation-aware losses

**Content to write:** Data-driven diagonal, channel normalization, symmetry, and full GASF-structure losses. Describe intended constraint and observed optimization tradeoff.

**Experiments:** Historical 3k notebook-auxiliary/MSE/diagonal matrix, root post-hoc calibration and ChannelNorm/MTFSym, Week 04 six-run matrix, Week 05 six-run matrix.

**Figures/tables:** Objective definitions and outcome table, including NaN failures.

### 5.2.3 Latent GAF diffusion

**Content to write:** Autoencoder, latent dimensions, latent U-Net, computational motivation, and incomplete evaluation.

**Experiments:** 14×14 and 28×28 branches including bad baseline.

**Figures/tables:** Autoencoder diagram; reconstruction/sample grid.

### 5.2.4 Decoding to trajectories

**Content to write:** Contrast legacy learned inverse with constrained analytic diagonal inverse and start heatmap soft-argmax. Derive sigmoid inverse and cumulative integration.

**Experiments:** Decoder studies, Week 04 inverse test, Week 06 decode.

**Figures/tables:** End-to-end GASF generation/decode diagram and error-source decomposition.

### 5.2.5 GASF consistency and road guidance

**Content to write:** Sampling-time gradient guidance, differentiable decoding, diagonal blend, clipping, endpoint cap, and the failure mechanism near bounds.

**Experiments:** Week 05 consistency guidance and Week 06 road-guidance variants.

**Figures/tables:** Guidance objective diagram; unguided/full-channel/fixed/endpoint-cap result table.

# Chapter 6 — Experimental evaluation

## 6.1 Experimental protocol

**Content to write:** Hardware/scheduler environment if recoverable, seeds, optimizer, batch, effective batch, train steps, EMA, checkpoint selection, reverse steps, sample counts, and split definitions. State where only one seed exists.

**Experiments:** All principal full runs; exclude smoke tests from the primary protocol.

**Figures/tables:** Master hyperparameter table and experiment-to-artifact table.

## 6.2 Image-space generation results

**Content to write:** Begin with the report-only 128×128 and 224×224 3k sanity experiments, then present root PureMSE/diagonal, step/sampler ablations, post-hoc calibration, ChannelNorm, MTFSym, and the Week 05 sigmoid/loss checkpoint sweep. Explain checkpoint selection at 40k for sig1.5 diagonal. Keep calibration visually/semantically separate because it uses real moments and does not improve the learned model.

**Experiments:** P1.0–P1.4 and P5.1.

**Figures/tables:**

- Root FID table.
- Week 05 FID-by-checkpoint table/line plot.
- Representative comparison grid.
- Channel-correlation/structure diagnostics.

## 6.3 Inverse reconstruction results

**Content to write:** Start with the report-only CNN-FC/naive/relative/absolute-ResNet validation, then decoder architecture/channel losses, absolute-versus-delta representation, the duplicate β=0.2 run outcome, Hybrid V1–V3, oracle similarity upper bounds, centroid correction, structure-aware ResNet, drift-aware correction, and sequence latent alignment.

**Experiments:** P2.2–P3.3.

**Figures/tables:** Consolidated ADE/FDE table; best/median/worst reconstruction examples; drift correction histogram. Keep root and Week 03 comparisons grouped by matched setup.

## 6.4 Map-aware reconstruction

**Content to write:** Affine/map construction, early/late fusion, oracle versus start-centered conditioning, nearest-road metrics, and refinement sweep.

**Experiments:** P3.4–P3.6.

**Figures/tables:** Map alignment example; oracle/deployable fusion table; raw/refined/ground-truth overlay.

## 6.5 GASF trajectory-space generation

**Content to write:** Decode the best-FID model, compare step/path/displacement/extent/support, diagnose map mismatch, and report corrected OSM metrics and rejection rates.

**Experiments:** P6.1–P6.2.

**Figures/tables:** Generated-versus-real empirical overlay; OSM coverage figure; filter comparison; accepted contact sheet; quantitative table.

## 6.6 Raw trajectory diffusion results

**Content to write:** Compare six models within representation, then absolute versus delta using validity and extent. Make the central result explicit: absolute U-Net has 98.5% support acceptance and best support distance but substantially smaller displacement/extent than real.

**Experiments:** P6.3.

**Figures/tables:** Six-model table; `raw_diffusion_best_median_worst_support.png`; model-specific contact sheet for absolute U-Net.

## 6.7 Guidance and post-processing

**Content to write:** Raw guidance, GASF consistency guidance, full-channel road guidance failure, fixed-diagonal repair, endpoint cap, and rejection filtering. Discuss the difference between improving a constraint conditional on acceptance and improving the generator.

**Experiments:** P5.2, P6.2, P6.4, P6.5.

**Figures/tables:** Guidance progression table and overlay panels.

## 6.8 Direct representation comparison

**Content to write:** This should be the headline comparison, limited to matched Week 05/06 data where possible. Do not compare FID 44.55 numerically against acceptance 98.5%; instead place metric families side by side.

**Experiments:** Best GASF (sig1.5 diagonal 40k), best raw absolute U-Net, best raw delta U-Net, and real reference.

**Figures/tables:** One table containing image metric availability, analytic invertibility, acceptance, support mean/p95, displacement, bbox, compute/memory, and conditioning assumptions. Missing cells must remain “not measured.”

# Chapter 7 — Failure analysis

## 7.1 Legacy representation information loss

**Content to write:** Hilbert scalarization, sequence squeeze/duplication, uint8 quantization, and the need for a learned inverse.

**Experiments:** P0.1, P2.2–P3.3.

**Figures/tables:** Error-source diagram and PNG-versus-float examples.

## 7.2 Sign ambiguity and inverse conditioning

**Content to write:** `[-1,1]` diagonal ambiguity; `[0,1]` exactness on real data; square-root/logit instability at generated bounds.

**Experiments:** P0.2 and P6.5.

**Figures/tables:** Sign table; diagonal saturation histograms (new summary plot may be required).

## 7.3 Pixel manifold violation

**Content to write:** A valid GASF matrix is determined by a length-T sequence but occupies a T×T image. Independent pixel noise and approximate denoising need not preserve global algebraic consistency. Explain why diagonal or symmetry penalties are necessary but insufficient.

**Experiments:** Root constraint ablations, Week 04 NaN losses, Week 05 FID, Week 06 decode.

**Figures/tables:** Real versus generated structural residuals; image-FID versus trajectory-validity juxtaposition.

## 7.4 Cumulative drift and physical constraint violation

**Content to write:** Relate local delta ADE to integrated ADE/FDE, extent explosion in GASF decode, and lower validity of raw delta models.

**Experiments:** Multibranch β=0, drift heads, GASF decoded statistics, raw delta models.

**Figures/tables:** Delta-error integration schematic and endpoint-error examples.

## 7.5 Metric mismatch

**Content to write:** Natural-image FID on scientific images, small 80-sample root evaluations, no multi-seed intervals, and lack of a unified trajectory precision/recall score. State that best FID at 40k worsened later while train loss continued.

**Experiments:** Root and Week 05 FID sweeps.

**Figures/tables:** FID-over-step plot and metric limitations table.

## 7.6 Map-frame mismatch

**Content to write:** The Week 03 affine transform was correct for its legacy source but wrong for Week 05. Detail the initial 0.0395% coverage and 2,247 m gap, new graph, and offset search. Treat this as a reproducibility lesson.

**Experiments:** P3.4 and P6.1.

**Figures/tables:** Coverage plots and offset heatmap.

## 7.7 Failed optimization and systems attempts

**Content to write:** Preserve negative and operational results:

- GASF-structure NaNs near steps 490/620.
- Historical 128×128 and 224×224 sanity outputs and original notebooks no longer present, leaving report-only provenance.
- Safe debug stopped too early.
- Float run time limit at 33,390.
- Initial `[-1,1]` metric KeyError.
- Latent bad baseline and missing latent evaluation.
- First raw jobs lost due to absent log directory, followed by successful resubmission.
- Full-channel GASF guidance worsening and diagonal saturation.
- Raw/consistency guidance neutral results.

**Experiments:** The failure table in `experiment_inventory.md`.

**Figures/tables:** One failure ledger with cause, evidence, fix/mitigation, and outcome. Do not include every smoke sample in the body; place paths in an appendix.

## 7.8 Oracle conditioning and post-selection

**Content to write:** Future bbox in map-aware models/refinement, real-start conditioning for delta raw sampling, rejection rates, and the distinction between base-generator quality and post-selected quality.

**Experiments:** Map fusion/refinement, raw delta, rejection filters.

**Figures/tables:** Assumption/availability table.

# Chapter 8 — Discussion

## 8.1 Answer to the main research question

**Content to write:** For this dataset and implementation, raw absolute sequences are the strongest current representation for generation. They provide direct inversion and local physical meaning, and the 1D U-Net learns empirical support well. GASF remains useful for structural analysis and possibly discriminative tasks but imposes a difficult image-manifold problem for generation.

## 8.2 Pixel space versus trajectory space

**Content to write:** Compare where the learning objective acts, how errors propagate, constraint locality, parameter/data efficiency, and evaluation alignment.

**Experiments:** Best GASF, raw absolute, raw delta, and guidance studies.

**Figures/tables:** Central comparison table/diagram.

## 8.3 Why classification evidence does not transfer automatically

**Content to write:** Discriminative sufficiency versus generative sufficiency; invariances; redundant global texture; exact reconstruction requirement; low-dimensional manifold; physical semantics absent from pixel neighborhoods.

**Experiments:** Structure-aware decoder improvements show that texture has information, while Week 06 shows it is hard to sample consistently.

## 8.4 What the raw result does and does not establish

**Content to write:** High empirical validity/precision, but likely mode shrinkage from small extents. No claim of superior diversity, route coverage, or likelihood until matched distributional metrics are run.

## 8.5 Role of maps

**Content to write:** Maps help evaluation and can help oracle-conditioned reconstruction. Sampling guidance was unreliable, suggesting maps may be better used as explicit conditioning during training than as a late gradient correction.

## 8.6 Threats to validity

**Content to write:** Two data sources, single seeds, fixed-length bounce artifacts, sample-count differences, Inception FID mismatch, coordinate calibration, real-start delta conditioning, oracle crops, incomplete absolute-GASF/latent branches, and absent quantitative map matching.

**Figures/tables:** Threats-to-validity table categorized as internal, construct, external, and reproducibility threats.

# Chapter 9 — Future work

## 9.1 Matched final benchmark

**Content to write:** Re-sample best raw absolute, raw delta, delta-GASF, and absolute-GASF models on one fixed held-out protocol with identical seeds and sample count. Use the same real reference and map/support transform.

## 9.2 Trajectory-space precision, recall, and diversity

**Content to write:** Add a learned or domain-specific trajectory embedding; report precision/recall, MMD or energy distance, nearest-neighbor novelty, route/region coverage, pairwise diversity, displacement/extent histograms, and confidence intervals. Avoid relying on FID alone.

## 9.3 Complete absolute-position GASF evaluation

**Content to write:** Sample the retained 20k checkpoint, decode it, compute FID only as a secondary measure, and run the exact Week 06 trajectory/support benchmark. This isolates integration drift from image-manifold error.

## 9.4 Variable-length modeling without bounce artifacts

**Content to write:** Masked diffusion, duration conditioning, resampling by physical time/distance, or length-conditioned sequence models. Split by vehicle before creating windows.

## 9.5 Map-conditioned diffusion

**Content to write:** Train with raster/vector map context, start/end or route conditioning, and classifier-free guidance. Compare to sampling-time distance-field correction.

## 9.6 Physical and kinematic constraints

**Content to write:** Speed, acceleration, jerk, curvature, heading, non-self-intersection, and endpoint constraints in trajectory space; consider parameterizing velocity/heading rather than Cartesian deltas.

## 9.7 Multi-seed and statistical evaluation

**Content to write:** At least three seeds for finalists, bootstrap intervals over generated and real samples, paired sampling comparisons, and ablation power analysis.

## 9.8 Quantitative map matching

**Content to write:** Complete the existing LCSS map-matching path, retain match success/error statistics, compare nearest-road distance with route continuity, and record the exact graph/version/CRS.

## 9.9 Constrained GASF alternatives

**Content to write:** If GASF remains in scope, generate the underlying length-T angular sequence and construct GASF deterministically, or project every sample onto the valid Gramian manifold. This removes T² unconstrained degrees of freedom while preserving GASF-derived features.

# Chapter 10 — Conclusion

## 10.1 Summary of findings

**Content to write:** Concisely restate:

- float `[0,1]` GASF is invertible for valid real encodings;
- legacy GAF needs an imperfect learned inverse;
- image-space DDPM achieved best FID 44.55 but decoded paths were frequently invalid;
- strict empirical GASF acceptance was 9.2%;
- direct absolute 1D U-Net reached 98.5% acceptance and support 8.4/25.6;
- raw samples remain too short/small, so coverage is unfinished;
- structure/guidance/post-processing helped isolated constraints but did not close the overall gap.

## 10.2 Final answer to the research question

**Content to write:** State that the best representation is task-dependent, but for unconditional trajectory synthesis under the current evidence, direct absolute-coordinate modeling is preferable because training, decoding, constraints, and evaluation occupy the same physical space.

## 10.3 Closing perspective

**Content to write:** Emphasize that representation is part of the generative model. A transformation useful for classification is not automatically a generative coordinate system.

# Appendices

## Appendix A — Complete experiment ledger

Include a condensed version of `experiment_inventory.md`, with all failed, smoke, aborted, and incomplete runs.

## Appendix B — Hyperparameters and checkpoints

Include exact command/config values, scheduler job IDs where useful, checkpoint paths, selected checkpoint rationale, and software versions recoverable from the environment/configs.

## Appendix C — Additional quantitative tables

Include the full Week 05 FID sweep, decoder per-sample summaries, all raw/guided metrics, filter rule counts, and map-refinement sweep.

## Appendix D — Additional figures

Include representative training samples, channel diagnostics, decoder best/median/worst cases, all six raw model contact sheets, guidance variants, and selected map refinements. Do not insert thousands of per-sample figures; reference the artifact-family paths documented in `figure_inventory.md`.

## Appendix E — Reproducibility and artifact manifest

Record:

- Git commit and dirty-worktree state at thesis freeze.
- Exact input file checksums and schemas.
- Split metadata checksums.
- OSM graph source/date/bbox/CRS and affine/offset parameters.
- Environment/package versions.
- Random seeds.
- Checkpoint and generated-array hashes.
- A note that no `.ipynb` notebooks were present at the archaeology date.

## Recommended core thesis figures

1. End-to-end raw versus GASF pipeline diagram.
2. Real GASF channels and symmetry decomposition.
3. `[0,1]` versus `[-1,1]` diagonal-inverse illustration.
4. Week 05 FID by checkpoint/loss/normalization.
5. Best legacy learned-inverse reconstruction examples.
6. OSM coverage failure and corrected alignment.
7. Decoded GASF versus real empirical-map overlay.
8. Raw six-model best/median/worst support comparison.
9. GASF rejection-filter acceptance/metric comparison.
10. Final matched table/figure comparing GASF, raw absolute, raw delta, and real statistics.

## Recommended core thesis tables

1. Dataset and preprocessing comparison.
2. Representation properties and invertibility.
3. Principal model hyperparameters.
4. Legacy DDPM FID and constraint ablations.
5. Learned decoder ADE/FDE.
6. Week 05 FID checkpoint sweep.
7. GASF decoded trajectory statistics and acceptance.
8. Six raw diffusion models and real reference.
9. Guidance/post-processing outcomes.
10. Failure ledger and threats to validity.

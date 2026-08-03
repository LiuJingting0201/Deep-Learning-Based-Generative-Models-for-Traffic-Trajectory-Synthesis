# Figure inventory

## Scope and accounting

There are **22,115 PNG files** in the workspace. The count below accounts for every PNG. Because most files are mechanically repeated dataset images, checkpoint samples, or per-sample sweep panels, this inventory records those sets as exact directory/filename families with counts. Singleton and thesis-ready summary figures are listed individually.

| Location | PNG count | Dominant content |
|---|---:|---|
| `results/` | 12 | Early decoder comparison examples |
| `results_hpc/` | 3,177 | Root/Week 02 DDPM, latent, decoder, and diagnostic figures |
| `week03/` | 9,669 | Decoder, structure, OSM, and especially refinement-sweep panels |
| `week04/` | 6,486 | Constrained-GASF dataset images and DDPM samples |
| `week05/` | 2,566 | Six 50k GASF runs, absolute-GASF run, normalization plots |
| `week06/` | 141 | Trajectory/map comparisons, filters, raw models, guidance |
| `Thesis_eval/` | 64 | Extra root DDPM sample sets |
| **Total** | **22,115** | — |

No notebook output cells can be recovered because no `.ipynb` file exists. An older report names notebooks and result figures that are no longer present; these are listed separately below and are not included in the 22,115-file count.

## Priority thesis-ready figures

The “shows” descriptions in this section were checked against the actual images, not inferred only from filenames.

| Filename | Location | What it shows | Recommended thesis placement |
|---|---|---|---|
| `raw_diffusion_best_median_worst_support.png` | `week06/results/raw_diffusion_visual_comparison/` | A 4×3 best/median/worst nearest-support comparison for absolute/delta U-Net, dilated ResNet, and ResNet samples on the empirical road map. It makes both high-validity absolute paths and catastrophic delta outliers visible. | Ch. 6.6, main six-model raw comparison; also Ch. 7.4 |
| `generated_vs_real_on_empirical_real_map.png` | `week06/results/trajectory_visualizations_empirical_support_metric_real1000/` | Side-by-side selected decoded GASF and real paths on the empirical map. Generated paths visibly overshoot the observed network and extent. | Ch. 6.5 and Ch. 7.3; central evidence figure |
| `real_vs_osm_coverage_osm.png` | `week06/results/week05_real_map_coverage/` | Week 05 real points and bbox far north of the reused Week 03 OSM bbox; directly visualizes the coordinate/map mismatch. | Ch. 7.6 |
| `real_trajectory_extent_raw_xy.png` | `week06/results/week05_real_map_coverage/` | The Week 05 trajectory extent in its raw XY frame. | Ch. 3.1 or Ch. 7.6 |
| `real_sample_overlay_osm.png` | `week06/results/week05_real_map_coverage/` | Example real paths over the incorrectly reused OSM graph. | Ch. 7.6 |
| `offset_search_fine_heatmap.png` | `week06/results/week05_map_offset_search/` | Fine-grid objective surface for XY translation search. | Ch. 3.6/6.5, coordinate calibration |
| `offset_search_fine_heatmap.png` | `week06/results/week05_map_offset_search_wide/` | Wider translation-search surface. | Appendix failure diagnostics |
| `offset_search_fine_heatmap.png` | `week06/results/week05_map_offset_search_extra_wide/` | Extra-wide search confirming the alignment basin. | Appendix failure diagnostics |
| `filter_metric_comparison.png` | `week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_empirical_support_mean25_p95_75/` | Six boxplots comparing real, all generated, and accepted samples for support mean/p95, displacement, bbox width/height, and max step. Acceptance fixes support but also changes geometry. | Ch. 6.5/6.7; preferred filter figure |
| `filter_metric_comparison.png` | `week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p95_strict_offset_dx220_dy-720/` | Same metric comparison for aligned strict OSM filtering. | Appendix or Ch. 6.7 |
| `filter_metric_comparison.png` | `week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p95_strict/` | Strict OSM filter before applying the selected offset. | Ch. 7.6 appendix; demonstrates calibration sensitivity |
| `filter_metric_comparison.png` | `week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_p99_loose/` | Real/generated/accepted metrics for the loose p99 filter. | Appendix rejection-threshold ablation |
| `accepted_contact_sheet_osm.png` | `week06/results/accepted_generated_on_osm_offset_dx220_dy-720/` | Twelve accepted GASF-generated trajectories over OSM with road mean/p95 labels; several accepted paths still contain loops or sharp artifacts. | Ch. 6.5; qualitative accepted-set audit |
| `accepted_all_overlay_osm.png` | Same directory | All accepted trajectories over the aligned OSM network. | Ch. 6.5 or appendix |
| `accepted_representative_overlay_osm.png` | Same directory | Representative accepted trajectories on OSM. | Ch. 6.5 |
| `generated_contact_sheet.png` | `week06/results/trajectory_visualizations/` | Contact sheet of unguided decoded GASF trajectories. | Ch. 6.5 qualitative baseline |
| `generated_vs_real_overlay.png` | Same directory | Selected unguided decoded GASF paths beside/over real examples. | Ch. 6.5 |
| `generated_all_overlay.png` | `week06/results/trajectory_visualizations_accepted_p95_strict_offset_dx220_dy-720/` | All strict accepted decoded paths in a common frame. | Appendix rejection results |
| `generated_vs_real_overlay.png` | Same directory | Strict accepted paths versus real paths. | Ch. 6.5/6.7 |
| `generated_vs_real_on_empirical_real_map.png` | `week06/results/raw_absolute_unet1d_visualizations_empirical_support/` | Absolute 1D U-Net generated paths versus real paths on the empirical map. | Ch. 6.6; best raw model qualitative figure |
| `generated_contact_sheet_empirical_real_map.png` | Same directory | Contact sheet for absolute 1D U-Net samples with empirical map context. | Ch. 6.6 or appendix |
| `generated_vs_real_on_empirical_real_map.png` | `week06/results/raw_delta_unet1d_visualizations_empirical_support/` | Delta 1D U-Net samples versus real paths; visibly greater off-support behavior. | Ch. 6.6 paired with absolute U-Net |
| `generated_vs_real_on_empirical_real_map.png` | `week06/results/week05_sig15_mse_diag_step_040000_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1_visualizations_empirical_support/` | Best endpoint-capped GASF road-guidance samples versus real paths. | Ch. 6.7, best guided GASF |
| `representative_comparison_grid.png` | `results_hpc/checkpoint50000_sampling_steps_150_analysis/` | Six generated legacy-GAF variants: raw/channel-normalized/MTF-symmetric crossed with PureMSE/diagonal. Visually exposes block/stripe artifacts and strong variant differences. | Ch. 6.2; root constraint ablation |
| `representative_comparison_grid.png` | `results_hpc/ddpm_50k_sampling_steps_150_analysis/` | Matched PureMSE versus diagonal legacy DDPM samples at 150 steps. | Ch. 6.2 |
| `representative_comparison_grid.png` | `results_hpc/ddpm_50k_sampling_steps_50_matched_analysis/` | Matched PureMSE versus diagonal samples at 50 steps. | Ch. 6.2 sampler-step ablation |
| `representative_comparison_grid.png` | `results_hpc/channelNorm_50k_sampling_steps_150_analysis/` | Channel-normalized PureMSE/diagonal samples at 150 steps. | Appendix channel normalization |
| `representative_comparison_grid.png` | `results_hpc/channelNorm_50k_sampling_steps_50_analysis/` | Channel-normalized samples at 50 steps. | Appendix sampler/channel ablation |
| `representative_comparison_grid.png` | `results_hpc/channel_diagnostics/calibrated_image_quality_analysis/` | Post-hoc calibrated sample comparison. | Appendix; calibration did not establish trajectory validity |
| `sample_000102_structure_components.png` | `week03/results/channel_structure_stats/visualizations/` | Twelve panels: raw/projected/residual/error views for symmetric GASF, antisymmetric GADF, and locally averaged/symmetry-diagnosed MTF. GASF residual is visually zero; GADF has small structured residual. | Ch. 2.5 or Ch. 4.2.1; preferred representation figure |
| `01_sample_001550_raw_refined_gt.png` | `week03/results/baseline_raw_resnet18_bs16_map_refinement_oracle_best_r25_wr5_wref0p02_wc0p05/best10_plots/raw_refined_gt/` | Ground truth, raw prediction, and refined path for a selected improvement case; title gives ADE/FDE and road-distance change. | Ch. 6.4; label clearly as oracle refinement |
| `vehicle_car1759_alignment.png` | `week03/results/map_alignment_diagnostics/` | Raw/affine/OSM alignment for a representative legacy vehicle. | Ch. 3.6/6.4 |
| `hist_endpoint_correction_improvement.png` | `week03/results/drift_length_diagnostics/plots/decomp_resnet18/test/` | Distribution of endpoint improvement from deterministic drift correction. | Ch. 7.4 |
| `hist_integrated_ade_original_scaled_endpoint.png` | Same directory | Integrated ADE for original versus endpoint-scaled correction. | Ch. 7.4 |
| `hist_original_trajectory_length_ratio.png` | Same directory | Predicted/ground-truth trajectory-length ratio distribution. | Ch. 7.4 |
| `dx_distribution.png` | `week05/data/delta_displacement_paired_regularized_bounce224/normalization_plots/` | Delta-x distribution after bounce/truncate preparation. | Ch. 3.3/3.4 |
| `dy_distribution.png` | Same directory | Delta-y distribution after bounce/truncate preparation. | Ch. 3.3/3.4 |

## Repeated Week 06 visualization families

For each directory below, the five-file pattern is:

- `generated_contact_sheet_empirical_real_map.png`
- `generated_on_empirical_real_map.png`
- `generated_vs_real_on_empirical_real_map.png`
- `real_contact_sheet_empirical_real_map.png`
- `real_on_empirical_real_map.png`

Each shows, respectively, generated small multiples, generated overlay, generated-versus-real comparison, real small multiples, and real overlay on the empirical map.

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week06/results/raw_absolute_temporal_resnet_visualizations_empirical_support/*.png` | 5 | Unguided absolute temporal ResNet | Ch. 6.6 / appendix |
| `week06/results/raw_absolute_temporal_resnet_dilated_visualizations_empirical_support/*.png` | 5 | Unguided absolute dilated ResNet | Ch. 6.6 / appendix |
| `week06/results/raw_absolute_unet1d_visualizations_empirical_support/*.png` | 5 | Unguided absolute 1D U-Net | Ch. 6.6 main/appendix |
| `week06/results/raw_delta_temporal_resnet_visualizations_empirical_support/*.png` | 5 | Unguided delta temporal ResNet | Ch. 6.6 / appendix |
| `week06/results/raw_delta_temporal_resnet_dilated_visualizations_empirical_support/*.png` | 5 | Unguided delta dilated ResNet | Ch. 6.6 / appendix |
| `week06/results/raw_delta_unet1d_visualizations_empirical_support/*.png` | 5 | Unguided delta 1D U-Net | Ch. 6.6 main/appendix |
| `week06/results/raw_guided/*_visualizations_empirical_support/*.png` | 30 | Six raw models with support guidance strength 0.1 | Ch. 6.7 appendix |
| `week06/results/week05_sig15_mse_diag_step_040000_road_guidance_strength{0p1,1,10,1000}_visualizations_empirical_support/*.png` | 20 | Full-channel GASF road guidance over four strengths | Ch. 7.3/7.7 appendix |
| `week06/results/week05_sig15_mse_diag_step_040000_road_guidance_fixed_diagonal_blend0p1_clip37p5_strength{0p1,1,10}_visualizations_empirical_support/*.png` | 15 | Fixed-diagonal/clipped road guidance | Ch. 6.7 appendix |
| `week06/results/week05_sig15_mse_diag_step_040000_road_guidance_endpointcap1916_diagonal_blend0p25_clip37p5_strength1_visualizations_empirical_support/*.png` | 5 | Best endpoint-capped GASF guidance | Ch. 6.7 |
| `week06/results/trajectory_visualizations_empirical_real_map/*.png` | 5 | Initial GASF empirical-map visualization | Ch. 6.5 appendix |
| `week06/results/trajectory_visualizations_empirical_support_metric_real1000/*.png` | 5 | GASF evaluated against 1,000-real support map | Ch. 6.5 main |

Other Week 06 families:

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week06/results/trajectory_visualizations/*.png` | 5 | Unguided GASF raw-coordinate contact sheets/overlays | Ch. 6.5 |
| `week06/results/trajectory_visualizations_offset_dx220_dy-720/*.png` | 5 | GASF paths under selected map offset | Ch. 6.5/7.6 |
| `week06/results/trajectory_visualizations_accepted_p95_strict_offset_dx220_dy-720/*.png` | 6 | Real/generated/all overlays for strict accepted samples | Ch. 6.5/6.7 |
| `week06/results/accepted_generated_on_osm_offset_dx220_dy-720/*.png` | 3 | Accepted trajectories on OSM | Ch. 6.5 |
| `week06/results/week05_real_map_coverage/*.png` | 3 | Map mismatch diagnosis | Ch. 7.6 |
| `week06/data/maps_week05/week05_osm_coverage.png` | 1 | Coverage of the replacement Week 05 OSM graph | Ch. 3.6/7.6 |
| `week06/results/week05_map_offset_search*/offset_search_fine_heatmap.png` | 3 | Fine heatmaps at three search extents | Ch. 7.6 / appendix |
| `week06/results/week05_sig15_mse_diag_step_040000_rejection_filter_*/filter_metric_comparison.png` | 4 | Four rejection-policy comparisons | Ch. 6.7 / appendix |
| `week06/results/raw_diffusion_visual_comparison/raw_diffusion_best_median_worst_support.png` | 1 | Consolidated raw-model comparison | Ch. 6.6 main |

These rows account for all **141 Week 06 PNGs**.

## Week 05 figure families

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week05/results/week05_sig10_mse/**/*.png` | 400 | Periodic samples across the full sigmoid-1.0 MSE run | Appendix training progression; select a few only |
| `week05/results/week05_sig10_mse_diag/**/*.png` | 400 | Sigmoid-1.0 diagonal-loss samples | Appendix |
| `week05/results/week05_sig10_mse_sym/**/*.png` | 400 | Sigmoid-1.0 symmetry-loss samples | Appendix failure/ablation |
| `week05/results/week05_sig15_mse/**/*.png` | 400 | Sigmoid-1.5 MSE samples | Appendix |
| `week05/results/week05_sig15_mse_diag/**/*.png` | 400 | Sigmoid-1.5 diagonal-loss samples, including selected model progression | Ch. 6.2; select 40k/50k panels |
| `week05/results/week05_sig15_mse_sym/**/*.png` | 400 | Sigmoid-1.5 symmetry-loss samples | Appendix |
| `week05/results/week05_abspos_sig15_mse_diag_20k/**/*.png` | 160 | Periodic absolute-position GASF samples | Ch. 9.3/appendix; evaluation incomplete |
| `week05/data/delta_displacement_paired/normalization_plots/{dx,dy}_distribution.png` | 2 | Original variable delta distributions | Ch. 3.3 appendix |
| `week05/data/delta_displacement_paired_window224/normalization_plots/{dx,dy}_distribution.png` | 2 | Sliding-window delta distributions | Ch. 3.3 |
| `week05/data/delta_displacement_paired_regularized_bounce224/normalization_plots/{dx,dy}_distribution.png` | 2 | Bounce/truncate delta distributions | Ch. 3.3 main |

These rows account for all **2,566 Week 05 PNGs**.

## Week 04 figure families

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week04/data_constrained_gasf_dxdy_start/**/*.png` | 3,181 | `[0,1]` constrained R=GASF(dx), G=GASF(dy), B=start heatmap dataset images plus sanity plots | Ch. 4.2.2; choose representative/sanity only |
| `week04/data_constrained_gasf_dxdy_start_neg11/**/*.png` | 3,161 | `[-1,1]` constrained-GASF dataset images plus diagnostics | Ch. 4.2.3 appendix |
| `week04/runs_light/**/*.png` | 112 | Periodic samples from Week 04 DDPM variants | Ch. 5.2.2/appendix |
| `week04/resampled_samples/minmax01_mse_checkpoint40000/ddpm_100_steps/step40000_ddpm100_sample{000..015}.png` | 16 | Independent samples from 40k `[0,1]` MSE checkpoint using 100 reverse steps | Appendix sampler examples |
| `week04/resampled_samples/minmax01_mse_checkpoint40000/ddpm_150_steps/step40000_ddpm150_sample{000..015}.png` | 16 | Same checkpoint using 150 reverse steps | Appendix sampler examples |

No PNGs are stored in the Week 04 debug/smoke directories; those attempts are documented by logs/checkpoints. These rows account for all **6,486 Week 04 PNGs**.

## Week 03 figure families

### Summary and model-diagnostic families

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week03/results/channel_structure_stats/visualizations/sample_*_structure_components.png` | 8 | Channel projection/residual/error panels | Ch. 4.2.1; choose one or two |
| `week03/results/channel_structure_stats_smoke/visualizations/*.png` | 4 | Smoke equivalents | Do not use in main text |
| `week03/results/map_alignment_diagnostics/vehicle_*_alignment.png` | 12 | Vehicle-level raw/affine/OSM alignment | Ch. 3.6/6.4; choose one |
| `week03/results/drift_length_diagnostics/plots/**/*.png` | 30 | 24 per-example drift panels and six histograms across hard/decomp models | Ch. 7.4; prefer histograms |
| `week03/results/baseline_raw_resnet18_bs16/**/*.png` | 44 | Loss curve, three histograms, and per-sample reconstruction plots | Ch. 6.3 appendix |
| `week03/results/experiment_b_hard_resnet18_bs16/**/*.png` | 44 | Hard-projection decoder diagnostics | Ch. 6.3 appendix |
| `week03/results/experiment_c_decomp_resnet18_bs16/**/*.png` | 44 | Decomposed decoder diagnostics | Ch. 6.3 appendix |
| `week03/results/experiment_d_decomp_mtf_local_resnet18_bs16/**/*.png` | 44 | Best structure-decoder diagnostics | Ch. 6.3; select examples |
| `week03/results/experiment_e1_drift_aware_raw_resnet18_bs16/**/*.png` | 10 | Component/total losses and raw/corrected error histograms | Ch. 6.3/7.4 |
| `week03/results/experiment_e1b_drift_aware_corrtarget_raw_resnet18_bs16/**/*.png` | 10 | Correction-target version diagnostics | Appendix |
| `week03/results/experiment_e2_drift_aware_decomp_mtf_local_resnet18_bs16/**/*.png` | 10 | Best E2 correction diagnostics | Ch. 6.3 |
| `week03/results/experiment_e2_drift_aware_decomp_mtf_local_resnet18_bs16_corr02_mag1e5/**/*.png` | 10 | Strong E2 diagnostics | Ch. 7.7 appendix |
| `week03/results/sequence_latent/**/*.png` | 22 | Sequence-autoencoder and image-to-latent losses/evaluations | Ch. 6.3 appendix |
| `week03/results/distance_field_debug/**/*.png` | 40 | Distance-field construction/debug views | Ch. 2.7 appendix |

### Map-aware decoder families

Each full 118-image directory contains loss/histogram summaries plus many per-sample reconstruction plots for its paired oracle/start-centered configurations.

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week03/results/map_aware_decoder/**/*.png` | 118 | Partial-raster early-fusion oracle/start models | Appendix |
| `week03/results/map_aware_decoder_full/**/*.png` | 118 | Full-raster early-fusion oracle/start models | Ch. 6.4 appendix |
| `week03/results/map_aware_decoder_late_full/**/*.png` | 118 | Full late-fusion oracle/start models | Ch. 6.4; select both crop modes |
| `week03/results/map_aware_decoder_late_smoke/**/*.png` | 59 | Late-fusion smoke outputs | Do not use in main results |

### Refinement families

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `week03/results/refinement_sweep_oracle/<108-configs>/**/*.png` | 8,640 | For each parameter setting: best cases, raw/refined/GT comparisons, optional map backgrounds, and debug panels | Ch. 6.4 appendix; never include all |
| `week03/results/baseline_raw_resnet18_bs16_map_refinement_oracle/**/*.png` | 50 | Baseline full refinement comparison set | Appendix |
| `week03/results/baseline_raw_resnet18_bs16_map_refinement_oracle_best_r25_wr5_wref0p02_wc0p05/**/*.png` | 50 | Selected full refinement best-10 panels in multiple comparison layouts | Ch. 6.4; choose 1–3 |
| `week03/results/smoke_refinement_sweep_oracle/**/*.png` | 8 | Refinement sweep smoke | Exclude from main text |
| `week03/results/smoke_baseline_raw_resnet18_bs16_map_refinement_oracle/**/*.png` | 5 | Baseline refinement smoke | Exclude |
| `week03/results/smoke_baseline_raw_resnet18_bs16_map_refinement_oracle_curvature/**/*.png` | 5 | Curvature smoke | Exclude |
| `week03/results/smoke_refinement_terminal/**/*.png` | 4 | Terminal smoke | Exclude |

### Remaining smoke families

| Location/pattern | Count | Thesis use |
|---|---:|---|
| `week03/results/smoke_train_hard_resnet18/**/*.png` | 44 | Engineering appendix only |
| `week03/results/smoke_train_decomp_resnet18/**/*.png` | 44 | Engineering appendix only |
| `week03/results/smoke_road_loss_oracle_raw_resnet18/**/*.png` | 44 | Engineering appendix only |
| `week03/results/smoke_train_e1_drift_aware_raw_resnet18/**/*.png` | 10 | Engineering appendix only |
| `week03/results/smoke_train_e1b_drift_aware_corrtarget/**/*.png` | 10 | Engineering appendix only |
| `week03/results/smoke_train_e2_drift_aware_decomp_mtf/**/*.png` | 10 | Engineering appendix only |

Together the Week 03 rows account for all **9,669 PNGs**.

## Root and `results_hpc` figure families

### Legacy pixel DDPM

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `results_hpc/ddpm_scratch_50000step_bs8_pureMSE/**/*.png` | 320 | Periodic 50k PureMSE samples plus resampled checkpoints | Ch. 6.2; select final samples |
| `results_hpc/ddpm_scratch_50000step_bs8_dataDrivenDiag_precomputed/**/*.png` | 320 | Matched diagonal-loss samples | Ch. 6.2 |
| `results_hpc/ddpm_scratch_50000step_bs8_channelNorm_pureMSE/**/*.png` | 240 | ChannelNorm PureMSE samples | Appendix ablation |
| `results_hpc/ddpm_scratch_50000step_bs8_channelNorm_dataDrivenDiag_precomputed/**/*.png` | 240 | ChannelNorm diagonal samples | Appendix ablation |
| `results_hpc/ddpm_scratch_50000step_bs8_mtfSym_pureMSE/**/*.png` | 160 | MTF-symmetry PureMSE samples | Appendix ablation |
| `results_hpc/ddpm_scratch_50000step_bs8_mtfSym_dataDrivenDiag_precomputed/**/*.png` | 160 | MTF-symmetry diagonal samples | Appendix ablation |
| `results_hpc/ddpm_scratch_10000step_bs8_pureMSE/**/*.png` | 20 | Early 10k PureMSE samples | Historical appendix |
| `results_hpc/ddpm_scratch_10000step_bs8_dataDrivenDiag_precomputed/**/*.png` | 40 | Early 10k diagonal samples | Historical appendix |
| `results_hpc/ddpm_scratch_10step/**/*.png` | 4 | Ten-step smoke samples | Exclude from scientific results |
| `results_hpc/channel_norm_smoke/**/*.png` | 8 | ChannelNorm/no-ChannelNorm smoke | Exclude |
| `results_hpc/channel_norm_diag_smoke/**/*.png` | 4 | Diagonal ChannelNorm smoke | Exclude |

### Image diagnostic/evaluation directories

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `results_hpc/ddim_150step_ablation_analysis/**/*.png` | 168 | DDIM samples, calibrated samples, channel heatmaps | Ch. 6.2 appendix |
| `results_hpc/channel_diagnostics/**/*.png` | 164 | Raw/calibrated samples, real/generated correlation heatmaps, comparison grid | Ch. 6.2 |
| `results_hpc/channelNorm_channel_diagnostics_50/**/*.png` | 163 | ChannelNorm diagnostics at 50 reverse steps | Appendix |
| `results_hpc/channelNorm_channel_diagnostics_150/**/*.png` | 163 | ChannelNorm diagnostics at 150 steps | Appendix |
| `results_hpc/checkpoint50000_channel_diagnostics_150/*.png` | 7 | Correlation heatmaps for six variants and real | Ch. 6.2; select composite candidates |
| `results_hpc/mtfSym_channel_diagnostics_150/*.png` | 7 | MTF-symmetry diagnostic heatmaps | Appendix |
| Seven `*sampling*analysis/representative_comparison_grid.png` singleton directories | 7 | Model/sampler comparison grids | Ch. 6.2 and appendix |

The seven singleton directories are `ddpm_50k_comparison_analysis`, `ddpm_50k_sampling_steps_150_analysis`, `ddpm_50k_sampling_steps_50_matched_analysis`, `checkpoint50000_sampling_steps_150_analysis`, `channelNorm_50k_sampling_steps_50_analysis`, `channelNorm_50k_sampling_steps_150_analysis`, and `mtfSym_50k_sampling_steps_150_analysis`.

### Latent and decoder figures

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `results_hpc/latent_ddpm_delta_224_v2/**/*.png` | 84 | 14×14 latent DDPM training/best/final samples | Ch. 5.2.3 appendix |
| `results_hpc/latent_ddpm_delta_224_latent28/**/*.png` | 84 | 28×28 latent DDPM samples | Ch. 5.2.3 appendix |
| `results_hpc/latent_ddpm_delta_224_bad_baseline_20260516_184723/**/*.png` | 42 | Known bad latent baseline | Ch. 7.7 appendix only |
| `results_hpc/gaf_autoencoder_delta_224/**/*.png` | 5 | Autoencoder reconstructions/training views | Ch. 5.2.3 |
| `results_hpc/gaf_autoencoder_delta_224_latent28/**/*.png` | 5 | 28×28 autoencoder figures | Ch. 5.2.3 |
| `results_hpc/delta_displacement_resnet18_decoder_ablation/**/*.png` | 84 | Absolute/delta decoder error plots and examples | Ch. 6.3 appendix |
| `results_hpc/hybrid_multitask_delta_decoder/**/*.png` | 81 | V1 evaluation examples/histograms | Ch. 6.3 appendix |
| `results_hpc/hybrid_multitask_delta_decoder_v2/**/*.png` | 47 | V2 reconstruction/error figures | Ch. 6.3; select examples |
| `results_hpc/hybrid_multitask_delta_decoder_v3_geometry_mild/**/*.png` | 47 | Mild geometry figures | Appendix |
| `results_hpc/hybrid_multitask_delta_decoder_v3_geometry_strong/**/*.png` | 47 | Strong geometry figures | Appendix |
| `results_hpc/delta_centroid_correction_decoder_{smoke,lc005,lc010,lc020}/**/*.png` | 324 | Four 81-image correction families | Ch. 7.4 appendix; smoke separate from full |
| `results_hpc/decoder_{simplecnn,midfusion,multibranch,channel}_*/**/*.png` | 132 | Eleven 12-image decoder-ablation families | Ch. 6.3 appendix; use best mid-fusion examples |
| `results/decoder_multibranch_delta/comparison_plots/val_sample_*.png` | 12 | Early validation trajectory reconstruction comparisons | Historical appendix |

These root/`results_hpc` rows account for **3,189 PNGs**: 3,177 under `results_hpc` plus 12 under `results`.

## `Thesis_eval` sample families

| Location/pattern | Count | What it shows | Thesis placement |
|---|---:|---|---|
| `Thesis_eval/pureMSE_checkpoint_final_samples_50/sample_{000000..000015}.png` | 16 | Final PureMSE, 50 reverse steps | Appendix |
| `Thesis_eval/pureMSE_checkpoint_final_samples_100/sample_{000000..000015}.png` | 16 | Final PureMSE, 100 reverse steps | Appendix |
| `Thesis_eval/dataDrivenDiag_checkpoint_final_samples_50/sample_{000000..000015}.png` | 16 | Final diagonal model, 50 steps | Appendix |
| `Thesis_eval/dataDrivenDiag_checkpoint_final_samples_100/sample_{000000..000015}.png` | 16 | Final diagonal model, 100 steps | Appendix |

No additional metric file accompanies these 64 images, so they should not be treated as a separate quantitative experiment.

## Figures that should be generated before thesis freeze

The repository has many qualitative images but lacks a few high-value aggregate figures:

1. A single Week 05 FID-versus-checkpoint line plot for all six models, with the selected 40k checkpoint marked.
2. Real versus generated histograms/violin plots for displacement, bbox width/height, path length, max step, and support for best GASF, best raw absolute, best raw delta, and real data.
3. A trajectory-space precision/recall or MMD plot with bootstrap intervals.
4. A real-versus-generated GASF diagonal-bound and decoded-range histogram.
5. An explicit raw-versus-GASF pipeline diagram with dimensionality (`2T` versus `3T²`) and error injection points.
6. A `[0,1]` versus `[-1,1]` diagonal-inverse diagram.
7. A compact failure timeline showing legacy inverse, structure-loss NaN, map-frame correction, and guidance outcomes.

These are recommendations only; no new figures were created during this documentation task.

## Historical figures referenced by reports but absent now

These entries satisfy the archaeology requirement without pretending the files remain available.

| Reported filename/pattern | Reported location | What it reportedly showed | Thesis treatment |
|---|---|---|---|
| DDPM samples and `generated_100/*.png` | `results/ddpm_sanity_128/` | 128×128 refactoring sanity samples at 1k–3k | Recreate only if checkpoint/data can be recovered; otherwise table only |
| Generated/sample figures | `results/ddpm_delta_displacement_224_sanity/` | Notebook-auxiliary 3k delta-GAF samples | Historical failure table; artifact absent |
| Generated/sample figures | `results/ddpm_delta_displacement_224_sanity_mse_only/` | MSE-only 3k samples, reportedly more balanced RGB | Historical failure table; artifact absent |
| Generated/sample figures | `results/ddpm_delta_displacement_224_data_driven_diag_3k/` | Data-driven diagonal 3k samples | Historical ablation table; artifact absent |
| Qualitative grids/error plots | `experiments/decoder_no_speed_original_config/` | Original-style CNN-FC reconstructions and error analysis | Do not cite a nonexistent file; use recovered numeric table |
| Qualitative grids/comparisons | `experiments/decoder_no_speed_relative_target/` | Relative-target/oracle-start comparisons | Report-only |
| `test_ADE_distribution_comparison.png` | `experiments/decoder_no_speed_resnet18_ablation/comparison_with_original/` | CNN-FC versus ResNet18 ADE distribution | Recreate from data only if source artifacts are recovered |
| `test_FDE_distribution_comparison.png` | Same reported directory | CNN-FC versus ResNet18 FDE distribution | Same |
| `best_median_worst_side_by_side_grid.png` | Same reported directory | Qualitative CNN-FC/ResNet18 comparison | Same |
| Alignment diagnostics and comparison plots | `experiments/decoder_no_speed_delta_displacement_resnet18_ablation/` | Absolute versus delta representation | Numeric results can be used; original plots absent |

The historical report also describes notebook-era map-channel, invalid-point-penalty, binary/BCE, false-positive, and LSTM-correction variants, but provides neither surviving filenames nor quantitative results. They belong in the textual failure archaeology, not the thesis figure list.

# Quantitative results summary

## Reading this document

All values below were recovered from retained logs, JSON/CSV summaries, Markdown audits, and generated-array evaluations. No new model evaluation was run. Unless explicitly stated otherwise, trajectory distances are in the workspace's planar coordinate units (treated as metres by the OSM work). Results should only be compared within tables whose dataset, sample population, coordinate transform, and metric are matched.

Important comparability limits:

- Image FID is computed on GAF/GASF images using Inception features; it is not a trajectory-validity score.
- Legacy decoder results use 3,159 Hilbert/GAF examples. Week 05/06 generation uses a newer 12,061-trajectory source.
- Week 03 oracle-bbox map results use future trajectory extent and are upper bounds.
- The Week 03 full refinement report uses a different retained evaluation population/scale from the decoder test table.
- No uncertainty across training seeds is available.
- No retained quantitative LCSS map-matching result was found; “road” and “support” tables are nearest-distance evaluations.

## Dataset and preprocessing statistics

### Legacy dataset

| Quantity | Value | Evidence/source |
|---|---:|---|
| Vehicle trajectories | 3,159 | Root paired-data and Week 03/04 reports |
| Fixed sequence length | 224 | Dataset builders/configs |
| Total retained points | 707,616 | Week 03 affine fit (`3159 × 224`) |
| Train / validation / test | 2,527 / 315 / 317 | Root decoder split metadata |
| Legacy MTF bins | 8 | `src/cnr_trajectory/encoding/gaf.py` |
| Hilbert recursion tolerance | 0.0037 | Root config/README |

### Week 05/06 dataset

| Quantity | Value | Evidence/source |
|---|---:|---|
| CSV rows | 2,189,591 | Week 05 preparation log |
| Vehicle IDs | 12,061 | Week 05 preparation log |
| Original length min / max | 13 / 657 | Dataset report |
| Original length mean / median | 181.543 / 173 | Dataset report |
| Naturally length 224 | 40 | Dataset report |
| Split train / validation / test | 9,648 / 1,206 / 1,207 | Week 05 metadata |
| Sliding windows, length 224, stride 112 | 3,877 | Windowing report |
| Short vehicles skipped by windowing | 8,738 (72.45%) | Windowing report |
| Bounce-extended / truncated / unchanged | 8,738 / 3,283 / 40 | Bounce report |
| Variable-data maximum reconstruction error | `3.18e-12` | Pairing check |
| Bounce-data maximum reconstruction error | `3.41e-12` | Pairing check |
| Absolute x mean / std | 1626.710 / 641.207 | Absolute normalization artifact |
| Absolute y mean / std | 2951.006 / 419.397 | Absolute normalization artifact |

## Representation and reconstruction tests

### Constrained `[0,1]` GASF alignment

| Alignment mode | Mean ADE | Interpretation |
|---|---:|---|
| A | `3.37e-17` | Correct convention; numerical zero |
| B | 9.1718 | Off-by-one convention; invalid |
| C | `5.16e-14` | Correct equivalent convention; numerical zero |

The maximum float reconstruction error was approximately `1e-5`–`1e-4`. Quantized PNG reconstruction produced ADE from about 0.5 to above 10 and maximum errors around 23, depending on the tested sample. The exact per-sample values are retained in Week 04 sanity outputs and should be cited individually if a thesis figure uses those cases.

### `[-1,1]` diagonal sign ambiguity

| Axis | Diagonal magnitude MAE | Positive-sign reconstruction MAE | Fraction negative |
|---|---:|---:|---:|
| dx | `1.94e-9` | 0.1540 | 32.10% |
| dy | `1.55e-9` | 0.1326 | 30.75% |

Maximum signed error was approximately 2 in normalized units. Oracle sign recovery was exact. This demonstrates that the diagonal identifies magnitude but not sign under symmetric normalization.

### Week 04 delta distribution

| Statistic | Value |
|---|---:|
| Training dx observations | 566,048 |
| dx minimum / maximum | -34.42 / 34.39 |
| `[0,1]` normalized mean / std | 0.5114 / 0.1372 |
| Fraction within 0.05 of 0.5 | 53.18% |

## Legacy pixel DDPM image results

### Historical refactoring sanity baseline (report-only)

The named result directory is no longer present; values come from `docs/weekly_report_refactoring_validation.md`.

| 128×128 checkpoint | FID, 100 samples |
|---|---:|
| 1,000 | 288.81 |
| 2,000 | 254.43 |
| 3,000/final | 226.14 |

The 3,000-step run used batch 4, learning rate `2e-4`, EMA, took 884.49 s at 3.39 steps/s, and ended at loss 0.004508. It is an engineering sanity result.

### Historical 224×224 delta-GAF DDPM sanity matrix (report-only)

| Metric | Notebook auxiliary | MSE-only | Data-driven diagonal |
|---|---:|---:|---:|
| Final loss | 0.03244 | 0.0138568 | 0.01529 |
| FID (100 samples) | 197.456 | 137.419 | **134.957** |
| KID mean | 0.2276 | 0.1199 | **0.1167** |
| KID std | 0.00648 | Not stated in summary | 0.0051 |
| Channel mean absolute difference | 0.0990 | **0.0602** | 0.0621 |
| Image diversity ratio | 0.7000 | **0.7661** | 0.7558 |
| Pixel variance ratio | 0.8017 | 0.5394 | 0.5389 |
| Generated symmetry error | **0.1593** | 0.2053 | 0.1971 |
| B diagonal mean (real 0.8871) | 0.5042 | 0.4688 | **0.7379** |
| Generated mean delta magnitude (real 9.17) | 6.438 | 7.320 | **7.498** |
| Generated mean path length (real 2054.49) | 1442.13 | 1639.67 | **1679.43** |
| Out-of-range point ratio | 0.3846 | 0.3204 | **0.3166** |
| Out-of-range sample ratio | 0.69 | **0.57** | 0.61 |
| Jump ratio above real p95 | 0.000179 | 0.0000897 | **0.0** |
| Path-length variance ratio | 0.1223 | 0.1397 | **0.1460** |

All were 3,000-step, batch-2, accumulation-2, learning-rate-`1e-4`, EMA runs with 100 samples. The slight diagonal FID/KID advantage is not a clean overall win; MSE-only has better sample acceptance/diversity. The named result directories are absent now.

### Matched FID: model loss and reverse-step count

The primary matched audit used 80 generated images per method: 40 from checkpoint 25,000 and 40 from checkpoint 50,000. It is therefore a matched **checkpoint-aggregate**, not a checkpoint-50k-only estimate.

| Training loss | DDPM 50 steps | DDPM 150 steps | DDIM 150 steps |
|---|---:|---:|---:|
| PureMSE | 78.03 | **74.62** | 93.41 |
| DataDrivenDiag | 93.93 | **81.03** | 98.71 |

Conclusions: more ancestral DDPM steps helped both models; PureMSE beat diagonal supervision; DDIM underperformed the matched DDPM sampler.

### Sample-provenance sensitivity in legacy FID

| Evaluation set | Samples/model | PureMSE FID | DataDrivenDiag FID | Use |
|---|---:|---:|---:|---|
| Matched checkpoint 25k+50k, DDPM 150 | 80 | **74.6189** | **81.0313** | Primary matched aggregate |
| Checkpoint 50k only, DDPM 150 | 40 | 87.9959 | 99.0539 | Checkpoint-specific, high variance |
| Periodic `samples/` directory | 80 | 86.3372 | 84.5526 | Mixed/unmatched diagnostic; not primary |

This spread is why all legacy FID claims must name both sample provenance and reverse-step protocol.

### 150-step architectural/loss variants

| Variant | PureMSE FID | DataDrivenDiag FID | Main structural outcome |
|---|---:|---:|---|
| Raw channels | **74.62** | **81.03** | Lowest FID in this phase |
| ChannelNorm | 130.62 | 137.00 | Better mean matching, substantially worse FID |
| MTF symmetry | 84.98 | 93.96 | Better MTF symmetry, worse saturation/FID |

### Channel diagnostics

| Dataset/model | RGB channel mean | MTF symmetry error | Blue saturation where recorded |
|---|---|---:|---:|
| Real | `[-0.3830, 0.0004, -0.1905]` | 0.0673 | — |
| Raw PureMSE | `[0.0137, 0.0033, -0.0719]` | 0.2715 | — |
| Raw diagonal | `[0.1939, -0.0365, -0.2593]` | 0.2280 | — |
| MTF-sym PureMSE | — | 0.1207 | 0.4423 |
| MTF-sym diagonal | — | 0.1124 | 0.3866 |

Real channel standard deviations were GASF 0.4845, GADF 0.4361, and MTF 0.8526. The generated means show that visually plausible samples still differ substantially from the real channel distribution.

### Post-hoc channel affine calibration

| Model | Raw FID | Calibrated FID | Raw RGB mean | Calibrated RGB mean | Raw→calibrated global symmetry |
|---|---:|---:|---|---|---|
| PureMSE | 74.6189 | **63.2486** | `[0.0137,0.0033,-0.0719]` | `[-0.3641,0.0011,-0.1623]` | 0.2738→0.2940 |
| DataDrivenDiag | 81.0313 | **67.6198** | `[0.1939,-0.0365,-0.2593]` | `[-0.3631,0.0006,-0.2037]` | 0.2892→0.2907 |

The calibration is \(x'_c=((x_c-\mu^{gen}_c)/\sigma^{gen}_c)\sigma^{real}_c+\mu^{real}_c\). It demonstrates that channel marginal mismatch drives part of FID, but it uses real distribution moments and does not improve the learned generator. Calibrated diagonal means were `[-0.3633,0.0028,0.6077]` and `[-0.3859,0.0044,0.7701]`.

## Latent image model results

| Experiment | Quantitative result | Evaluation status |
|---|---:|---|
| 28×28 GAF autoencoder | MAE 0.039219; MSE 0.006421 | Reconstruction only |
| 14×14 latent DDPM | Best rolling loss 0.075972 | No FID/KID/trajectory metrics |
| 28×28 latent DDPM | Best rolling loss 0.061122 | No FID/KID/trajectory metrics |
| Marked bad latent baseline | No valid scientific score | Failed/debug artifact |

These training losses cannot support a claim that latent generation improved sample quality.

## Learned legacy-GAF inverse

### Historical absolute/relative decoder validation (report-only)

| Model/target | Test ADE | Test FDE | Other evidence |
|---|---:|---:|---|
| Naive train-mean trajectory | 640.89 | 730.67 | Non-learned baseline |
| Original-style CNN-FC absolute | 160.30 | 201.58 | Pred/GT test variance ratio 0.852 |
| Relative target + oracle start | 212.17 | 316.28 | Worse in aggregate; some worst cases improved |
| Scratch ResNet18 absolute | **120.9169** | **186.5896** | ADE p95 423.88 vs CNN-FC 538.38 |

For CNN-FC, test ADE correlations were 0.8556 with start error, 0.8121 with endpoint error, and 0.9825 with centroid error, but only 0.0149–0.0849 with length/tortuosity/bbox/sharp turns. These values are retained only in the Markdown report; the named `experiments/` tree is absent.

### Historical absolute versus delta ResNet18 representation

| Metric | Absolute x/y ResNet18 | Delta integrated from true start | Change/interpretation |
|---|---:|---:|---|
| Raw ADE | **120.92** | 135.29 | Absolute better |
| Raw FDE | **186.59** | 224.34 | Absolute better |
| Start-aligned ADE | 180.44 | **135.29** | Delta better |
| Centroid-aligned ADE | 86.60 | **72.64** | Delta better |
| Scale-normalized shape ADE | 80.90 | **55.15** | Delta better |
| ADE maximum | 1562.45 | **454.62** | Delta reduces catastrophic tail |
| FDE maximum | 1772.32 | **854.54** | Delta reduces catastrophic tail |

Delta-space ADE/FDE was 3.8768/6.4908, predicted/GT path-length ratio 0.9487, delta variance ratio 0.7516, predicted near-zero ratio 0 versus ground truth 0.2010. This shows the raw-position versus local-shape tradeoff.

### Decoder ablation: test integrated trajectory error

| Decoder/input/loss | Integrated ADE | Integrated FDE | Note |
|---|---:|---:|---|
| Mid-fusion RGB, β=0.05 | **139.43** | **217.99** | Best selected decoder ablation |
| Mid-fusion RGB, β=0.20 | 141.37 | 234.06 | Higher integrated weight hurt FDE |
| Mid-fusion normalized, β=0.05 | 145.58 | 237.31 | Normalization did not help |
| Simple CNN RGB, β=0.05 | 149.34 | 232.32 | Baseline |
| Simple CNN normalized, β=0.05 | 158.56 | 249.70 | Worse |
| Multibranch, β=0 | 208.27 | 356.35 | Delta ADE 6.59; severe cumulative drift |
| Multibranch, β=0.05 | 162.92 | 257.63 | Integrated term helps |
| Multibranch, β=0.20 | 164.93 | 272.77 | Too much weight hurts |
| Multibranch, β=0.20, second local run | **153.43** | **249.62** | Retained separately in `results/`; different run outcome |
| R/GASF channel only | 205.26 | 335.30 | Weak alone |
| G/GADF channel only | 183.81 | 296.22 | Strongest single channel |
| B/MTF channel only | 397.77 | 720.23 | Failed modality |
| Absolute-coordinate ResNet18 ablation | 135.29 | 224.34 | Adjacent root experiment |

The common training setting was 100 epochs, batch 16, learning rate 0.001, and split 2,527/315/317.

### Hybrid and geometry-aware decoder results

| Model | Delta ADE | Integrated ADE | Integrated FDE | Oracle-start ADE | Conclusion |
|---|---:|---:|---:|---:|---|
| Hybrid V1 | 8.841 | 125.01 | 170.00 | 524.72 | Learned heads compensate errors; oracle start exposes mismatch |
| Hybrid V2 | 5.377 | **107.85** | **147.20** | 158.46 | Best root-era hybrid |
| Hybrid V3 mild geometry | 5.449 | 124.95 | 164.06 | 173.09 | Geometry terms did not improve V2 |
| Hybrid V3 strong geometry | 5.351 | 132.31 | 184.92 | 185.00 | Stronger geometry worse |

### Learned centroid correction

| Correction loss weight | Corrected test ADE | Ground-truth-correction ADE (approx.) | Conclusion |
|---|---:|---:|---|
| 0.05 | 226.84 | 99–105 | Learned correction poor |
| 0.10 | 215.63 | 99–105 | Slight improvement |
| 0.20 | 209.84 | 99–105 | Best learned correction, still weak |

### Oracle translation/similarity upper bounds

| Prediction source | Raw ADE/FDE | Centroid-translation ADE/FDE | Similarity ADE/FDE | Mean rotation | Mean scale | Mean translation norm |
|---|---|---|---|---:|---:|---:|
| Delta ResNet18 retained prediction path | 173.41 / 297.28 | 97.48 / 169.45 | **55.47 / 94.46** | 6.47° | 1.114 | 376.92 |
| Hybrid V1 | 125.01 / 170.00 | 73.77 / 120.72 | **57.01 / 92.20** | 3.18° | 1.062 | 187.12 |

Similarity alignment improved ADE over raw for 317/317 delta samples and 316/317 Hybrid samples. It is an oracle upper bound, not an inference method; its delta raw numbers come from its own retained prediction path and differ from the primary 135.29/224.34 summary.

### Week 03 structure-aware ResNet18 decoders

| Input variant | Best epoch | Delta ADE | Delta FDE | Integrated ADE | Integrated FDE |
|---|---:|---:|---:|---:|---:|
| Raw RGB | 711 | 4.289 | 6.981 | 151.56 | 255.95 |
| Hard symmetry/antisymmetry projection | 599 | 4.156 | 6.963 | 143.85 | 247.90 |
| Structural decomposition | 798 | **3.945** | **6.492** | 140.16 | 235.19 |
| Decomposition + local MTF | 611 | 4.012 | 6.789 | **138.10** | **230.69** |

Measured source-channel structure:

| Property | Mean error |
|---|---:|
| GASF symmetry | 0 |
| GADF antisymmetry | 0.000884 |
| MTF symmetry | 0.0336 |

### Drift-aware correction

| Model | Raw ADE/FDE | Corrected ADE/FDE | Correction norm ratio | Result |
|---|---|---|---:|---|
| E1 raw drift head | 154.17 / 268.30 | 150.90 / 262.78 | — | ~2% gain; train degradation |
| E1b correction target | 146.87 / 249.56 | 142.23 / 241.82 | 0.1465 | ~3% gain |
| E2 decomp+MTF, weight 0.1 | 135.37 / 230.69 | **131.15 / 224.09** | 0.1687 | Best Week 03 correction |
| E2 strong, weight 0.2 | 140.26 / 234.44 | 137.20 / 229.19 | — | Worse than base E2 |

### Sequence latent alignment

| Stage/model | Integrated ADE | Integrated FDE | Conclusion |
|---|---:|---:|---|
| Trajectory AE, latent 64 | **2.341** | **9.342** | Sequence autoencoding is accurate |
| Trajectory AE, latent 128 | 2.517 | 11.960 | No benefit from larger latent |
| Image→latent rowwise | 239.080 | 431.800 | Failed |
| Image→latent temporal convolution | 178.650 | 319.650 | Better but weak |
| Image→latent temporal conv, kernel 7/depth 3 | **137.795** | **239.577** | Best alignment variant |
| Image→latent Transformer | 146.600 | 254.415 | Competitive but worse |
| MTF-only Transformer | 542.484 | 988.853 | Failed modality |

## OSM, map alignment, and refinement

### Week 03 map artifacts

| Quantity | Value |
|---|---:|
| Affine-fit points / trajectories | 707,616 / 3,159 |
| Approximate XY→UTM translation | `[+506600, +4150650]` |
| Maximum affine residual | `<1e-6` |
| Initial OSM graph nodes / edges | 1,465 / 2,672 |
| CRS | EPSG:32633 |
| Local rasters per crop mode | 3,159 |

### Map-aware decoder comparison

| Map fusion/crop | Integrated ADE | Integrated FDE | Valid at inference? |
|---|---:|---:|---|
| Early concat, partial oracle bbox | 159.86 | 266.72 | No: future bbox |
| Early concat, partial start centered | 168.20 | 293.44 | Yes |
| Early concat, full oracle bbox | 155.69 | 258.72 | No: future bbox |
| Early concat, full start centered | 168.20 | 293.44 | Yes |
| Late fusion, full oracle bbox | **123.25** | **185.39** | No: future bbox |
| Late fusion, full start centered | **141.27** | **250.29** | Yes |

For the late-fusion oracle-bbox model, mean nearest-road distance was 12.55 m for predictions versus 1.82 m for ground truth. Fraction farther than 10 m was 0.462 versus 0.00112.

### Oracle-map refinement

The 108-setting sweep crossed radius `{10,15,25}`, road weight `{0.5,1,2,5}`, reference weight `{0.02,0.05,0.1}`, and curvature `{0.05,0.1,0.2}` on 200 samples.

| Result | Raw | Refined | Change |
|---|---:|---:|---:|
| Full-application ADE | 68.230 | 66.424 | -1.806 |
| Full-application FDE | 121.55 | 119.40 | -2.15 |
| Mean road distance | 30.25 | 15.59 | -14.66 |
| Fraction farther than 10 m | 0.4566 | 0.1706 | -0.2860 |

Selected configuration: radius 25, road weight 5, reference weight 0.02, curvature weight 0.05. In the 200-sample sweep it reduced the off-road-over-10 fraction by 0.290 and changed ADE by `-2.49`. The table is an oracle-crop result and is not directly comparable to the decoder table.

## Week 04 constrained-GASF DDPM training

| Normalization | Loss variant | Last/observed status | Last loss | Valid checkpoint |
|---|---|---|---:|---|
| `[0,1]` min-max | MSE | Stopped around 40,610 | 0.000578 | Yes, 40k |
| `[0,1]` min-max | Symmetry | Stopped around 40.6k | 0.000609 | Yes, 40k |
| `[0,1]` min-max | GASF structure | NaN, first recorded near 490 | NaN | No scientific checkpoint |
| `[-1,1]` | MSE | Stopped around 40.6k | 0.000615 | Yes, 40k |
| `[-1,1]` | Symmetry | Stopped around 40.6k | 0.000640 | Yes, 40k |
| `[-1,1]` | GASF structure | NaN, first recorded near 620 | NaN | No scientific checkpoint |
| Float `[0,1]` | MSE | Scheduler time limit at 33,390 | Partial log | Partial only |
| Safe structure debug | Protected structure | Stopped at 170/180 | Finite so far | Inconclusive |

Common target setting: 50k steps, batch 16, accumulation 2 (effective 32), learning rate `2e-4`, 1,000 diffusion steps, EMA, checkpoints/samples every 2k. No FID or decoded trajectory metric was retained for these runs.

## Week 05 larger-data GASF diffusion

### FID over checkpoints (1,000 samples, 50 reverse steps)

| Model | 40k | 42k | 44k | 46k | 48k | 50k | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sigmoid 1.0, MSE | 78.7878 | 79.7034 | 76.4782 | 73.1179 | 71.2523 | 69.6000 | 69.6000 @50k |
| Sigmoid 1.0, diagonal | **52.8173** | 55.2542 | 55.8920 | 54.0297 | 53.7218 | 58.3155 | 52.8173 @40k |
| Sigmoid 1.0, symmetry | 143.7027 | 141.7514 | 141.4227 | 139.8515 | 139.3601 | 137.4864 | 137.4864 @50k |
| Sigmoid 1.5, MSE | 77.9133 | 77.1691 | 73.7081 | 71.3687 | **67.9659** | 68.2091 | 67.9659 @48k |
| Sigmoid 1.5, diagonal | **44.5496** | 44.8652 | 47.6087 | 48.0252 | 48.8892 | 52.4143 | **44.5496 @40k** |
| Sigmoid 1.5, symmetry | 141.3163 | 138.4636 | 133.9564 | **129.1660** | 133.0479 | 132.7624 | 129.1660 @46k |

### Final 50k training-loss components

| Model | Total/final loss | Noise MSE | Auxiliary loss |
|---|---:|---:|---:|
| Sigmoid 1.0, MSE | 0.000807 | 0.000807 | — |
| Sigmoid 1.0, diagonal | 0.004644 | 0.001147 | Diagonal 0.069941 |
| Sigmoid 1.0, symmetry | 0.000937 | 0.000888 | Symmetry 0.000986 |
| Sigmoid 1.5, MSE | 0.000912 | 0.000912 | — |
| Sigmoid 1.5, diagonal | 0.006084 | 0.001494 | Diagonal 0.091788 |
| Sigmoid 1.5, symmetry | 0.001033 | 0.000993 | Symmetry 0.000807 |

The lowest training loss and lowest FID select different objectives/checkpoints. This is additional evidence that the auxiliary terms and FID do not measure the same property.

### Consistency guidance

| Guidance scale range | FID range | Mean consistency change | Conclusion |
|---|---:|---:|---|
| 0–0.5 | 44.548–44.551 | 0.03634191 → 0.03633951 | No material effect |

### Absolute-position GASF, 20k

| Total loss | Noise MSE | Diagonal loss | FID | Trajectory evaluation |
|---:|---:|---:|---|---|
| 0.00280264 | 0.00038791 | 0.0482946 | Missing | Missing |

## Week 06 decoded GASF trajectory results

### Unguided selected GASF model versus real data

| Metric | Real | Generated GASF | Direction of mismatch |
|---|---:|---:|---|
| Mean step length | 11.18 | 12.47 | Slightly high |
| Mean path length | 2492 | 2781 | High |
| Net displacement | 876.86 | 2350 | Severely high |
| Bounding-box width | 972.61 | 1665.61 | High |
| Bounding-box height | 681.24 | 1413.89 | High |
| Mean decoded out-of-range fraction | — | 0.3505 | Severe inverse/manifold violation |
| Empirical support mean | Near-support reference | 417.55 | Severe |
| Empirical support p95 | Near-support reference | 1059.47 | Severe |
| Fraction support distance >20 | — | 0.6869 | Severe |

### OSM alignment diagnosis

| Measurement | Before correction | After graph/offset correction |
|---|---:|---:|
| Real points covered by initial graph | 0.0395% | New graph covers dataset bbox |
| Initial graph | 1,465 nodes / 2,672 edges | 1,848 nodes / 3,580 edges |
| Northward coverage gap | 2,247 m | Removed by new graph |
| Real road mean | 25.51 m | 15.49 m |
| Real road p95 | 70.16 m | 41.15 m |
| Selected offset | — | dx +220, dy -720 |

After alignment, generated road mean/p95 was 220.28/644.65 m versus real 15.24/40.40 m.

### Rejection-filter outcomes

| Policy | Accepted | Acceptance rate | Accepted support mean/p95 where available |
|---|---:|---:|---|
| Strict OSM p95 | 128 / 1,000 | 12.8% | — |
| Strict empirical support | 92 / 1,000 | 9.2% | 16.16 / 41.56 |
| Loose p99 | 228 / 1,000 | 22.8% | — |

For the strict OSM filter, recorded failed-rule counts include road mean 763, road p95 740, and decoded out-of-range 701. Rules overlap, so counts do not sum to 1,000.

## Raw trajectory diffusion

### Training losses

| Representation | Architecture | Final loss | Minimum logged loss |
|---|---|---:|---:|
| Absolute | Temporal ResNet | 0.01545 | 0.01374 |
| Absolute | Dilated temporal ResNet | 0.01412 | 0.01250 |
| Absolute | 1D U-Net | **0.01126** | **0.01010** |
| Delta | Temporal ResNet | 0.05899 | 0.05180 |
| Delta | Dilated temporal ResNet | 0.05455 | 0.04801 |
| Delta | 1D U-Net | **0.03866** | **0.03347** |

Loss values across absolute and delta normalization spaces are not directly comparable; compare architectures within a representation.

### Physical/support metrics on 1,000 samples

| Representation/model | Acceptance | Support mean | Support p95 | Displacement | Bbox width | Bbox height | Max step |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real reference | — | — | — | **876.9** | **972.6** | **681.2** | **26.9** |
| Absolute ResNet | 92.7% | 10.4 | 31.8 | 562.2 | 615.8 | 451.5 | 24.4 |
| Absolute dilated ResNet | 95.9% | 10.0 | 30.7 | 592.1 | 645.4 | 462.3 | 23.9 |
| Absolute 1D U-Net | **98.5%** | **8.4** | **25.6** | 549.7 | 613.5 | 448.5 | 23.7 |
| Delta ResNet | 61.0% | 43.1 | 103.0 | 456.0 | 531.5 | 361.0 | 12.0 |
| Delta dilated ResNet | 59.2% | 42.3 | 101.5 | 422.1 | 513.9 | 364.0 | 11.9 |
| Delta 1D U-Net | 60.5% | 42.8 | 103.5 | 425.2 | 502.5 | 364.3 | 11.4 |

Absolute 1D U-Net is best on empirical validity, but all raw models understate displacement and extent. Therefore the current result is high **precision/validity**, not demonstrated distributional recall.

### Raw diffusion guidance at strength 0.1

| Model | Unguided support mean/p95 | Guided support mean/p95 | Effect |
|---|---|---|---|
| Absolute ResNet | 10.4 / 31.8 | 10.61 / 32.95 | Worse |
| Absolute dilated ResNet | 10.0 / 30.7 | 10.15 / 31.05 | Slightly worse |
| Absolute 1D U-Net | 8.4 / 25.6 | 8.319 / 25.406 | Negligible improvement |
| Delta family | ~42–43 / ~101–104 | ~44–46 / ~105–109 | Worse |

## GASF road-guidance studies

| Variant | Key setting | Support mean | Support p95 | Other result | Conclusion |
|---|---|---:|---:|---|---|
| Unguided selected GASF | — | 417.55 | 1059.47 | Off>20 0.6869 | Baseline poor |
| Full-channel guidance | strengths 0.1/1/10/1000 | ~551–600 | — | Diagonal p99 near 1; deltas up to 85 | Worse/unstable |
| Fixed diagonal | blend 0.1, clip 37.5, best strength 1 | 381.93 | 986.77 | Saturation removed | Small improvement, still poor |
| Endpoint capped | cap 1916, blend 0.25, clip 37.5, strength 1 | **270.25** | **706.73** | Step 10.85; displacement 1960.04; bbox 1391.63×1215.49; off>20 0.645 | Best guided GASF, still far from raw/real |

The distance-field unit smoke test reduced energy from 16 to 11.6476 and mean distance from 4 to 3.4129; four tests passed. Thus the guidance primitive works in isolation, but gradients through the GASF inverse produce harmful tradeoffs.

Recorded endpoint-capped fractions of generated samples whose mean support distance was below each threshold were:

| Threshold | Fraction below threshold |
|---|---:|
| 50 | 31.4% |
| 100 | 42.2% |
| 200 | 55.2% |
| 500 | 80.0% |

## Best results by narrowly defined criterion

| Criterion | Best retained result | Necessary caveat |
|---|---|---|
| Legacy uncalibrated image FID | PureMSE DDPM, 150 steps: 74.62 | 80 images; no physical inverse validation |
| Legacy calibrated image FID | PureMSE post-hoc calibration: 63.25 | Uses real moments; diagnostic, not generator improvement |
| Week 05 image FID | Sigmoid 1.5 + diagonal, 40k: 44.5496 | Natural-image feature space, not trajectory space |
| Learned legacy-image reconstruction | Hybrid V2: ADE/FDE 107.85/147.20 | Separate supervised inverse; surprising oracle-start behavior |
| Week 03 corrected reconstruction | E2: ADE/FDE 131.15/224.09 | Weaker than Hybrid V2; different architecture study |
| Deployable map-aware reconstruction | Late start-centered: 141.27/250.29 | Oracle version is better but invalid at inference |
| Oracle map refinement | Road mean 30.25→15.59; off>10 0.4566→0.1706 | Oracle future crop and different evaluation population |
| GASF decoded acceptance | 22.8% loose; 9.2% strict empirical | Low yield; filtering hides base failure |
| Raw trajectory validity | Absolute 1D U-Net: 98.5%, support 8.4/25.6 | Extent/displacement substantially too small |
| Guided GASF support | Endpoint-cap: 270.25/706.73 | Still orders of magnitude worse than raw absolute models |

## Metrics requested but absent or incomplete

| Requested quantity | Status |
|---|---|
| FID | Available for root pixel and Week 05 GASF experiments; absent for latent/Week 04/absolute-GASF branches |
| KID | Historical 3k delta-GAF values exist in a report; no retained primary artifacts or final cross-model KID table |
| Training loss | Available for principal DDPM/decoder/raw runs; not consistently normalized across phases |
| Training steps | Available for principal runs and failure logs |
| Acceptance rate | Available for Week 06 decoded GASF filters and raw diffusion |
| Map matching error | No retained LCSS result; nearest-road/support metrics only |
| Reconstruction error | Available for representation tests and learned decoders |
| Trajectory statistics | Available for Week 06 GASF/raw samples, but not as one fully matched precision/recall evaluation |
| Diversity/coverage | Some image/novelty code exists; no matched trajectory-space coverage/recall result across raw and GASF |
| Multi-seed confidence intervals | Missing |
| Held-out conditional likelihood | Missing |

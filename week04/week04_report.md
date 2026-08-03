# Week 4 Progress Summary – Constrained GASF Representation

## Background

At the end of Week 3, the project direction changed significantly.

Previous work focused on:

```text
Trajectory
→ Image Representation
→ DDPM
→ Neural Decoder
→ Trajectory
```

including:

* CNN decoder
* ResNet18 decoder
* Differential displacement representation
* Multi-branch decoder
* Transformer decoder
* Post-processing refinement

The original goal was to verify whether image-based trajectory representations preserve enough information to reconstruct trajectories.

During the latest meeting, a new observation emerged:

> GASF is mathematically invertible through its diagonal.

Therefore, the project focus shifted from:

```text
Learning-based decoding
```

to:

```text
Analytical constrained decoding
```

The proposed new pipeline becomes:

```text
Trajectory
→ dx, dy
→ GASF_dx, GASF_dy
→ DDPM
→ GASF diagonal
→ analytical inverse
→ trajectory
→ map matching
```

This removes the need for a learned decoder.

---

# Initial Design Decision

The first task was to redesign the dataset representation.

The previous dataset used:

```text
R = GASF(Hilbert)
G = GADF(Hilbert)
B = MTF(Hilbert)
```

The new constrained representation follows the meeting notes:

```text
R = GASF(dx)
G = GASF(dy)
B = Initial Point Heatmap
```

where:

```text
dx = x[t] - x[t-1]
dy = y[t] - y[t-1]
```

and:

```text
Initial Point = trajectory start location
```

The objective is to preserve:

```text
Relative motion:
dx, dy

Absolute location:
start point
```

simultaneously.

---

# Dataset Construction

A new dataset builder was implemented:

```text
build_constrained_gasf_dataset.py
```

Output dataset:

```text
week04/data_constrained_gasf_dxdy_start
```

Generated contents:

```text
images/
arrays/
labels_absolute/
labels_delta_displacement/
starts/
splits/
metadata.csv
README.md
constrained_gasf_normalization.json
sanity_checks/
```

Dataset size:

```text
3159 trajectories
```

For every sample:

```text
absolute trajectory
→ start point

delta displacement
→ dx sequence
→ dy sequence

dx → GASF_dx
dy → GASF_dy

stack:
[GASF_dx,
 GASF_dy,
 start_heatmap]
```

Image size:

```text
224 × 224 × 3
```

---

# First Analytical Inverse Validation

The first sanity check attempted:

```text
Encoded image
→ diagonal extraction
→ inverse GASF
→ dx, dy recovery
→ trajectory integration
```

Unexpected result:

```text
ADE ≈ 0.5 ~ 11+
```

Example:

```text
sample_000009
ADE ≈ 11.5
FDE ≈ 19.6
```

Initially suspected causes:

* timestep alignment mismatch
* delta definition mismatch
* integration offset

---

# Alignment Investigation

Three reconstruction modes were tested:

Mode A:

```text
xy[t]
=
xy[t-1]
+
delta[t]
```

Mode B:

```text
xy[t]
=
xy[t-1]
+
delta[t-1]
```

Mode C:

```text
xy
=
start
+
cumsum(delta)
```

Results:

```text
Mode B failed badly
ADE ≈ 8 ~ 14
```

while:

```text
Mode A ≈ Mode C
```

Conclusion:

```text
No timestep alignment problem.
```

The delta convention was consistent.

---

# Delta Consistency Verification

A dedicated script was created:

```text
check_delta_consistency.py
```

Goal:

Verify:

```text
stored_delta
==
abs_xy[t] - abs_xy[t-1]
```

Results:

```text
3159 samples checked

mean_mse = 0
global_max_error = 0

3159 / 3159 samples
matched within 1e-6
```

Conclusion:

```text
labels_delta_displacement
is exactly the trajectory difference.
```

The dataset labels are correct.

---

# Discovery: PNG Quantization Error

The next hypothesis was:

```text
GASF inverse is correct
but PNG storage destroys precision.
```

Dataset builder was upgraded:

New output:

```text
arrays/
```

containing:

```text
float32 representations
before PNG quantization
```

Sanity check was modified to compare:

```text
Float reconstruction
vs
PNG reconstruction
```

Results:

### Float Arrays

```text
max error ≈ 1e-5 ~ 1e-4
```

### PNG Images

```text
ADE ≈ 0.5 ~ 10+
```

Conclusion:

```text
Analytical inverse works.

The previous reconstruction errors
were caused by uint8 image quantization.
```

This is a major result.

---

# Current Status

The following statement has now been verified:

```text
Trajectory
→ dx, dy
→ GASF_dx, GASF_dy
→ diagonal inverse
→ trajectory
```

is effectively lossless under float precision.

Observed reconstruction accuracy:

```text
max error:
1e-5 ~ 1e-4
```

which is close to numerical precision.

---

# Distribution Analysis

After validating invertibility, the next concern became:

```text
Is the representation suitable for DDPM?
```

Current normalization:

```text
global min-max
→ [0,1]
```

Statistics:

```text
dx_min = -34.42
dx_max = 34.39

Total dx samples:
566048
```

Distribution:

```text
|dx_norm - 0.5| <= 0.01
33.25%

|dx_norm - 0.5| <= 0.02
40.03%

|dx_norm - 0.5| <= 0.05
53.18%
```

More than half the samples fall inside:

```text
[0.45, 0.55]
```

around the center.

---

# Interpretation

This revealed an important issue.

The representation is:

```text
Mathematically invertible
✓
```

but:

```text
Potentially difficult for DDPM
?
```

because:

```text
Most displacement values collapse
around 0.5
```

after global min-max normalization.

Consequences:

```text
Different motions
→ similar normalized values
→ similar GASF textures
```

which may encourage DDPM to learn:

```text
average-looking structures
```

rather than meaningful trajectory dynamics.

---

# Discussion: [0,1] vs [-1,1]

The meeting notes allow:

```text
Normalization to a fixed interval
such as [0,1] or [-1,1].
```

Current implementation:

```text
[0,1]
```

Advantages:

```text
Direct diagonal inverse

x
=
sqrt((diag+1)/2)
```

No sign ambiguity.

Strict analytical reconstruction remains possible.

Disadvantages:

```text
Strong concentration near 0.5
```

for the current trajectory distribution.

Potential alternative:

```text
[-1,1]
```

Advantages:

```text
Preserves directional structure
better.
```

Disadvantages:

```text
Diagonal inverse loses sign.

x
=
± sqrt((diag+1)/2)
```

Therefore:

```text
Strict analytical invertibility
is lost.
```

For this reason, the current plan is:

```text
Keep [0,1] as the main baseline.

Evaluate [-1,1]
only as a representation-distribution ablation.
```

---

# Current Conclusions

Verified:

```text
✓ Dataset construction completed.

✓ New constrained representation implemented.

✓ Analytical inverse validated.

✓ Delta labels verified.

✓ Quantization problem identified.

✓ Representation is effectively lossless
  in float precision.

✓ Current [0,1] normalization compresses
  many values near 0.5.
```

Not yet verified:

```text
✗ DDPM training on the new representation.

✗ FID behaviour.

✗ Generated GASF legality.

✗ Map-conditioned generation.
```

---

# Next Planned Experiment

## Experiment A

Representation Ablation

Compare:

```text
Version A:
global min-max → [0,1]

Version B:
global min-max → [-1,1]
```

Measure:

```text
distribution spread
histograms
visual diversity
GASF texture diversity
```

Goal:

Determine whether the current concentration around 0.5 is a fundamental limitation of the representation.

---

## Experiment B

Small DDPM Baseline

After finalizing the representation:

```text
GASF_dx
GASF_dy
start heatmap
```

train a small DDPM and evaluate:

```text
generated samples
trajectory recovery
consistency metrics
```

before moving to constrained losses or map conditioning.

---

# Main Takeaway

The most important Week 4 result so far is:

```text
The constrained GASF representation
is analytically invertible.

The reconstruction errors previously observed
were caused by image quantization,
not by information loss in the representation itself.
```
## Invertibility Analysis of the [-1,1] Normalization Variant

### Motivation

One concern raised during the Week 4 discussion was whether the proposed normalization range could affect the analytical invertibility of the GASF representation. Since the original pipeline uses a constrained normalization designed to preserve analytical decoding from the GASF diagonal, an ablation study was performed using the standard ([-1,1]) normalization commonly adopted in the GASF literature.

The goal was to determine whether trajectories normalized to ([-1,1]) remain strictly recoverable from the diagonal elements of the generated GASF image.

---

### Theoretical Analysis

For standard GASF, a normalized signal (x_i \in [-1,1]) is transformed as

[
G_{ij}=\cos(\phi_i+\phi_j),
\qquad
\phi_i=\arccos(x_i).
]

The diagonal entries satisfy

[
G_{ii}
======

# \cos(2\phi_i)

2x_i^2-1.
]

Therefore,

[
x_i^2
=====

\frac{G_{ii}+1}{2},
]

which leads to

[
|x_i|
=====

\sqrt{\frac{G_{ii}+1}{2}}.
]

This result shows that the diagonal uniquely determines the magnitude of the normalized value, but does not contain information about its sign.

Consequently,

[
x_i
===

\pm
\sqrt{\frac{G_{ii}+1}{2}},
]

and the sign ambiguity cannot be resolved from the diagonal alone.

---

### Experimental Verification

An invertibility analysis was performed on the newly generated dataset using the ([-1,1]) normalization.

#### Absolute-Value Reconstruction

Using

[
|x_i|
=====

\sqrt{\frac{G_{ii}+1}{2}},
]

the reconstruction errors were:

* Mean absolute diagonal reconstruction error (dx): (1.94\times10^{-9})
* Mean absolute diagonal reconstruction error (dy): (1.55\times10^{-9})

These values are essentially numerical precision errors, confirming that the diagonal accurately preserves the magnitude information.

#### Signed Reconstruction Using Positive Branch

Assuming

[
x_i=
+\sqrt{\frac{G_{ii}+1}{2}},
]

the reconstruction errors became:

* Mean signed reconstruction error (dx): 0.1540
* Mean signed reconstruction error (dy): 0.1326
* Maximum observed error: 2.0

This demonstrates that a substantial amount of sign information is lost.

#### Oracle Sign Recovery

When the true sign of each sample was provided externally (oracle experiment), the reconstruction errors returned to:

* Mean signed reconstruction error (dx): (1.94\times10^{-9})
* Mean signed reconstruction error (dy): (1.55\times10^{-9})

This confirms that the GASF representation itself is numerically correct and that the only missing information is the sign.

#### Sign Distribution

The dataset contains a significant proportion of negative values:

* Fraction of negative dx samples: 32.10%
* Fraction of negative dy samples: 30.75%

Therefore, sign ambiguity cannot be ignored in practice.

---

### Conclusion

The ([-1,1]) normalized GASF representation is not strictly invertible from the diagonal alone.

The diagonal permits exact recovery of

[
|x_i|,
]

but cannot uniquely determine the sign of (x_i). As a result, analytical decoding from the diagonal is no longer guaranteed.

Therefore:

* The ([-1,1]) version is suitable as a representation-distribution ablation study.
* It is not suitable for the main analytical decoding framework.
* The constrained normalization used in the primary pipeline remains the only strictly analytically invertible representation currently available.

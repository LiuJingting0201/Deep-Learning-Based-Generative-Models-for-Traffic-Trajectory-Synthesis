# Week 4 Plan: Constrained GASF Representation for Trajectory Generation

## 1. Background

In the previous weeks, the project mainly focused on learning-based decoding from image representations back to trajectories. Several decoder-based experiments were tested, including CNN, ResNet18, multi-branch structures, Transformer-based variants, and post-processing refinement. These experiments helped verify that image-based GAF representations preserve trajectory-related information, especially when using differential displacement rather than absolute coordinates.

However, based on the latest meeting and the new proposed direction, the focus should now shift from learned decoding to analytical constrained decoding. The key idea is that GASF is mathematically invertible through its diagonal. Therefore, instead of training a neural decoder to recover trajectories from generated images, the new pipeline should generate constrained GASF images and recover trajectories analytically from the generated GASF diagonals.

## 2. Main Goal

The goal of Week 4 is to construct a new dataset representation that follows the proposed constrained GASF pipeline.

Each trajectory should be encoded as a three-channel image:

```text
Channel 0: GASF of dx increments
Channel 1: GASF of dy increments
Channel 2: initial-point heatmap
```

This replaces the previous representation:

```text
Channel 0: GASF of Hilbert sequence
Channel 1: GADF of Hilbert sequence
Channel 2: MTF of Hilbert sequence
```

The new representation is more suitable for analytical trajectory recovery because the diagonal of each GASF channel can be used to reconstruct the original normalized dx and dy sequences.

## 3. Dataset Construction

The existing paired dataset already contains:

```text
labels_absolute/
labels_delta_displacement/
starts/
splits/
metadata.csv
encoding_normalization.json
```

The new dataset should be built from:

```text
absolute_xy = labels_absolute/sample.npy
delta_xy = labels_delta_displacement/sample.npy
```

For each sample:

```text
initial_point = absolute_xy[0]
dx_series = delta_xy[:, 0]
dy_series = delta_xy[:, 1]
```

Then:

```text
dx_series → global normalization → GASF_dx
dy_series → global normalization → GASF_dy
initial_point → start-point heatmap
```

The final image should have shape:

```text
[224, 224, 3]
```

or internally:

```text
[3, 224, 224]
```

depending on the training pipeline.

## 4. Normalization

Global normalization should be used, not per-trajectory normalization.

The normalization statistics should be computed from the training set only:

```text
dx_min, dx_max
dy_min, dy_max
x_min, x_max
y_min, y_max
```

They should be saved into:

```text
constrained_gasf_normalization.json
```

The normalization formula can be:

```text
dx_norm = (dx - dx_min) / (dx_max - dx_min)
dy_norm = (dy - dy_min) / (dy_max - dy_min)
```

The normalized values should be in `[0, 1]`, matching the existing pyts GASF setting.

## 5. GASF Encoding

The old code already contains the useful GASF tool:

```python
GramianAngularField(method="summation", sample_range=(0, 1))
```

But the input should be changed.

Old input:

```text
Hilbert sequence
```

New input:

```text
dx sequence
dy sequence
```

The new encoding should be:

```python
gasf_dx = gaf.fit_transform(dx_norm[None, :])[0]
gasf_dy = gaf.fit_transform(dy_norm[None, :])[0]
```

GADF and MTF should not be used in the new strict version.

## 6. Initial-Point Heatmap

The third channel should encode the initial position of the trajectory.

The initial point:

```text
x0, y0 = absolute_xy[0]
```

should be mapped to image coordinates using the global coordinate range:

```text
px = (x0 - x_min) / (x_max - x_min) * 223
py = (y0 - y_min) / (y_max - y_min) * 223
```

Then a Gaussian heatmap should be generated around `(px, py)`.

This heatmap preserves absolute position information, because dx and dy increments alone only describe relative motion.

## 7. Sanity Check

Before training any DDPM model, the new representation must be verified through analytical decoding.

For each generated real encoded sample:

```text
GASF_dx diagonal → recover dx_norm
GASF_dy diagonal → recover dy_norm
denormalize dx, dy
integrate from initial point
compare reconstructed trajectory with original absolute trajectory
```

The reconstruction error should be close to zero. If not, the likely problems are:

```text
delta sequence length mismatch
delta definition mismatch
wrong normalization range
wrong initial displacement convention
coordinate order mismatch
```

## 8. Week 4 Experimental Plan

### Step 1: Create the new dataset

Write a new script:

```text
scripts/build_constrained_gasf_dataset.py
```

Output directory:

```text
data_constrained_gasf_dxdy_start/
  images/
  labels_absolute/
  labels_delta_displacement/
  starts/
  splits/
  metadata.csv
  constrained_gasf_normalization.json
  sanity_checks/
```

### Step 2: Run analytical decoding sanity check

Create a script or integrated function to verify:

```text
real constrained GASF image
→ diagonal extraction
→ dx/dy reconstruction
→ trajectory integration
→ comparison with ground truth
```

Save:

```text
sanity_check_metrics.csv
trajectory comparison plots
```

### Step 3: Train a small DDPM baseline

Once the dataset is verified, train a simple DDPM on the new three-channel representation.

Initial configuration:

```text
image_size = 224
channels = 3
batch_size = 8 if possible
max_train_steps = 10000 or smaller for quick test
aux_loss_mode = none for first baseline
```

### Step 4: Evaluate generated image legality

For generated samples, compute:

```text
diagonal valid ratio
symmetry error
GASF consistency error
decoded trajectory success rate
trajectory length distribution
```

The purpose is to check whether generated images are not only visually realistic, but also mathematically compatible with valid GASF structure.

## 9. Expected Contribution

This week’s work changes the project from a learned-decoder pipeline to a constrained analytical-decoding pipeline.

The new direction can be summarized as:

```text
Generate image representations, but recover trajectories through mathematical constraints.
```

This is more coherent than treating GAF images as ordinary RGB images, because GASF matrices are not arbitrary images. They must lie on a constrained manifold determined by an underlying time series.

The final goal is:

```text
trajectory dataset
→ dx/dy/start representation
→ constrained GASF image
→ DDPM generation
→ diagonal-based analytical decoding
→ reconstructed trajectory
→ map matching
```

## 10. Immediate Next Action

The first action is to build and verify the new dataset:

```text
GASF_dx + GASF_dy + start-point heatmap
```

No new model should be trained before the analytical decoding sanity check passes.

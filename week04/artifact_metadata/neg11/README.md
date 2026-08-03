# Constrained GASF dx/dy/start Dataset, [-1,1] Ablation

This dataset uses:

R = GASF(dx) with global min-max scaling to [-1,1]
G = GASF(dy) with global min-max scaling to [-1,1]
B = initial-point heatmap

Representation name: `GASF_dx_GASF_dy_start_heatmap_neg11`
Image size: `224`
Start heatmap sigma: `3.0`
Normalization file: `constrained_gasf_normalization.json`

PNG images in `images/` are DDPM-compatible training inputs.
Float32 arrays in `arrays/` preserve the same saved channels before uint8 quantization.

Important: this version is NOT analytically invertible from the GASF diagonal
alone because `diag = 2x^2 - 1` only recovers `abs(x)`. Sign information is
lost. This dataset is intended only as a representation-distribution ablation.

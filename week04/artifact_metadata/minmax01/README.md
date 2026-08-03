# Constrained GASF dx/dy/start Dataset

This dataset uses:

R = GASF(dx)
G = GASF(dy)
B = initial-point heatmap

It replaces the previous GASF/GADF/MTF representation.
It is designed for analytical diagonal decoding.

Representation name: `GASF_dx_GASF_dy_start_heatmap`
Image size: `224`
Start heatmap sigma: `3.0`
Normalization file: `constrained_gasf_normalization.json`

PNG images in `images/` are the DDPM training inputs.
Float32 arrays in `arrays/` preserve the same channels before uint8 quantization
and are used for exact analytical decoding verification.

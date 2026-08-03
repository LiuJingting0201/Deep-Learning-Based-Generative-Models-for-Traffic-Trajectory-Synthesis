# Week 4 DDPM Ablation Notes

The current constrained GASF DDPM ablation has three runs for each normalization variant:

- `none`: pure DDPM noise-prediction MSE.
- `symmetry`: MSE plus GASF symmetry loss on channels 0 and 1.
- `gasf_structure`: MSE plus symmetry loss plus GASF self-consistency loss rebuilt from the predicted diagonal.

The structure losses only apply to channels 0 and 1. Channel 2 is the initial-point heatmap and is not constrained by GASF losses.

Normalization variants:

- `minmax01`: original global min-max displacement normalization to `[0,1]`.
- `neg11`: global min-max displacement normalization to `[-1,1]`; this variant is not analytically signed-invertible from the GASF diagonal alone.

TODO: compare the three trained runs using FID, sample visualization, symmetry error, GASF self-consistency error, and diagonal decoded trajectory quality.

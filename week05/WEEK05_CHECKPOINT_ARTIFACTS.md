# Week05 Checkpoint Artifacts

This file records the large Week05 model artifacts that should be preserved on
external storage. These weights are intentionally not committed to ordinary Git:
individual checkpoint files are larger than GitHub's 100 MB file limit, and
`git-lfs` is not installed in the current environment.

## Primary GASF DDPM Run

Run directory:

```text
week05/results/week05_sig15_mse_diag
```

Best image-space checkpoint used by Week06 decoding and guidance experiments:

```text
week05/results/week05_sig15_mse_diag/checkpoint-40000
```

Final checkpoint:

```text
week05/results/week05_sig15_mse_diag/checkpoint-final
```

Large weight files inside each checkpoint:

```text
unet/diffusion_pytorch_model.safetensors
ema_unet/diffusion_pytorch_model.safetensors
training_state.pt
```

Related sample directory used by Week06:

```text
week05/results_npy_samples/week05_sig15_mse_diag
```

## Full Week05 Result Archive

The external-drive export should preserve:

```text
week05/data
week05/results
week05/results_npy_samples
week05/fid_results
week05/guidance_consistency_results
week05/guidance_consistency_results_gradclip010
week05/logs
week05/scripts
week05/slurm
```

The helper script below copies these artifacts, plus Week06 reports and result
summaries, to a mounted external drive:

```text
week05/scripts/export_week05_artifacts_to_drive.sh
```

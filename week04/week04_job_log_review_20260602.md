# Week04 job log review - 2026-06-02

Checked logs under `week04/logs` for the six running Slurm jobs:

| job id | run | data scaling | aux loss | last observed step | NaN? | latest checkpoint seen | recommendation |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1748669 | `week04_gasf_minmax01_mse` | minmax01 | none / MSE only | 40560 | no | `checkpoint-40000` | Can continue to 50000; early stop is acceptable if saving GPU time. |
| 1748670 | `week04_gasf_minmax01_symmetry` | minmax01 | symmetry | 40550 | no | `checkpoint-40000` | Can continue to 50000; early stop is acceptable if saving GPU time. |
| 1748671 | `week04_gasf_minmax01_gasf_structure` | minmax01 | gasf_structure | 41420 | yes, first seen at step 490 | `checkpoint-40000`, but training has been NaN for almost all logged steps | Stop now. This run is not useful as-is. |
| 1748672 | `week04_gasf_neg11_mse` | neg11 | none / MSE only | 40560 | no | `checkpoint-40000` | Can continue to 50000; early stop is acceptable if saving GPU time. |
| 1748673 | `week04_gasf_neg11_symmetry` | neg11 | symmetry | 40560 | no | `checkpoint-40000` | Can continue to 50000; early stop is acceptable if saving GPU time. |
| 1748674 | `week04_gasf_neg11_gasf_structure` | neg11 | gasf_structure | 41420 | yes, first seen at step 620 | `checkpoint-40000`, but training has been NaN for almost all logged steps | Stop now. This run is not useful as-is. |

## Decision

The two `gasf_structure` jobs were cancelled after review:

```bash
scancel 1748671 1748674
```

Post-cancel queue check first showed only these jobs still running: `1748669`, `1748670`, `1748672`, and `1748673`.

After the follow-up decision to stop all remaining jobs, the four healthy jobs were also cancelled:

```bash
scancel 1748669 1748670 1748672 1748673
```

Final `squeue -u jliu` check returned no running jobs. The healthy runs can be resumed from their `checkpoint-40000` directories if needed.

## Local lightweight copy

The run outputs were copied back into `/home/jliu/Thesis/week04/runs_light`.

Copied contents:

- Four healthy runs include `checkpoint-40000`, `train_log.jsonl`, and `samples/step_040000_*.png`.
- Two failed `gasf_structure` runs include only `train_log.jsonl` and `samples/step_040000_*.png` for diagnostics.
- The lightweight archive is about 6.8G.

See `week04/runs_light/README.md` for the exact contents and NaN diagnosis.

The four MSE/symmetry jobs are healthy. They are at about 40.5k / 50k steps, with `save_every=2000`, so the latest durable checkpoint is step 40000. Letting them finish should take roughly another 4-5 hours based on the current runtime. If GPU time matters more than getting exactly 50000-step checkpoints, cancelling these four after step 40000 is defensible because their losses have already plateaued around the `5e-4` range.

## Evidence

- `1748669` latest line: `step=40560 total_loss=0.000555 mse_loss=0.000555`.
- `1748670` latest line: `step=40550 total_loss=0.000556 mse_loss=0.000537 gasf_symmetry_loss=0.001932 gasf_consistency_loss=0.003867`.
- `1748671` latest line: `step=41420 total_loss=nan mse_loss=nan gasf_symmetry_loss=nan gasf_consistency_loss=nan`; first NaN observed at `step=490`.
- `1748672` latest line: `step=40560 total_loss=0.000504 mse_loss=0.000504`.
- `1748673` latest line: `step=40560 total_loss=0.000540 mse_loss=0.000532 gasf_symmetry_loss=0.000832 gasf_consistency_loss=0.003594`.
- `1748674` latest line: `step=41420 total_loss=nan mse_loss=nan gasf_symmetry_loss=nan gasf_consistency_loss=nan`; first NaN observed at `step=620`.

The `.err` files for the two `gasf_structure` jobs also show repeated inference/sample progress plus:

```text
RuntimeWarning: invalid value encountered in cast
```

That warning is consistent with generated samples containing invalid values after NaN training.

## Notes

- All runs used `max_train_steps=50000`, `sample_every=2000`, and `save_every=2000`.
- The healthy jobs have no `nan` lines in the logs checked.
- The `gasf_structure` objective likely needs a numerical guard or a lower/warmer-started structure loss before rerunning.

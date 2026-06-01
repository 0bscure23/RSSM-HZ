# RSSM-HZ Experiment Snapshot (2026-06-01)

This page is a lightweight online-viewable snapshot of the experiments that are currently considered reproducible or diagnostically useful. Large H5 datasets, `.mat` prediction dumps, and `.pth` checkpoints are intentionally excluded from git.

## Confirmed Baselines

| Dataset | Method | PSNR | SAM | ERGAS | Q | Note |
|---|---:|---:|---:|---:|---:|---|
| GF2 | WFANet reproduction | 50.0684 | 0.6641 | 0.5899 | 0.9088 | Q4 |
| GF2 | RSSM-HZ clean baseline | 48.4208 | 0.7752 | 0.7183 | 0.8882 | older pre-innovation result |
| GF2 | RSSM-HZ innovation-z | 48.9536 | 0.7363 | 0.6753 | 0.8956 | confirmed current best before band-corr |
| QB | WFANet reproduction | 38.6932 | 4.3781 | 3.5485 | 0.8463 | Q4 |
| QB | RSSM-HZ clean baseline | 37.6160 | 4.7206 | 4.0317 | 0.8268 | older pre-innovation result |
| QB | RSSM-HZ innovation-z | 37.9858 | 4.6096 | 3.8598 | 0.8322 | confirmed current best before band-corr |

Main takeaway: innovation-z and the corrected training/evaluation protocol significantly improve GF2/QB over the older RSSM-HZ baseline, but RSSM-HZ is still behind WFANet on these two original 64x64-training datasets.

## z-state Diagnostics

The old Phase-B `z` branch was nearly unused. The innovation-order version makes `z` measurable on GF2.

| Dataset | z mode | PSNR | SAM | ERGAS | Q4 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| GF2 | prior | 48.9606 | 0.7370 | 0.6750 | 0.8955 | best PSNR among z modes |
| GF2 | posterior | 48.9536 | 0.7363 | 0.6753 | 0.8956 | main checkpoint result |
| GF2 | zero | 48.9313 | 0.7383 | 0.6771 | 0.8944 | z contributes a small but real gain |
| QB | prior | 37.9919 | 4.6089 | 3.8575 | 0.8322 | tiny PSNR gain |
| QB | posterior | 37.9858 | 4.6096 | 3.8598 | 0.8322 | main checkpoint result |
| QB | zero | 37.9823 | 4.6143 | 3.8631 | 0.8320 | z contribution is weak |

## Error Analysis

Error summaries are stored under:

```text
docs/metrics/gf2_error_analysis_summary.json
docs/metrics/qb_error_analysis_summary.json
```

Key observations:

- GF2: LMS to RSSM-HZ greatly reduces LL error, but the remaining error is concentrated in a hard multispectral band.
- QB: LL error is also reduced, but band-specific residuals remain large; improving Q4 alone is not enough because PSNR/SAM/ERGAS are still behind WFANet.
- Therefore, the current next direction is not "make z larger"; it is lightweight band-aware spectral correction and low-complexity local frequency fusion.

## Experiments Rejected Or Deprioritized

| Experiment | Finding |
|---|---|
| Tiled 64x64 inference on GF2 256x256 test images | Worse than full-image inference: PSNR dropped from about 48.95 to 48.18. |
| LocalFrequencyMixer epoch-1 probe | GF2/QB early test metrics were below innovation-z best, so the long run was stopped. |
| z-residual auxiliary head | It activated `z`, but did not improve final metrics. |
| Geometric augmentation | Not consistently beneficial on GF2/QB. |

## Current WIP

The repository now contains code for a zero-initialized `BandAwareCorrection` head and a `head_only` freeze mode. These are still experimental as of this snapshot.

Early probe:

| Dataset | WIP method | PSNR | SAM | ERGAS | Q4 | Status |
|---|---:|---:|---:|---:|---:|---|
| GF2 | band-corr head-only epoch 1 | 48.9563 | 0.7362 | 0.6751 | 0.8956 | tiny positive signal |
| QB | band-corr mse+band epoch 1 | 37.9790 | 4.6117 | 3.8625 | 0.8322 | Q4 tiny positive, PSNR negative |

The WIP code is included because it is zero-initialized and disabled by default, but it should not be claimed as a final improvement until the running 80-epoch experiments finish.

Follow-up WIP added after the first head-only probe:

- `head_reduce` freeze mode trains `reduce + band_corr + out_act + fused_weight`, allowing the 32-channel fused representation to be remapped to the four GF2/QB bands while keeping the recurrent backbone frozen.
- `launch_headreduce_round_20260601.sh` runs GF2/QB with head-reduce using L1-only and SAM+LL variants.
- `launch_level_ll_round_20260601.sh` additionally enables per-level `LevelLLCorrection`, so low-frequency/spectral bias can be corrected before hierarchical IDWT instead of only at the final output.
- `launch_level_ll_state_round_20260601.sh` is the stronger follow-up: it keeps shallow encoders frozen but fine-tunes recurrent fusion blocks, gates, per-level LL correction, and output heads with a small LR.
- `launch_level_ll_state_focus_20260601.sh` keeps only the two most promising state-level variants active: GF2 L1 and QB SAM+LL.
- `launch_level_ll_state_from_reduce_20260601.sh` starts state-level fine-tuning from the improved per-level-LL reduce checkpoints instead of the older innovation-z checkpoints.
- These variants still use only convolutional/local operations and keep the intended linear-complexity advantage over WFANet-style global attention.

## Uploaded Lightweight Metric JSONs

```text
docs/metrics/gf2_innovation_z_100ep_metrics.json
docs/metrics/qb_innovation_z_100ep_metrics.json
docs/metrics/gf2_innovation_prior_metrics.json
docs/metrics/gf2_innovation_zero_metrics.json
docs/metrics/qb_innovation_prior_metrics.json
docs/metrics/qb_innovation_zero_metrics.json
docs/metrics/gf2_error_analysis_summary.json
docs/metrics/qb_error_analysis_summary.json
```

These files are intended for online review and audit. Full checkpoints and datasets remain local.

# RSSM-HZ Current Status

Updated: 2026-06-05.

For the latest GF2/QB innovation-z metrics and online-friendly JSON summaries,
see [docs/experiment_results_20260601.md](docs/experiment_results_20260601.md).
For the Mamba/WFM/Channel-DWT continuation summary, see
[docs/mamba_frequency_mixer_20260602.md](docs/mamba_frequency_mixer_20260602.md).

This note summarizes the current state of the RSSM-HZ experiments so the project can be handed to another model or collaborator for review.

## 1. Goal

The project starts from WFANet-style wavelet pansharpening and replaces the expensive attention-heavy fusion idea with a coarse-to-fine recurrent state update over wavelet scales.

The intended core idea is:

1. Encode PAN and LRMS/LMS into feature maps.
2. Apply multi-level DWT.
3. Treat wavelet levels as a sequence from coarse to fine.
4. Use a deterministic hidden state `h` and optional stochastic latent `z` to propagate cross-scale context.
5. Fuse LL and high-frequency subbands at each level.
6. Reconstruct the HRMS output through hierarchical IDWT and an LMS residual.

## 2. Current Architecture

Main file:

```text
rssm_hz_wfanet.py
```

Important modules:

```text
WaveletPyramid
WaveletReconstructor
RSSMHzCell
CrossScaleFusionHz
RSSMWaveletFusionHz
RSSMHWViTHZ
```

The current main path is feature-space 3-level DWT:

```text
PAN/LMS/LRMS
  -> shallow feature encoding
  -> 3-level DWT
  -> Level 2 -> Level 1 -> Level 0 recurrent fusion
  -> hierarchical IDWT
  -> reduce 32 -> C
  -> HRMS = PReLU(fused_weight * fused_C + MS_up)
```

For each level:

```text
h_i = GRU([obs_i, z_in], h_in)

Phase A:
  z_i = 0, KL = 0

Phase B:
  z_i ~ q(z_i | h_i, obs_i)
  inference uses prior mean from p(z_i | h_i)
```

LL fusion:

```text
fused_LL_i = Decode(h_i, z_i) * Gate(obs_i) + MS_LL_i
```

High-frequency fusion:

```text
PAN_HF_i -> 1x1 projection
z_i -> z_to_gate(z_i)
gate_in = Cat(fused_LL_i, MS_LL_i, Projected PAN_HF_i, z_gate_i)

fused_LH_i = MS_LH_i + alpha_LH_i * Projected PAN_LH_i
fused_HL_i = MS_HL_i + alpha_HL_i * Projected PAN_HL_i
fused_HH_i = MS_HH_i + alpha_HH_i * Projected PAN_HH_i
```

## 3. Main Training Strategy

The most stable no-distillation strategy so far:

```text
64x64 crop pretraining
  -> validation-best checkpoint selection
  -> full-image / larger-image fine-tuning when available
  -> tiled or full evaluation depending on memory
```

Distillation was tested earlier and improved WV3 somewhat, but it is not currently preferred for paper use because it depends on WFANet as a teacher.

## 4. Strongest Current Results

### PanScale-style datasets

These are the strongest no-distillation results currently available.

| Dataset | Method | PSNR | SAM | ERGAS | Q/Q8 | Status |
|---|---:|---:|---:|---:|---:|---|
| jilin | WFANet repro | 39.4040 | 1.1523 | 1.1817 | 0.9529 | baseline |
| jilin | RSSM-HZ best | 39.5423 | 1.1394 | 1.1597 | 0.9540 | better on all four |
| landsat | WFANet repro | 43.3821 | 1.6079 | 2.5809 | 0.6886 | baseline |
| landsat | RSSM-HZ best | 43.5499 | 1.6014 | 2.4377 | 0.6910 | better on all four |
| skysat | WFANet repro | 46.6013 | 1.2393 | 1.1091 | 0.8346 | baseline |
| skysat | RSSM-HZ balanced | 46.7152 | 1.2381 | 1.1435 | 0.8364 | PSNR/SAM/Q better, ERGAS worse |
| skysat | RSSM-HZ ERGAS-oriented | 46.7222 | 1.2415 | 1.1146 | 0.8332 | ERGAS closer, Q/SAM worse |

Interpretation:

RSSM-HZ is already competitive on PanScale-style data, especially when the model can be trained or fine-tuned on larger spatial inputs such as 200x200.

### Original WFANet datasets

These results are still weaker than the local WFANet reproduction.

Important source note:

The WFANet numbers below are from local reproduction / local evaluation files,
not copied directly from the WFANet paper table. In particular:

- WV3 comes from `plp/WFANet/eval_wv3_package/wv3_metrics_summary.json`.
- GF2 comes from `plp/WFANet/results_rssm_hz/gf2_wfanet_v2/test_metrics.json`.
- QB comes from `plp/WFANet/results_rssm_hz/qb_wfanet_v2/test_metrics.json`.

Paper-reported numbers can differ because of MATLAB evaluation details, Q4/Q8
implementation, crop/border handling, and dynamic-range conventions. Use the
local WFANet reproduction for apples-to-apples comparisons with RSSM-HZ runs.

| Dataset | Method | PSNR | SAM | ERGAS | Q/Q8 |
|---|---:|---:|---:|---:|---:|
| WV3 | WFANet repro | 39.2442 | 2.8751 | 2.1041 | 0.8786 |
| WV3 | RSSM-HZ no-distill best | 38.9393 | 2.9354 | 2.1739 | 0.8755 |
| WV3 | RSSM-HZ continued fine-tune | 38.9397 | 2.9383 | 2.1746 | 0.8755 |
| GF2 | WFANet repro | 50.0684 | 0.6641 | 0.5899 | 0.9088 |
| GF2 | RSSM-HZ older baseline | 48.4208 | 0.7752 | 0.7183 | 0.8882 |
| GF2 | RSSM-HZ innovation-z | 48.9536 | 0.7363 | 0.6753 | 0.8956 |
| GF2 | RSSM-HZ Mamba/WFM/ChDWT best | 49.0793 | 0.7277 | 0.6651 | 0.8974 |
| GF2 | RSSM-HZ doc-FMamba l1 160ep | 49.0383 | 0.7303 | 0.6682 | 0.8968 |
| GF2 | RSSM-HZ doc-FMamba mseband 160ep | 49.0395 | 0.7304 | 0.6682 | 0.8969 |
| QB | WFANet repro | 38.6932 | 4.3781 | 3.5485 | 0.8463 |
| QB | RSSM-HZ older baseline | 37.6153 | 4.7196 | 4.0308 | 0.8273 |
| QB | RSSM-HZ innovation-z | 37.9858 | 4.6096 | 3.8598 | 0.8322 |
| QB | RSSM-HZ Mamba/WFM/ChDWT best | 38.0518 | 4.5846 | 3.8293 | 0.8340 |
| QB | RSSM-HZ doc-FMamba mseband 160ep | 38.0161 | 4.5950 | 3.8451 | 0.8330 |
| QB | RSSM-HZ doc-FMamba samll 160ep | 38.0373 | 4.5914 | 3.8375 | 0.8330 |

Important metric note:

For GF2/QB, use `q_win_size=4` for fair Q4 comparison. Some older scripts
printed the metric under `Q8` even when the window size was 4, so always check
the stored `q_win_size` field in each metrics JSON.

Interpretation as of 2026-06-05:

- The best no-distillation GF2/QB line is still the incremental
  Mamba/WFM/Channel-DWT version, not the full doc-FMamba rewrite.
- The full doc-FMamba / WSLM-style blueprint was trained for 160 epochs on
  GF2/QB and did not improve over the Mamba/WFM/Channel-DWT version.
- Extending doc-FMamba training from 80 to 160 epochs did not fix the gap:
  GF2 l1 changed from about 49.041 PSNR at 80 epochs to 49.038 PSNR at
  160 epochs on the final test set.
- Therefore, the current evidence suggests that the WFANet gap on GF2/QB is
  structural / protocol-related rather than simply caused by under-training.

## 5. Current Open Problems

### Problem A: 64x64-only training limits the recurrent wavelet idea

For 3-level DWT:

| Input size | Level sizes |
|---|---|
| 64x64 | 32 -> 16 -> 8 |
| 200x200 | 100 -> 50 -> 25 |
| 256x256 | 128 -> 64 -> 32 |

On 64x64 patches, the coarsest level is only 8x8. The model still has low-frequency information, but the coarse state has limited scene-level spatial structure. On 200x200 or larger inputs, the coarse state keeps a richer layout and the recurrent coarse-to-fine design has more room to help.

This is likely one reason RSSM-HZ looks stronger on PanScale than on the original WV3/GF2/QB 64x64 training protocol.

### Problem B: Low-frequency correction is conservative

RSSM-HZ does not lack low-frequency information. It explicitly uses `PAN_LL_i` and `MS_LL_i`, and LL fusion is:

```text
fused_LL_i = Decode(h_i, z_i) * Gate(obs_i) + MS_LL_i
```

The issue is that the LL path is residual and conservative. It relies heavily on `MS_LL_i` and final `MS_up`, so it may under-correct low-frequency or spectral bias on GF2/QB. WFANet-style attention fusion may have stronger low-frequency/color correction capacity.

### Problem C: Two-level DWT did not solve the issue

Landsat diagnostic experiments compared 2-level and 3-level DWT under 64x64 crop training:

| Experiment | PSNR | SAM | ERGAS | Q |
|---|---:|---:|---:|---:|
| crop64 + 2-level | 43.2047 | 1.6357 | 2.5000 | 0.6755 |
| crop64 + 3-level | 43.2003 | 1.6399 | 2.5088 | 0.6787 |
| crop64 align4 + 3-level | 43.4466 | 1.6141 | 2.4733 | 0.6876 |

This suggests that simply changing 3-level DWT to 2-level DWT is not enough.

## 6. Recently Completed Experiments

### Mamba/WFM/Channel-DWT continuation

This was the most useful incremental branch after innovation-z.

| Dataset | Run tag | PSNR | SAM | ERGAS | Q4 |
|---|---|---:|---:|---:|---:|
| GF2 | `gf2_mamba_wfm_chdwt_l1_80ep_from120` | 49.0793 | 0.7277 | 0.6651 | 0.8974 |
| GF2 | `gf2_mamba_wfm_chdwt_mseband_80ep_from120` | 49.0762 | 0.7283 | 0.6654 | 0.8973 |
| QB | `qb_mamba_wfm_chdwt_mseband_80ep_from120` | 38.0314 | 4.5916 | 3.8375 | 0.8341 |
| QB | `qb_mamba_wfm_chdwt_samll_80ep_from120` | 38.0518 | 4.5846 | 3.8293 | 0.8340 |

Conclusion:

This branch gives a consistent small improvement over innovation-z, but the
remaining gap to WFANet is still large on GF2 and visible on QB.

### Doc-FMamba / WSLM-style blueprint

The doc-faithful rewrite was implemented and then tested with 160-epoch
Phase-B fine-tuning.

| Dataset | Run tag | PSNR | SAM | ERGAS | Q4 |
|---|---|---:|---:|---:|---:|
| GF2 | `gf2_doc_fmamba_l1_80ep_gpu3r1` | 49.0383 | 0.7303 | 0.6682 | 0.8968 |
| GF2 | `gf2_doc_fmamba_mseband_160ep_gpu0` | 49.0395 | 0.7304 | 0.6682 | 0.8969 |
| QB | `qb_doc_fmamba_mseband_160ep_gpu1` | 38.0161 | 4.5950 | 3.8451 | 0.8330 |
| QB | `qb_doc_fmamba_samll_160ep_gpu2` | 38.0373 | 4.5914 | 3.8375 | 0.8330 |

Conclusion:

The full doc-FMamba branch is not the current best direction. It is slightly
worse than the simpler Mamba/WFM/Channel-DWT branch, despite the longer
160-epoch fine-tuning. Keep it as a negative / diagnostic experiment rather
than the main paper path.

## 7. Suggested Next Directions

Recommended next experiments:

1. Return to the Mamba/WFM/Channel-DWT version as the current strongest
   GF2/QB baseline, rather than continuing the doc-FMamba branch.
2. Add targeted low-frequency / spectral correction only if it is residual-safe
   and evaluated with a fixed recipe across GF2/QB.
3. Analyze band-wise residuals and low-frequency bias on GF2/QB; previous error
   analysis suggested that remaining errors are concentrated in hard
   multispectral bands.
4. Try PanScale pretraining transfer for GF2/QB and compare with from-scratch
   GF2/QB checkpoints.
5. If original large scenes for WV3/GF2/QB can be obtained, train or fine-tune
   on larger crops instead of only 64x64 patches.
6. Keep no-distillation as the main paper setting.
7. Use a single fixed training recipe when possible to avoid dataset-specific
   tuning concerns.

Not recommended as the main paper route:

1. Pseudo-stitching unrelated 64x64 patches into fake large images.
2. Image interpolation from 64x64 to larger sizes as a substitute for real large-image training.
3. Relying on WFANet teacher distillation for headline results.
4. Continuing to scale the doc-FMamba / WSLM-style branch without a more
   specific diagnosis, because 160-epoch runs did not improve over the simpler
   Mamba/WFM/Channel-DWT branch.

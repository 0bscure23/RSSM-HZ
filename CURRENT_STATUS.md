# RSSM-HZ Current Status

Updated: 2026-06-01.

For the latest GF2/QB innovation-z metrics and online-friendly JSON summaries,
see [docs/experiment_results_20260601.md](docs/experiment_results_20260601.md).

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

These results are still weaker than WFANet.

| Dataset | Method | PSNR | SAM | ERGAS | Q/Q8 |
|---|---:|---:|---:|---:|---:|
| WV3 | WFANet repro | 39.2442 | 2.8751 | 2.1041 | 0.8786 |
| WV3 | RSSM-HZ no-distill best | 38.9393 | 2.9354 | 2.1739 | 0.8755 |
| WV3 | RSSM-HZ continued fine-tune | 38.9397 | 2.9383 | 2.1746 | 0.8755 |
| GF2 | WFANet repro | 50.0684 | 0.6641 | 0.5899 | 0.9088 |
| GF2 | RSSM-HZ older baseline | 48.4208 | 0.7752 | 0.7183 | 0.8882 |
| GF2 | RSSM-HZ innovation-z | 48.9536 | 0.7363 | 0.6753 | 0.8956 |
| QB | WFANet repro | 38.6932 | 4.3781 | 3.5485 | 0.8463 |
| QB | RSSM-HZ older baseline | 37.6153 | 4.7196 | 4.0308 | 0.8273 |
| QB | RSSM-HZ innovation-z | 37.9858 | 4.6096 | 3.8598 | 0.8322 |

Important metric note:

For GF2/QB, use `q_win_size=4` for fair Q4 comparison. Some older scripts
printed the metric under `Q8` even when the window size was 4, so always check
the stored `q_win_size` field in each metrics JSON.

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

## 6. Experiments Currently Running

The current low-cost experiment is PanScale checkpoint transfer to GF2/QB.

| Run tag | Target | Init checkpoint |
|---|---|---|
| `transfer_gf2_from_jilin_20260526` | GF2 | jilin PanScale best |
| `transfer_gf2_from_skysat_20260526` | GF2 | skysat PanScale best |
| `transfer_qb_from_jilin_20260526` | QB | jilin PanScale best |
| `transfer_qb_from_skysat_20260526` | QB | skysat PanScale best |

Purpose:

Check whether PanScale pretraining gives RSSM-HZ a useful cross-scale prior for the original 4-channel datasets.

## 7. Suggested Next Directions

Recommended next experiments:

1. Add a lightweight low-frequency / spectral correction head after IDWT or at each LL level.
2. Try PanScale pretraining transfer for GF2/QB and compare with from-scratch GF2/QB checkpoints.
3. If original large scenes for WV3/GF2/QB can be obtained, train or fine-tune on larger crops instead of only 64x64 patches.
4. Keep no-distillation as the main paper setting.
5. Use a single fixed training recipe when possible to avoid dataset-specific tuning concerns.

Not recommended as the main paper route:

1. Pseudo-stitching unrelated 64x64 patches into fake large images.
2. Image interpolation from 64x64 to larger sizes as a substitute for real large-image training.
3. Relying on WFANet teacher distillation for headline results.

# RSSM-HZ: Recurrent State-Space Model for Wavelet-based Pansharpening

This repository contains the code for our RSSM-HZ pansharpening method, which achieves state-of-the-art performance by modeling coarse-to-fine wavelet fusion as a recurrent state-space process.

## Key Results (Jilin Dataset, PanScale Benchmark)

| Method | Q8 | PSNR | SAM | ERGAS |
|--------|-----|------|-----|-------|
| **RSSM-HZ (ours)** | **0.9529** | **39.48** | **1.13** | **1.17** |
| WFANet | 0.9529 | 39.40 | 1.15 | 1.18 |

## Quick Start

```bash
# Phase A: h-only deterministic training (800 epochs)
python train_rssmhz_crop.py \
    --gpu 0 --epochs 800 --batch-size 12 \
    --hidden-dim 128 --latent-dim 48 \
    --phase a --run-tag my_experiment

# Phase B: h+z stochastic fine-tuning (optional)
python train_rssmhz_crop.py \
    --gpu 0 --epochs 200 --batch-size 12 \
    --hidden-dim 128 --latent-dim 48 \
    --phase b --init-ckpt results_rssm_hz/my_experiment/checkpoints/rssm_hz_best.pth \
    --run-tag my_experiment_phaseB
```

## File Structure

```
├── rssm_hz_wfanet.py          # Core model architecture
├── net_torch.py               # DWT/IDWT wavelet transform + WFANet baseline
├── train_rssmhz_crop.py       # Crop-based training (optimal method)
├── train_rssm_hz.py           # Full-image training (reference)
├── train_wfanet_jilin_crop.py # WFANet baseline training
├── evaluate_wv3_metrics.py    # PSNR/SAM/ERGAS/Q8 evaluation
├── convert_panscale_to_h5.py  # Data preprocessing
├── super_para_panscale.yml    # Jilin dataset config
└── super_para.yml             # WV3 dataset config
```

## Documentation

- [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) — Detailed code walkthrough with diagrams
- [PROJECT_REPORT.md](PROJECT_REPORT.md) — Project report with experiment history

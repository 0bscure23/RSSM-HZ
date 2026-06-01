# RSSM-HZ: Coarse-to-Fine Recurrent Wavelet Fusion for Pansharpening

This repository contains the code for RSSM-HZ, a pansharpening model that treats multi-level wavelet subbands as a coarse-to-fine sequence and updates a hidden state across scales.

The project is still under active experimentation. The strongest current evidence is on PanScale-style datasets, where RSSM-HZ reaches or exceeds the local WFANet reproduction while using a recurrent wavelet fusion path instead of WFANet's heavier attention-style fusion. On the original WV3/GF2/QB protocol, RSSM-HZ is close on WV3 but still behind on GF2/QB.

## Current Snapshot

See [CURRENT_STATUS.md](CURRENT_STATUS.md) for the latest experiment summary, open issues, and next steps.

For the GF2/QB innovation-z update and lightweight metric files, see
[docs/experiment_results_20260601.md](docs/experiment_results_20260601.md).

Note: experimental modules such as `BandAwareCorrection` are zero-initialized
and disabled by default. They are included for reproducibility of current
ablation work, but only the innovation-z results in the snapshot are treated as
confirmed improvements.

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

For validation-driven full-image fine-tuning:

```bash
python train_rssm_hz.py \
    --config super_para_panscale.yml \
    --gpu 0 \
    --run-tag jilin_full_finetune \
    --train-path Dataset/PanScale_H5/jilin/jilin_train_v2.h5 \
    --val-path Dataset/PanScale_H5/jilin/jilin_val_v2.h5 \
    --test-path Dataset/PanScale_H5/jilin/jilin_test200.h5 \
    --init-ckpt results_rssm_hz/jilin_h128_800ep/checkpoints/rssm_hz_best.pth \
    --epochs 200 --batch-size 4 \
    --hidden-dim 128 --latent-dim 48 \
    --phase a --lr-scale 0.01 \
    --val-every 20 --best-metric overall
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
├── matlab_eval/               # MATLAB/Octave-style reduced-resolution evaluation helpers
├── super_para_panscale.yml    # Jilin dataset config
├── super_para_gf2_2047.yml    # GF2 4-channel reduced-resolution config
├── super_para_qb.yml          # QB 4-channel reduced-resolution config
└── super_para.yml             # WV3 dataset config
```

## Documentation

- [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) — Detailed code walkthrough with diagrams
- [PROJECT_REPORT.md](PROJECT_REPORT.md) — Project report with experiment history
- [CURRENT_STATUS.md](CURRENT_STATUS.md) — Current progress, results, limitations, and discussion points

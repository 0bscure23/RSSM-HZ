# Clean Equal-Budget Protocol + Shared-Recurrent RSSM-HZ + Dihedral Augmentation

This branch holds the cleaned main line of the RSSM-HZ pansharpening study. It
replaces the earlier tangle of warm-started / distilled / dataset-specific
finetunes with a single from-scratch, equal-budget protocol so that
architecture choices can be compared fairly, and adds the augmentation lever
that closes most of the remaining gap to WFANet.

## Components

- **`train_phase0_clean.py`** — from-scratch equal-budget trainer. L1-only loss,
  batch 32, 240 epochs, cosine LR (9e-4), deterministic state, Q4 validation.
  One question: *under the same budget, how do the architecture families
  compare?* Optional `--augment` enables dihedral (rot90/flip) geometric
  augmentation, applied identically to PAN/MS/GT/LMS via a dedicated RNG so
  no-aug runs stay bit-reproducible. rot90/flip are exact symmetries of nadir
  satellite imagery and spectrum-neutral, so they cannot bias SAM.
- **`eval_phase0_fullframe.py`** — full-frame 256px re-evaluation. Tiled
  inference underestimates PSNR by ~1 dB (WFANet -1.03, two-stage RSSM -1.41),
  and all reference numbers are full-frame, so every checkpoint is re-scored
  full-frame here.
- **`rssm_hz_wfanet.py`** — `WFANetTwoStageRSSMFusion` with
  `share_scale_recurrent` (a single weight-shared coarse-to-fine recurrent state
  cell — the K-step scale-recurrence identity, ~40% fewer params than the
  unshared two-stage), dw-window-attention ConvGRU gates, and level-LL
  correction.
- **`net_torch.py`** — WFANet reproduction (DWT/IDWT, MFFA attention) used as the
  baseline arm.

## Main RSSM line

`rssm_phase1_shared_dw_window`: two-stage shared recurrent cell + dw-window-attn
gates + level-LL correction, deterministic, ~1.99M params.

```bash
# RSSM main line with augmentation
python train_phase0_clean.py --dataset gf2 --model-kind rssm_phase1_shared_dw_window \
    --augment --epochs 240 --gpu 0 --seed 1 --run-tag <tag>
# WFANet baseline arm (same protocol)
python train_phase0_clean.py --dataset gf2 --model-kind wfanet \
    --augment --epochs 240 --gpu 1 --seed 1 --run-tag <tag>
# full-frame re-evaluation (do not use the in-run tiled TEST numbers)
python eval_phase0_fullframe.py --pattern '<tag-glob>' --device cuda:0
```

## Key finding

Under the clean protocol the remaining RSSM-vs-WFANet gap is a **generalization**
problem, not a capacity one (RSSM reaches lower train L1 yet tests worse). Adding
HF-synthesis capacity does not help; the decisive lever is **dihedral
augmentation**, which the over-fitting RSSM benefits from far more than the
(under-trained) WFANet:

| dataset | RSSM no-aug | RSSM +aug | WFANet no-aug | WFANet +aug | gap no-aug → +aug |
|---|---|---|---|---|---|
| GF2 | 49.19 | 50.09 | 49.91 | 50.19 | −0.72 → −0.10 |
| QB  | 38.19 | 38.58 | 38.77 | 38.70 | −0.58 → −0.12 |

(full-frame PSNR, 240ep; longer-budget confirmation in progress)

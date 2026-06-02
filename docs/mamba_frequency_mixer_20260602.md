# Mamba Frequency Mixer Experiments

This branch preserves the RSSM-HZ variant that adds a true `mamba_ssm`
windowed high-frequency mixer on top of the previous WFM + channel-Haar setup.

## What Changed

- Added optional `MambaFrequencyMixer` modules for LH/HL/HH high-frequency
  subbands.
- Kept the existing dependency-free `WindowedFrequencyMixer` as a baseline/proxy.
- Added `ChannelHaarSpectralAdapter` for 1D channel-wise spectral residual
  adaptation on the MS stream.
- Exposed CLI switches:
  - `--use-mamba-frequency-mixer`
  - `--mamba-window-size`
  - `--mamba-hidden-scale`
  - `--mamba-d-state`
  - `--mamba-d-conv`
  - `--mamba-expand`
  - `--use-windowed-frequency-mixer`
  - `--use-channel-dwt-adapter`
- Included the Mamba continuation launchers:
  - `launch_mamba_wfm_chdwt_20260602.sh`
  - `launch_mamba_wfm_chdwt_wait_20260602.sh`

## Environment

The Mamba branch was validated in a cloned environment named `wfanet_mamba`:

```bash
conda create --name wfanet_mamba --clone wfanet
```

Installed wheels:

```bash
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install einops
pip install --no-deps huggingface_hub==0.36.0
pip install --no-deps regex safetensors tokenizers==0.20.3 transformers==4.46.3
```

Smoke test:

```bash
python -c "import torch; from mamba_ssm.modules.mamba_simple import Mamba; m=Mamba(d_model=32,d_state=16,d_conv=4,expand=2).cuda(); x=torch.randn(2,64,32,device='cuda'); y=m(x); print(y.shape, torch.isfinite(y).all().item())"
```

Expected output:

```text
torch.Size([2, 64, 32]) True
```

## Results

The Mamba continuation was trained for 80 epochs from the previous 120-epoch
WFM+channel-DWT checkpoints. All results are reduced-resolution code metrics
with `q_win_size=4`.

| Dataset | Run | PSNR | SAM | ERGAS | Q4 |
|---|---|---:|---:|---:|---:|
| GF2 | `gf2_mamba_wfm_chdwt_l1_80ep_from120` | 49.0793 | 0.7277 | 0.6651 | 0.8974 |
| GF2 | `gf2_mamba_wfm_chdwt_mseband_80ep_from120` | 49.0762 | 0.7283 | 0.6654 | 0.8973 |
| QB | `qb_mamba_wfm_chdwt_mseband_80ep_from120` | 38.0314 | 4.5916 | 3.8375 | 0.8341 |
| QB | `qb_mamba_wfm_chdwt_samll_80ep_from120` | 38.0518 | 4.5846 | 3.8293 | 0.8340 |

## Interpretation

The true Mamba mixer gives a consistent but small gain over the dependency-free
windowed mixer. It is useful as a lightweight frequency residual branch, but it
is not yet equivalent to the full FMamba/WSLM architecture proposed in the
research note. In particular, this branch does not yet implement cross-modal
Mamba parameter modulation or manifold-preservation regularization.

# Doc FMamba / WSLM Blueprint Implementation

This branch implements the larger architecture suggested by
`RSSM-HZ 改进以超越 WFAnet.docx` as optional code paths. It is intentionally
kept behind CLI switches so the previously validated RSSM-HZ and Mamba mixer
branches remain reproducible.

## Implemented Pieces

- **Modality-specific wavelet front-end**
  - PAN still uses the existing 2D Haar/DWT wavelet pyramid.
  - MS/LMS gains a channel-wise 1D Haar spectral adapter through
    `ChannelHaarSpectralAdapter`.
  - `--use-doc-fmamba` automatically enables the channel-wise adapter.

- **FMamba-style cross-modal frequency fusion**
  - `DocFMambaFrequencyMixer` keeps PAN high-frequency and MS high-frequency
    streams separate.
  - PAN and MS streams each pass through Mamba.
  - PAN stream gates/modulates the MS stream, approximating the document's
    cross-modal selective-state modulation without editing `mamba_ssm`
    internals.

- **WSLM-style local state mixing**
  - `WindowedStateMambaMixer` applies local window Mamba to recurrent hidden
    states.
  - The output projection is zero-initialized so the module starts as a safe
    no-op residual path.

- **Doc-inspired losses**
  - `--w-frequency`: joint multi-level LL/HF wavelet consistency loss.
  - `--w-mp`: low-dimensional manifold preservation loss aligning prediction
    affinity with PAN and LMS affinity structure.

## Important Switches

```bash
--use-doc-fmamba
--doc-window-size 4
--doc-hidden-scale 1.0
--doc-mamba-d-state 16
--doc-mamba-d-conv 4
--doc-mamba-expand 2
--w-frequency 0.02
--w-mp 0.001
--mp-pool-size 8
```

## Notes

This is a faithful engineering implementation of the document's direction, but
not a literal reimplementation of every theoretical equation. In particular,
`mamba_ssm.modules.Mamba` does not expose internal B/C transition parameters for
direct external modulation, so the code uses a practical equivalent:
PAN-conditioned cross gating over the MS Mamba stream.

The branch has only been syntax-checked. It has not yet been trained or
evaluated.

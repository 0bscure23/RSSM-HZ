# MATLAB-Style Evaluation Results

Date: 2026-05-20

Environment:

- MATLAB-compatible runtime: GNU Octave 10.3.0
- Octave package: `octave-image 2.20.0`
- Metric implementation: DLPan-Toolbox MATLAB reduced-resolution metrics

Protocol:

- `ratio = 4`
- `L = 11`
- `data_range = 2047`
- `Qblocks_size = 32`
- `flag_cut_bounds = 1`
- `dim_cut = 21`
- `th_values = 1`

| Dataset | Run | PSNR↑ | SAM↓ | ERGAS↓ | Q2n/Q4↑ | Q_avg↑ | SCC↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QB | `qb_h128_phaseA_lf_240x160` | 38.434769 | 4.508714 | 3.716627 | 0.935255 | 0.934027 | 0.981670 |
| GF2 | `gf2_h128_phaseA_lf_lightloss_2047_240x160` | 48.209938 | 0.813980 | 0.729438 | 0.977347 | 0.979118 | 0.988693 |

Notes:

- `Q2n/Q4` is the MATLAB-style multi-band Q index from DLPan's `q2n.m`.
- PSNR is computed by this wrapper as mean PSNR over spectral bands, because DLPan's `indexes_evaluation.m` does not output PSNR.
- Octave warnings about invalid UTF-8 come from legacy comments in DLPan MATLAB files and do not affect numeric results.

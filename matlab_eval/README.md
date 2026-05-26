# MATLAB-Style Reduced-Resolution Evaluation

This folder provides a MATLAB/Octave-compatible evaluation wrapper for RSSM-HZ
pansharpening predictions.

## Metric Source

The core reduced-resolution metrics come from DLPan-Toolbox:

- `indexes_evaluation.m`
- `SAM.m`
- `ERGAS.m`
- `q2n.m`
- `Q.m`
- `SCC.m`

The wrapper additionally computes mean-band PSNR because DLPan's
`indexes_evaluation.m` does not report PSNR.

Default protocol:

- `ratio = 4`
- `L = 11`
- `data_range = 2047`
- `Qblocks_size = 32`
- `flag_cut_bounds = 1`
- `dim_cut = 21`
- `th_values = 1`

`Q2n` is the reported `Q4` for 4-channel QB/GF2 data and `Q8` for 8-channel
WV3 data.

## Prepare Eval Pack

```bash
python matlab_eval/prepare_reduced_eval_pack.py \
  --pred-dir results_rssm_hz/<run>/eval/pred \
  --test-h5 ../WFANet/Dataset/<dataset>/test_<dataset>_multiExm1.h5 \
  --out-dir results_rssm_hz/<run>/matlab_eval_pack
```

The pack contains:

- `pred/pred_XX.mat` with variable `sr`
- `ref/ref_XX.mat` with variable `gt`
- `index_map.csv`

## Run With Octave

```bash
bash matlab_eval/run_reduced_eval_octave.sh \
  results_rssm_hz/<run>/matlab_eval_pack \
  results_rssm_hz/<run>/matlab_metrics
```

## Run With MATLAB

```bash
matlab -nodisplay -nosplash -r "addpath('matlab_eval'); eval_reduced_matlab('results_rssm_hz/<run>/matlab_eval_pack', '../_external_eval_refs/DLPan-Toolbox/02-Test-toolbox-for-traditional-and-DL(Matlab)', 'results_rssm_hz/<run>/matlab_metrics', 2047, 4, 11, 32, 1, 21, 1); exit;"
```

## Outputs

- `matlab_metrics_summary.txt`
- `matlab_metrics_summary.mat`
- `matlab_metrics_per_sample.csv`

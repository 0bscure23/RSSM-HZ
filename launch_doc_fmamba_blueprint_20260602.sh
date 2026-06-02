#!/usr/bin/env bash
set -euo pipefail

cd /ssd2/lizhy_workspace/plp/RSSM-HZ
mkdir -p output_log /ssd2/lizhy_workspace/tmp/rssmhz_tmp
export TMPDIR=/ssd2/lizhy_workspace/tmp/rssmhz_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HDF5_USE_FILE_LOCKING=FALSE

PY=/ssd2/lizhy_workspace/anaconda3/envs/wfanet_mamba/bin/python

COMMON=(
  --config super_para_original4.yml
  --phase b
  --use-conv-gru
  --z-update-order innovation
  --z-eval-mode posterior
  --hidden-dim 128
  --latent-dim 48
  --epochs 80
  --batch-size 32
  --num-workers 0
  --val-every 20
  --val-batch-size 8
  --best-metric overall
  --q-win-size 4
  --w-edge 0
  --w-wavelet-hf 0
  --w-kl 0
  --phase-b-freeze-mode gate_head_reduce
  --phase-b-lr-scale 0.035
  --phase-b-ramp-epochs 1
  --no-loss-clamp
  --use-level-ll-corr
  --use-band-corr
  --band-corr-kernel-size 7
  --band-corr-hidden 64
  --use-doc-fmamba
  --doc-window-size 4
  --doc-hidden-scale 1.0
  --doc-mamba-d-state 16
  --doc-mamba-d-conv 4
  --doc-mamba-expand 2
  --w-frequency 0.02
  --w-mp 0.001
  --mp-pool-size 8
  --save-every 999
)

"$PY" -u train_rssm_hz.py --gpu 0 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/gf2_mamba_wfm_chdwt_l1_80ep_from120/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/gf2/train_gf2.h5 \
  --val-path Dataset/gf2/valid_gf2.h5 \
  --test-path Dataset/gf2/test_gf2_multiExm1.h5 \
  --w-sam 0 \
  --w-ll 0 \
  --run-tag gf2_doc_fmamba_l1_80ep \
  > output_log/gf2_doc_fmamba_l1_80ep_20260602.log 2>&1 &
pids=("$!")
echo "started gf2_doc_fmamba_l1_80ep pid=${pids[-1]}"

"$PY" -u train_rssm_hz.py --gpu 1 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/gf2_mamba_wfm_chdwt_mseband_80ep_from120/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/gf2/train_gf2.h5 \
  --val-path Dataset/gf2/valid_gf2.h5 \
  --test-path Dataset/gf2/test_gf2_multiExm1.h5 \
  --w-sam 0 \
  --w-ll 0 \
  --w-mse 10 \
  --w-band-balanced 0.002 \
  --run-tag gf2_doc_fmamba_mseband_80ep \
  > output_log/gf2_doc_fmamba_mseband_80ep_20260602.log 2>&1 &
pids+=("$!")
echo "started gf2_doc_fmamba_mseband_80ep pid=${pids[-1]}"

"$PY" -u train_rssm_hz.py --gpu 2 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/qb_mamba_wfm_chdwt_mseband_80ep_from120/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/qb/train_qb.h5 \
  --val-path Dataset/qb/valid_qb.h5 \
  --test-path Dataset/qb/test_qb_multiExm1.h5 \
  --w-sam 0.01 \
  --w-ll 0.02 \
  --w-mse 20 \
  --w-band-balanced 0.005 \
  --run-tag qb_doc_fmamba_mseband_80ep \
  > output_log/qb_doc_fmamba_mseband_80ep_20260602.log 2>&1 &
pids+=("$!")
echo "started qb_doc_fmamba_mseband_80ep pid=${pids[-1]}"

"$PY" -u train_rssm_hz.py --gpu 3 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/qb_mamba_wfm_chdwt_samll_80ep_from120/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/qb/train_qb.h5 \
  --val-path Dataset/qb/valid_qb.h5 \
  --test-path Dataset/qb/test_qb_multiExm1.h5 \
  --w-sam 0.01 \
  --w-ll 0.02 \
  --run-tag qb_doc_fmamba_samll_80ep \
  > output_log/qb_doc_fmamba_samll_80ep_20260602.log 2>&1 &
pids+=("$!")
echo "started qb_doc_fmamba_samll_80ep pid=${pids[-1]}"

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

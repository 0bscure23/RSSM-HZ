#!/usr/bin/env bash
set -euo pipefail

cd /ssd2/lizhy_workspace/plp/WFANet
mkdir -p output_log /ssd2/lizhy_workspace/tmp/wfanet_tmp
export TMPDIR=/ssd2/lizhy_workspace/tmp/wfanet_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

CONDA=/ssd2/lizhy_workspace/anaconda3/bin/conda
PY_CMD=("$CONDA" run --no-capture-output -n wfanet python -u train_rssm_hz.py)

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
  --val-every 10
  --val-batch-size 8
  --best-metric overall
  --q-win-size 4
  --w-sam 0
  --w-edge 0
  --w-wavelet-hf 0
  --w-ll 0
  --w-kl 0
  --phase-b-freeze-mode head_only
  --phase-b-lr-scale 0.5
  --phase-b-ramp-epochs 1
  --no-loss-clamp
  --use-band-corr
  --band-corr-kernel-size 5
  --band-corr-hidden 32
  --save-every 999
)

GF2=(
  --init-ckpt results_rssm_hz/gf2_innovation_z_100ep/checkpoints/rssm_hz_best.pth
  --train-path Dataset/gf2/train_gf2.h5
  --val-path Dataset/gf2/valid_gf2.h5
  --test-path Dataset/gf2/test_gf2_multiExm1.h5
)

QB=(
  --init-ckpt results_rssm_hz/qb_innovation_z_100ep/checkpoints/rssm_hz_best.pth
  --train-path Dataset/qb/train_qb.h5
  --val-path Dataset/qb/valid_qb.h5
  --test-path Dataset/qb/test_qb_multiExm1.h5
)

"${PY_CMD[@]}" --gpu 0 "${COMMON[@]}" "${GF2[@]}" \
  --run-tag gf2_bandcorr_head_l1_80ep \
  > output_log/gf2_bandcorr_head_l1_20260601.log 2>&1 &
echo "started gf2_bandcorr_head_l1_80ep pid=$!"

"${PY_CMD[@]}" --gpu 1 "${COMMON[@]}" "${QB[@]}" \
  --run-tag qb_bandcorr_head_l1_80ep \
  > output_log/qb_bandcorr_head_l1_20260601.log 2>&1 &
echo "started qb_bandcorr_head_l1_80ep pid=$!"

"${PY_CMD[@]}" --gpu 2 "${COMMON[@]}" "${GF2[@]}" \
  --w-mse 50 \
  --w-band-balanced 0.02 \
  --run-tag gf2_bandcorr_head_mseband_80ep \
  > output_log/gf2_bandcorr_head_mseband_20260601.log 2>&1 &
echo "started gf2_bandcorr_head_mseband_80ep pid=$!"

"${PY_CMD[@]}" --gpu 3 "${COMMON[@]}" "${QB[@]}" \
  --w-mse 50 \
  --w-band-balanced 0.02 \
  --run-tag qb_bandcorr_head_mseband_80ep \
  > output_log/qb_bandcorr_head_mseband_20260601.log 2>&1 &
echo "started qb_bandcorr_head_mseband_80ep pid=$!"

wait

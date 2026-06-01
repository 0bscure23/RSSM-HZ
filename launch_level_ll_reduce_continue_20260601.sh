#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
mkdir -p output_log /ssd2/lizhy_workspace/tmp/wfanet_tmp
export TMPDIR=/ssd2/lizhy_workspace/tmp/wfanet_tmp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HDF5_USE_FILE_LOCKING=FALSE

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
  --epochs 50
  --batch-size 32
  --num-workers 0
  --val-every 10
  --val-batch-size 8
  --best-metric overall
  --q-win-size 4
  --w-edge 0
  --w-wavelet-hf 0
  --w-kl 0
  --phase-b-freeze-mode head_reduce
  --phase-b-lr-scale 0.08
  --phase-b-ramp-epochs 1
  --no-loss-clamp
  --use-level-ll-corr
  --use-band-corr
  --band-corr-kernel-size 7
  --band-corr-hidden 64
  --save-every 999
)

"${PY_CMD[@]}" --gpu 0 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/gf2_level_ll_reduce_l1_60ep/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/gf2/train_gf2.h5 \
  --val-path Dataset/gf2/valid_gf2.h5 \
  --test-path Dataset/gf2/test_gf2_multiExm1.h5 \
  --w-sam 0 \
  --w-ll 0 \
  --run-tag gf2_level_ll_reduce_l1_cont_50ep \
  > output_log/gf2_level_ll_reduce_l1_cont_20260601.log 2>&1 &
echo "started gf2_level_ll_reduce_l1_cont_50ep pid=$!"

"${PY_CMD[@]}" --gpu 1 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/qb_level_ll_reduce_samll_60ep/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/qb/train_qb.h5 \
  --val-path Dataset/qb/valid_qb.h5 \
  --test-path Dataset/qb/test_qb_multiExm1.h5 \
  --w-sam 0.01 \
  --w-ll 0.02 \
  --run-tag qb_level_ll_reduce_samll_cont_50ep \
  > output_log/qb_level_ll_reduce_samll_cont_20260601.log 2>&1 &
echo "started qb_level_ll_reduce_samll_cont_50ep pid=$!"

"${PY_CMD[@]}" --gpu 2 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/gf2_level_ll_reduce_l1_60ep/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/gf2/train_gf2.h5 \
  --val-path Dataset/gf2/valid_gf2.h5 \
  --test-path Dataset/gf2/test_gf2_multiExm1.h5 \
  --w-sam 0 \
  --w-ll 0 \
  --w-mse 20 \
  --w-band-balanced 0.005 \
  --run-tag gf2_level_ll_reduce_mseband_cont_50ep \
  > output_log/gf2_level_ll_reduce_mseband_cont_20260601.log 2>&1 &
echo "started gf2_level_ll_reduce_mseband_cont_50ep pid=$!"

"${PY_CMD[@]}" --gpu 3 "${COMMON[@]}" \
  --init-ckpt results_rssm_hz/qb_level_ll_reduce_samll_60ep/checkpoints/rssm_hz_best_val.pth \
  --train-path Dataset/qb/train_qb.h5 \
  --val-path Dataset/qb/valid_qb.h5 \
  --test-path Dataset/qb/test_qb_multiExm1.h5 \
  --w-sam 0.01 \
  --w-ll 0.02 \
  --w-mse 20 \
  --w-band-balanced 0.005 \
  --run-tag qb_level_ll_reduce_mseband_cont_50ep \
  > output_log/qb_level_ll_reduce_mseband_cont_20260601.log 2>&1 &
echo "started qb_level_ll_reduce_mseband_cont_50ep pid=$!"

wait

#!/bin/bash
# Phase B launcher - run after Phase A completes
# Usage: bash launch_phaseB.sh <phaseA_run_tag> <gpu> <suffix>

PHASE_A_TAG=${1:-jilin_phaseA_v2}
GPU=${2:-0}
SUFFIX=${3:-phaseB}

PHASE_A_CKPT="results_rssm_hz/${PHASE_A_TAG}/checkpoints/rssm_hz_best.pth"

if [ ! -f "$PHASE_A_CKPT" ]; then
    echo "ERROR: Phase A checkpoint not found: $PHASE_A_CKPT"
    exit 1
fi

echo "Launching Phase B from: $PHASE_A_CKPT"
echo "GPU: $GPU, Tag: ${PHASE_A_TAG}_${SUFFIX}"

nohup python3 -u train_rssm_hz.py \
  --config super_para_panscale.yml \
  --phase b \
  --init-ckpt "$PHASE_A_CKPT" \
  --epochs 160 \
  --batch-size 16 \
  --lr-scale 1.0 \
  --phase-b-lr-scale 0.3 \
  --phase-b-ramp-epochs 80 \
  --phase-b-freeze-mode shallow \
  --w-kl 1e-4 \
  --w-sam 0.10 --w-edge 0.05 --w-wavelet-hf 0.12 \
  --use-ema --ema-decay 0.999 \
  --train-path Dataset/PanScale_H5/jilin/jilin_train_v2.h5 \
  --val-path Dataset/PanScale_H5/jilin/jilin_val_v2.h5 \
  --test-path Dataset/PanScale_H5/jilin/jilin_test200.h5 \
  --val-every 10 --best-metric q8 \
  --gpu "$GPU" --num-workers 4 \
  --run-tag "${PHASE_A_TAG}_${SUFFIX}" \
  > "output_log/${PHASE_A_TAG}_${SUFFIX}.log" 2>&1 &

echo "Phase B PID: $!"
echo "Log: output_log/${PHASE_A_TAG}_${SUFFIX}.log"

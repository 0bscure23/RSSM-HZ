#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash matlab_eval/run_reduced_eval_octave.sh <pack_dir> <out_dir> [data_range] [ratio] [L] [Qblocks] [crop_flag] [crop_dim] [threshold_flag]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${REPO_DIR}/../.." && pwd)"

PACK_DIR="$1"
OUT_DIR="$2"
DATA_RANGE="${3:-2047}"
RATIO="${4:-4}"
L_BITS="${5:-11}"
QBLOCKS="${6:-32}"
CROP_FLAG="${7:-1}"
CROP_DIM="${8:-21}"
THRESHOLD_FLAG="${9:-1}"

OCTAVE_ENV="${OCTAVE_ENV:-${WORKSPACE_DIR}/.conda_envs/octave_eval}"
CONDA_BIN="${CONDA_BIN:-${WORKSPACE_DIR}/anaconda3/bin/conda}"
DLPAN_DIR="${DLPAN_DIR:-${WORKSPACE_DIR}/plp/_external_eval_refs/DLPan-Toolbox/02-Test-toolbox-for-traditional-and-DL(Matlab)}"

"${CONDA_BIN}" run -p "${OCTAVE_ENV}" octave --no-gui --quiet --eval \
  "addpath('${SCRIPT_DIR}'); eval_reduced_matlab('${PACK_DIR}', '${DLPAN_DIR}', '${OUT_DIR}', ${DATA_RANGE}, ${RATIO}, ${L_BITS}, ${QBLOCKS}, ${CROP_FLAG}, ${CROP_DIM}, ${THRESHOLD_FLAG});"

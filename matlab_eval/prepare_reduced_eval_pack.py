#!/usr/bin/env python3
"""Prepare a MATLAB/Octave reduced-resolution evaluation package.

The RSSM-HZ evaluator exports predictions as pred_XX.mat files containing
`sr` in HWC layout. This script exports matching reference files from the
PanCollection-style H5 test set so MATLAB only needs to load .mat pairs.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {x.shape}")
    return np.transpose(x, (1, 2, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="Directory with pred_XX.mat files containing variable sr")
    parser.add_argument("--test-h5", required=True, help="PanCollection-style test H5 with gt/pan/ms/lms")
    parser.add_argument("--out-dir", required=True, help="Output package directory")
    parser.add_argument("--copy-pred", action="store_true", help="Copy prediction mats into out-dir/pred")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    test_h5 = Path(args.test_h5).resolve()
    out_dir = Path(args.out_dir).resolve()
    ref_dir = out_dir / "ref"
    out_pred_dir = out_dir / "pred"
    ref_dir.mkdir(parents=True, exist_ok=True)
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("pred_*.mat"))
    if not pred_files:
        raise FileNotFoundError(f"No pred_*.mat files found in {pred_dir}")

    with h5py.File(test_h5, "r") as h5:
        required = ["gt", "pan", "ms", "lms"]
        missing = [k for k in required if k not in h5]
        if missing:
            raise KeyError(f"Missing keys in {test_h5}: {missing}")

        n = h5["gt"].shape[0]
        if len(pred_files) > n:
            raise ValueError(f"More predictions ({len(pred_files)}) than H5 samples ({n})")

        rows = []
        for i, pred_path in enumerate(pred_files):
            ref_path = ref_dir / f"ref_{i:02d}.mat"
            ref = {
                "gt": chw_to_hwc(h5["gt"][i]).astype(np.float64),
                "pan": chw_to_hwc(h5["pan"][i]).astype(np.float64),
                "ms": chw_to_hwc(h5["ms"][i]).astype(np.float64),
                "lms": chw_to_hwc(h5["lms"][i]).astype(np.float64),
            }
            sio.savemat(ref_path, ref)

            target_pred = out_pred_dir / f"pred_{i:02d}.mat"
            if args.copy_pred:
                shutil.copy2(pred_path, target_pred)
            else:
                if not target_pred.exists():
                    target_pred.symlink_to(pred_path)

            rows.append(
                {
                    "index": i,
                    "pred_mat": f"pred/pred_{i:02d}.mat",
                    "ref_mat": f"ref/ref_{i:02d}.mat",
                }
            )

    with (out_dir / "index_map.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "pred_mat", "ref_mat"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} pairs in {out_dir}")


if __name__ == "__main__":
    main()

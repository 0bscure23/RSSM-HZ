"""
Convert PanScale TIFF dataset to H5 format for RSSM/WFANet training.

Wald's protocol (reduced-resolution):
  - gt  = original high-res MS (ground truth)
  - pan = original high-res PAN
  - ms  = gt downsampled by scale_factor (low-res MS)
  - lms = ms upsampled back to original resolution

For cross-scale subsets (fjilin/flandsat/fskysat) where PAN > MS:
  The original MS is already at lower resolution — we save it as ms,
  upsample it as lms, and save the original PAN as pan. There's no gt.
"""

import argparse
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def bicubic_downsample(img_hwc, scale=4):
    """Downsample HWC uint8 image by integer scale using area interpolation."""
    from skimage.transform import resize
    h, w, c = img_hwc.shape
    new_h, new_w = h // scale, w // scale
    # Use anti-aliased downsampling
    return resize(img_hwc, (new_h, new_w, c), order=1, anti_aliasing=True,
                  preserve_range=True).astype(np.uint8)


def bicubic_upsample(img_hwc, target_shape):
    """Upsample HWC uint8 image to target H,W."""
    from skimage.transform import resize
    h, w = target_shape[:2]
    return resize(img_hwc, (h, w, img_hwc.shape[2]), order=1, anti_aliasing=False,
                  preserve_range=True).astype(np.uint8)


def load_split(ms_dir, pan_dir, num_samples):
    """Load all TIFF pairs from a split directory."""
    ms_list = []
    pan_list = []
    for i in range(1, num_samples + 1):
        ms = np.array(Image.open(os.path.join(ms_dir, f"{i}.tif")))
        pan = np.array(Image.open(os.path.join(pan_dir, f"{i}.tif")))
        ms_list.append(ms)
        pan_list.append(pan)
    return np.stack(ms_list, axis=0), np.stack(pan_list, axis=0)


def create_h5_same_scale(output_path, ms_images, pan_images, scale=4):
    """
    Create H5 file for same-scale data using Wald's protocol.

    ms_images: (N, H, W, C) uint8 — original high-res MS (= gt)
    pan_images: (N, H, W) uint8 — original high-res PAN
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n, h, w, c = ms_images.shape
    h_lr = h // scale
    w_lr = w // scale

    gt_all = np.zeros((n, c, h, w), dtype=np.float32)
    pan_all = np.zeros((n, 1, h, w), dtype=np.float32)
    ms_all = np.zeros((n, c, h_lr, w_lr), dtype=np.float32)
    lms_all = np.zeros((n, c, h, w), dtype=np.float32)

    for i in tqdm(range(n), desc=f"Processing {output_path}"):
        gt = ms_images[i]  # (H, W, C)
        pan = pan_images[i]  # (H, W)

        # Downsample GT to create low-res MS
        ms_lr = bicubic_downsample(gt, scale=scale)  # (H/4, W/4, C)

        # Upsample LR MS back to create LMS
        lms = bicubic_upsample(ms_lr, (h, w))  # (H, W, C)

        gt_all[i] = gt.transpose(2, 0, 1)  # (C, H, W)
        pan_all[i, 0] = pan
        ms_all[i] = ms_lr.transpose(2, 0, 1)
        lms_all[i] = lms.transpose(2, 0, 1)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("gt", data=gt_all)
        h5.create_dataset("pan", data=pan_all)
        h5.create_dataset("ms", data=ms_all)
        h5.create_dataset("lms", data=lms_all)

    print(f"Saved {n} samples to {output_path}: gt={gt_all.shape}, pan={pan_all.shape}, "
          f"ms={ms_all.shape}, lms={lms_all.shape}")


def create_h5_cross_scale(output_path, ms_images, pan_images):
    """
    Create H5 file for cross-scale data (no ground truth).
    PAN is larger than MS by the scale ratio.

    ms_images: (N, H_ms, W_ms, C) uint8
    pan_images: (N, H_pan, W_pan) uint8
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n = ms_images.shape[0]
    c = ms_images.shape[3]
    h_pan, w_pan = pan_images.shape[1], pan_images.shape[2]
    h_ms, w_ms = ms_images.shape[1], ms_images.shape[2]

    pan_all = np.zeros((n, 1, h_pan, w_pan), dtype=np.float32)
    ms_all = np.zeros((n, c, h_ms, w_ms), dtype=np.float32)
    lms_all = np.zeros((n, c, h_pan, w_pan), dtype=np.float32)

    for i in tqdm(range(n), desc=f"Processing {output_path}"):
        ms = ms_images[i]  # (H_ms, W_ms, C)
        pan = pan_images[i]  # (H_pan, W_pan)

        # Upsample MS to PAN resolution for LMS
        lms = bicubic_upsample(ms, (h_pan, w_pan))

        pan_all[i, 0] = pan
        ms_all[i] = ms.transpose(2, 0, 1)
        lms_all[i] = lms.transpose(2, 0, 1)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("pan", data=pan_all)
        h5.create_dataset("ms", data=ms_all)
        h5.create_dataset("lms", data=lms_all)

    print(f"Saved {n} samples to {output_path}: pan={pan_all.shape}, "
          f"ms={ms_all.shape}, lms={lms_all.shape}")


def convert_subset(panscale_root, subset, out_dir):
    """Convert one PanScale subset to H5."""
    src_dir = os.path.join(panscale_root, subset)
    if not os.path.isdir(src_dir):
        print(f"Skipping {subset}: not found at {src_dir}")
        return

    out_subdir = os.path.join(out_dir, subset)
    os.makedirs(out_subdir, exist_ok=True)

    is_cross_scale = subset.startswith("f")  # fjilin, flandsat, fskysat

    splits = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    for split_name in sorted(splits):
        split_dir = os.path.join(src_dir, split_name)
        ms_dir = os.path.join(split_dir, "ms")
        pan_dir = os.path.join(split_dir, "pan")

        if not os.path.isdir(ms_dir) or not os.path.isdir(pan_dir):
            print(f"  Skipping {split_name}: ms/ or pan/ missing")
            continue

        num_samples = len(os.listdir(ms_dir))
        print(f"\nLoading {subset}/{split_name}: {num_samples} pairs...")
        ms_images, pan_images = load_split(ms_dir, pan_dir, num_samples)

        out_path = os.path.join(out_subdir, f"{subset}_{split_name}.h5")

        if is_cross_scale:
            create_h5_cross_scale(out_path, ms_images, pan_images)
        else:
            create_h5_same_scale(out_path, ms_images, pan_images, scale=4)


def main():
    parser = argparse.ArgumentParser(description="Convert PanScale TIFF to H5")
    parser.add_argument("--panscale-root", default="Dataset/PanScale")
    parser.add_argument("--out-dir", default="Dataset/PanScale_H5")
    parser.add_argument("--subsets", nargs="+", default=["jilin", "landsat", "skysat",
                                                          "fjilin", "flandsat", "fskysat"])
    args = parser.parse_args()

    for subset in args.subsets:
        convert_subset(args.panscale_root, subset, args.out_dir)

    print("\nDone. H5 files saved to", args.out_dir)


if __name__ == "__main__":
    main()

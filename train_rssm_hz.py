import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.io import savemat
from torch.utils.data import DataLoader, Dataset

from evaluate_wv3_metrics import calculate_metrics
from rssm_hz_wfanet import RSSMHWViTHZ
from net_torch import DWT_2D, HWViT


class PanDataset(Dataset):
    def __init__(self, path, ratio=2047.0, max_samples=None):
        super().__init__()
        with h5py.File(path, "r") as h5:
            gt = h5["gt"][:]
            pan = h5["pan"][:]
            ms = h5["ms"][:]
            lms = h5["lms"][:]

        if max_samples is not None:
            gt = gt[:max_samples]
            pan = pan[:max_samples]
            ms = ms[:max_samples]
            lms = lms[:max_samples]

        self.gt = gt
        self.pan = pan
        self.ms = ms
        self.lms = lms
        self.ratio = ratio

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, idx):
        gt = torch.from_numpy(self.gt[idx] / self.ratio).float()
        pan = torch.from_numpy(self.pan[idx] / self.ratio).float()
        ms = torch.from_numpy(self.ms[idx] / self.ratio).float()
        lms = torch.from_numpy(self.lms[idx] / self.ratio).float()
        return gt, pan, ms, lms


class RandomCropPanDataset(PanDataset):
    def __init__(self, path, ratio=2047.0, max_samples=None, crop_size=0, crop_align=1, repeat=4):
        super().__init__(path, ratio=ratio, max_samples=max_samples)
        self.crop_size = int(crop_size)
        self.crop_align = max(1, int(crop_align))
        self.repeat = max(1, int(repeat))
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive for RandomCropPanDataset")
        if self.gt.shape[-1] != self.pan.shape[-1] or self.gt.shape[-2] != self.pan.shape[-2]:
            raise ValueError("gt and pan spatial sizes must match for random cropping")
        self.scale = int(round(self.gt.shape[-1] / self.ms.shape[-1]))
        if self.scale <= 0 or self.gt.shape[-1] % self.ms.shape[-1] != 0:
            raise ValueError("gt/ms spatial ratio must be an integer")
        if self.crop_size % self.scale != 0:
            raise ValueError(f"crop_size must be divisible by scale={self.scale}")

    def __len__(self):
        return self.gt.shape[0] * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % self.gt.shape[0]
        _, h, w = self.gt[base_idx].shape
        if self.crop_size > h or self.crop_size > w:
            raise ValueError(f"crop_size={self.crop_size} exceeds sample size {(h, w)}")

        max_y = h - self.crop_size
        max_x = w - self.crop_size
        y = np.random.randint(0, max_y // self.crop_align + 1) * self.crop_align
        x = np.random.randint(0, max_x // self.crop_align + 1) * self.crop_align
        ms_y = y // self.scale
        ms_x = x // self.scale
        ms_size = self.crop_size // self.scale

        gt = self.gt[base_idx, :, y:y + self.crop_size, x:x + self.crop_size]
        pan = self.pan[base_idx, :, y:y + self.crop_size, x:x + self.crop_size]
        ms = self.ms[base_idx, :, ms_y:ms_y + ms_size, ms_x:ms_x + ms_size]
        lms = self.lms[base_idx, :, y:y + self.crop_size, x:x + self.crop_size]
        return (
            torch.from_numpy(gt / self.ratio).float(),
            torch.from_numpy(pan / self.ratio).float(),
            torch.from_numpy(ms / self.ratio).float(),
            torch.from_numpy(lms / self.ratio).float(),
        )


def ssim_loss(pred, gt, win_size=11, data_range=1.0):
    """Differentiable SSIM loss (1 - SSIM)."""
    from torch.nn.functional import conv2d
    C = pred.shape[1]
    # 1D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(win_size, device=pred.device, dtype=pred.dtype) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_1d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)
    kernel = kernel_1d.expand(C, 1, win_size, win_size)

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu_p = conv2d(pred, kernel, padding=win_size // 2, groups=C)
    mu_g = conv2d(gt, kernel, padding=win_size // 2, groups=C)
    mu_pp = mu_p * mu_p
    mu_gg = mu_g * mu_g
    mu_pg = mu_p * mu_g

    sigma_pp = conv2d(pred * pred, kernel, padding=win_size // 2, groups=C) - mu_pp
    sigma_gg = conv2d(gt * gt, kernel, padding=win_size // 2, groups=C) - mu_gg
    sigma_pg = conv2d(pred * gt, kernel, padding=win_size // 2, groups=C) - mu_pg

    ssim_map = ((2 * mu_pg + C1) * (2 * sigma_pg + C2)) / (
        (mu_pp + mu_gg + C1) * (sigma_pp + sigma_gg + C2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


def sobel_edges(x):
    c = x.shape[1]
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    ex = F.conv2d(x, kx, padding=1, groups=c)
    ey = F.conv2d(x, ky, padding=1, groups=c)
    return torch.sqrt(ex * ex + ey * ey + 1e-8)


def sam_loss(pred, gt):
    b, c, h, w = pred.shape
    p = pred.permute(0, 2, 3, 1).reshape(-1, c)
    g = gt.permute(0, 2, 3, 1).reshape(-1, c)
    dot = torch.sum(p * g, dim=1)
    np_ = torch.norm(p, dim=1)
    ng = torch.norm(g, dim=1)
    cosang = torch.clamp(dot / (np_ * ng + 1e-8), -1.0, 1.0)
    ang = torch.acos(cosang)
    return torch.nan_to_num(ang.mean(), nan=0.0, posinf=0.0, neginf=0.0)


def band_balanced_l1_loss(pred, gt):
    """Per-band normalized L1 to keep hard spectral bands from being washed out."""
    diff = (pred - gt).abs().mean(dim=(0, 2, 3))
    scale = gt.detach().std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-3)
    return (diff / scale).mean()


def ms_fidelity_loss(pred, ms):
    """Reduced-resolution consistency: HRMS prediction should downsample to LRMS."""
    pred_lr = F.adaptive_avg_pool2d(pred, ms.shape[-2:])
    return F.l1_loss(pred_lr, ms)


def pan_fidelity_loss(pred, pan):
    """Simple PAN consistency using the mean multispectral intensity proxy."""
    pred_pan = pred.mean(dim=1, keepdim=True)
    return F.l1_loss(pred_pan, pan)


def psnr_tensor(pred, gt, data_range=1.0):
    mse = torch.mean((pred - gt) ** 2).item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10((data_range ** 2) / mse)


def ergas_tensor(pred, gt, ratio=4.0):
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2, dim=(0, 2, 3)))
    mean_gt = torch.mean(gt, dim=(0, 2, 3)).abs().clamp_min(1e-8)
    term = (rmse / mean_gt) ** 2
    return (100.0 / ratio) * torch.sqrt(torch.mean(term)).item()


def q8_numpy(pred_hwc, gt_hwc, win_size=8):
    from scipy.ndimage import uniform_filter

    pred = pred_hwc.astype(np.float64)
    gt = gt_hwc.astype(np.float64)
    b = pred.shape[2]

    q_maps = []
    for i in range(b):
        x = pred[:, :, i]
        y = gt[:, :, i]

        ux = uniform_filter(x, size=win_size, mode="reflect")
        uy = uniform_filter(y, size=win_size, mode="reflect")
        vx = uniform_filter(x * x, size=win_size, mode="reflect") - ux * ux
        vy = uniform_filter(y * y, size=win_size, mode="reflect") - uy * uy
        cov = uniform_filter(x * y, size=win_size, mode="reflect") - ux * uy

        num = 4.0 * cov * ux * uy
        den = (vx + vy) * (ux * ux + uy * uy) + 1e-12
        q_maps.append(num / den)

    return float(np.mean(np.stack(q_maps, axis=2)))


def wavelet_hf_loss(pred, gt, dwt_module, l1_loss):
    pred_w = dwt_module(pred)
    gt_w = dwt_module(gt)

    c = pred.shape[1]
    pred_lh = pred_w[:, c:2 * c]
    pred_hl = pred_w[:, 2 * c:3 * c]
    pred_hh = pred_w[:, 3 * c:4 * c]

    gt_lh = gt_w[:, c:2 * c]
    gt_hl = gt_w[:, 2 * c:3 * c]
    gt_hh = gt_w[:, 3 * c:4 * c]

    return (l1_loss(pred_lh, gt_lh) + l1_loss(pred_hl, gt_hl) + l1_loss(pred_hh, gt_hh)) / 3.0


def wavelet_hf_loss_multilevel(pred, gt, dwt_module, l1_loss, level_weights):
    pred_cur = pred
    gt_cur = gt
    loss = pred.new_tensor(0.0)
    weight_sum = 0.0

    for weight in level_weights:
        if weight <= 0:
            pred_cur = dwt_module(pred_cur)[:, : pred_cur.shape[1]]
            gt_cur = dwt_module(gt_cur)[:, : gt_cur.shape[1]]
            continue

        pred_w = dwt_module(pred_cur)
        gt_w = dwt_module(gt_cur)
        c = pred_cur.shape[1]

        pred_lh = pred_w[:, c:2 * c]
        pred_hl = pred_w[:, 2 * c:3 * c]
        pred_hh = pred_w[:, 3 * c:4 * c]

        gt_lh = gt_w[:, c:2 * c]
        gt_hl = gt_w[:, 2 * c:3 * c]
        gt_hh = gt_w[:, 3 * c:4 * c]

        level_loss = (l1_loss(pred_lh, gt_lh) + l1_loss(pred_hl, gt_hl) + l1_loss(pred_hh, gt_hh)) / 3.0
        loss = loss + float(weight) * level_loss
        weight_sum += float(weight)

        pred_cur = pred_w[:, :c]
        gt_cur = gt_w[:, :c]

    if weight_sum <= 0:
        return pred.new_tensor(0.0)
    return loss / weight_sum


def wavelet_ll_loss_multilevel(pred, gt, dwt_module, l1_loss, level_weights):
    pred_cur = pred
    gt_cur = gt
    loss = pred.new_tensor(0.0)
    weight_sum = 0.0

    for weight in level_weights:
        pred_w = dwt_module(pred_cur)
        gt_w = dwt_module(gt_cur)
        c = pred_cur.shape[1]

        pred_ll = pred_w[:, :c]
        gt_ll = gt_w[:, :c]

        if weight > 0:
            loss = loss + float(weight) * l1_loss(pred_ll, gt_ll)
            weight_sum += float(weight)

        pred_cur = pred_ll
        gt_cur = gt_ll

    if weight_sum <= 0:
        return pred.new_tensor(0.0)
    return loss / weight_sum


def save_prediction_mats(pred_scaled, out_dir):
    pred_dir = os.path.join(out_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)

    pred_np = pred_scaled.numpy().transpose(0, 2, 3, 1)
    for i in range(pred_np.shape[0]):
        savemat(os.path.join(pred_dir, f"pred_{i:02d}.mat"), {"sr": pred_np[i]})


def aggregate_z_diagnostics(records):
    if not records:
        return {}

    by_level = {}
    for record in records:
        level = int(record.get("level", -1))
        by_level.setdefault(level, []).append(record)

    per_level = []
    skip_keys = {"level", "force_zero_z"}
    for level in sorted(by_level.keys()):
        items = by_level[level]
        summary = {
            "level": int(level),
            "num_batches": int(len(items)),
            "force_zero_z": bool(any(item.get("force_zero_z", False) for item in items)),
        }
        numeric_keys = sorted(
            key
            for key in items[0].keys()
            if key not in skip_keys and isinstance(items[0].get(key), (int, float))
        )
        for key in numeric_keys:
            values = [float(item[key]) for item in items if key in item]
            if values:
                summary[key] = float(np.mean(values))
        per_level.append(summary)

    return {
        "num_records": int(len(records)),
        "per_level": per_level,
    }


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = {}
        for k, v in state_dict.items():
            shadow = v.detach().clone()
            if device is not None:
                shadow = shadow.to(device)
            self.shadow[k] = shadow

    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def load_state_dict_flexible(model, state_dict):
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_in_model"))
            continue
        if model_state[key].shape != value.shape:
            # Backward compatibility: older gates used
            # [fused_ll, ll_ms, pan_lh, pan_hl, pan_hh] = 5 * C inputs.
            # Newer gates append z_gate, so we copy the old weights and keep
            # the z_gate slice at zero. This preserves old checkpoints exactly.
            if (
                "high_gate" in key
                and key.endswith(".weight")
                and value.ndim == 4
                and model_state[key].ndim == 4
                and value.shape[0] == model_state[key].shape[0]
                and value.shape[2:] == model_state[key].shape[2:]
                and value.shape[1] < model_state[key].shape[1]
            ):
                padded = torch.zeros_like(model_state[key])
                padded[:, : value.shape[1], :, :] = value
                filtered_state[key] = padded
                continue
            skipped.append((key, f"shape_mismatch {tuple(value.shape)} -> {tuple(model_state[key].shape)}"))
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    return missing, unexpected, skipped


def apply_phase_b_freeze(model, freeze_mode):
    if freeze_mode == "none":
        return []

    if freeze_mode == "fusion_only":
        for param in model.parameters():
            param.requires_grad = False
        if getattr(model, "use_wfanet_two_stage", False):
            trainable_modules = [
                ("rssm_fusion.coarse_stage", model.rssm_fusion.coarse_stage),
                ("rssm_fusion.fine_stage", model.rssm_fusion.fine_stage),
                ("rssm_fusion.state_up_h", model.rssm_fusion.state_up_h),
                ("rssm_fusion.state_up_z", model.rssm_fusion.state_up_z),
                ("rssm_fusion.final_combine", model.rssm_fusion.final_combine),
                ("rssm_fusion.final_refine", model.rssm_fusion.final_refine),
            ]
        else:
            trainable_modules = [
                ("rssm_fusion.conv_fusion_lh", model.rssm_fusion.conv_fusion_lh),
                ("rssm_fusion.conv_fusion_hl", model.rssm_fusion.conv_fusion_hl),
                ("rssm_fusion.conv_fusion_hh", model.rssm_fusion.conv_fusion_hh),
            ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        return [name for name, _ in trainable_modules]

    if freeze_mode == "shallow":
        frozen_modules = [
            ("pan_raise", model.pan_raise),
            ("ms_upsample", model.ms_upsample),
            ("ms_act", model.ms_act),
            ("ms_raise", model.ms_raise),
        ]
        for _, module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        return [name for name, _ in frozen_modules]

    if getattr(model, "use_wfanet_two_stage", False):
        fusion = model.rssm_fusion
        lowfreq_modules = []
        if getattr(model, "lowfreq_corr", None) is not None:
            lowfreq_modules.append(("lowfreq_corr", model.lowfreq_corr))
        if getattr(model, "band_corr", None) is not None:
            lowfreq_modules.append(("band_corr", model.band_corr))
        if getattr(model, "channel_dwt_adapter", None) is not None:
            lowfreq_modules.append(("channel_dwt_adapter", model.channel_dwt_adapter))

        stage_gate_modules = []
        for stage_name, stage in (("coarse_stage", fusion.coarse_stage), ("fine_stage", fusion.fine_stage)):
            stage_gate_modules.append((f"rssm_fusion.{stage_name}.pan_high_to_ms", stage.pan_high_to_ms))
            stage_gate_modules.append((f"rssm_fusion.{stage_name}.z_to_gate", stage.z_to_gate))
            if getattr(stage, "separate_subband_gates", False):
                stage_gate_modules.extend([
                    (f"rssm_fusion.{stage_name}.high_gate_lh", stage.high_gate_lh),
                    (f"rssm_fusion.{stage_name}.high_gate_hl", stage.high_gate_hl),
                    (f"rssm_fusion.{stage_name}.high_gate_hh", stage.high_gate_hh),
                ])
            else:
                stage_gate_modules.append((f"rssm_fusion.{stage_name}.high_gate", stage.high_gate))

        def _freeze_all_then_enable(trainable_modules):
            for param in model.parameters():
                param.requires_grad = False
            for _, module in trainable_modules:
                for param in module.parameters():
                    param.requires_grad = True
            if hasattr(model, "fused_weight") and freeze_mode in {"head_only", "head_reduce", "gate_head_reduce"}:
                model.fused_weight.requires_grad = True
            names = [name for name, _ in trainable_modules]
            if hasattr(model, "fused_weight") and freeze_mode in {"head_only", "head_reduce", "gate_head_reduce"}:
                names.append("fused_weight")
            return names

        if freeze_mode == "head_only":
            return _freeze_all_then_enable([
                *lowfreq_modules,
                ("out_act", model.out_act),
            ])
        if freeze_mode == "head_reduce":
            return _freeze_all_then_enable([
                ("reduce", model.reduce),
                *lowfreq_modules,
                ("out_act", model.out_act),
            ])
        if freeze_mode == "gate_head_reduce":
            return _freeze_all_then_enable([
                *stage_gate_modules,
                ("reduce", model.reduce),
                *lowfreq_modules,
                ("out_act", model.out_act),
            ])
        if freeze_mode == "state_gate_head":
            return _freeze_all_then_enable([
                ("rssm_fusion.coarse_stage.state_fusion", fusion.coarse_stage.state_fusion),
                ("rssm_fusion.fine_stage.state_fusion", fusion.fine_stage.state_fusion),
                ("rssm_fusion.state_up_h", fusion.state_up_h),
                ("rssm_fusion.state_up_z", fusion.state_up_z),
                *stage_gate_modules,
                ("reduce", model.reduce),
                *lowfreq_modules,
                ("out_act", model.out_act),
            ])
        if freeze_mode == "state_high":
            return _freeze_all_then_enable([
                ("rssm_fusion.coarse_stage", fusion.coarse_stage),
                ("rssm_fusion.fine_stage", fusion.fine_stage),
                ("rssm_fusion.state_up_h", fusion.state_up_h),
                ("rssm_fusion.state_up_z", fusion.state_up_z),
                ("rssm_fusion.final_combine", fusion.final_combine),
                ("rssm_fusion.final_refine", fusion.final_refine),
                ("reduce", model.reduce),
                *lowfreq_modules,
                ("out_act", model.out_act),
            ])

    # Auto-detect gate module names based on model configuration.
    fusion = model.rssm_fusion
    if hasattr(fusion, 'separate_subband_gates') and fusion.separate_subband_gates:
        gate_modules = [
            ("rssm_fusion.high_gate_lh", fusion.high_gate_lh),
            ("rssm_fusion.high_gate_hl", fusion.high_gate_hl),
            ("rssm_fusion.high_gate_hh", fusion.high_gate_hh),
        ]
    else:
        gate_modules = [
            ("rssm_fusion.high_gate", fusion.high_gate),
        ]

    # Include ConvFusion modules when learnable fusion is enabled.
    fusion_modules = []
    if hasattr(fusion, 'learnable_fusion') and fusion.learnable_fusion:
        fusion_modules = [
            ("rssm_fusion.conv_fusion_lh", fusion.conv_fusion_lh),
            ("rssm_fusion.conv_fusion_hl", fusion.conv_fusion_hl),
            ("rssm_fusion.conv_fusion_hh", fusion.conv_fusion_hh),
        ]
        if getattr(fusion, "residual_learnable_fusion", False):
            fusion_modules.extend([
                ("rssm_fusion.conv_fusion_beta_lh", fusion.conv_fusion_beta_lh),
                ("rssm_fusion.conv_fusion_beta_hl", fusion.conv_fusion_beta_hl),
                ("rssm_fusion.conv_fusion_beta_hh", fusion.conv_fusion_beta_hh),
            ])

    if hasattr(model, 'image_space_wavelet') and model.image_space_wavelet:
        img_wav_modules = [
            ("pan_subband_raise", model.pan_subband_raise),
            ("ms_subband_raise", model.ms_subband_raise),
        ]
    else:
        img_wav_modules = []

    lowfreq_modules = []
    if getattr(model, "lowfreq_corr", None) is not None:
        lowfreq_modules.append(("lowfreq_corr", model.lowfreq_corr))
    if getattr(model, "band_corr", None) is not None:
        lowfreq_modules.append(("band_corr", model.band_corr))
    if getattr(model, "channel_dwt_adapter", None) is not None:
        lowfreq_modules.append(("channel_dwt_adapter", model.channel_dwt_adapter))
    for idx, block in enumerate(getattr(fusion, "fusion_blocks", [])):
        if getattr(block, "level_ll_corr", None) is not None:
            lowfreq_modules.append((f"rssm_fusion.fusion_blocks.{idx}.level_ll_corr", block.level_ll_corr))

    freq_modules = []
    if getattr(fusion, "use_local_freq_mixer", False):
        freq_modules.extend([
            ("rssm_fusion.local_mixer_lh", fusion.local_mixer_lh),
            ("rssm_fusion.local_mixer_hl", fusion.local_mixer_hl),
            ("rssm_fusion.local_mixer_hh", fusion.local_mixer_hh),
        ])
    if getattr(fusion, "use_windowed_freq_mixer", False):
        freq_modules.extend([
            ("rssm_fusion.window_mixer_lh", fusion.window_mixer_lh),
            ("rssm_fusion.window_mixer_hl", fusion.window_mixer_hl),
            ("rssm_fusion.window_mixer_hh", fusion.window_mixer_hh),
        ])
    if getattr(fusion, "use_mamba_freq_mixer", False):
        freq_modules.extend([
            ("rssm_fusion.mamba_mixer_lh", fusion.mamba_mixer_lh),
            ("rssm_fusion.mamba_mixer_hl", fusion.mamba_mixer_hl),
            ("rssm_fusion.mamba_mixer_hh", fusion.mamba_mixer_hh),
        ])

    if freeze_mode == "state_high":
        for param in model.parameters():
            param.requires_grad = False

        trainable_modules = [
            ("rssm_fusion.fusion_blocks", model.rssm_fusion.fusion_blocks),
            ("rssm_fusion.state_up_h", model.rssm_fusion.state_up_h),
            ("rssm_fusion.state_up_z", model.rssm_fusion.state_up_z),
            ("rssm_fusion.pan_high_to_ms", model.rssm_fusion.pan_high_to_ms),
            *gate_modules,
            *fusion_modules,
            ("rssm_fusion.z_to_gate", model.rssm_fusion.z_to_gate),
            *freq_modules,
            *img_wav_modules,
            ("reduce", model.reduce),
            *lowfreq_modules,
            ("out_act", model.out_act),
        ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        return [name for name, _ in trainable_modules]

    if freeze_mode == "state_gate_head":
        for param in model.parameters():
            param.requires_grad = False

        trainable_modules = [
            ("rssm_fusion.fusion_blocks", model.rssm_fusion.fusion_blocks),
            ("rssm_fusion.state_up_h", model.rssm_fusion.state_up_h),
            ("rssm_fusion.state_up_z", model.rssm_fusion.state_up_z),
            *gate_modules,
            *fusion_modules,
            ("rssm_fusion.z_to_gate", model.rssm_fusion.z_to_gate),
            *freq_modules,
            ("reduce", model.reduce),
            *lowfreq_modules,
            ("out_act", model.out_act),
        ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        return [name for name, _ in trainable_modules]

    if freeze_mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False

        trainable_modules = [
            *lowfreq_modules,
            ("out_act", model.out_act),
        ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        if hasattr(model, "fused_weight"):
            model.fused_weight.requires_grad = True
        names = [name for name, _ in trainable_modules]
        if hasattr(model, "fused_weight"):
            names.append("fused_weight")
        return names

    if freeze_mode == "head_reduce":
        for param in model.parameters():
            param.requires_grad = False

        trainable_modules = [
            ("reduce", model.reduce),
            *lowfreq_modules,
            ("out_act", model.out_act),
        ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        if hasattr(model, "fused_weight"):
            model.fused_weight.requires_grad = True
        names = [name for name, _ in trainable_modules]
        if hasattr(model, "fused_weight"):
            names.append("fused_weight")
        return names

    if freeze_mode == "gate_head_reduce":
        for param in model.parameters():
            param.requires_grad = False

        trainable_modules = [
            ("rssm_fusion.pan_high_to_ms", model.rssm_fusion.pan_high_to_ms),
            *gate_modules,
            ("rssm_fusion.z_to_gate", model.rssm_fusion.z_to_gate),
            *freq_modules,
            ("reduce", model.reduce),
            *lowfreq_modules,
            ("out_act", model.out_act),
        ]
        for _, module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        if hasattr(model, "fused_weight"):
            model.fused_weight.requires_grad = True
        names = [name for name, _ in trainable_modules]
        if hasattr(model, "fused_weight"):
            names.append("fused_weight")
        return names

    raise ValueError(f"Unsupported freeze mode: {freeze_mode}")


def evaluate_dataset(
    model,
    config,
    device,
    dataset_path,
    out_dir=None,
    max_samples=None,
    eval_clamp=False,
    export_preds=False,
    batch_size=1,
    num_workers=0,
    q_win_size=8,
    tile_size=0,
    tile_overlap=0,
    results_filename="rssm_hz_results.mat",
    metrics_filename="rssm_hz_metrics.json",
    collect_z_diagnostics=False,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset: {dataset_path}")

    eval_ds = PanDataset(dataset_path, ratio=float(config["ratio"]), max_samples=max_samples)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    if collect_z_diagnostics and hasattr(model, "set_z_diagnostics"):
        model.set_z_diagnostics(True)

    preds = []
    gts = []
    z_diag_records = []
    with torch.no_grad():
        for gt, pan, ms, lms in eval_loader:
            out, _, _ = tiled_forward(
                model,
                pan.to(device, non_blocking=True),
                ms.to(device, non_blocking=True),
                lms.to(device, non_blocking=True),
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            if collect_z_diagnostics and hasattr(model, "get_z_diagnostics"):
                z_diag_records.extend(model.get_z_diagnostics())
            preds.append(out.cpu())
            gts.append(gt.cpu())

    if collect_z_diagnostics and hasattr(model, "set_z_diagnostics"):
        model.set_z_diagnostics(False)

    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)

    # Keep evaluation numerically valid even if a checkpoint contains rare non-finite outputs.
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    if eval_clamp:
        pred = pred.clamp(0.0, 1.0)

    pred_scaled = pred * float(config["ratio"])
    gt_scaled = gt * float(config["ratio"])

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        savemat(os.path.join(out_dir, results_filename), {"results": pred_scaled.numpy()})
        if export_preds:
            save_prediction_mats(pred_scaled, out_dir)

    eval_metrics = calculate_metrics(
        pred_scaled.numpy(),
        gt_scaled.numpy(),
        ratio=4.0,
        data_range=float(config["ratio"]),
        q_win_size=int(q_win_size),
    )

    metrics = {
        "PSNR": float(eval_metrics["PSNR"]),
        "PSNR_global": float(eval_metrics["PSNR_global"]),
        "SAM": float(eval_metrics["SAM"]),
        "ERGAS": float(eval_metrics["ERGAS"]),
        "Q": float(eval_metrics["Q"]),
        "Q8": float(eval_metrics["Q"]),
        f"Q{int(q_win_size)}": float(eval_metrics["Q"]),
        "q_win_size": int(q_win_size),
        "num_samples": int(pred_scaled.shape[0]),
        "eval_clamp": bool(eval_clamp),
        "eval_tile_size": int(tile_size),
        "eval_tile_overlap": int(tile_overlap),
    }

    z_diagnostics = aggregate_z_diagnostics(z_diag_records)
    if z_diagnostics:
        metrics["z_diagnostics"] = z_diagnostics

    if out_dir is not None:
        with open(os.path.join(out_dir, metrics_filename), "w") as f:
            json.dump(metrics, f, indent=2)
        if z_diagnostics:
            with open(os.path.join(out_dir, "rssm_hz_z_diagnostics.json"), "w") as f:
                json.dump(z_diagnostics, f, indent=2)

    return metrics


def _tile_starts(length, tile_size, step):
    if tile_size >= length:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tiled_forward(model, pan, ms, lms, tile_size, tile_overlap):
    """Run a full-resolution sample through the model by high-res tiles.

    GF2/QB train/val patches are 64x64 while their test samples are 256x256.
    Tiled inference lets us test whether RSSM-HZ prefers the training spatial
    scale without changing the learned model. Tile starts are kept /4 aligned
    so PAN/LMS crops match the corresponding LRMS crop exactly.
    """
    _, _, height, width = pan.shape
    if tile_size <= 0 or (tile_size >= height and tile_size >= width):
        out, kl, z_residuals = model(pan, ms, lms)
        return out, kl, z_residuals
    if tile_size % 4 != 0 or tile_overlap % 4 != 0:
        raise ValueError("--eval-tile-size and --eval-tile-overlap must be divisible by 4")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("--eval-tile-overlap must satisfy 0 <= overlap < tile_size")

    step = tile_size - tile_overlap
    ys = _tile_starts(height, tile_size, step)
    xs = _tile_starts(width, tile_size, step)
    accum = pan.new_zeros(pan.shape[0], lms.shape[1], height, width)
    weight = pan.new_zeros(1, 1, height, width)
    kl_terms = []

    for y in ys:
        y2 = y + tile_size
        my, my2 = y // 4, y2 // 4
        for x in xs:
            x2 = x + tile_size
            mx, mx2 = x // 4, x2 // 4
            out_tile, kl_tile, _ = model(
                pan[:, :, y:y2, x:x2],
                ms[:, :, my:my2, mx:mx2],
                lms[:, :, y:y2, x:x2],
            )
            accum[:, :, y:y2, x:x2] += out_tile
            weight[:, :, y:y2, x:x2] += 1.0
            kl_terms.append(kl_tile.detach())

    out = accum / weight.clamp_min(1.0)
    kl = torch.stack(kl_terms).mean() if kl_terms else out.new_tensor(0.0)
    return out, kl, None


def evaluate(model, config, device, out_dir, test_path=None, max_test_samples=None, eval_clamp=False,
             export_preds=False, q_win_size=8, collect_z_diagnostics=False, tile_size=0, tile_overlap=0):
    if test_path is None:
        test_path = os.path.join("Dataset", "WV3", "test_wv3_multiExm1.h5")
    return evaluate_dataset(
        model,
        config,
        device,
        dataset_path=test_path,
        out_dir=out_dir,
        max_samples=max_test_samples,
        eval_clamp=eval_clamp,
        export_preds=export_preds,
        q_win_size=q_win_size,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        collect_z_diagnostics=collect_z_diagnostics,
        batch_size=1,
        num_workers=0,
    )


def compute_metric_score(metrics, args):
    if args.best_metric == "loss":
        raise ValueError("loss-based selection should be handled outside compute_metric_score")
    if args.best_metric == "psnr":
        return float(metrics["PSNR"])
    if args.best_metric == "q8":
        return float(metrics["Q8"])
    if args.best_metric == "sam":
        return -float(metrics["SAM"])
    if args.best_metric == "ergas":
        return -float(metrics["ERGAS"])
    if args.best_metric == "overall":
        return (
            args.best_psnr_weight * float(metrics["PSNR"])
            + args.best_q8_weight * float(metrics["Q8"])
            - args.best_sam_weight * float(metrics["SAM"])
            - args.best_ergas_weight * float(metrics["ERGAS"])
        )
    raise ValueError(f"Unsupported best_metric: {args.best_metric}")


def build_checkpoint_state(epoch, model, optimizer, config, args, extra=None, ema=None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "args": vars(args),
    }
    if ema is not None:
        state["model_ema"] = ema.state_dict()
    if extra:
        state.update(extra)
    return state


def main():
    parser = argparse.ArgumentParser(description="Train/eval improved RSSM+WFANet (h+z)")
    parser.add_argument("--config", default="super_para.yml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--state-mode", choices=["h", "hz"], default="hz", help="h: deterministic only; hz: deterministic+stochastic")
    parser.add_argument("--phase", choices=["single", "a", "b"], default="single", help="a: deterministic pretrain, b: stochastic finetune")
    parser.add_argument("--init-ckpt", default=None, help="Load model weights from checkpoint before training")
    parser.add_argument("--resume-ckpt", default=None, help="Resume model+optimizer from checkpoint")
    parser.add_argument("--phase-b-lr-scale", type=float, default=0.3, help="LR scale used in phase b")
    parser.add_argument("--phase-b-ramp-epochs", type=int, default=80, help="KL warmup epochs in phase b")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Global LR scale for cautious finetuning")
    parser.add_argument(
        "--phase-b-freeze-mode",
        choices=["none", "shallow", "state_high", "state_gate_head", "fusion_only", "head_only", "head_reduce", "gate_head_reduce"],
        default="none",
        help="Optional parameter freezing strategy used in phase b",
    )
    parser.add_argument("--separate-subband-gates", action="store_true", default=True,
                        help="Use per-subband (LH/HL/HH) high-frequency gates")
    parser.add_argument("--shared-high-gate", action="store_true", default=False,
                        help="Use a single shared high-frequency gate (overrides --separate-subband-gates)")
    parser.add_argument("--use-conv-gru", action="store_true", default=False,
                        help="Use 2D ConvGRU instead of per-pixel GRUCell for state updates")
    parser.add_argument(
        "--state-conv-type",
        choices=[
            "plain",
            "dw_large",
            "convnext_dw",
            "ms_dilated",
            "deformable",
            "window_attn",
            "dw_window_attn",
            "swin_window_attn",
        ],
        default="plain",
        help="Convolution operator inside ConvGRU gates/candidate",
    )
    parser.add_argument(
        "--state-kernel-size",
        type=int,
        default=3,
        help="Kernel size for plain/dw_large/convnext_dw ConvGRU operators",
    )
    parser.add_argument(
        "--freq-state-mode",
        choices=["mixed", "simple", "split"],
        default="mixed",
        help=(
            "mixed: old PAN_LL/LH/HL/HH concat state; "
            "simple: LL updates state and HF modulates ConvGRU gates; "
            "split: independent LL/LH/HL/HH recurrent fusion blocks"
        ),
    )
    parser.add_argument("--learnable-fusion", action="store_true", default=False,
                        help="Replace additive PAN injection with learnable ConvFusion blocks")
    parser.add_argument("--residual-learnable-fusion", action="store_true", default=False,
                        help="Use ConvFusion as a zero-init residual correction on top of gated PAN injection")
    parser.add_argument("--signed-hf-gate", action="store_true", default=False,
                        help="Add a signed correction to the pretrained positive high-frequency gate")
    parser.add_argument("--hf-gate-scale", type=float, default=1.0,
                        help="Scale applied to high-frequency PAN injection gates")
    parser.add_argument("--image-space-wavelet", action="store_true", default=False,
                        help="Perform wavelet decomposition on raw images instead of feature maps")
    parser.add_argument("--use-lowfreq-corr", action="store_true", default=False,
                        help="Enable zero-init low-frequency/spectral correction head before residual output")
    parser.add_argument("--use-sdem-lite", action="store_true", default=False,
                        help="Enable zero-init lightweight PAN spatial detail enhancement before output reduction")
    parser.add_argument("--use-wfanet-two-stage", action="store_true", default=False,
                        help="Use WFANet-style coarse/fine two-stage RSSM fusion instead of recursive multi-level fusion")
    parser.add_argument("--share-scale-recurrent", action="store_true", default=False,
                        help="Share the coarse-to-fine recurrent state fusion block across scales/stages")
    parser.add_argument("--phase1-preset", action="store_true", default=False,
                        help="Enable the Phase-1 RSSM-HZ preset: two-stage + simple state + shared dw-window recurrent cell + local HF mixer")
    parser.add_argument("--use-state-spatial-mixer", action="store_true", default=False,
                        help="Enable zero-init local spatial mixer on recurrent hidden states")
    parser.add_argument("--use-level-ll-corr", action="store_true", default=False,
                        help="Enable zero-init per-level LL/spectral correction before IDWT")
    parser.add_argument("--use-band-corr", action="store_true", default=False,
                        help="Enable zero-init per-band spectral correction head before output activation")
    parser.add_argument("--band-corr-kernel-size", type=int, default=5,
                        help="Depthwise kernel size for per-band spectral correction")
    parser.add_argument("--band-corr-hidden", type=int, default=32,
                        help="Hidden channels for per-band spectral correction")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="Weight for knowledge distillation loss (L1 between student and teacher outputs)")
    parser.add_argument("--teacher-ckpt", type=str, default="checkpoints/WFANet_best.pth",
                        help="Path to WFANet teacher checkpoint for distillation")
    parser.add_argument("--w-sam", type=float, default=0.08)
    parser.add_argument("--w-edge", type=float, default=0.05)
    parser.add_argument("--w-wavelet-hf", type=float, default=0.08)
    parser.add_argument("--w-ll", type=float, default=0.0, help="Multi-level wavelet LL loss weight")
    parser.add_argument("--w-ssim", type=float, default=0.0, help="SSIM loss weight")
    parser.add_argument("--w-kl", type=float, default=5e-5)
    parser.add_argument(
        "--wavelet-level-weights",
        nargs=3,
        type=float,
        default=[1.0, 0.5, 0.25],
        metavar=("FINE", "MID", "COARSE"),
        help="Level weights for 3-level wavelet HF supervision from fine to coarse",
    )
    parser.add_argument("--eval-clamp", action="store_true", help="Clamp predictions to [0,1] before evaluation/export")
    parser.add_argument("--no-loss-clamp", action="store_true",
                        help="Use raw predictions for training losses; evaluation clamp remains controlled by --eval-clamp")
    parser.add_argument("--q-win-size", type=int, default=8,
                        help="Window size for Q metric. Use 8 for WV3/PanScale comparability, 4 for GF2/QB Q4.")
    parser.add_argument("--eval-tile-size", type=int, default=0,
                        help="Optional high-resolution tile size for final/eval-only inference; 0 disables tiling.")
    parser.add_argument("--eval-tile-overlap", type=int, default=0,
                        help="High-resolution overlap for tiled inference; must be divisible by 4.")
    parser.add_argument("--export-eval-preds", action="store_true", help="Export per-sample pred_XX.mat files for MATLAB-style evaluation")
    parser.add_argument("--use-ema", action="store_true", help="Track an EMA copy of model weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--train-path", default=None, help="Training H5 path")
    parser.add_argument("--test-path", default=None, help="Test H5 path")
    parser.add_argument("--val-path", default=None, help="Validation H5 path; defaults to Dataset/WV3/valid_wv3.h5 if present")
    parser.add_argument("--train-crop-size", type=int, default=0, help="Random high-resolution crop size for training; 0 uses full samples")
    parser.add_argument("--train-crop-align", type=int, default=4, help="Align crop origin to this high-resolution stride")
    parser.add_argument("--train-crop-repeat", type=int, default=4, help="Virtual repeats per image when random crop training is enabled")
    parser.add_argument("--val-every", type=int, default=0, help="Run validation every N epochs; 0 disables validation")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional cap on validation samples")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--val-num-workers", type=int, default=0, help="Validation dataloader workers")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load a checkpoint and run final evaluation without training")
    parser.add_argument(
        "--best-metric",
        choices=["loss", "psnr", "q8", "sam", "ergas", "overall"],
        default="loss",
        help="Checkpoint selection rule",
    )
    parser.add_argument("--best-psnr-weight", type=float, default=1.0)
    parser.add_argument("--best-q8-weight", type=float, default=10.0)
    parser.add_argument("--best-sam-weight", type=float, default=1.0)
    parser.add_argument("--best-ergas-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--z-eval-mode", choices=["prior", "posterior", "zero"], default="prior",
                        help="Eval-time z source in hz mode: prior mean, posterior mean from PAN/MS obs, or zero")
    parser.add_argument("--z-update-order", choices=["legacy", "innovation"], default="legacy",
                        help="legacy: GRU sees obs and previous z before current z; innovation: current z is inferred before GRU")
    parser.add_argument("--z-zero-levels", nargs="*", type=int, default=None,
                        help="Optional DWT levels whose z is forced to zero, e.g. 2 or 1 0")
    parser.add_argument("--collect-z-diagnostics", action="store_true",
                        help="During final eval, save per-level z magnitude and z-vs-zero LL/HF gate deltas")
    parser.add_argument("--augment-geometric", action="store_true",
                        help="Apply random hflip/vflip/rot90 to gt/pan/ms/lms during training")
    parser.add_argument("--use-z-residual-head", action="store_true",
                        help="Enable per-level z residual prediction heads")
    parser.add_argument("--w-z-res-ll", type=float, default=0.03,
                        help="Weight for z residual LL loss")
    parser.add_argument("--w-z-res-hf", type=float, default=0.01,
                        help="Weight for z residual HF loss")
    parser.add_argument("--use-local-frequency-mixer", action="store_true",
                        help="Enable zero-init local frequency residual mixers for LH/HL/HH")
    parser.add_argument("--lfm-kernel-size", type=int, default=3,
                        help="Depthwise kernel size for local frequency mixer")
    parser.add_argument("--lfm-hidden-scale", type=float, default=1.0,
                        help="Hidden channel scale for local frequency mixer")
    parser.add_argument("--use-linear-frequency-attention", action="store_true",
                        help="Enable O(N*C^2) WFANet-style linear frequency attention in two-stage fusion")
    parser.add_argument("--linear-attn-heads", type=int, default=4,
                        help="Number of heads for linear frequency attention")
    parser.add_argument("--use-windowed-frequency-mixer", action="store_true",
                        help="Enable dependency-free windowed selective-scan mixers for LH/HL/HH")
    parser.add_argument("--wfm-window-size", type=int, default=8,
                        help="Local window size for windowed frequency selective scan")
    parser.add_argument("--wfm-hidden-scale", type=float, default=1.0,
                        help="Hidden channel scale for windowed frequency mixer")
    parser.add_argument("--use-mamba-frequency-mixer", action="store_true",
                        help="Enable true mamba_ssm window mixers for LH/HL/HH")
    parser.add_argument("--mamba-window-size", type=int, default=8,
                        help="Local window size for true Mamba frequency mixer")
    parser.add_argument("--mamba-hidden-scale", type=float, default=1.0,
                        help="Hidden channel scale for true Mamba frequency mixer")
    parser.add_argument("--mamba-d-state", type=int, default=16,
                        help="Mamba state dimension for frequency mixer")
    parser.add_argument("--mamba-d-conv", type=int, default=4,
                        help="Mamba local convolution width for frequency mixer")
    parser.add_argument("--mamba-expand", type=int, default=2,
                        help="Mamba expansion ratio for frequency mixer")
    parser.add_argument("--use-channel-dwt-adapter", action="store_true",
                        help="Enable 1D channel-wise Haar spectral adapter for MS_up")
    parser.add_argument("--channel-dwt-hidden", type=int, default=32,
                        help="Hidden channels for channel-wise Haar spectral adapter")
    parser.add_argument("--w-mse", type=float, default=0.0,
                        help="Optional MSE loss weight for PSNR-oriented finetuning")
    parser.add_argument("--w-band-balanced", type=float, default=0.0,
                        help="Optional per-band normalized L1 loss weight")
    parser.add_argument("--w-ms-fidelity", type=float, default=0.0,
                        help="Optional LRMS consistency loss weight: downsample(pred) should match input MS")
    parser.add_argument("--w-pan-fidelity", type=float, default=0.0,
                        help="Optional PAN consistency loss weight using mean(pred bands) as a lightweight intensity proxy")
    args = parser.parse_args()

    if args.residual_learnable_fusion:
        args.learnable_fusion = True

    if args.phase1_preset:
        args.use_wfanet_two_stage = True
        args.use_conv_gru = True
        args.freq_state_mode = "simple"
        args.state_conv_type = "dw_window_attn"
        args.state_kernel_size = 7
        args.share_scale_recurrent = True
        args.use_local_frequency_mixer = True
        args.use_level_ll_corr = True

    if args.phase == "a":
        args.state_mode = "h"
        args.w_kl = 0.0
    elif args.phase == "b":
        args.state_mode = "hz"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_tag = args.run_tag or time.strftime("rssm_hz_%Y%m%d_%H%M%S")
    out_root = os.path.join("results_rssm_hz", run_tag)
    ckpt_dir = os.path.join(out_root, "checkpoints")
    eval_dir = os.path.join(out_root, "eval")
    val_dir = os.path.join(out_root, "val")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    epochs = args.epochs or int(config["epochs"])
    batch_size = args.batch_size or int(config["batch_size"])

    separate_subband = args.separate_subband_gates and not args.shared_high_gate
    model = RSSMHWViTHZ(
        L_up_channel=int(config.get("L_up_channel", 8)),
        pan_channel=1,
        pan_target_channel=int(config["pan_target_channel"]),
        ms_target_channel=int(config["ms_target_channel"]),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        deterministic_only=(args.state_mode == "h"),
        separate_subband_gates=separate_subband,
        use_conv_gru=args.use_conv_gru,
        state_conv_type=args.state_conv_type,
        state_kernel_size=args.state_kernel_size,
        freq_state_mode=args.freq_state_mode,
        learnable_fusion=args.learnable_fusion,
        signed_hf_gate=args.signed_hf_gate,
        hf_gate_scale=args.hf_gate_scale,
        residual_learnable_fusion=args.residual_learnable_fusion,
        image_space_wavelet=args.image_space_wavelet,
        use_lowfreq_corr=args.use_lowfreq_corr,
        use_sdem_lite=args.use_sdem_lite,
        use_state_spatial_mixer=args.use_state_spatial_mixer,
        use_level_ll_corr=args.use_level_ll_corr,
        use_band_corr=args.use_band_corr,
        band_corr_kernel_size=args.band_corr_kernel_size,
        band_corr_hidden=args.band_corr_hidden,
        z_eval_mode=args.z_eval_mode,
        z_update_order=args.z_update_order,
        z_zero_levels=args.z_zero_levels,
        use_z_residual_head=args.use_z_residual_head,
        use_local_freq_mixer=args.use_local_frequency_mixer,
        lfm_kernel_size=args.lfm_kernel_size,
        lfm_hidden_scale=args.lfm_hidden_scale,
        use_linear_freq_attention=args.use_linear_frequency_attention,
        linear_attn_heads=args.linear_attn_heads,
        use_windowed_freq_mixer=args.use_windowed_frequency_mixer,
        wfm_window_size=args.wfm_window_size,
        wfm_hidden_scale=args.wfm_hidden_scale,
        use_mamba_freq_mixer=args.use_mamba_frequency_mixer,
        mamba_window_size=args.mamba_window_size,
        mamba_hidden_scale=args.mamba_hidden_scale,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        use_channel_dwt_adapter=args.use_channel_dwt_adapter,
        channel_dwt_hidden=args.channel_dwt_hidden,
        use_wfanet_two_stage=args.use_wfanet_two_stage,
        share_scale_recurrent=args.share_scale_recurrent,
    ).to(device)

    # If training fresh, zero-init ms_upsample + fused_weight for LMS-start.
    # Skip when loading a checkpoint (init_ckpt/resume_ckpt will overwrite).
    if not args.init_ckpt and not args.resume_ckpt:
        if hasattr(model, 'ms_upsample') and hasattr(model.ms_upsample, '0'):
            ms_conv = model.ms_upsample[0]
            if isinstance(ms_conv, nn.Conv2d):
                nn.init.zeros_(ms_conv.weight)
                if ms_conv.bias is not None:
                    nn.init.zeros_(ms_conv.bias)
                print("ms_upsample zero-initialized for LMS-start")
        if hasattr(model, 'fused_weight'):
            model.fused_weight.data.fill_(0.0)
            print("fused_weight=0 for LMS-start")

    start_epoch = 0
    best_loss = float("inf")
    best_val_score = -float("inf")
    val_history = []
    best_ckpt_path = os.path.join(ckpt_dir, "rssm_hz_best.pth")
    best_loss_ckpt_path = os.path.join(ckpt_dir, "rssm_hz_best_loss.pth")
    best_val_ckpt_path = os.path.join(ckpt_dir, "rssm_hz_best_val.pth")

    if args.init_ckpt:
        init_obj = torch.load(args.init_ckpt, map_location="cpu")
        init_state = init_obj["model"] if isinstance(init_obj, dict) and "model" in init_obj else init_obj
        missing, unexpected, skipped = load_state_dict_flexible(model, init_state)
        print(f"init_ckpt loaded: {args.init_ckpt}")
        print(
            f"init missing_keys={len(missing)} unexpected_keys={len(unexpected)} "
            f"skipped_keys={len(skipped)}"
        )
        if skipped:
            print("init skipped sample:", skipped[:5])

    resume_obj = None
    if args.resume_ckpt:
        resume_obj = torch.load(args.resume_ckpt, map_location="cpu")
        resume_state = resume_obj["model"] if isinstance(resume_obj, dict) and "model" in resume_obj else resume_obj
        missing, unexpected, skipped = load_state_dict_flexible(model, resume_state)
        print(f"resume_ckpt loaded model: {args.resume_ckpt}")
        print(
            f"resume missing_keys={len(missing)} unexpected_keys={len(unexpected)} "
            f"skipped_keys={len(skipped)}"
        )
        if skipped:
            print("resume skipped sample:", skipped[:5])
        if isinstance(resume_obj, dict) and "epoch" in resume_obj:
            start_epoch = int(resume_obj["epoch"])
            print(f"resume start_epoch={start_epoch}")
        if isinstance(resume_obj, dict) and "best_loss" in resume_obj:
            best_loss = float(resume_obj["best_loss"])
        if isinstance(resume_obj, dict) and "best_val_score" in resume_obj:
            best_val_score = float(resume_obj["best_val_score"])

    if hasattr(model, "set_z_eval_mode"):
        model.set_z_eval_mode(args.z_eval_mode)

    if args.eval_only:
        print(
            f"eval_only run_tag={run_tag} device={device} state_mode={args.state_mode} "
            f"z_eval_mode={args.z_eval_mode} z_update_order={args.z_update_order} "
            f"z_zero_levels={args.z_zero_levels}"
        )
        metrics = evaluate(
            model,
            config,
            device,
            eval_dir,
            test_path=args.test_path,
            max_test_samples=args.max_test_samples,
            eval_clamp=args.eval_clamp,
            export_preds=args.export_eval_preds,
            q_win_size=args.q_win_size,
            collect_z_diagnostics=args.collect_z_diagnostics,
            tile_size=args.eval_tile_size,
            tile_overlap=args.eval_tile_overlap,
        )
        print("===== eval only =====")
        for k, v in metrics.items():
            if k == "z_diagnostics":
                print(f"{k}: saved to {os.path.join(eval_dir, 'rssm_hz_z_diagnostics.json')}")
            else:
                print(f"{k}: {v}")
        return

    frozen_names = []
    if args.phase == "b" and args.phase_b_freeze_mode != "none":
        frozen_names = apply_phase_b_freeze(model, args.phase_b_freeze_mode)
        print(f"phase b freeze mode={args.phase_b_freeze_mode}: {', '.join(frozen_names)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {total_params:,} trainable: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr_max"]),
        weight_decay=float(config["weight_decay"]),
        betas=(0.9, 0.999),
    )

    if args.resume_ckpt and isinstance(resume_obj, dict) and "optimizer" in resume_obj:
        if args.phase == "b" and args.phase_b_freeze_mode != "none":
            print("resume optimizer skipped because freeze mode changes trainable parameter groups")
        else:
            optimizer.load_state_dict(resume_obj["optimizer"])
            print("resume optimizer loaded")

    ema = ModelEMA(model, args.ema_decay) if args.use_ema else None
    if ema is not None and isinstance(resume_obj, dict) and "model_ema" in resume_obj:
        ema.load_state_dict(resume_obj["model_ema"], device=device)
        print("resume EMA loaded")

    l1_loss = nn.L1Loss()
    dwt_loss = DWT_2D().to(device)

    if args.train_path:
        train_path = args.train_path
    else:
        train_path = os.path.join("Dataset", "WV3", "train_wv3-001.h5")
        if not os.path.exists(train_path):
            train_path = os.path.join("Dataset", "WV3", "train_wv3.h5")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train set: {args.train_path or 'train_wv3-001.h5/train_wv3.h5'}")

    if args.train_crop_size > 0:
        train_ds = RandomCropPanDataset(
            train_path,
            ratio=float(config["ratio"]),
            max_samples=args.max_train_samples,
            crop_size=args.train_crop_size,
            crop_align=args.train_crop_align,
            repeat=args.train_crop_repeat,
        )
        print(
            f"random crop training enabled: crop_size={args.train_crop_size} "
            f"align={args.train_crop_align} repeat={args.train_crop_repeat}"
        )
    else:
        train_ds = PanDataset(train_path, ratio=float(config["ratio"]), max_samples=args.max_train_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_path = args.val_path
    if val_path is None:
        default_val_path = os.path.join("Dataset", "WV3", "valid_wv3.h5")
        if os.path.exists(default_val_path):
            val_path = default_val_path
    val_enabled = bool(val_path) and args.val_every > 0 and args.best_metric != "loss"
    if val_enabled:
        print(
            f"validation enabled: path={val_path} every={args.val_every} "
            f"best_metric={args.best_metric} max_val_samples={args.max_val_samples}"
        )

    history = {
        "total": [], "l1": [], "mse": [], "band": [], "sam": [], "edge": [],
        "wave": [], "ll": [], "ssim": [], "ms_fid": [], "pan_fid": [], "distill": [], "kl": [],
        "z_res_ll": [], "z_res_hf": []
    }

    # ---- Load WFANet teacher for distillation ----
    teacher = None
    if args.distill_weight > 0:
        teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher_config = teacher_ckpt.get("config", config) if isinstance(teacher_ckpt, dict) else config
        teacher_state = teacher_ckpt.get("model", teacher_ckpt) if isinstance(teacher_ckpt, dict) else teacher_ckpt
        teacher = HWViT(
            L_up_channel=int(config.get("L_up_channel", 8)),
            pan_channel=1,
            pan_target_channel=int(config["pan_target_channel"]),
            ms_target_channel=int(config["ms_target_channel"]),
            head_channel=int(config.get("head_channel", config["ms_target_channel"] // 2)),
            dropout=float(config.get("dropout", 0.085)),
        ).to(device)
        teacher.load_state_dict(teacher_state, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Teacher model loaded: {args.teacher_ckpt}")

    print(
        f"run_tag={run_tag} device={device} epochs={epochs} batch_size={batch_size} "
        f"state_mode={args.state_mode} phase={args.phase} distill_weight={args.distill_weight} "
        f"z_eval_mode={args.z_eval_mode} z_update_order={args.z_update_order} z_zero_levels={args.z_zero_levels} "
        f"freq_state_mode={args.freq_state_mode} state_conv_type={args.state_conv_type} "
        f"state_kernel={args.state_kernel_size} "
        f"use_lfm={args.use_local_frequency_mixer} use_linear_attn={args.use_linear_frequency_attention} "
        f"use_wfm={args.use_windowed_frequency_mixer} "
        f"use_chdwt={args.use_channel_dwt_adapter} use_wfanet_two_stage={args.use_wfanet_two_stage} "
        f"share_scale_recurrent={args.share_scale_recurrent} phase1_preset={args.phase1_preset} "
        f"w_mse={args.w_mse} w_band={args.w_band_balanced} "
        f"w_ms_fid={args.w_ms_fidelity} w_pan_fid={args.w_pan_fidelity}"
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()

        # Snapshot a known-good state for in-epoch recovery when encountering non-finite values.
        last_good_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # warmup + cosine decay
        warmup_epochs = min(10, max(1, epochs // 20))
        if epoch < warmup_epochs:
            lr = float(config["lr_max"]) * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            min_lr = float(config.get("lr_min", config["lr_max"] * 0.1))
            lr = min_lr + 0.5 * (float(config["lr_max"]) - min_lr) * (1.0 + np.cos(np.pi * progress))

        lr = lr * args.lr_scale
        if args.phase == "b":
            lr = lr * args.phase_b_lr_scale
        for g in optimizer.param_groups:
            g["lr"] = lr

        total_meter = 0.0
        l1_meter = 0.0
        mse_meter = 0.0
        band_meter = 0.0
        sam_meter = 0.0
        edge_meter = 0.0
        wave_meter = 0.0
        ll_meter = 0.0
        ssim_meter = 0.0
        ms_fid_meter = 0.0
        pan_fid_meter = 0.0
        distill_meter = 0.0
        z_res_ll_meter = 0.0
        z_res_hf_meter = 0.0
        kl_meter = 0.0
        bad_step_count = 0

        kl_weight_base = 0.0 if args.state_mode == "h" else args.w_kl
        if args.phase == "b":
            kl_beta = kl_weight_base * min(1.0, (epoch + 1) / max(1, args.phase_b_ramp_epochs))
        else:
            kl_beta = kl_weight_base * min(1.0, (epoch + 1) / max(1, warmup_epochs * 2))

        for step, (gt, pan, ms, lms) in enumerate(train_loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

            gt = gt.to(device, non_blocking=True)
            pan = pan.to(device, non_blocking=True)
            ms = ms.to(device, non_blocking=True)
            lms = lms.to(device, non_blocking=True)

            # Geometric augmentation: apply same hflip/vflip/rot90 to all four tensors
            if args.augment_geometric:
                if torch.rand(1).item() < 0.5:
                    gt = gt.flip(-1)
                    pan = pan.flip(-1)
                    ms = ms.flip(-1)
                    lms = lms.flip(-1)
                if torch.rand(1).item() < 0.5:
                    gt = gt.flip(-2)
                    pan = pan.flip(-2)
                    ms = ms.flip(-2)
                    lms = lms.flip(-2)
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    gt = torch.rot90(gt, k, [-2, -1])
                    pan = torch.rot90(pan, k, [-2, -1])
                    ms = torch.rot90(ms, k, [-2, -1])
                    lms = torch.rot90(lms, k, [-2, -1])

            optimizer.zero_grad(set_to_none=True)

            pred_raw, kl_loss, z_residuals = model(pan, ms, lms)
            loss_z_res_ll = pred_raw.new_tensor(0.0)
            loss_z_res_hf = pred_raw.new_tensor(0.0)
            if args.use_z_residual_head and z_residuals is not None and model.training:
                # Compute GT wavelet targets in feature space (32ch per subband)
                gt_feat = model.ms_raise(gt)
                lms_feat = model.ms_raise(lms)
                gt_pyr = model.wavelet(gt_feat)
                lms_pyr = model.wavelet(lms_feat)
                # z_residuals is [coarsest, ..., finest], gt_pyr is [finest, ..., coarsest]
                levels = len(z_residuals)
                for i, (r_ll, r_lh, r_hl, r_hh) in enumerate(z_residuals):
                    g_ll, g_lh, g_hl, g_hh = gt_pyr[levels - 1 - i]
                    l_ll, l_lh, l_hl, l_hh = lms_pyr[levels - 1 - i]
                    loss_z_res_ll = loss_z_res_ll + F.l1_loss(r_ll, g_ll - l_ll)
                    loss_z_res_hf = loss_z_res_hf + (
                        F.l1_loss(r_lh, g_lh - l_lh) +
                        F.l1_loss(r_hl, g_hl - l_hl) +
                        F.l1_loss(r_hh, g_hh - l_hh)
                    ) / 3.0
            if (not torch.isfinite(pred_raw).all()) or (not torch.isfinite(kl_loss)):
                bad_step_count += 1
                model.load_state_dict(last_good_state)
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-6)
                continue

            pred = pred_raw if args.no_loss_clamp else pred_raw.clamp(0.0, 1.0)
            loss_l1 = l1_loss(pred, gt)
            loss_mse = F.mse_loss(pred, gt) if args.w_mse > 0 else pred.new_tensor(0.0)
            loss_band = band_balanced_l1_loss(pred, gt) if args.w_band_balanced > 0 else pred.new_tensor(0.0)
            loss_ms_fid = ms_fidelity_loss(pred, ms) if args.w_ms_fidelity > 0 else pred.new_tensor(0.0)
            loss_pan_fid = pan_fidelity_loss(pred, pan) if args.w_pan_fidelity > 0 else pred.new_tensor(0.0)
            loss_sam = sam_loss(pred, gt) if args.w_sam > 0 else pred.new_tensor(0.0)
            loss_edge = l1_loss(sobel_edges(pred), sobel_edges(gt)) if args.w_edge > 0 else pred.new_tensor(0.0)
            loss_wave = (
                wavelet_hf_loss_multilevel(pred, gt, dwt_loss, l1_loss, args.wavelet_level_weights)
                if args.w_wavelet_hf > 0 else pred.new_tensor(0.0)
            )
            loss_ll = (
                wavelet_ll_loss_multilevel(pred, gt, dwt_loss, l1_loss, args.wavelet_level_weights)
                if args.w_ll > 0 else pred.new_tensor(0.0)
            )
            loss_ssim = ssim_loss(pred, gt) if args.w_ssim > 0 else pred.new_tensor(0.0)

            loss_distill = pred.new_tensor(0.0)
            if args.distill_weight > 0 and teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(pan, ms, lms)
                loss_distill = l1_loss(pred, teacher_out.clamp(0.0, 1.0))

            kl_safe = torch.nan_to_num(kl_loss, nan=0.0, posinf=1e3, neginf=0.0)
            loss = (
                loss_l1
                + args.w_sam * loss_sam
                + args.w_mse * loss_mse
                + args.w_band_balanced * loss_band
                + args.w_ms_fidelity * loss_ms_fid
                + args.w_pan_fidelity * loss_pan_fid
                + args.w_edge * loss_edge
                + args.w_wavelet_hf * loss_wave
                + args.w_ll * loss_ll
                + args.w_ssim * loss_ssim
                + args.distill_weight * loss_distill
                + kl_beta * kl_safe
                + args.w_z_res_ll * loss_z_res_ll
                + args.w_z_res_hf * loss_z_res_hf
            )

            if not torch.isfinite(loss):
                bad_step_count += 1
                model.load_state_dict(last_good_state)
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-6)
                continue

            loss.backward()

            # Fix non-finite grads in-place instead of discarding the whole step.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=False)
            nan_grad_params = 0
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    nan_grad_params += 1
            if nan_grad_params > 0 and bad_step_count == 0:
                pass  # non-finite grads were auto-fixed
            optimizer.step()
            if ema is not None:
                ema.update(model)

            if (step + 1) % 20 == 0:
                last_good_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            total_meter += loss.item()
            l1_meter += loss_l1.item()
            mse_meter += loss_mse.item()
            band_meter += loss_band.item()
            sam_meter += loss_sam.item()
            edge_meter += loss_edge.item()
            wave_meter += loss_wave.item()
            ll_meter += loss_ll.item()
            ssim_meter += loss_ssim.item()
            ms_fid_meter += loss_ms_fid.item()
            pan_fid_meter += loss_pan_fid.item()
            distill_meter += loss_distill.item()
            kl_meter += kl_safe.item()
            z_res_ll_meter += loss_z_res_ll.item()
            z_res_hf_meter += loss_z_res_hf.item()

        steps_done = step + 1
        avg_total = total_meter / max(1, steps_done)
        avg_l1 = l1_meter / max(1, steps_done)
        avg_mse = mse_meter / max(1, steps_done)
        avg_band = band_meter / max(1, steps_done)
        avg_sam = sam_meter / max(1, steps_done)
        avg_edge = edge_meter / max(1, steps_done)
        avg_wave = wave_meter / max(1, steps_done)
        avg_ll = ll_meter / max(1, steps_done)
        avg_ssim = ssim_meter / max(1, steps_done)
        avg_ms_fid = ms_fid_meter / max(1, steps_done)
        avg_pan_fid = pan_fid_meter / max(1, steps_done)
        avg_distill = distill_meter / max(1, steps_done)
        avg_kl = kl_meter / max(1, steps_done)

        history["total"].append(avg_total)
        history["l1"].append(avg_l1)
        history["mse"].append(avg_mse)
        history["band"].append(avg_band)
        history["sam"].append(avg_sam)
        history["edge"].append(avg_edge)
        history["wave"].append(avg_wave)
        history["ll"].append(avg_ll)
        history["ssim"].append(avg_ssim)
        history["ms_fid"].append(avg_ms_fid)
        history["pan_fid"].append(avg_pan_fid)
        history["z_res_ll"].append(z_res_ll_meter / max(1, steps_done))
        history["z_res_hf"].append(z_res_hf_meter / max(1, steps_done))
        history["distill"].append(avg_distill)
        history["kl"].append(avg_kl)

        if avg_total < best_loss:
            best_loss = avg_total
            best_loss_state = build_checkpoint_state(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                config=config,
                args=args,
                extra={"best_loss": best_loss},
                ema=ema,
            )
            torch.save(best_loss_state, best_loss_ckpt_path)
            if not val_enabled:
                torch.save(best_loss_state, best_ckpt_path)

        if val_enabled and (((epoch + 1) % args.val_every == 0) or epoch == 0 or (epoch + 1) == epochs):
            raw_metrics = evaluate_dataset(
                model,
                config,
                device,
                dataset_path=val_path,
                out_dir=None,
                max_samples=args.max_val_samples,
                eval_clamp=args.eval_clamp,
                export_preds=False,
                batch_size=args.val_batch_size,
                num_workers=args.val_num_workers,
                q_win_size=args.q_win_size,
            )
            raw_score = compute_metric_score(raw_metrics, args)
            selected_source = "raw"
            selected_metrics = raw_metrics
            selected_score = raw_score

            ema_metrics = None
            ema_score = None
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                ema_metrics = evaluate_dataset(
                    model,
                    config,
                    device,
                    dataset_path=val_path,
                    out_dir=None,
                    max_samples=args.max_val_samples,
                    eval_clamp=args.eval_clamp,
                    export_preds=False,
                    batch_size=args.val_batch_size,
                    num_workers=args.val_num_workers,
                    q_win_size=args.q_win_size,
                )
                ema_score = compute_metric_score(ema_metrics, args)
                if ema_score > selected_score:
                    selected_source = "ema"
                    selected_metrics = ema_metrics
                    selected_score = ema_score
                ema.restore(model)

            val_record = {
                "epoch": epoch + 1,
                "selected_source": selected_source,
                "selected_score": float(selected_score),
                "raw_metrics": raw_metrics,
                "raw_score": float(raw_score),
            }
            if ema_metrics is not None:
                val_record["ema_metrics"] = ema_metrics
                val_record["ema_score"] = float(ema_score)
            val_history.append(val_record)
            with open(os.path.join(val_dir, "val_history.json"), "w") as f:
                json.dump(val_history, f, indent=2)
            with open(os.path.join(val_dir, f"val_epoch_{epoch + 1:03d}.json"), "w") as f:
                json.dump(val_record, f, indent=2)

            print(
                f"val epoch {epoch + 1:03d}: source={selected_source} score={selected_score:.6f} "
                f"PSNR={selected_metrics['PSNR']:.6f} SAM={selected_metrics['SAM']:.6f} "
                f"ERGAS={selected_metrics['ERGAS']:.6f} Q{args.q_win_size}={selected_metrics['Q8']:.6f}"
            )

            if selected_score > best_val_score:
                best_val_score = float(selected_score)
                if selected_source == "ema":
                    ema.store(model)
                    ema.copy_to(model)
                best_val_state = build_checkpoint_state(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    args=args,
                    extra={
                        "best_loss": best_loss,
                        "best_val_score": best_val_score,
                        "best_val_metrics": selected_metrics,
                        "best_source": selected_source,
                    },
                    ema=ema,
                )
                torch.save(best_val_state, best_val_ckpt_path)
                torch.save(best_val_state, best_ckpt_path)
                if selected_source == "ema":
                    ema.restore(model)

        if (epoch + 1) % args.save_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            torch.save(
                build_checkpoint_state(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    args=args,
                    extra={"best_loss": best_loss, "best_val_score": best_val_score},
                    ema=ema,
                ),
                os.path.join(ckpt_dir, f"rssm_hz_epoch_{epoch + 1}.pth"),
            )

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1:03d}/{epochs} "
            f"loss={avg_total:.6f} l1={avg_l1:.6f} mse={avg_mse:.6f} band={avg_band:.6f} sam={avg_sam:.6f} "
            f"edge={avg_edge:.6f} wave={avg_wave:.6f} ll={avg_ll:.6f} "
            f"ssim={avg_ssim:.6f} ms_fid={avg_ms_fid:.6f} pan_fid={avg_pan_fid:.6f} "
            f"distill={avg_distill:.6f} kl={avg_kl:.6f} bad_steps={bad_step_count} "
            f"beta={kl_beta:.6e} lr={lr:.3e} time={dt:.1f}s"
        )

    with open(os.path.join(out_root, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if os.path.exists(best_ckpt_path):
        best_obj = torch.load(best_ckpt_path, map_location="cpu")
        best_state = best_obj["model"] if isinstance(best_obj, dict) and "model" in best_obj else best_obj
        model.load_state_dict(best_state, strict=False)

    metrics = evaluate(
        model,
        config,
        device,
        eval_dir,
        test_path=args.test_path,
        max_test_samples=args.max_test_samples,
        eval_clamp=args.eval_clamp,
        export_preds=args.export_eval_preds,
        q_win_size=args.q_win_size,
        collect_z_diagnostics=args.collect_z_diagnostics,
        tile_size=args.eval_tile_size,
        tile_overlap=args.eval_tile_overlap,
    )
    print("===== final eval =====")
    for k, v in metrics.items():
        if k == "z_diagnostics":
            print(f"{k}: saved to {os.path.join(eval_dir, 'rssm_hz_z_diagnostics.json')}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

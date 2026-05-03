"""Train RSSM-HZ on jilin using random 64x64 crops. Supports Phase A (h-only) and Phase B (h+z)."""
import os, time, yaml, math, argparse, json
import h5py, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rssm_hz_wfanet import RSSMHWViTHZ
from net_torch import HWViT
from evaluate_wv3_metrics import calculate_metrics

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CropDataset(Dataset):
    def __init__(self, pan, gt, ms, lms, pan_size=64):
        self.pan = pan; self.gt = gt; self.ms = ms; self.lms = lms
        self.pan_size = pan_size; self.ms_size = pan_size // 4
    def __len__(self):
        return len(self.pan) * 4
    def __getitem__(self, idx):
        i = idx % len(self.pan)
        _, _, H, W = self.pan.shape
        y = np.random.randint(0, max(1, H - self.pan_size + 1))
        x = np.random.randint(0, max(1, W - self.pan_size + 1))
        return (self.pan[i, :, y:y+self.pan_size, x:x+self.pan_size],
                self.gt[i, :, y:y+self.pan_size, x:x+self.pan_size],
                self.ms[i, :, y//4:y//4+self.ms_size, x//4:x//4+self.ms_size],
                self.lms[i, :, y:y+self.pan_size, x:x+self.pan_size])

def tiled_forward(model, pan, ms, lms, tile_size=64, pad=8):
    _, _, H, W = pan.shape
    C_out = ms.shape[1]
    scale = H // ms.shape[2]
    ts_pan = (tile_size // scale) * scale; ts_ms = ts_pan // scale
    tiles_h = (H + ts_pan - 1) // ts_pan; tiles_w = (W + ts_pan - 1) // ts_pan
    H_pad = tiles_h * ts_pan; W_pad = tiles_w * ts_pan
    pan_p = F.pad(pan, (0, W_pad - W, 0, H_pad - H), mode='reflect')
    ms_p = F.pad(ms, (0, W_pad//scale - ms.shape[3], 0, H_pad//scale - ms.shape[2]), mode='reflect')
    lms_p = F.pad(lms, (0, W_pad - W, 0, H_pad - H), mode='reflect')
    pan_ext = F.pad(pan_p, (pad, pad, pad, pad), mode='reflect')
    ms_ext = F.pad(ms_p, (pad//scale, pad//scale, pad//scale, pad//scale), mode='reflect')
    lms_ext = F.pad(lms_p, (pad, pad, pad, pad), mode='reflect')
    output = torch.zeros(1, C_out, H, W, device=pan.device)
    weight = torch.zeros(1, 1, H, W, device=pan.device)
    for ti in range(tiles_h):
        for tj in range(tiles_w):
            pi, pj = ti*ts_pan+pad, tj*ts_pan+pad
            mi, mj = ti*ts_ms+pad//scale, tj*ts_ms+pad//scale
            p_tile = pan_ext[:,:,pi-pad:pi+ts_pan+pad,pj-pad:pj+ts_pan+pad]
            m_tile = ms_ext[:,:,mi-pad//scale:mi+ts_ms+pad//scale,mj-pad//scale:mj+ts_ms+pad//scale]
            l_tile = lms_ext[:,:,pi-pad:pi+ts_pan+pad,pj-pad:pj+ts_pan+pad]
            out_tile, _ = model(p_tile, m_tile, l_tile)
            out_crop = out_tile.clamp(0,1)[:,:,pad:pad+ts_pan,pad:pad+ts_pan]
            oi, oj = ti*ts_pan, tj*ts_pan
            eh, ew = min(ts_pan, H-oi), min(ts_pan, W-oj)
            output[:,:,oi:oi+eh,oj:oj+ew] += out_crop[:,:,:eh,:ew]
            weight[:,:,oi:oi+eh,oj:oj+ew] += 1.0
    return output / weight.clamp_min(1.0)

def load_state_dict_flexible(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr-scale", type=float, default=1.0)
parser.add_argument("--hidden-dim", type=int, default=96)
parser.add_argument("--latent-dim", type=int, default=32)
parser.add_argument("--run-tag", default="jilin_rssmhz_crop")
parser.add_argument("--phase", choices=["a", "b"], default="a")
parser.add_argument("--init-ckpt", default=None, help="Phase A checkpoint for Phase B")
parser.add_argument("--phase-b-freeze-mode", choices=["none", "shallow", "state_gate_head"], default="shallow")
parser.add_argument("--phase-b-lr-scale", type=float, default=0.3)
parser.add_argument("--phase-b-ramp-epochs", type=int, default=80)
parser.add_argument("--w-kl", type=float, default=1e-4)
parser.add_argument("--distill-weight", type=float, default=0.0)
parser.add_argument("--teacher-ckpt", default="results_rssm_hz/wfanet_jilin_crop/checkpoints/WFANet_jilin_best.pth")
parser.add_argument("--val-every", type=int, default=20)
parser.add_argument("--crop-size", type=int, default=64, help="Random crop size for PAN (MS = crop_size//4)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda")

with open(os.path.join(ROOT_DIR, "super_para_panscale.yml")) as f:
    cfg = yaml.safe_load(f)

print("Loading data...", flush=True)
with h5py.File(os.path.join(ROOT_DIR, "Dataset/PanScale_H5/jilin/jilin_train_v2.h5"), 'r') as f:
    train_pan = torch.from_numpy(f['pan'][:] / float(cfg["ratio"])).float()
    train_gt = torch.from_numpy(f['gt'][:] / float(cfg["ratio"])).float()
    train_ms = torch.from_numpy(f['ms'][:] / float(cfg["ratio"])).float()
    train_lms = torch.from_numpy(f['lms'][:] / float(cfg["ratio"])).float()

with h5py.File(os.path.join(ROOT_DIR, "Dataset/PanScale_H5/jilin/jilin_val_v2.h5"), 'r') as f:
    val_pan = torch.from_numpy(f['pan'][:] / float(cfg["ratio"])).float()
    val_gt = torch.from_numpy(f['gt'][:] / float(cfg["ratio"])).float()
    val_ms = torch.from_numpy(f['ms'][:] / float(cfg["ratio"])).float()
    val_lms = torch.from_numpy(f['lms'][:] / float(cfg["ratio"])).float()

with h5py.File(os.path.join(ROOT_DIR, "Dataset/PanScale_H5/jilin/jilin_test200.h5"), 'r') as f:
    test_pan = torch.from_numpy(f['pan'][:] / float(cfg["ratio"])).float()
    test_gt = torch.from_numpy(f['gt'][:] / float(cfg["ratio"])).float()
    test_ms = torch.from_numpy(f['ms'][:] / float(cfg["ratio"])).float()
    test_lms = torch.from_numpy(f['lms'][:] / float(cfg["ratio"])).float()

C = train_ms.shape[1]
det_only = (args.phase == "a")
print(f"Train: {len(train_pan)}, Val: {len(val_pan)}, Test: {len(test_pan)}", flush=True)
print(f"Phase: {args.phase}, deterministic_only={det_only}", flush=True)

model = RSSMHWViTHZ(
    L_up_channel=C, pan_channel=1,
    pan_target_channel=int(cfg["pan_target_channel"]),
    ms_target_channel=int(cfg["ms_target_channel"]),
    hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
    deterministic_only=det_only,
).to(device)

# Load Phase A checkpoint for Phase B
if args.init_ckpt:
    ckpt = torch.load(args.init_ckpt, map_location="cpu")
    ckpt_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_flexible(model, ckpt_state)
    print(f"Loaded checkpoint: {args.init_ckpt}", flush=True)

# Phase B freeze
if args.phase == "b" and args.phase_b_freeze_mode == "shallow":
    frozen = []
    for name, mod in [("pan_raise", model.pan_raise), ("ms_upsample", model.ms_upsample),
                       ("ms_act", model.ms_act), ("ms_raise", model.ms_raise)]:
        if hasattr(model, name):
            for p in mod.parameters():
                p.requires_grad = False
            frozen.append(name)
    print(f"Phase B freeze: {frozen}", flush=True)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {total:,} trainable: {trainable:,}", flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
    lr=float(cfg["lr_max"]) * args.lr_scale, weight_decay=float(cfg["weight_decay"]), betas=(0.9, 0.999))

criterion = nn.L1Loss()
train_ds = CropDataset(train_pan, train_gt, train_ms, train_lms, pan_size=args.crop_size)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

# Teacher for distillation
teacher = None
if args.distill_weight > 0 and os.path.exists(args.teacher_ckpt):
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_state = t_ckpt["model"] if isinstance(t_ckpt, dict) and "model" in t_ckpt else t_ckpt
    teacher = HWViT(L_up_channel=C, pan_channel=1,
                    pan_target_channel=int(cfg["pan_target_channel"]),
                    ms_target_channel=int(cfg["ms_target_channel"]),
                    head_channel=int(cfg["head_channel"]), dropout=float(cfg["dropout"])).to(device)
    teacher.load_state_dict(t_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {args.teacher_ckpt}", flush=True)

out_dir = os.path.join(ROOT_DIR, "results_rssm_hz", args.run_tag)
ckpt_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

best_q8 = -float("inf")
history = []

print("Starting training...", flush=True)
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss, n_steps = 0.0, 0
    t0 = time.time()

    # KL ramp-up for Phase B
    kl_beta = 0.0
    if args.phase == "b":
        kl_beta = args.w_kl * min(1.0, epoch / max(1, args.phase_b_ramp_epochs))

    for pan, gt, ms, lms in train_loader:
        pan, gt, ms, lms = pan.to(device), gt.to(device), ms.to(device), lms.to(device)
        optimizer.zero_grad()

        out, kl_loss = model(pan, ms, lms)
        pred = out.clamp(0, 1)
        loss_l1 = criterion(pred, gt)
        loss = loss_l1

        if kl_beta > 0:
            kl_safe = torch.nan_to_num(kl_loss, nan=0.0, posinf=1e3, neginf=0.0)
            loss = loss + kl_beta * kl_safe

        if args.distill_weight > 0 and teacher is not None:
            with torch.no_grad():
                t_out = teacher(pan, ms, lms).clamp(0, 1)
            loss = loss + args.distill_weight * criterion(pred, t_out)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()
        total_loss += loss.item()
        n_steps += 1

    # Cosine LR
    progress = epoch / args.epochs
    base_lr = float(cfg["lr_max"]) * args.lr_scale
    if args.phase == "b":
        base_lr *= args.phase_b_lr_scale
    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = max(lr, 1e-7)

    avg_loss = total_loss / max(1, n_steps)
    dt = time.time() - t0

    if epoch % args.val_every == 0 or epoch == 1 or epoch == args.epochs:
        model.eval()
        val_outs = []
        with torch.no_grad():
            for i in range(len(val_pan)):
                vo = tiled_forward(model, val_pan[i:i+1].to(device),
                                   val_ms[i:i+1].to(device), val_lms[i:i+1].to(device)).cpu()
                val_outs.append(vo)
        val_out = torch.cat(val_outs, dim=0)
        m = calculate_metrics((val_out * float(cfg["ratio"])).numpy(),
                              (val_gt * float(cfg["ratio"])).numpy(),
                              ratio=4.0, data_range=float(cfg["ratio"]), q_win_size=8)
        val_q8 = float(m["Q"])
        history.append({"epoch": epoch, "Q8": val_q8, "PSNR": float(m["PSNR"]),
                        "SAM": float(m["SAM"]), "ERGAS": float(m["ERGAS"])})
        print(f"epoch {epoch:03d} loss={avg_loss:.6f} lr={lr:.2e} kl_beta={kl_beta:.2e} dt={dt:.1f}s "
              f"val Q8={val_q8:.6f} PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f} ERGAS={m['ERGAS']:.4f}", flush=True)
        if val_q8 > best_q8:
            best_q8 = val_q8
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(ckpt_dir, "rssm_hz_best.pth"))
    else:
        print(f"epoch {epoch:03d} loss={avg_loss:.6f} lr={lr:.2e} kl_beta={kl_beta:.2e} dt={dt:.1f}s", flush=True)

    if epoch % 50 == 0:
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(ckpt_dir, f"rssm_hz_epoch_{epoch}.pth"))

# Final test eval
print(f"\nBest val Q8={best_q8:.6f}", flush=True)
model.load_state_dict(torch.load(os.path.join(ckpt_dir, "rssm_hz_best.pth"), map_location="cpu")["model"])
model.eval()

test_outs = []
with torch.no_grad():
    for i in range(len(test_pan)):
        to = tiled_forward(model, test_pan[i:i+1].to(device),
                           test_ms[i:i+1].to(device), test_lms[i:i+1].to(device)).cpu()
        test_outs.append(to)
test_out = torch.cat(test_outs, dim=0)
fm = calculate_metrics((test_out * float(cfg["ratio"])).numpy(),
                       (test_gt * float(cfg["ratio"])).numpy(),
                       ratio=4.0, data_range=float(cfg["ratio"]), q_win_size=8)
print("===== Test results =====", flush=True)
for k, v in fm.items():
    print(f"  {k}: {v:.6f}", flush=True)

with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
    json.dump({"test_metrics": {k: float(v) for k, v in fm.items()},
               "best_val_q8": float(best_q8), "val_history": history}, f, indent=2)
print("Done.", flush=True)

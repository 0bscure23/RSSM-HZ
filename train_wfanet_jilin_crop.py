"""Train WFANet on jilin using random crops and tiled inference."""
import os, time, yaml, math, argparse
import h5py, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from net_torch import HWViT
from evaluate_wv3_metrics import calculate_metrics
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CropDataset(Dataset):
    """Random 64x64 PAN crops with corresponding 16x16 MS crops."""
    def __init__(self, pan, gt, ms, lms, ratio, pan_size=64):
        self.pan = pan
        self.gt = gt
        self.ms = ms
        self.lms = lms
        self.ratio = ratio
        self.pan_size = pan_size
        self.ms_size = pan_size // 4

    def __len__(self):
        return len(self.pan) * 4  # multiple crops per image

    def __getitem__(self, idx):
        img_idx = idx % len(self.pan)
        _, _, H, W = self.pan.shape
        y = np.random.randint(0, H - self.pan_size + 1)
        x = np.random.randint(0, W - self.pan_size + 1)
        my, mx = y // 4, x // 4

        return (
            self.pan[img_idx, :, y:y+self.pan_size, x:x+self.pan_size],
            self.gt[img_idx, :, y:y+self.pan_size, x:x+self.pan_size],
            self.ms[img_idx, :, my:my+self.ms_size, mx:mx+self.ms_size],
            self.lms[img_idx, :, y:y+self.pan_size, x:x+self.pan_size],
        )


def tiled_forward(model, pan, ms, lms, tile_size=64, pad=8):
    """Tiled inference for large images to avoid attention OOM.
    pan: [1, 1, H, W], ms: [1, C, h, w], lms: [1, C, H, W]
    """
    _, _, H, W = pan.shape
    C_out = ms.shape[1]
    scale = H // ms.shape[2]  # typically 4

    ts_pan = (tile_size // scale) * scale  # divisible by scale
    ts_ms = ts_pan // scale

    # Round up to tile multiples
    tiles_h = (H + ts_pan - 1) // ts_pan
    tiles_w = (W + ts_pan - 1) // ts_pan
    H_pad = tiles_h * ts_pan
    W_pad = tiles_w * ts_pan
    h_pad = H_pad // scale
    w_pad = W_pad // scale

    # Pad the images
    pan_p = torch.nn.functional.pad(pan, (0, W_pad - W, 0, H_pad - H), mode='reflect')
    ms_p = torch.nn.functional.pad(ms, (0, w_pad - ms.shape[3], 0, h_pad - ms.shape[2]), mode='reflect')
    lms_p = torch.nn.functional.pad(lms, (0, W_pad - W, 0, H_pad - H), mode='reflect')

    # Extended padding for overlap
    pan_ext = torch.nn.functional.pad(pan_p, (pad, pad, pad, pad), mode='reflect')
    ms_ext = torch.nn.functional.pad(ms_p, (pad//scale, pad//scale, pad//scale, pad//scale), mode='reflect')
    lms_ext = torch.nn.functional.pad(lms_p, (pad, pad, pad, pad), mode='reflect')

    output = torch.zeros(1, C_out, H, W, device=pan.device)
    weight = torch.zeros(1, 1, H, W, device=pan.device)

    for ti in range(tiles_h):
        for tj in range(tiles_w):
            pi = ti * ts_pan + pad
            pj = tj * ts_pan + pad
            mi = ti * ts_ms + pad // scale
            mj = tj * ts_ms + pad // scale

            p_tile = pan_ext[:, :, pi-pad:pi+ts_pan+pad, pj-pad:pj+ts_pan+pad]
            m_tile = ms_ext[:, :, mi-pad//scale:mi+ts_ms+pad//scale, mj-pad//scale:mj+ts_ms+pad//scale]
            l_tile = lms_ext[:, :, pi-pad:pi+ts_pan+pad, pj-pad:pj+ts_pan+pad]

            out_tile = model(pan=p_tile, ms=m_tile, lms=l_tile).clamp(0, 1)
            out_crop = out_tile[:, :, pad:pad+ts_pan, pad:pad+ts_pan]

            # Place in output (handle boundary partial tiles)
            oi, oj = ti * ts_pan, tj * ts_pan
            eh = min(ts_pan, H - oi)
            ew = min(ts_pan, W - oj)
            output[:, :, oi:oi+eh, oj:oj+ew] += out_crop[:, :, :eh, :ew]
            weight[:, :, oi:oi+eh, oj:oj+ew] += 1.0

    return output / weight.clamp_min(1.0)


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
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
    val_pan_all = torch.from_numpy(f['pan'][:] / float(cfg["ratio"])).float()
    val_gt_all = torch.from_numpy(f['gt'][:] / float(cfg["ratio"])).float()
    val_ms_all = torch.from_numpy(f['ms'][:] / float(cfg["ratio"])).float()
    val_lms_all = torch.from_numpy(f['lms'][:] / float(cfg["ratio"])).float()

with h5py.File(os.path.join(ROOT_DIR, "Dataset/PanScale_H5/jilin/jilin_test200.h5"), 'r') as f:
    test_pan_all = torch.from_numpy(f['pan'][:] / float(cfg["ratio"])).float()
    test_gt_all = torch.from_numpy(f['gt'][:] / float(cfg["ratio"])).float()
    test_ms_all = torch.from_numpy(f['ms'][:] / float(cfg["ratio"])).float()
    test_lms_all = torch.from_numpy(f['lms'][:] / float(cfg["ratio"])).float()

C = train_ms.shape[1]
print(f"Train: {len(train_pan)}, Val: {len(val_pan_all)}, Test: {len(test_pan_all)} C={C}", flush=True)

model = HWViT(
    L_up_channel=C, pan_channel=1,
    pan_target_channel=int(cfg["pan_target_channel"]),
    ms_target_channel=int(cfg["ms_target_channel"]),
    head_channel=int(cfg["head_channel"]),
    dropout=float(cfg["dropout"]),
).to(device)

print(f"WFANet params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

optimizer = torch.optim.AdamW(
    [{'params': [p for n, p in model.named_parameters() if 'bias' not in n], 'weight_decay': float(cfg['weight_decay'])},
     {'params': [p for n, p in model.named_parameters() if 'bias' in n]}],
    lr=args.lr, betas=(0.9, 0.999))

criterion = nn.L1Loss()

train_ds = CropDataset(train_pan, train_gt, train_ms, train_lms, float(cfg["ratio"]))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

out_dir = os.path.join(ROOT_DIR, "results_rssm_hz", "wfanet_jilin_crop")
ckpt_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

best_q8 = -float("inf")
best_epoch = 0
val_history = []

print("Starting training...", flush=True)
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    n_steps = 0
    t0 = time.time()

    for pan, gt, ms, lms in train_loader:
        pan, gt, ms, lms = pan.to(device), gt.to(device), ms.to(device), lms.to(device)
        optimizer.zero_grad()
        output = model(pan=pan, ms=ms, lms=lms)
        loss = criterion(output, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_steps += 1

    progress = epoch / args.epochs
    lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = max(lr, 1e-6)

    avg_loss = total_loss / max(1, n_steps)
    dt = time.time() - t0

    if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
        model.eval()
        val_outs = []
        with torch.no_grad():
            for i in range(len(val_pan_all)):
                vo = tiled_forward(model, val_pan_all[i:i+1].to(device),
                                   val_ms_all[i:i+1].to(device),
                                   val_lms_all[i:i+1].to(device)).cpu()
                val_outs.append(vo)
        val_out = torch.cat(val_outs, dim=0)
        val_out_scaled = val_out * float(cfg["ratio"])
        val_gt_scaled = val_gt_all * float(cfg["ratio"])
        m = calculate_metrics(val_out_scaled.numpy(), val_gt_scaled.numpy(),
                              ratio=4.0, data_range=float(cfg["ratio"]), q_win_size=8)
        val_q8 = float(m["Q"])
        val_history.append({"epoch": epoch, "Q8": val_q8, "PSNR": float(m["PSNR"]),
                            "SAM": float(m["SAM"]), "ERGAS": float(m["ERGAS"])})
        print(f"epoch {epoch:03d} loss={avg_loss:.6f} lr={lr:.2e} dt={dt:.1f}s "
              f"val Q8={val_q8:.6f} PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f} ERGAS={m['ERGAS']:.4f}", flush=True)
        if val_q8 > best_q8:
            best_q8 = val_q8
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": m},
                       os.path.join(ckpt_dir, "WFANet_jilin_best.pth"))
    else:
        print(f"epoch {epoch:03d} loss={avg_loss:.6f} lr={lr:.2e} dt={dt:.1f}s", flush=True)

    if epoch % 50 == 0:
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(ckpt_dir, f"WFANet_jilin_epoch_{epoch}.pth"))

# Final test eval (tiled)
print(f"\nBest val Q8={best_q8:.6f} at epoch {best_epoch}", flush=True)
model.load_state_dict(torch.load(os.path.join(ckpt_dir, "WFANet_jilin_best.pth"), map_location="cpu")["model"])
model.eval()

test_outs = []
with torch.no_grad():
    for i in range(len(test_pan_all)):
        to = tiled_forward(model, test_pan_all[i:i+1].to(device),
                           test_ms_all[i:i+1].to(device),
                           test_lms_all[i:i+1].to(device)).cpu()
        test_outs.append(to)
test_out = torch.cat(test_outs, dim=0)
test_out_scaled = test_out * float(cfg["ratio"])
test_gt_scaled = test_gt_all * float(cfg["ratio"])
fm = calculate_metrics(test_out_scaled.numpy(), test_gt_scaled.numpy(),
                       ratio=4.0, data_range=float(cfg["ratio"]), q_win_size=8)

print("===== WFANet jilin test results =====", flush=True)
for k, v in fm.items():
    print(f"  {k}: {v:.6f}", flush=True)

with open(os.path.join(out_dir, "wfanet_jilin_metrics.json"), "w") as f:
    json.dump({"test_metrics": {k: float(v) for k, v in fm.items()},
               "best_val_q8": float(best_q8), "best_epoch": best_epoch,
               "val_history": val_history}, f, indent=2)
print("Done.", flush=True)

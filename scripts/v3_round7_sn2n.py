#!/usr/bin/env python3
"""Round7：SN2N 的**忠实移植**（2D，逐 z 切片），训练 + 100% 覆盖推理。

为什么要有这个文件（改动原因）
------------------------------
Round5/6 把 "D1（间距 1 相位对角配对）" 标成了 "SN2N 原型"，那是错的。
SN2N 的真实构造在 `/data2/wjb/SN2N-main` 里可以逐行核对：

    SN2N/datagen.py:158-196  block2d
        ul=[0::2,0::2] ur=[0::2,1::2] dr=[1::2,1::2] dl=[1::2,0::2]
        left  = (ul+dr)/2        # 主对角均值，半分辨率
        right = (ur+dl)/2        # 副对角均值，半分辨率
    SN2N/datagen.py:674-708  block3d       —— 同样只在 xy 平面切，**z 轴不动**
    SN2N/datagen.py:338-386  fourier_inter —— 对称填充 + MATLAB interpft 频域补零，放大 2×
    SN2N/models.py:88-121    Unet_2d       —— 64/128/256/512/512，**末端 sigmoid**
    SN2N/trainer.py:61-98    Adam(lr=2e-4, betas=(0.5,0.999))，L1，
                             loss=(L1(f(A),B)+L1(f(B),A)+w·L1(f(B),f(A)))/(2+w), w=1
    SN2N/inference.py        推理时**整帧单次前向**（不是相位拼装）

与源码的两处**有意偏离**，写在这里而不是藏着：
1. `fourier_inter` 原实现在 64→32→64 这条路径上，其切片索引会切出 127 行再塞进 128 行，
   直接照抄会报维度错。这里按该函数的**意图**（MATLAB `interpft` 频域补零、2× 上采样）
   实现为 `interpft2x`，数值上是零填充插值本身，不引入原实现的边界裁剪。
2. percentile 归一化的参数取 `percentile_normalization.py` 里**实际执行的**
   `pmin=30, pmax=99.8`；同文件 docstring 写的是 20/99.9，两处不一致，以代码为准并在此声明。

本臂的输入 = `data/processed/v3_round7/<seq>/pd_svd_db.npy`（官方血流带 SVD 后的 PD，dB）。
SN2N 原生吃的是 0–255 图像，所以先把 dB 逐片 percentile 归一到 [0,1]，训练/推理后再反变换。

用法
----
    python -u scripts/v3_round7_sn2n.py --seq SA64  --epochs 200
    python -u scripts/v3_round7_sn2n.py --seq HSA64 --epochs 200
产物
----
    runs/v3_round7/sn2n_{seq}/{model.pt,meta.json}
    data/processed/v3_round7/{seq}/pd_sn2n_db.npy     （100% 覆盖，整帧单次前向）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/v3_round7"

PMIN, PMAX = 30.0, 99.8          # percentile_normalization.py 里实际执行的值


# ------------------------------------------------------------------ 网络 ----
class DoubleConv2d(nn.Module):
    """与 SN2N/models.py:DoubleConv_2d 同构：Conv-BN-LeakyReLU ×2。"""

    def __init__(self, cin, cout, mid=None):
        super().__init__()
        mid = mid or cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 3, padding=1), nn.BatchNorm2d(mid),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(mid, cout, 3, padding=1), nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.02, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Unet2d(nn.Module):
    """与 SN2N/models.py:Unet_2d 同构（bilinear=True，末端 sigmoid）。"""

    def __init__(self, cin=1, cout=1):
        super().__init__()
        self.inc = DoubleConv2d(cin, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2d(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2d(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2d(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2d(512, 512))

        def up(cin_, cout_):
            return nn.ModuleDict({
                "up": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                "conv": DoubleConv2d(cin_, cout_, cin_ // 2),
            })

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = nn.Conv2d(64, cout, 1)

    @staticmethod
    def _cat(block, x1, x2):
        x1 = block["up"](x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return block["conv"](torch.cat([x2, x1], dim=1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self._cat(self.up1, x5, x4)
        y = self._cat(self.up2, y, x3)
        y = self._cat(self.up3, y, x2)
        y = self._cat(self.up4, y, x1)
        return torch.sigmoid(self.outc(y))


# -------------------------------------------------------------- 子视图 ----
def block2d(img: np.ndarray):
    """SN2N datagen.py:block2d —— 2×2 块的两条对角均值，各半分辨率。"""
    ul, ur = img[0::2, 0::2], img[0::2, 1::2]
    dr, dl = img[1::2, 1::2], img[1::2, 0::2]
    return 0.5 * (ul + dr), 0.5 * (ur + dl)


def interpft2x(img: np.ndarray) -> np.ndarray:
    """MATLAB interpft 的 2× 频域零填充插值（`fourier_inter` 的意图）。"""
    n, m = img.shape
    spec = np.fft.fft2(img)
    out = np.zeros((2 * n, 2 * m), dtype=complex)
    half_n, half_m = n // 2, m // 2
    # 低频块保持在原位置（与 interpft 的 a[0:nyq] / a[nyq:] 拆分一致）
    out[:half_n, :half_m] = spec[:half_n, :half_m]
    out[:half_n, -half_m:] = spec[:half_n, -half_m:]
    out[-half_n:, :half_m] = spec[-half_n:, :half_m]
    out[-half_n:, -half_m:] = spec[-half_n:, -half_m:]
    y = np.real(np.fft.ifft2(out)) * 4.0
    return np.maximum(y, 0.0)      # fourier_inter 末尾对负值截 0


def percentile_norm(frame: np.ndarray):
    lo, hi = np.percentile(frame, PMIN), np.percentile(frame, PMAX)
    span = max(hi - lo, 1e-8)
    return np.clip((frame - lo) / span, 0.0, 1.0), (lo, hi)


def percentile_inv(x: np.ndarray, lo: float, hi: float):
    return x * (hi - lo) + lo


# ---------------------------------------------------------------- 训练 ----
def build_pairs(db_vol: np.ndarray):
    """每张 z 切片 → (A_up, B_up)，两者都是原尺寸；8 种翻转旋转各一份。"""
    pairs = []
    for z in range(db_vol.shape[0]):
        x01, _ = percentile_norm(db_vol[z])
        for k in range(4):
            for flip in (False, True):
                img = np.rot90(x01, k)
                if flip:
                    img = np.flipud(img)
                a, b = block2d(img)
                pairs.append((interpft2x(a), interpft2x(b)))
    return np.stack([p[0] for p in pairs])[:, None], np.stack([p[1] for p in pairs])[:, None]


def train(db_vol, epochs, bs, lr, device, seed, w=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A, B = build_pairs(db_vol)
    n = len(A)
    model = Unet2d().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)
    l1 = nn.L1Loss()
    At, Bt = torch.from_numpy(A).float(), torch.from_numpy(B).float()
    log = []
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            a, b = At[idx].to(device), Bt[idx].to(device)
            fa, fb = model(a), model(b)
            loss = (l1(fa, b) + l1(fb, a) + w * l1(fb, fa)) / (2 + w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        ep_loss = tot / max(nb, 1)
        sched.step(ep_loss)
        log.append(ep_loss)
        if ep == 1 or ep % 20 == 0:
            print(f"  epoch {ep:3d}/{epochs} loss={ep_loss:.5f} lr={opt.param_groups[0]['lr']:.2e}",
                  flush=True)
    return model, log


@torch.no_grad()
def predict_volume(model, db_vol, device, tta=1):
    """整帧单次前向 → 100% 覆盖。tta=8 时做 8 种翻转旋转再逆变换平均。"""
    out = np.zeros_like(db_vol, dtype=np.float64)
    for z in range(db_vol.shape[0]):
        x01, (lo, hi) = percentile_norm(db_vol[z])
        acc = np.zeros_like(x01)
        cnt = 0
        for k in range(tta if tta in (1, 4, 8) else 1):
            for flip in ((False,) if tta == 1 else (False, True)):
                if tta == 4 and flip:
                    continue
                img = np.rot90(x01, k % 4)
                if flip:
                    img = np.flipud(img)
                t = torch.from_numpy(np.ascontiguousarray(img))[None, None].float().to(device)
                y = model(t)[0, 0].cpu().numpy()
                y = np.flipud(y) if flip else y
                acc += np.rot90(y, -k)
                cnt += 1
        out[z] = percentile_inv(acc / cnt, lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tta", type=int, default=8, choices=[1, 4, 8])
    ap.add_argument("--out", default=None)
    # Round10 追加（改动原因）：要在**半卷**上训练同一套 SN2N，就必须能指定输入/输出路径。
    # 不给这两个参数时行为与原版逐位相同（读 Round7 的全卷 dB，写回原目录）。
    ap.add_argument("--pd-npy", default=None,
                    help="线性 PD 的 .npy；给定时按 Round7 同一规则换算成 dB 当输入。")
    ap.add_argument("--pred-npy", default=None,
                    help="线性 PD 输出的 .npy 路径（由预测 dB 反变换得到）。")
    args = ap.parse_args()

    src = DATA / args.seq
    pd_linear = None
    if args.pd_npy:
        pd_linear = np.load(args.pd_npy).astype(np.float64)
        # 与 scripts/v3_round7_prep.py::to_db 逐字同口径：10log10(PD/max)，零压到地板
        floor = -60.0
        db = 10.0 * np.log10(np.maximum(pd_linear, 1e-12) / pd_linear.max())
        db = np.maximum(np.nan_to_num(db, nan=floor, neginf=floor), floor)
    else:
        db = np.load(src / "pd_svd_db.npy").astype(np.float64)
    run = Path(args.out) if args.out else ROOT / f"runs/v3_round7/sn2n_{args.seq}"
    run.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[SN2N {args.seq}] vol {db.shape} epochs={args.epochs} bs={args.bs} tta={args.tta}")
    model, log = train(db, args.epochs, args.bs, args.lr, args.device, args.seed)
    torch.save({"model": model.state_dict()}, run / "model.pt")

    pred = predict_volume(model, db, args.device, tta=args.tta)
    if args.pred_npy:
        # dB → 线性 PD，与 to_db 严格互逆（except 被地板截断的部分）
        pd_pred = pd_linear.max() * 10.0 ** (pred / 10.0)
        Path(args.pred_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.pred_npy, np.maximum(pd_pred, 0.0).astype(np.float32))
    else:
        np.save(src / "pd_sn2n_db.npy", pred.astype(np.float32))
    meta = {
        "seq": args.seq, "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "seed": args.seed, "tta": args.tta, "pmin": PMIN, "pmax": PMAX,
        "loss_final": log[-1], "loss_first": log[0],
        "minutes": round((time.time() - t0) / 60, 2),
        "inference": "整帧单次前向（SN2N inference.py 口径）→ 100% 覆盖",
        "source": "/data2/wjb/SN2N-main（v0.3.5，逐行核对，见文件头）",
    }
    (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

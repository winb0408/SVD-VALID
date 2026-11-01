"""Round4-D 数值检验（**不训练**）：对角/互补子视图构造 vs 当前 Tetris 视图。

对照对象：Zhang et al. 2026, *SVD-SN2N*（bioRxiv `10.64898/2026.09.10.750668`）。
其 "Diagonal Resampling" 做法：在每个 2×2 像素块里取 `x1=(p1+p4)/2`、`x2=(p2+p3)/2`，
再用 Fourier 零填充插值把两个半分辨率伴图**放大回原网格**。论文给的理由很明确：
2×2 块的三种二分割里，**只有对角分割的两个伴图共享同一质心**，因此不引入半像素位移；
另外论文还实测了两伴图噪声的相关系数 0.39（并非 0），并说明这个偏离只会**少去噪**、不会造结构。

本项目的 Tetris 采样是 3D 的 2×2×2 块 + 4+4 角点互补 tet 配对（VALID 口径）：

    tet A = {0,1,2,4} 质心 (0.25,0.25,0.25)
    tet B = {3,5,6,7} 质心 (0.75,0.75,0.75)
    ⇒ 两伴图相差 (0.5,0.5,0.5) 体素的**体对角位移**，与论文强调的"零位移"不同。

在**已知真值**的仿真卷（low 档 e902–e905）上，对四种构造各量三件事：

  1. 信号几何：`std(S1−S2)/std(S1)`（两伴图信号是否同一）+ 与全分辨率真值的相对误差；
  2. 残余依赖：`corr(n1, n2)`，n = 伴图 − 该伴图的干净对应；
  3. 插值代价：Fourier 零填充**前后**的 `corr(n1, n2)`（论文声称插值不引入体素间相关）。

只量构造本身，不训练；"相关更低"不等于"训练更好"。

用法
----
    python -u scripts/v3_round4_d_subview.py
产物
----
    reports/v3_round4/d_subview.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load                                   # noqa: E402
from datasets.sampling import TETRIS_PATTERNS                        # noqa: E402

SCENES = [902, 903, 904, 905]
LEVEL = "low"
TET_A = list(TETRIS_PATTERNS[0].tolist())          # [0,1,2,4]
TET_B = sorted(set(range(8)) - set(TET_A))         # [3,5,6,7]
# 共享质心的 4+4 分割（"偶宇称"对"奇宇称"）：两半质心都是 (0.5,0.5,0.5)。
# 逐模式核验见 JSON `valid_patterns_centroid_shared`：VALID 的 8 个模式**没有一个是**共享质心的。
PAR_A = [0, 3, 5, 6]
PAR_B = [1, 2, 4, 7]


def corner_offset(i: int):
    return i // 4, (i // 2) % 2, i % 2


def even_crop(*arrs):
    """所有轴裁成偶数（相位/块平均要求等长），返回裁后的数组列表。"""
    shape = [min(a.shape) for a in arrs]
    shp = tuple((s // 2) * 2 for s in shape)
    return [a[:shp[0], :shp[1], :shp[2]] for a in arrs]


def blk3(vol: np.ndarray, corners) -> np.ndarray:
    """每 2×2×2 块内给定角点求平均 → 各轴 /2。"""
    D, H, W = vol.shape
    acc = None
    for i in corners:
        dz, dy, dx = corner_offset(i)
        ph = vol[dz::2, dy::2, dx::2].astype(np.float64)
        acc = ph if acc is None else acc + ph
    return acc / len(corners)


def blk2(vol: np.ndarray, which: int) -> np.ndarray:
    """论文 2D 对角：(y,x) 平面上 x1=(p1+p4)/2 或 x2=(p2+p3)/2 → y,x 各 /2。"""
    a = vol[:, 0::2, 0::2].astype(np.float64)
    b = vol[:, 1::2, 1::2].astype(np.float64)
    c = vol[:, 0::2, 1::2].astype(np.float64)
    d = vol[:, 1::2, 0::2].astype(np.float64)
    return 0.5 * (a + b) if which == 0 else 0.5 * (c + d)


def _up_axis(y: np.ndarray, ax: int) -> np.ndarray:
    n = y.shape[ax]
    F = np.fft.fftshift(np.fft.fft(y, axis=ax), axes=ax)
    pad = [(0, 0)] * y.ndim
    pad[ax] = (n // 2, n // 2)
    F2 = np.fft.ifftshift(np.pad(F, pad), axes=ax)
    return np.real(np.fft.ifft(F2, axis=ax)) * 2.0        # 长度 2n


def fourier_up(y: np.ndarray, axes) -> np.ndarray:
    """Fourier 零填充插值（带限），沿给定轴各放大 2 倍。"""
    for ax in axes:
        y = _up_axis(y, ax)
    return y


def stats(S1, S2, n1, n2, ves, bg) -> dict:
    dmis = S1 - S2
    out = {
        "signal_mismatch_rel": float(np.std(dmis[ves]) / max(np.std(S1[ves]), 1e-12)),
        "noise_corr_ves": float(np.corrcoef(n1[ves], n2[ves])[0, 1]),
        "noise_corr_bg": float(np.corrcoef(n1[bg], n2[bg])[0, 1]) if bg.any() else float("nan"),
        "noise_var_ves": float(np.var(n1[ves])),
        "n_ves": int(ves.sum()),
    }
    return out


def masks_for(grid_shape, valid, rho, mode: str):
    """在给定网格上重建掩膜：mode='blk3' / 'blk2' 用块平均，'full' 用原分辨率裁切。"""
    if mode == "full":
        v = valid.astype(np.float64)
        r = rho.astype(np.float64)
    elif mode == "blk3":
        v = blk3(valid.astype(np.float64), range(8))
        r = blk3(rho.astype(np.float64), range(8))
    else:  # blk2：(y,x) 上 4 像素平均（**先转 float 再相加**：bool+bool 在 numpy 里是 OR）
        vf = valid.astype(np.float64)
        rf = rho.astype(np.float64)
        v = 0.25 * (vf[:, 0::2, 0::2] + vf[:, 1::2, 1::2]
                    + vf[:, 0::2, 1::2] + vf[:, 1::2, 0::2])
        r = 0.25 * (rf[:, 0::2, 0::2] + rf[:, 1::2, 1::2]
                    + rf[:, 0::2, 1::2] + rf[:, 1::2, 0::2])
    vmask = v > 0.999
    ves = vmask & (r >= 0.1)
    bg = vmask & (r <= 0.0)
    assert vmask.shape == grid_shape, (vmask.shape, grid_shape)
    return ves, bg


def main() -> None:
    rec = {
        "prereg": "reviews/V3_PR1_round4_prereg.md §4",
        "paper": {
            "doi": "10.64898/2026.09.10.750668",
            "title": "SVD-SN2N: Self-Inspired Noise2Noise Learning for Denoising "
                     "Log-Compressed SVD-Filtered Ultrasound Imaging",
            "text_claim": "2×2 块的三种二分割里只有对角分割两伴图共享质心 ⇒ 不引入半像素位移",
            "their_measured": {"x1_x2_signal_corr": 0.960, "residual_corr_between_copies": 0.39,
                               "residual_neighbour_corr": {"axial": 0.09, "lateral": 0.31}},
        },
        "constructions": {
            "tetris3d": {"A_corners": TET_A, "B_corners": TET_B,
                         "centroid_A": [0.25, 0.25, 0.25], "centroid_B": [0.75, 0.75, 0.75],
                         "shift_vox": [0.5, 0.5, 0.5]},
            "paper2d_diagonal": {"centroid_shift_vox": [0.0, 0.0, 0.0]},
            "tetris3d_parity": {"A_corners": PAR_A, "B_corners": PAR_B,
                                "centroid_A": [0.5, 0.5, 0.5], "centroid_B": [0.5, 0.5, 0.5],
                                "shift_vox": [0.0, 0.0, 0.0]},
            "valid_patterns_centroid_shared": None,
        },
        "scenes": {},
    }

    # VALID 的 8 个 tetris 模式里有没有"共享质心"的？（逐模式核验，写进 JSON）
    cen = []
    for pat in TETRIS_PATTERNS.tolist():
        c = [sum(corner_offset(i)[k] for i in pat) / 4 for k in range(3)]
        comp = [i for i in range(8) if i not in pat]
        cc = [sum(corner_offset(i)[k] for i in comp) / 4 for k in range(3)]
        cen.append({"pattern": pat, "centroid_A": c, "centroid_B": cc,
                    "shared": bool(all(abs(a - b) < 1e-9 for a, b in zip(c, cc)))})
    rec["constructions"]["valid_patterns_centroid_shared"] = {
        "per_pattern": cen, "any_shared": any(c["shared"] for c in cen)}

    for s in SCENES:
        E = load(ROOT / f"data/processed/v3_round2_s0/{LEVEL}6/eval/s0_{LEVEL}_e{s}.npz")
        vol, truth, valid, rho = even_crop(E["vol"], E["truth_img"], E["valid"], E["rho"])
        row = {}

        # ---- 构造 1：3D 互补 tet（本项目 Tetris 口径）----
        for tag in ("raw", "fourier"):
            XA, XB = blk3(vol, TET_A), blk3(vol, TET_B)
            SA, SB = blk3(truth, TET_A), blk3(truth, TET_B)
            if tag == "fourier":
                XA, XB = fourier_up(XA, (0, 1, 2)), fourier_up(XB, (0, 1, 2))
                SA, SB = fourier_up(SA, (0, 1, 2)), fourier_up(SB, (0, 1, 2))
                ves, bg = masks_for(XA.shape, valid, rho, "full")
            else:
                ves, bg = masks_for(XA.shape, valid, rho, "blk3")
            row[f"tetris3d_{tag}"] = stats(SA, SB, XA - SA, XB - SB, ves, bg)
            # 单个伴图相对"8 角点全平均"信号的形变（0 = 伴图信号无失真）
            SAvg = blk3(truth, range(8))
            if tag == "fourier":
                SAvg = fourier_up(SAvg, (0, 1, 2))
            else:
                SAvg = SAvg[:SA.shape[0], :SA.shape[1], :SA.shape[2]]
            row[f"tetris3d_{tag}"]["signal_distortion_rel"] = float(
                np.std((SA - SAvg)[ves]) / max(np.std(SAvg[ves]), 1e-12))

        # ---- 构造 1b：**共享质心**的 4+4 分割（论文在 2D 里主张的那条性质，3D 推广）----
        # 2×2×2 块的 35 种二分割里，4+4 分割中只有"偶宇称 tet {0,3,5,6} 对奇宇称 {1,2,4,7}"
        # 两半质心同为 (0.5,0.5,0.5)。**VALID 的 8 个 tetris 模式里没有这一个**（逐模式核验见 JSON）。
        for tag in ("raw", "fourier"):
            XA, XB = blk3(vol, PAR_A), blk3(vol, PAR_B)
            SA, SB = blk3(truth, PAR_A), blk3(truth, PAR_B)
            if tag == "fourier":
                XA, XB = fourier_up(XA, (0, 1, 2)), fourier_up(XB, (0, 1, 2))
                SA, SB = fourier_up(SA, (0, 1, 2)), fourier_up(SB, (0, 1, 2))
                ves, bg = masks_for(XA.shape, valid, rho, "full")
            else:
                ves, bg = masks_for(XA.shape, valid, rho, "blk3")
            row[f"tetris3d_parity_{tag}"] = stats(SA, SB, XA - SA, XB - SB, ves, bg)
            SAvg = blk3(truth, range(8))
            if tag == "fourier":
                SAvg = fourier_up(SAvg, (0, 1, 2))
            row[f"tetris3d_parity_{tag}"]["signal_distortion_rel"] = float(
                np.std((SA - SAvg)[ves]) / max(np.std(SAvg[ves]), 1e-12))

        # ---- 构造 2：论文 2D 对角 ----
        for tag in ("raw", "fourier"):
            X1, X2 = blk2(vol, 0), blk2(vol, 1)
            S1, S2 = blk2(truth, 0), blk2(truth, 1)
            if tag == "fourier":
                X1, X2 = fourier_up(X1, (1, 2)), fourier_up(X2, (1, 2))
                S1, S2 = fourier_up(S1, (1, 2)), fourier_up(S2, (1, 2))
                ves, bg = masks_for(X1.shape, valid, rho, "full")
            else:
                ves, bg = masks_for(X1.shape, valid, rho, "blk2")
            row[f"paper2d_{tag}"] = stats(S1, S2, X1 - S1, X2 - S2, ves, bg)

        # ---- 参照：不做任何拆分时的原始噪声方差（供"降了多少"对照）----
        ves_f, _ = masks_for(vol.shape, valid, rho, "full")
        row["reference"] = {"noise_var_ves_full_res": float(np.var((vol - truth)[ves_f]))}
        rec["scenes"][f"e{s}"] = row
        print(f"[e{s}] " + json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                           for kk, vv in v.items()} for k, v in row.items()},
                                      ensure_ascii=False))

    keys = [k for k in next(iter(rec["scenes"].values())) if k != "reference"]
    rec["summary_median"] = {
        k: {kk: float(np.median([rec["scenes"][f"e{s}"][k][kk] for s in SCENES]))
            for kk in ("signal_mismatch_rel", "noise_corr_ves", "noise_corr_bg", "noise_var_ves")}
        for k in keys
    }
    p = ROOT / "reports/v3_round4/d_subview.json"
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec["summary_median"], indent=1, ensure_ascii=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

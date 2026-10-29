"""Round4 结构指标 v2：**段级可检测性**（segment-level detectability）。

定义与通过线见 `reviews/V3_PR1_round4_prereg_v2_metric.md`（先写死，再跑）。
v1（voxel 级 `recall@fpr5`，见 `scripts/v3_round4_metrics.py`）的已知答案控制**未通过**
（0/6 与 1/6），根因是分辨率：管径 < PSF，体素级上"弱血管 vs 背景"在无噪真值里就已重叠。

v2 把评价尺度换到**整条血管段**：段的分割只用 phantom 几何（`rho`），不看任何观测；
在段内取平均后再比背景噪声，PSF 的分辨率限制被平均掉——这是匹配滤波/CNR 的常规做法。

| 端点 | 定义 | 方向 |
|---|---|---|
| `seg_cnr_median` | `median_s (mean(pred[V_s]) − bg_mu) / bg_sd` | 越大越好 |
| `seg_cnr_p10` | 段级 CNR 的 10% 分位（最弱那批管子） | 越大越好 |
| `seg_recall_z2` | `mean_s(snr_s ≥ 2)`（z=2 固定，不调） | 越大越好 |
| `fp_segments` | `pred > bg_mu + 2·bg_sd` 在 `bg` 内的连通分量数（≥30 体素） | 越小越好 |

`n2n` / `rmse_img` / `bg_std_ratio` 沿用 v1 的可核验标量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load, masked_gauss, blend   # noqa: E402,F401  (re-export)

MIN_SEG_VOX = 30          # 段的最小体素数（预注册 v2 §2.1）
Z_DETECT = 2.0            # 段级检出阈值，固定不调
METRIC_VERSION_V2 = "R4-v2"


def segments(E: dict) -> dict:
    """只用 `rho` 切段：`valid & (rho ≥ 0.1)` 的连通分量，保留 ≥30 体素。结果缓存在 E 里。"""
    if "_segs_v2" in E:
        return E["_segs_v2"]
    m = E["valid"] & (E["rho"] >= 0.1)
    lab, n = label(m)
    idx, sizes = [], []
    if n:
        cnt = np.bincount(lab.ravel())
        for k in range(1, n + 1):
            if cnt[k] >= MIN_SEG_VOX:
                idx.append(np.flatnonzero(lab.ravel() == k))
                sizes.append(int(cnt[k]))
    rec = {"idx": idx, "sizes": np.array(sizes, dtype=int),
           "mask": m, "n": len(idx), "min_seg_vox": MIN_SEG_VOX}
    E["_segs_v2"] = rec
    return rec


def _seg_snr(pred_flat: np.ndarray, seg: dict, bg_mu: float, bg_sd: float) -> np.ndarray:
    return np.array([(float(pred_flat[i].mean()) - bg_mu) / bg_sd for i in seg["idx"]])


def metrics_v2(pred: np.ndarray, E: dict) -> dict:
    v, bg = E["valid"], E["bg"]
    seg = segments(E)
    flat = pred.ravel()
    bg_mu = float(pred[bg].mean()) if bg.any() else 0.0
    bg_sd = float(pred[bg].std()) if bg.any() else 0.0
    out = {"metric_version": METRIC_VERSION_V2, "n_seg": seg["n"],
           "bg_mu": bg_mu, "bg_sd": bg_sd}

    if seg["n"] and bg_sd > 0:
        snr = _seg_snr(flat, seg, bg_mu, bg_sd)
        out.update({
            "seg_cnr_median": float(np.median(snr)),
            "seg_cnr_p10": float(np.quantile(snr, 0.10)),
            "seg_cnr_mean": float(snr.mean()),
            "seg_recall_z2": float(np.mean(snr >= Z_DETECT)),
            "seg_snr": snr,
        })
        hit = np.zeros(pred.shape, bool)
        hit[bg] = pred[bg] > bg_mu + Z_DETECT * bg_sd
        lab, n = label(hit)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
        out["fp_segments"] = int(np.sum(sizes >= MIN_SEG_VOX))
    else:
        out.update({"seg_cnr_median": float("nan"), "seg_cnr_p10": float("nan"),
                    "seg_cnr_mean": float("nan"), "seg_recall_z2": float("nan"),
                    "seg_snr": np.array([]), "fp_segments": 0})

    # 与 v1 一致的可核验标量（不依赖结构解释）
    out["n2n"] = float(np.mean((pred[v] - E["vol2"][v]) ** 2) /
                       max(np.mean((E["vol"][v] - E["vol2"][v]) ** 2), 1e-12))
    out["rmse_img"] = float(np.sqrt(np.mean((pred[v] - E["truth_img"][v]) ** 2)))
    out["bg_std_ratio"] = (float(np.std(pred[bg]) / np.std(E["vol"][bg]))
                           if bg.any() and np.std(E["vol"][bg]) > 0 else float("nan"))
    out["width"] = "NOT_RESOLVABLE"
    return out


# ------------------------------------------------------------------ 控制臂
def controls_v2(E: dict, sigma_gauss: float, cut_frac: float = 0.30) -> dict:
    """六个已知答案臂。构造在预注册 v2 §2.2 里写死，这里只是实现。"""
    ref, valid, bg = E["truth_img"], E["valid"], E["bg"]
    seg = segments(E)

    # 负控制 A：抹掉体积最小的 cut_frac 段（确定性，不用随机数）
    cut = ref.copy()
    if seg["n"]:
        order = np.argsort(seg["sizes"], kind="mergesort")
        n_cut = max(1, int(round(cut_frac * seg["n"])))
        for j in order[:n_cut]:
            cut.ravel()[seg["idx"][j]] = 0.0
    cut[~valid] = 0.0

    # 负控制 B：背景里插一条直管，强度按构造 = bg_mu + 3·bg_sd（保证可检出）
    fake = ref.copy()
    fake[~valid] = 0.0
    bg_mu = float(ref[bg].mean()) if bg.any() else 0.0
    bg_sd = float(ref[bg].std()) if bg.any() else 0.0
    level = bg_mu + 3.0 * bg_sd
    zc, xc = ref.shape[0] // 2, ref.shape[2] // 2
    tube = np.zeros(ref.shape, bool)
    yy = np.arange(8, ref.shape[1] - 8)
    for y in yy:
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                z, x = zc + dz, xc + dx
                if 0 <= z < ref.shape[0] and 0 <= x < ref.shape[2]:
                    v = (z, y, x)
                    if bg[v] or not valid[v]:
                        tube[v] = True
    # 只在完整落在 bg 里的部分保留：断开的碎段不算"一条管"
    labt, nt = label(tube)
    if nt:
        sizes = np.bincount(labt.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        tube = labt == keep
    fake[tube] = np.maximum(fake[tube], level)
    fake[~valid] = 0.0

    return {
        "truth_img": ref,
        "input": E["vol"],
        "gauss": masked_gauss(E["vol"], valid, sigma_gauss),
        "zero": np.zeros_like(ref),
        "cut30": cut,
        "fake": fake,
    }

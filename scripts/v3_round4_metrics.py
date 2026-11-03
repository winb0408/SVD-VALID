"""Round4 结构保真指标：**把真值换成 PSF 匹配的成像真值**（`gt_imaging_pd`）。

## 为什么要有这个文件

R12（`scripts/v3_eval_microvessel.py`）的已知答案标定已经证伪了 `recall@fpr5` / AUC /
`iou_core` / SSIM 的单调性：**无噪真值自己在 recall 上输给含噪输入**。根因写在它的 docstring 里
——真值用的是锐利的 `gt_flow_pd`（未过 PSF），而所有预测都来自被 PSF 模糊过的 PD 图像。
拿锐利真值去做体素级"血管 vs 背景"可分性检验，等于让无噪真值承担了一个它不该承担的分辨率劣势。

Round4 把参照换成 `log1p(gt_imaging_pd/s)`（PSF 已含的无噪成像真值），并且**不假设**换了参照就
一定单调——这一点由 `scripts/v3_round4_metric_controls.py` 的已知答案表去检验，不通过就记
`METRIC_UNRESOLVED`（见 `reviews/V3_PR1_round4_prereg.md` §2.3）。

## 端点（冻结定义见预注册 §2.2）

| 名字 | 定义 |
|---|---|
| `n2n` | `mse(pred, obs2) / mse(obs1, obs2)`，`valid` 上 |
| `rmse_img` | 对 `truth_img` 的 RMSE（log1p 域，`valid` 上） |
| `recall_at_fpr5` | 阈值 = `quantile(pred[bg], 0.95)`，召回 = `mean(pred[weak] > thr)` |
| `f1_weak` | `2PR/(P+R)`，P = 命中数 / `pred[valid] > thr` 的总数 |
| `ncomp_ratio` | 预测前景（与 `core` 同体素预算 top-k）连通分量数 / `core` 连通分量数 |
| `fp_blobs` | `pred[bg] > thr` 的连通分量中体素 ≥8 的个数（"假血管计数"的替代） |
| `bg_std_ratio` | `std(pred[bg]) / std(input[bg])` |
| `iou_core` / `r_weak` / `r_core` | 沿用 R12 定义，只作描述 |

`width` **不做**：`PSF_PD σ=(4.24,3.35,5.61)` 体素 vs 血管半径 0.75/1.1/1.5 体素，
成像后半高宽由 PSF 决定，测出来的是 PSF 不是血管（与 R12 撤除理由一致，预注册 §1）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

WEAK_LO, WEAK_HI = 0.1, 0.5      # 与 R12 一致，未改
MIN_COMP_VOX = 8                 # 与 R12 一致
FPR_NOMINAL = 0.05               # 与 R12 一致
METRIC_VERSION = "R4-v1"


# ------------------------------------------------------------------ 读取
def load(npz: Path) -> dict:
    """读一卷仿真数据，全部换到 log1p 域。区域只看 `rho`，不看任何观测。"""
    d = np.load(npz)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s = float(d["s"])
    lg = lambda a: np.log1p(np.maximum(a.astype(np.float32), 0.0) / max(s, 1e-12))
    vol, vol2 = lg(d["pd_linear"]), lg(d["pd2_linear"])
    truth_img = lg(d["gt_imaging_pd"])          # ← PSF 匹配的无噪成像真值（本轮改的就是这里）
    flow = lg(d["gt_flow_pd"])                  # 锐利血流真值：保留下来只作诊断对照
    for a in (vol, vol2, truth_img, flow):
        a[~valid] = 0.0
    return {
        "vol": vol, "vol2": vol2, "truth_img": truth_img, "flow": flow,
        "valid": valid, "rho": rho, "s": s, "npz": str(npz),
        "core": valid & (rho >= 0.5),
        "deep": valid & (rho >= 0.9),
        "weak": valid & (rho >= WEAK_LO) & (rho < WEAK_HI),
        "bg": valid & (rho <= 0.0),
    }


# ------------------------------------------------------------------ 小工具
def _pearson(a, b) -> float:
    if a.size < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _ncomp(mask: np.ndarray) -> int:
    lab, n = label(mask)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(np.sum(sizes >= MIN_COMP_VOX))


def _topk_mask(pred: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """与 `core` 同体素预算的前景：所有臂同一预算，防"把图糊平就赢"。"""
    out = np.zeros_like(pred, bool)
    if k <= 0:
        return out
    flat = pred[valid]
    thr = float(np.sort(flat)[-k]) if flat.size >= k else float(flat.min())
    out[valid] = pred[valid] >= thr
    return out


# ------------------------------------------------------------------ 主端点
def metrics(pred: np.ndarray, E: dict) -> dict:
    """`pred` 必须与 `E["vol"]` 同形状、同一 log1p 尺度。"""
    v, core, weak, bg, deep = E["valid"], E["core"], E["weak"], E["bg"], E["deep"]
    ref = E["truth_img"]

    out = {
        "metric_version": METRIC_VERSION,
        "n2n": float(np.mean((pred[v] - E["vol2"][v]) ** 2) /
                     max(np.mean((E["vol"][v] - E["vol2"][v]) ** 2), 1e-12)),
        "rmse_img": float(np.sqrt(np.mean((pred[v] - ref[v]) ** 2))),
        "rmse_input": float(np.sqrt(np.mean((E["vol"][v] - ref[v]) ** 2))),
        "r_weak": _pearson(pred[weak], ref[weak]),
        "r_core": _pearson(pred[core], ref[core]),
        "n_weak": int(weak.sum()), "n_core": int(core.sum()), "n_bg": int(bg.sum()),
        "bg_std_ratio": (float(np.std(pred[bg]) / np.std(E["vol"][bg]))
                         if bg.any() and np.std(E["vol"][bg]) > 0 else float("nan")),
        "width": "NOT_RESOLVABLE",
        "width_reason": ("PSF_PD σ=(4.24,3.35,5.61) voxel 远宽于血管半径 0.75–1.5 voxel，"
                         "成像后半高宽由 PSF 决定、对方法不敏感"),
    }

    if bg.sum() and weak.sum():
        thr = float(np.quantile(pred[bg], 1.0 - FPR_NOMINAL))
        hit = pred > thr
        out["thr"] = thr
        out["recall_at_fpr5"] = float(np.mean(pred[weak] > thr))
        out["fpr_actual"] = float(np.mean(pred[bg] > thr))
        n_hit = int((hit & v).sum())
        out["precision_weak"] = float((hit & weak).sum() / n_hit) if n_hit else float("nan")
        p, r = out["precision_weak"], out["recall_at_fpr5"]
        out["f1_weak"] = float(2 * p * r / (p + r)) if p and r and (p + r) > 0 else 0.0
        lab, n = label(hit & bg)
        sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
        out["fp_blobs"] = int(np.sum(sizes >= MIN_COMP_VOX))
        out["fp_voxels"] = int((hit & bg).sum())
    else:
        out.update({"thr": None, "recall_at_fpr5": float("nan"), "fpr_actual": float("nan"),
                    "precision_weak": float("nan"), "f1_weak": float("nan"),
                    "fp_blobs": 0, "fp_voxels": 0})

    k = int(core.sum())
    if k > 0:
        fg = _topk_mask(pred, v, k)
        out["iou_core"] = float((fg & core).sum() / max((fg | core).sum(), 1))
        out["ncomp_pred"] = _ncomp(fg)
        out["ncomp_truth"] = _ncomp(core)
        out["ncomp_ratio"] = (out["ncomp_pred"] / out["ncomp_truth"]
                              if out["ncomp_truth"] else float("nan"))
    else:
        out.update({"iou_core": float("nan"), "ncomp_pred": 0, "ncomp_truth": 0,
                    "ncomp_ratio": float("nan")})
    return out


# ------------------------------------------------------------------ 算子
def masked_gauss(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """只在 `valid` 内平滑（**归一化卷积**，与 A5 的 `blur()` 逐字同口径）。

    为什么必须同口径：σ*（low 0.6 / vlow 0.5）是和 A5 那个算子一起冻结的。
    换一种边界处理（例如掩膜外填均值）会得到另一条 n2n-σ 曲线，σ* 就不再是同一个东西。
    """
    if sigma <= 0:
        return x.copy()
    m = valid.astype(np.float32)
    num = gaussian_filter(x * m, sigma=sigma, mode="nearest")
    den = gaussian_filter(m, sigma=sigma, mode="nearest")
    y = np.where(den > 1e-6, num / np.maximum(den, 1e-6), x).astype(np.float32)
    y[~valid] = 0.0
    return y


def wiener_shrink(x: np.ndarray, valid: np.ndarray, w: int) -> np.ndarray:
    """Phase4 ①：局部 Wiener 型收缩 `out = mu + max(0, 1 − σ²/lv)(x − mu)`。

    σ² = `median(lv[valid])`——本数据绝大多数体素是背景，这个估计是数据驱动的、不用 obs2。
    """
    from scipy.ndimage import uniform_filter
    xf = np.where(valid, x, 0.0).astype(np.float64)
    mu = uniform_filter(xf, w)
    lv = np.maximum(uniform_filter(xf * xf, w) - mu ** 2, 0.0)
    s2 = float(np.median(lv[valid])) if valid.any() else 0.0
    b = np.clip(1.0 - s2 / np.maximum(lv, 1e-12), 0.0, 1.0)
    y = (mu + b * (xf - mu)).astype(np.float32)
    y[~valid] = 0.0
    return y


def blend(x: np.ndarray, pred: np.ndarray, beta: float) -> np.ndarray:
    """网络臂的 Pareto 旋钮：β=0 回到输入，β=1 即原输出。"""
    y = (x + beta * (pred - x)).astype(np.float32)
    return y


# ------------------------------------------------------------------ 已知答案
def known_answers(E: dict, sigma_gauss: float, seed: int = 0) -> dict:
    """正负控制臂。全部**只改真值**，不碰任何观测，保证"答案已知"。

    - `truth_img`：完美答案
    - `input`：什么都不做
    - `gauss`：固定 σ* 的掩膜高斯（A5 冻结值）
    - `cut`：把真值里最大的一条血管连通域整条抹掉（断枝负控制）
    - `fake`：在真值背景区造一条直管（造假负控制）
    """
    ref, valid, bg = E["truth_img"], E["valid"], E["bg"]

    cut = ref.copy()
    lab, n = label(E["core"])
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        big = int(np.argmax(sizes)) + 1
        cut[lab == big] = 0.0
        cut[~valid] = 0.0

    fake = ref.copy()
    fake[~valid] = 0.0
    # 在背景里放一条沿 y 的直管：位置由 rho 决定（确定性），强度 = 弱血管中位强度
    zc, xc = ref.shape[0] // 2, ref.shape[2] // 2
    level = float(np.median(ref[E["weak"]])) if E["weak"].any() else 0.0
    tube = np.zeros_like(ref, bool)
    ys = np.linspace(10, ref.shape[1] - 10, ref.shape[1] - 20).astype(int)
    for y in ys:
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                z, x = zc + dz, xc + dx
                if 0 <= z < ref.shape[0] and 0 <= x < ref.shape[2] and bg[z, y, x]:
                    tube[z, y, x] = True
    fake[tube] = np.maximum(fake[tube], level)
    fake[~valid] = 0.0

    return {
        "truth_img": ref,
        "input": E["vol"],
        "gauss": masked_gauss(E["vol"], valid, sigma_gauss),
        "cut": cut,
        "fake": fake,
    }

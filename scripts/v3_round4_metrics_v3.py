"""Round4 结构指标 v3：**偏差—方差分解**（对 PSF 匹配真值的代数分解）。

定义与通过线见 `reviews/V3_PR1_round4_prereg_v3_metric.md`（先写死，再跑）。

v1（voxel 级检测）与 v2（段级 CNR）的已知答案控制都失败，且失败形态指向同一根因：
管径 ≪ PSF、噪声在段尺度被平均掉 ⇒ **任何阈值化检测量在本仿真上都没有判别区间**。
v3 不做阈值化，只做 `rmse² = bias² + var` 的分解——正确性由代数恒等式与构造臂的严格关系检验。

用法见 `scripts/v3_round4_metric_checks_v3.py`。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load, masked_gauss, wiener_shrink, blend   # noqa: E402,F401 (re-export)
from v3_round4_metrics_v2 import segments                                # noqa: E402,F401 (re-export)

METRIC_VERSION_V3 = "R4-v3"


def _decomp(e: np.ndarray, mask: np.ndarray) -> dict:
    x = e[mask]
    if x.size == 0:
        return {"bias": float("nan"), "var": float("nan"), "rmse": float("nan"), "n": 0}
    b = float(np.mean(x))
    v = float(np.var(x))
    return {"bias": b, "var": v, "rmse": float(np.sqrt(np.mean(x ** 2))), "n": int(x.size)}


def metrics_v3(pred: np.ndarray, E: dict) -> dict:
    v = E["valid"]
    ves = v & (E["rho"] >= 0.1)
    bg = v & (E["rho"] <= 0.0)
    e = (pred - E["truth_img"]).astype(np.float64)

    out = {"metric_version": METRIC_VERSION_V3,
           "ves": _decomp(e, ves), "bg": _decomp(e, bg),
           "all": _decomp(e, v)}
    out["overall_rmse"] = out["all"]["rmse"]
    out["bias_ves"] = out["ves"]["bias"]
    out["var_ves"] = out["ves"]["var"]
    out["bias_bg"] = out["bg"]["bias"]
    out["var_bg"] = out["bg"]["var"]

    # 可核验标量（与 v1 同定义，便于与历史结论对照）
    out["n2n"] = float(np.mean((pred[v] - E["vol2"][v]) ** 2) /
                       max(np.mean((E["vol"][v] - E["vol2"][v]) ** 2), 1e-12))
    out["bg_std_ratio"] = (float(np.std(pred[bg]) / np.std(E["vol"][bg]))
                           if bg.any() and np.std(E["vol"][bg]) > 0 else float("nan"))
    out["rmse_vs_input"] = float(np.sqrt(np.mean((E["vol"][v] - E["truth_img"][v]) ** 2)))

    # 逐段幅度保留率（**描述性**，不作判据）：10% 分位 = 最弱那批管子保留了多少
    seg = segments(E)
    if seg["n"]:
        tref = E["truth_img"].ravel()
        pr = pred.ravel()
        ret = []
        for i in seg["idx"]:
            t = float(tref[i].mean())
            if t > 1e-6:
                ret.append(float(pr[i].mean()) / t)
        out["ret_p10"] = float(np.quantile(ret, 0.10)) if ret else float("nan")
        out["ret_median"] = float(np.median(ret)) if ret else float("nan")
    else:
        out["ret_p10"] = out["ret_median"] = float("nan")

    out["width"] = "NOT_RESOLVABLE"
    return out


# ------------------------------------------------------------------ 控制臂
def controls_v3(E: dict, sigma_gauss: float, cut_frac: float = 0.30) -> dict:
    """四个构造臂 + 输入 + 高斯。全部**只改真值或已知算子**，答案可解析。"""
    ref, valid = E["truth_img"], E["valid"]
    seg = segments(E)

    cut = ref.copy()
    if seg["n"]:
        order = np.argsort(seg["sizes"], kind="mergesort")
        for j in order[:max(1, int(round(cut_frac * seg["n"])))]:
            cut.ravel()[seg["idx"][j]] = 0.0
    cut[~valid] = 0.0

    return {
        "truth_img": ref,
        "zero": np.zeros_like(ref),
        "input": E["vol"],
        "gauss": masked_gauss(E["vol"], valid, sigma_gauss),
        "cut30": cut,
    }

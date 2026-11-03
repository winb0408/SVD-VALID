#!/usr/bin/env python3
"""Round8 判定用的补充分析：固定 Gaussian / Wiener 对照 + 输入输出配准 + 稳健背景统计。

为什么单独写一个脚本而不改 `v3_round8_zero_train_full_infer.py`
--------------------------------------------------------------
那份脚本是审查方提供的产物，**保持原样不覆盖**。它已经产出了 7 种输出的全部存储与
基础指标；本脚本只补两项它没有、但止损判定必须有的东西：

1. 审查的止损规则写明"若棋盘消失但完整推理仍输给固定 Gaussian/Wiener，判为
   `INFERENCE_ARTIFACT_FIXED_METHOD_NO_GAIN`"。要下这个判定，就必须真的把这两个
   固定对照算出来，且必须用**同一 SVD 输入、同一线性 PD 标定参考、同一 dB 窗口**，
   不给网络列单独调参。
2. 审查要求统计里包含"**输入输出配准**"，原脚本只有仿射斜率/截距，没有配准。

指标定义（phase_bias / y_seam）**直接 import 原脚本**，不重写，避免口径漂移。
另补一组稳健背景统计（MAD、p99、超阈占比）：`std` 经 logeps 反变换（指数）后对少数
离群体素极不稳健，单看 std 比会把结论放大或缩小，两种都报。

产物（追加到同一目录，不覆盖原文件）：
    reports/v3_round8_zero_train/{seq}_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

D8 = ROOT / "data/processed/v3_round8_zero_train"
R8 = ROOT / "reports/v3_round8_zero_train"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())["logeps"]

from v3_round8_zero_train_full_infer import phase_bias, y_seam  # noqa: E402


def to_pd(y, s):
    """标准化 logeps → 线性 PD，与 v3_round8_zero_train_full_infer.py 同一路径。"""
    from v3_round4_b1_prep import inv
    return np.maximum(inv("logeps", y * NORM["std"] + NORM["mean"], s), 0.0)


def to_std(pd, s, valid):
    """线性 PD → 标准化 logeps（固定对照回写用；锥外按 fwd 的约定置 log(EPS)）。"""
    from v3_round4_b1_prep import fwd
    return (fwd("logeps", pd, s, valid).astype(np.float64) - NORM["mean"]) / NORM["std"]


# ---------------------------------------------------------------- 固定对照 ----
def gauss_blur(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x.astype(np.float64), sigma=sigma, mode="nearest")


def wiener2d(pd_vol, noise_var):
    """每张 2D 帧上的固定 Wiener（scipy.signal.wiener，窗口固定 3×3）。

    噪声方差由**输入背景 ROI** 一次性估定、对所有方法相同，不从输出回读，
    因此不是"按输出调参"。选 2D 是因为这笔数据本来就是逐帧采集的，
    固定滤波器基线理应按帧定义。
    """
    from scipy.signal import wiener
    out = np.empty_like(pd_vol, dtype=np.float64)
    for z in range(pd_vol.shape[0]):
        out[z] = wiener(pd_vol[z], noise=noise_var, mysize=3)
    return out


# ----------------------------------------------------------------- 配准 ------
def registration(pd_a, pd_b, roi):
    """在 ±2 体素内搜索使相关最大的整数位移，并做抛物线亚体素细化。

    报的是"a 相对 b 移动多少最像"。严重位移 = 空间错位，是审查明确要看的失败模式。
    """
    def corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        den = np.sqrt((u * u).sum() * (v * v).sum())
        return float((u * v).sum() / den) if den > 0 else 0.0

    grid = {}
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                sa = tuple(slice(max(0, d), n + min(0, d))
                           for d, n in zip((dz, dy, dx), pd_a.shape))
                sb = tuple(slice(max(0, -d), n + min(0, -d))
                           for d, n in zip((dz, dy, dx), pd_a.shape))
                m = roi[sa] & roi[sb]
                if m.sum() < 100:
                    continue
                grid[(dz, dy, dx)] = corr(pd_a[sa][m], pd_b[sb][m])
    best = max(grid, key=grid.get)
    sub = []
    for ax in range(3):
        vals = []
        for o in (-1, 0, 1):
            k = list(best)
            k[ax] += o
            vals.append(grid.get(tuple(k), np.nan))
        if all(np.isfinite(vals)) and vals[1] >= vals[0] and vals[1] >= vals[2]:
            den = vals[0] - 2 * vals[1] + vals[2]
            sub.append(float(0.5 * (vals[0] - vals[2]) / den) if abs(den) > 1e-12 else 0.0)
        else:
            sub.append(0.0)
    return {"best_integer_shift": list(best), "corr_at_best": grid[best],
            "corr_at_zero_shift": grid.get((0, 0, 0), float("nan")),
            "subvoxel_offset": sub}


def _corr(u, v):
    u, v = u - u.mean(), v - v.mean()
    den = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / den) if den > 0 else float("nan")


def bg_stats(pd, bg):
    v = pd[bg].astype(np.float64)
    med = np.median(v)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "mad": float(np.median(np.abs(v - med))),
            "p99": float(np.percentile(v, 99)),
            "frac_gt_10x_own_median": float((v > 10 * med).mean())}


def phase_residual_stds(y, source, mask):
    """8 个相位类上仿射残差的标准差（审查要求"残差均值/方差"，原脚本只有均值）。"""
    from v3_round8_zero_train_full_infer import parity_slices
    x, v = source[mask].astype(np.float64), y[mask].astype(np.float64)
    slope = np.dot(x - x.mean(), v - v.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-12)
    residual = y - slope * source - (v.mean() - slope * x.mean())
    out = []
    for i in range(8):
        sl = parity_slices(i)
        m = mask[sl]
        out.append(float(residual[sl][m].std()) if m.any() else float("nan"))
    return out


def roughness(y, roi):
    """掩膜内梯度粗糙度：mean|∇y|²（y = 标准化 logeps，不经指数反变换）。

    `bg_std_ratio` 走的是 `inv("logeps")` = exp，少数离群体素会把 std 放大成几倍到几十倍，
    不能单独作为"噪声放大"的证据。粗糙度在标准化域里算、单调、无指数，是同一件事的稳健版本。
    """
    acc, n = 0.0, 0
    for ax in range(3):
        d = np.diff(y, axis=ax)
        m = roi[tuple(slice(0, s - 1) if a == ax else slice(None)
                      for a, s in enumerate(y.shape))]
        if m.any():
            acc += float((d[m] ** 2).sum())
            n += int(m.sum())
    return acc / max(n, 1)


def main():
    seqs = sys.argv[1:] or ["SA64", "HSA64"]
    for seq in seqs:
        src = json.loads((R8 / f"{seq}_comparison.json").read_text())
        meta = json.loads((ROOT / f"data/processed/v3_round6/{seq}/metadata.json").read_text())
        s = float(meta["s"])
        mask = np.load(ROOT / f"data/processed/official/{seq}/valid_mask.npy").astype(bool)
        mask = mask[:src["shape"][0]]

        std = {f.name.replace("_logeps_standardized.npy", ""):
               np.load(f).astype(np.float64)
               for f in sorted(D8.glob(f"{seq}/*_logeps_standardized.npy"))}

        pd_in = to_pd(std["svd_input"], s)
        refmax = max(float(pd_in[mask].max()), 1e-12)
        bright = mask & (pd_in >= np.quantile(pd_in[mask], 0.90))
        bg = mask & (pd_in <= np.quantile(pd_in[mask], 0.30))
        noise_var = float(np.var(pd_in[bg]))
        bg_in = bg_stats(pd_in, bg)

        cand = {k: v for k, v in std.items()}
        for sg in (0.5, 1.0, 1.5):
            cand[f"fixed_gauss_s{sg:g}"] = gauss_blur(std["svd_input"], sg)
        cand["fixed_wiener2d"] = to_std(wiener2d(pd_in, noise_var), s, mask)

        out = {
            "sequence": seq,
            "reference": "同一 SVD 输入的锥内最大线性 PD，所有方法（含固定对照）共用",
            "refmax_linear_pd": refmax,
            "mask_voxels": int(mask.sum()),
            "bright_roi": "pd_in ≥ 锥内 p90（按输入选，不按输出选）",
            "background_roi": "pd_in ≤ 锥内 p30（按输入选，不按输出选）",
            "wiener_noise_var_from_input_background": noise_var,
            "input_background": bg_in,
            "input_bright_mean_linear_pd": float(pd_in[bright].mean()),
            "methods": {},
        }
        y_in = std["svd_input"]
        rough_in = roughness(y_in, mask)
        rough_in_bright = roughness(y_in, bright)
        rough_in_bg = roughness(y_in, bg)
        for name, y in cand.items():
            pd = to_pd(y, s)
            db = 10 * np.log10(np.maximum(pd, 1e-12) / refmax)
            b = bg_stats(pd, bg)
            out["methods"][name] = {
                **phase_bias(y, y_in, mask),
                "phase_residual_stds": phase_residual_stds(y, y_in, mask),
                "seam_y_db": y_seam(db, mask),
                "core_mean_ratio": float(pd[bright].mean()
                                          / max(pd_in[bright].mean(), 1e-12)),
                "bg_std_ratio": float(b["std"] / max(bg_in["std"], 1e-12)),
                "bg_mad_ratio": float(b["mad"] / max(bg_in["mad"], 1e-12)),
                "bg_p99_ratio": float(b["p99"] / max(bg_in["p99"], 1e-12)),
                "bg_std_ratio_std_space": float(y[bg].std() / max(y_in[bg].std(), 1e-12)),
                "roughness_ratio_all": float(roughness(y, mask) / max(rough_in, 1e-12)),
                "roughness_ratio_bright": float(roughness(y, bright)
                                                / max(rough_in_bright, 1e-12)),
                "roughness_ratio_bg": float(roughness(y, bg) / max(rough_in_bg, 1e-12)),
                "bg": b,
                "corr_vs_input_whole_mask": _corr(y[mask], y_in[mask]),
                "corr_vs_input_whole_mask_db": _corr(db[mask], 10 * np.log10(
                    np.maximum(pd_in, 1e-12) / refmax)[mask]),
                "registration_vs_input": registration(pd, pd_in, bright),
            }
        # 同强度公平比较：给每个网络输出配一个 core 保持率最接近的高斯 σ*，
        # 在"同样的亮区信号损失"下比背景噪声。避免"你拿过度模糊的高斯来比"的反驳。
        fine = np.round(np.arange(0.30, 2.55, 0.05), 2)
        gauss_curve = []
        for sg in fine:
            y = gauss_blur(std["svd_input"], sg)
            pdg = to_pd(y, s)
            gauss_curve.append({
                "sigma": float(sg),
                "core_mean_ratio": float(pdg[bright].mean()
                                         / max(pd_in[bright].mean(), 1e-12)),
                "bg_std_ratio": float(bg_stats(pdg, bg)["std"] / max(bg_in["std"], 1e-12)),
                "roughness_ratio_bg": float(roughness(y, bg) / max(rough_in_bg, 1e-12)),
                "roughness_ratio_bright": float(roughness(y, bright)
                                                / max(rough_in_bright, 1e-12)),
            })
        matched = {}
        for name, v in out["methods"].items():
            if not (name.startswith("full_") or name.startswith("legacy")):
                continue
            g = min(gauss_curve, key=lambda c: abs(c["core_mean_ratio"]
                                                  - v["core_mean_ratio"]))
            matched[name] = {
                "core_mean_ratio_net": v["core_mean_ratio"],
                "matched_gauss_sigma": g["sigma"],
                "core_mean_ratio_gauss": g["core_mean_ratio"],
                "bg_std_ratio_net": v["bg_std_ratio"],
                "bg_std_ratio_gauss": g["bg_std_ratio"],
                "bg_std_ratio_penalty": v["bg_std_ratio"] / max(g["bg_std_ratio"], 1e-12),
                "roughness_ratio_bg_net": v["roughness_ratio_bg"],
                "roughness_ratio_bg_gauss": g["roughness_ratio_bg"],
                "roughness_bg_penalty": v["roughness_ratio_bg"]
                                         / max(g["roughness_ratio_bg"], 1e-12),
                "roughness_ratio_bright_net": v["roughness_ratio_bright"],
                "roughness_ratio_bright_gauss": g["roughness_ratio_bright"],
            }
        out["matched_strength_vs_fixed_gaussian"] = matched
        out["gaussian_strength_curve"] = gauss_curve

        (R8 / f"{seq}_analysis.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"{seq}: {len(cand)} methods -> {R8 / f'{seq}_analysis.json'}")


if __name__ == "__main__":
    main()

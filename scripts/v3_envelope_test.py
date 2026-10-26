"""V3 Phase 1：背景的慢结构是"乘性本底"还是"加性杂波"？

这条区分决定了 Gate 1 用哪条 ACF：
  * 乘性（完全发育散斑 / 平滑灵敏度本底）：慢结构就是局部均值，是**本底信号**，
    去噪器的任务是压掉围绕它的快速散斑，不该把它当噪声扣掉 → 用"去平台"的快速 ACF。
  * 加性（残余杂波叠加）：慢结构是与信号无关的加性项，盲点法原理上去不掉，但它**该被去掉**
    → 用原始 ACF，泄漏上界取 0.81–1.00，必须加大间距。

判据是局部方差随局部均值的斜率（线性 PD 域，5×5×5 窗口）：
    完全发育散斑  PD = g·N,  N 为指数分布  =>  Var ∝ mean^2   => 斜率 ≈ 2
    加性白噪声    PD = g + N                =>  Var ∝ const    => 斜率 ≈ 0

同时给出第三个读数：慢结构在空间上是否跟着组织/结构走（与全图平滑图的秩相关）。

用法：python -u scripts/v3_envelope_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OFFICIAL = Path("data/processed/official")


def boxmean(a, k):
    out = a.astype(np.float64)
    r = k // 2
    for ax in range(3):
        p = np.pad(out, [(r, r) if i == ax else (0, 0) for i in range(3)], mode="edge")
        c = np.cumsum(p, axis=ax)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=ax)), c], axis=ax)
        n = out.shape[ax]
        out = (np.take(c, np.arange(k, k + n), axis=ax)
               - np.take(c, np.arange(0, n), axis=ax)) / k
    return out


def variance_slope(P, mask, k=5, nbins=10):
    """log(local var) ~ log(local mean) 的斜率。"""
    M = np.maximum(boxmean(P, k), 1e-12)
    R = P - M
    V = np.maximum(boxmean(R * R, k), 1e-30)
    m, v = M[mask], V[mask]
    edges = np.quantile(m, np.linspace(0, 1, nbins + 1))
    rows = []
    for i in range(nbins):
        sel = (m >= edges[i]) & (m < edges[i + 1] if i < nbins - 1 else m <= edges[i + 1])
        if sel.sum() < 200:
            continue
        rows.append({"bin": i, "mean": float(np.median(m[sel])),
                     "var": float(np.median(v[sel])), "n": int(sel.sum())})
    if len(rows) < 3:
        return float("nan"), rows
    x = np.log([r["mean"] for r in rows])
    y = np.log([r["var"] for r in rows])
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, rows


def main():
    out = {}
    for seq in ("SA64", "HSA64"):
        P = np.load(OFFICIAL / seq / "pd_linear.npy").astype(np.float64)
        M = np.load(OFFICIAL / seq / "valid_mask.npy").astype(bool)
        lg = np.log(np.maximum(P, 1e-12))
        # 窗口必须完全落在有效体内，否则恒 0 的死体素会污染局部均值/方差
        interior = boxmean(M.astype(np.float64), 5) > 0.999
        # 背景：有效体内最低 40% 分位（留出足够的动态范围做斜率回归）
        bg = interior & (P <= np.quantile(P[interior], 0.40))
        slope, rows = variance_slope(P, bg)
        # 慢结构是否跟组织走：背景慢分量 vs 全图平滑图
        slow_bg = boxmean(lg, 9)
        tissue = boxmean(np.where(M, lg, 0.0), 9)
        rr = float(np.corrcoef(slow_bg[bg], tissue[bg])[0, 1])
        out[seq] = {"slope": slope, "bins": rows, "slow_vs_tissue_r": rr,
                    "n_bg": int(bg.sum())}
        print(f"== {seq}  n_bg={bg.sum()}")
        print(f"   log-variance vs log-mean 斜率 = {slope:+.3f}   "
              f"(2=完全发育散斑/乘性, 0=加性)")
        print(f"   慢分量 vs 组织平滑图 秩相关 = {rr:+.3f}")
        print(f"   {'bin':>3} {'local mean':>12} {'local var':>12} {'var/mean^2':>11}")
        for r in rows:
            print(f"   {r['bin']:>3} {r['mean']:>12.4e} {r['var']:>12.4e} "
                  f"{r['var']/max(r['mean']**2,1e-30):>11.4f}")
    Path("data/processed/v3_envelope_test.json").write_text(json.dumps(out, indent=1))
    print("\nwrote data/processed/v3_envelope_test.json")


if __name__ == "__main__":
    main()

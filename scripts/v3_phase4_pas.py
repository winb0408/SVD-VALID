"""PAS：PSF 感知 + 正交子采样变差函数外推的**自适应收缩**（本项目自己的方法）。

## 为什么要有它（L48 定的方向）

L48 的结论是：① 的 Wiener 收缩已经过判据（0.754），但它有**三个与本项目无关的短板**，
而这三个短板正好是本项目自己的两块资产（标定过的 PSF / 3D 正交自采样结构）能补的：

1. **各向同性窗口**：PSF 实测是 (σz, σy, σx) = (3.0, 2.37, 3.97)，各轴相关长度差 1.7 倍。
   用同一个立方窗口，短相关轴被过度平均（糊），长相关轴平均不足（噪声没压下去）。
   → 改用**与 PSF 形状匹配的各向异性高斯核**。
2. **噪声方差取全局中位数**：`σ² = median(局部方差)`。这在模拟器里勉强成立
   （噪声近似空间均匀），真实数据里深度增益一变就不成立。
   → 改用**逐体素的变差函数外推**给出空间分辨的 σ̂²(x)。
3. **局部方差本身被信号梯度污染**：局部方差里既有噪声也有信号起伏，在边缘处严重高估，
   于是 b 在边缘保持接近 1（噪声留下）。→ 变差函数外推对**局部线性/二次**的信号是
   **无偏**的（下面有推导），这是逐体素噪声估计的关键改进。

## 变差函数外推（本方法的核心，且用到正交采样的 lag-2 对）

结构函数（variogram）`D(d) = E[(x(v+d) − x(v))²] = 2σ² + [S(v+d) − S(v)]²`。
若信号在局部可近似为二次，沿轴 a 展开 `[S(v+d)−S(v)]² ≈ c_a d²`，于是

    D_a(1) = 2σ² + c_a
    D_a(2) = 2σ² + 4c_a

解得 **σ̂²_a = (4·D_a(1) − D_a(2)) / 6**：信号项精确消掉，只剩噪声。三个轴取平均
（各轴按 PSF 的相关长度给权重）。**lag-2 的体素对，正是 3D 正交自采样里跨相位的对**，
也就是说这个估计器只用本项目采样结构天然提供的两种距离。

`D_a(1)`、`D_a(2)` 都在 PSF 匹配核下做局部平均，所以 σ̂² 是**逐体素**的（单次采集可部署，
不需要第二次观测、不需要训练）。

## 四个可消融的档位（用来证明"提升来自项目自己的成分"）

    A  isotropic + 全局 σ²   = ① 基线（0.754）
    B  PSF 核   + 全局 σ²     ← 只换核
    C  PSF 核   + 全局 σ²(variogram)
    D  PSF 核   + 逐体素 σ̂²(x) ← 本方法
    E  分区 oracle（背景/血管各一个 b）← 参考上界，不可部署

用法：
    python scripts/v3_phase4_pas.py \
        --obs1 data/processed/v3_sim/sim_seed1.npz \
        --obs2 data/processed/v3_sim/sim_seed1_n201_s1.npz \
        --obs-mean .../n202_s1.npz .../n203_s1.npz .../n204_s1.npz \
        --out reports/v3_phase4_2026-10-21/pas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase3_eval import M_EVAL, usable, support_mask          # noqa: E402
from v3_phase4_scoreboard import to_log, score                   # noqa: E402

PSF = (3.0, 2.37, 3.97)          # 标定值，轴序 (z, y, x)，见 data/processed/v3_sim/calibration.json


def local_stats(x, sig):
    """PSF 匹配核下的局部均值与局部方差。"""
    m = gaussian_filter(x, sig, mode="nearest")
    m2 = gaussian_filter(x * x, sig, mode="nearest")
    return m, np.maximum(m2 - m * m, 0.0)


def diff_sq(x, ax, lag):
    """沿轴 ax 的 lag 距离上的逐体素平方差，边界用近邻复制填充。"""
    a = [slice(None)] * 3
    b = [slice(None)] * 3
    a[ax] = slice(lag, None)
    b[ax] = slice(0, x.shape[ax] - lag)
    dif = x[tuple(a)] - x[tuple(b)]
    sq = dif * dif
    out = np.empty_like(x)
    fill = [slice(None)] * 3
    fill[ax] = slice(lag, None)
    out[tuple(fill)] = sq
    head = [slice(None)] * 3
    head[ax] = slice(0, lag)
    out[tuple(head)] = np.take(sq, [0], axis=ax)   # 前 lag 层用首层复制
    return out


def variogram_sigma2(x, sig):
    """逐体素噪声方差：三轴 (4·D(1) − D(2))/6 的平均。"""
    acc = np.zeros_like(x)
    for ax in range(3):
        d1 = gaussian_filter(diff_sq(x, ax, 1), sig, mode="nearest")
        d2 = gaussian_filter(diff_sq(x, ax, 2), sig, mode="nearest")
        acc += (4.0 * d1 - d2) / 6.0
    return acc / 3.0


def oracle_b(x, obs2, ves, far, grid):
    """分区 oracle：背景区一个 b、血管区一个 b，各自最小化与 obs2 的平方误差。"""
    b = np.zeros_like(x)
    out = {}
    for name, m in (("far", far), ("ves", ves)):
        # b = argmin Σ (b(x−mu) − (obs2−mu))²  ⇒  Σ(x−mu)(obs2−mu) / Σ(x−mu)²
        num = float(np.sum((obs2[m] - grid[m]) * (x[m] - grid[m])))
        den = float(np.sum((x[m] - grid[m]) ** 2))
        bb = num / den if den > 0 else 1.0
        b[m] = bb
        out[name] = bb
    b[~(far | ves)] = out["ves"]        # 其余体素（边缘过渡带）沿用血管档
    return b, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--obs-mean", nargs="*", default=[])
    ap.add_argument("--kappas", nargs="+", type=float, default=[0.35, 0.5, 0.65, 0.8, 1.0],
                    help="PSF 核的尺度系数：sigma = kappa * PSF")
    ap.add_argument("--window", type=int, default=7, help="① 基线用的各向同性均匀窗口")
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/pas")
    ap.add_argument("--save-volumes", action="store_true",
                    help="把各档输出与 b 图存成 npz，供画图脚本用")
    args = ap.parse_args()

    d1 = np.load(args.obs1)
    shape = tuple((n // 32) * 32 for n in usable(d1["pd_linear"].shape, M_EVAL))
    crop = tuple(slice(0, n) for n in shape)
    pd1 = d1["pd_linear"][crop].astype(np.float32)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))

    obs1 = to_log(pd1, s)
    obs2 = to_log(np.load(args.obs2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)
    obs_mean = None
    if args.obs_mean:
        acc = pd1.astype(np.float64).copy()
        for p in args.obs_mean:
            acc += np.load(p)["pd_linear"][crop].astype(np.float64)
        obs_mean = to_log((acc / (1 + len(args.obs_mean))).astype(np.float32), s)

    common = support_mask(shape, M_EVAL) & valid
    ves = vessels & common
    far = (common & (distance_transform_edt(~vessels) >= 4)
           & (distance_transform_edt(valid) >= 2))
    masks = {"common": common, "ves": ves, "far": far}

    ref = score(obs1, obs1, obs2, truth, masks)
    ceiling = score(obs_mean, obs1, obs2, truth, masks)["n2n_ratio"] if obs_mean is not None else None
    x = obs1.astype(np.float64)

    def judge(pred, tag, b=None):
        m = score(pred.astype(np.float32), obs1, obs2, truth, masks)
        gap = (1.0 - m["n2n_ratio"]) / (1.0 - ceiling) if ceiling and ceiling < 1 else None
        ok = (m["n2n_ratio"] <= 0.95 and m["r_ves"] >= ref["r_ves"] - 0.01
              and gap is not None and gap >= 0.50)
        rec = {"tag": tag, "gap_closed": gap, "pass": bool(ok), **m}
        if b is not None:
            rec["b_bg_mean"] = float(b[far].mean())
            rec["b_ves_mean"] = float(b[ves].mean())
        return rec

    rows = []
    vols = {}

    # ---- A: ① 基线，各向同性均匀窗口 + 全局 σ² = median(局部方差)
    mu_u, lv_u = local_stats_uniform(x, args.window)
    sig2_u = float(np.median(lv_u[common]))
    b_A = np.clip(1.0 - sig2_u / np.maximum(lv_u, 1e-9), 0.0, 1.0)
    pred_A = mu_u + b_A * (x - mu_u)
    rows.append(judge(pred_A, f"A_iso_w{args.window}", b_A))
    vols["A"] = (pred_A, b_A)

    for kap in args.kappas:
        sig = tuple(kap * p for p in PSF)
        mu, lv = local_stats(x, sig)
        sg_var = variogram_sigma2(x, sig)

        # ---- B: PSF 核 + 全局 σ²（仍用局部方差中位数）
        sig2_B = float(np.median(lv[common]))
        b_B = np.clip(1.0 - sig2_B / np.maximum(lv, 1e-9), 0.0, 1.0)
        rows.append(judge(mu + b_B * (x - mu), f"B_psf_k{kap}_globalLV", b_B))
        vols[f"B_{kap}"] = (mu + b_B * (x - mu), b_B)

        # ---- C: PSF 核 + 全局 σ²（用变差函数估，标量）
        sig2_C = float(np.median(np.maximum(sg_var[common], 0.0)))
        b_C = np.clip(1.0 - sig2_C / np.maximum(lv, 1e-9), 0.0, 1.0)
        rows.append(judge(mu + b_C * (x - mu), f"C_psf_k{kap}_globalVa", b_C))
        vols[f"C_{kap}"] = (mu + b_C * (x - mu), b_C)

        # ---- D: PSF 核 + 逐体素 σ̂²(x)（本方法）
        s2 = np.maximum(sg_var, 0.0)
        b_D = np.clip(1.0 - s2 / np.maximum(lv, 1e-9), 0.0, 1.0)
        rows.append(judge(mu + b_D * (x - mu), f"D_psf_k{kap}_voxVa", b_D))
        vols[f"D_{kap}"] = (mu + b_D * (x - mu), b_D)

    # ---- E: 分区 oracle（参考上界）
    mu_best, lv_best = local_stats(x, tuple(0.65 * p for p in PSF))
    b_E, b_E_regions = oracle_b(x, obs2.astype(np.float64), ves, far, mu_best)
    rows.append(judge(mu_best + b_E * (x - mu_best), "E_oracle_regions", b_E))
    vols["E"] = (mu_best + b_E * (x - mu_best), b_E)

    rows.sort(key=lambda r: r["n2n_ratio"])
    print(f"{'档位':<26}{'n2n':>7s}{'gap':>7s}{'r_ves':>7s}{'bg':>7s}{'ves':>7s}"
          f"{'slope':>7s}{'b_bg':>7s}{'b_ves':>7s}  判定")
    for r in rows:
        print(f"{r['tag']:<26}{r['n2n_ratio']:7.3f}{r['gap_closed']:7.3f}{r['r_ves']:7.3f}"
              f"{r['n2n_ratio_bg']:7.3f}{r['n2n_ratio_ves']:7.3f}{r['slope_vs_input']:7.3f}"
              f"{r['b_bg_mean']:7.2f}{r['b_ves_mean']:7.2f}  "
              f"{'通过' if r['pass'] else '不通过'}")

    print(f"\n输入 n2n=1.000  r_ves={ref['r_ves']:.3f}   天花板={ceiling:.3f}   "
          f"oracle 分区 b={b_E_regions}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pas.json").write_text(json.dumps(
        {"psf": PSF, "kappas": args.kappas, "iso_window": args.window,
         "ceiling_n2n": ceiling, "input": ref, "oracle_regions": b_E_regions,
         "rows": rows}, indent=1, ensure_ascii=False))
    if args.save_volumes:
        np.savez_compressed(out / "volumes.npz",
                            obs1=obs1, obs2=obs2, truth=truth, valid=valid,
                            vessels=vessels, far=far, **{f"pred_{k}": v[0].astype(np.float32)
                                                         for k, v in vols.items()},
                            **{f"b_{k}": v[1].astype(np.float32) for k, v in vols.items()})
        print(f"wrote {out}/volumes.npz")
    print(f"wrote {out}/pas.json")


def local_stats_uniform(x, w):
    m = uniform_filter(x, w)
    m2 = uniform_filter(x * x, w)
    return m, np.maximum(m2 - m * m, 0.0)


if __name__ == "__main__":
    main()

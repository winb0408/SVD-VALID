"""TIRe：**时间交错自参考噪声估计** + 自适应收缩（本项目自己的方法）。

## 一句话

判据要的是"两次独立观测之差"，而一次采集里**奇偶帧两半**之差正好给出同一个量的
无偏估计：(pd_1 − pd_2)² 就是 Var(pd)。于是不需要第二次采集、不需要训练，
就能得到**逐体素、与判据同一种噪声**的方差估计，再拿它做信号依赖收缩。

## 为什么这是本项目的方法（而不是又一个经典滤波）

① 的 Wiener 收缩（0.754）靠 `σ² = median(局部方差)`，它有两个结构性缺陷：

1. **局部方差把"结构"当噪声**：血管边缘处 lv 巨大 → b≈1 → 噪声留下；背景里
   lv 又同时含噪声与平滑结构 → σ² 被高估/低估，只能取一个全局中位数代偿。
2. **空间平滑的噪声看不见**：本数据里的杂波是空间大尺度（σ=(5,8,8)）但**逐次实现不同**，
   它就是噪声；任何只看空间统计的估计器都会把它当信号。

时间交错正好补这两点：两半之差里**只有随帧变化的部分**，与它在空间上是否平滑无关。
这与本项目"用交错子采样的差异当参考"的思路是同一个动作，只是把交错从空间搬到时间；
空间交错（3D 正交相位）那一路在 L47 已被证明只能预测到 ~0.5 的斜率，时间交错这一路
不用于预测目标值，只用于**估噪**，因此没有盲点上限。

## 无偏性（先验推导，后面用已知答案核对）

`pd = Σ_t |filt_t|²`，按帧奇偶拆成 `pd1 + pd2 = pd`，两半同分布且独立：
`Var(pd) = Var(pd1) + Var(pd2) = 2Var(pd1)`，而 `Var(pd1 − pd2) = 2Var(pd1)`，
所以 **E[(pd1 − pd2)²] = Var(pd)**。log 域按 delta 法 `σ²_log = σ²_pd/(s+pd)²`。

## 档位（用于证明"提升来自噪声估计，不是来自挑窗口"）

    T0  ① 基线：均匀窗口 + 全局 median(局部方差)
    T1  全局 median(交错 σ̂²) + 均匀窗口      ← 只换噪声估计
    T2  逐体素交错 σ̂²(x) + 均匀窗口          ← 再换空间分辨
    T3  逐体素交错 σ̂²(x) + PSF 匹配核        ← 本方法
    T4  线性域功率相减（不收缩，直接减噪声功率）
    E   分区 oracle（参考上界）

用法：
    python scripts/v3_phase4_tire.py \
        --obs1 data/processed/v3_sim_halves/sim_seed1.npz \
        --obs2 data/processed/v3_sim/sim_seed1_n201_s1.npz \
        --obs-mean ... \
        --out reports/v3_phase4_2026-10-21/tire --save-volumes
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

PSF = (3.0, 2.37, 3.97)          # 标定值，轴序 (z, y, x)


def local_uniform(x, w):
    m = uniform_filter(x, w)
    v = np.maximum(uniform_filter(x * x, w) - m * m, 0.0)
    return m, v


def local_psf(x, sig):
    m = gaussian_filter(x, sig, mode="nearest")
    v = np.maximum(gaussian_filter(x * x, sig, mode="nearest") - m * m, 0.0)
    return m, v


def oracle_b(x, obs2, ves, far, grid):
    """分区 oracle：背景一个 b、血管一个 b，各自最小化与 obs2 的平方误差。"""
    b = np.zeros_like(x)
    out = {}
    for name, m in (("far", far), ("ves", ves)):
        num = float(np.sum((obs2[m] - grid[m]) * (x[m] - grid[m])))
        den = float(np.sum((x[m] - grid[m]) ** 2))
        bb = num / den if den > 0 else 1.0
        b[m] = bb
        out[name] = bb
    b[~(far | ves)] = out["ves"]
    return b, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim_halves/sim_seed1.npz",
                    help="必须带 pd_half1/pd_half2（用 v3_simulator.py --emit-halves 生成）")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--obs-mean", nargs="*", default=[])
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 7, 9])
    ap.add_argument("--kappas", nargs="+", type=float, default=[0.5, 0.65, 0.8, 1.0])
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/tire")
    ap.add_argument("--save-volumes", action="store_true")
    args = ap.parse_args()

    d1 = np.load(args.obs1)
    if "pd_half1" not in d1.files:
        raise SystemExit(f"{args.obs1} 里没有 pd_half1/pd_half2；"
                         "用 scripts/v3_simulator.py --emit-halves 重新生成")
    shape = tuple((n // 32) * 32 for n in usable(d1["pd_linear"].shape, M_EVAL))
    crop = tuple(slice(0, n) for n in shape)

    pd1 = d1["pd_linear"][crop].astype(np.float64)
    h1 = d1["pd_half1"][crop].astype(np.float64)
    h2 = d1["pd_half2"][crop].astype(np.float64)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))

    obs1 = to_log(pd1.astype(np.float32), s)
    obs2 = to_log(np.load(args.obs2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)
    obs_mean = None
    if args.obs_mean:
        acc = pd1.copy()
        for p in args.obs_mean:
            acc += np.load(p)["pd_linear"][crop].astype(np.float64)
        obs_mean = to_log((acc / (1 + len(args.obs_mean))).astype(np.float32), s)

    common = support_mask(shape, M_EVAL) & valid
    ves = vessels & common
    far = (common & (distance_transform_edt(~vessels) >= 4)
           & (distance_transform_edt(valid) >= 2))
    masks = {"common": common, "ves": ves, "far": far}

    x = obs1.astype(np.float64)
    ref = score(obs1, obs1, obs2, truth, masks)
    ceiling = score(obs_mean, obs1, obs2, truth, masks)["n2n_ratio"] if obs_mean is not None else None

    # ---------- 交错噪声估计（线性域）----------
    sig2_pd = (h1 - h2) ** 2                     # E[·] = Var(pd)，逐体素
    sig2_pd_s = gaussian_filter(sig2_pd, tuple(0.5 * p for p in PSF), mode="nearest")
    sig2_log_vox = sig2_pd_s / (s + pd1) ** 2    # delta 法换到 log 域

    # ---------- 已知答案核对：与"两次独立观测之差"比 ----------
    sig2_true_vox = 0.5 * (x - obs2.astype(np.float64)) ** 2
    def med(a, m):
        return float(np.median(a[m]))
    calib = {
        "sigma2_log_est_median_common": med(sig2_log_vox, common),
        "sigma2_log_true_median_common": med(sig2_true_vox, common),
        "sigma2_log_est_median_far": med(sig2_log_vox, far),
        "sigma2_log_true_median_far": med(sig2_true_vox, far),
        "sigma2_log_est_median_ves": med(sig2_log_vox, ves),
        "sigma2_log_true_median_ves": med(sig2_true_vox, ves),
        "note": "估/真 的比值接近 1 = 交错估计无偏；分区比值的**形状**是否一致，是关键",
    }
    calib["ratio_common"] = (calib["sigma2_log_est_median_common"]
                             / max(calib["sigma2_log_true_median_common"], 1e-12))
    calib["ratio_far"] = calib["sigma2_log_est_median_far"] / max(
        calib["sigma2_log_true_median_far"], 1e-12)
    calib["ratio_ves"] = calib["sigma2_log_est_median_ves"] / max(
        calib["sigma2_log_true_median_ves"], 1e-12)
    print("已知答案核对（交错 σ̂² 对 (obs1−obs2)²/2）：")
    for k in ("common", "far", "ves"):
        print(f"  {k:7s} 估={calib[f'sigma2_log_est_median_{k}']:.3e} "
              f"真={calib[f'sigma2_log_true_median_{k}']:.3e} "
              f"比值={calib[f'ratio_{k}']:.3f}")

    rows = []
    vols = {}

    def judge(pred, tag, b=None, extra=None):
        m = score(pred.astype(np.float32), obs1, obs2, truth, masks)
        gap = (1.0 - m["n2n_ratio"]) / (1.0 - ceiling) if ceiling and ceiling < 1 else None
        ok = (m["n2n_ratio"] <= 0.95 and m["r_ves"] >= ref["r_ves"] - 0.01
              and gap is not None and gap >= 0.50)
        rec = {"tag": tag, "gap_closed": gap, "pass": bool(ok), **m}
        if b is not None:
            rec["b_bg_mean"] = float(b[far].mean())
            rec["b_ves_mean"] = float(b[ves].mean())
        if extra:
            rec.update(extra)
        rows.append(rec)
        return rec

    # ---- T0: ① 基线
    for w in args.windows[:1]:
        mu, lv = local_uniform(x, w)
        sig2 = float(np.median(lv[common]))
        b = np.clip(1.0 - sig2 / np.maximum(lv, 1e-9), 0.0, 1.0)
        pred = mu + b * (x - mu)
        judge(pred, f"T0_iso_w{w}_globalLV", b)
        vols["T0"] = (pred, b)

    sig2_glob = float(np.median(np.maximum(sig2_log_vox[common], 0.0)))
    for w in args.windows:
        mu, lv = local_uniform(x, w)
        # ---- T1: 全局交错 σ² + 均匀窗口
        b1 = np.clip(1.0 - sig2_glob / np.maximum(lv, 1e-9), 0.0, 1.0)
        judge(mu + b1 * (x - mu), f"T1_iso_w{w}_globTIRe", b1)
        vols[f"T1_w{w}"] = (mu + b1 * (x - mu), b1)
        # ---- T2: 逐体素交错 σ̂²(x) + 均匀窗口
        b2 = np.clip(1.0 - np.maximum(sig2_log_vox, 0.0) / np.maximum(lv, 1e-9), 0.0, 1.0)
        judge(mu + b2 * (x - mu), f"T2_iso_w{w}_voxTIRe", b2)
        vols[f"T2_w{w}"] = (mu + b2 * (x - mu), b2)

    for kap in args.kappas:
        mu, lv = local_psf(x, tuple(kap * p for p in PSF))
        # ---- T3: 逐体素交错 σ̂²(x) + PSF 匹配核
        b3 = np.clip(1.0 - np.maximum(sig2_log_vox, 0.0) / np.maximum(lv, 1e-9), 0.0, 1.0)
        judge(mu + b3 * (x - mu), f"T3_psf_k{kap}_voxTIRe", b3)
        vols[f"T3_k{kap}"] = (mu + b3 * (x - mu), b3)

    # ---- T4: 线性域功率相减（不做收缩）
    pd_den = np.maximum(pd1 - sig2_pd_s, 0.0)
    judge(np.log1p(pd_den / s), "T4_linear_subtract")

    # ---- E: 分区 oracle
    mu_o, _ = local_psf(x, tuple(0.65 * p for p in PSF))
    bE, reg = oracle_b(x, obs2.astype(np.float64), ves, far, mu_o)
    judge(mu_o + bE * (x - mu_o), "E_oracle_regions", bE)
    vols["E"] = (mu_o + bE * (x - mu_o), bE)

    rows.sort(key=lambda r: r["n2n_ratio"])
    print(f"\n{'档位':<26}{'n2n':>7s}{'gap':>7s}{'r_ves':>7s}{'bg':>7s}{'ves':>7s}"
          f"{'slope':>7s}{'b_bg':>6s}{'b_ves':>6s}  判定")
    for r in rows:
        print(f"{r['tag']:<26}{r['n2n_ratio']:7.3f}{r['gap_closed']:7.3f}{r['r_ves']:7.3f}"
              f"{r['n2n_ratio_bg']:7.3f}{r['n2n_ratio_ves']:7.3f}{r['slope_vs_input']:7.3f}"
              f"{r.get('b_bg_mean', float('nan')):6.2f}{r.get('b_ves_mean', float('nan')):6.2f}"
              f"  {'通过' if r['pass'] else '不通过'}")
    print(f"\n输入 n2n=1.000  r_ves={ref['r_ves']:.3f}  天花板={ceiling:.3f}  "
          f"oracle 分区 b={ {k: round(v, 3) for k, v in reg.items()} }")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tire.json").write_text(json.dumps(
        {"psf": PSF, "windows": args.windows, "kappas": args.kappas,
         "noise_estimator": "TIRe: σ̂²_pd=(pd_half1−pd_half2)², σ̂²_log=σ̂²_pd/(s+pd)²",
         "calibration": calib, "ceiling_n2n": ceiling, "input": ref,
         "oracle_regions": reg, "rows": rows}, indent=1, ensure_ascii=False))
    if args.save_volumes:
        np.savez_compressed(
            out / "volumes.npz", obs1=obs1, obs2=obs2, truth=truth, valid=valid,
            vessels=vessels, far=far, sig2_log=sig2_log_vox.astype(np.float32),
            sig2_true=sig2_true_vox.astype(np.float32),
            **{f"pred_{k}": v[0].astype(np.float32) for k, v in vols.items()},
            **{f"b_{k}": v[1].astype(np.float32) for k, v in vols.items()})
        print(f"wrote {out}/volumes.npz")
    print(f"wrote {out}/tire.json")


if __name__ == "__main__":
    main()

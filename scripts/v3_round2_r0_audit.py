"""R0：真实 Dryad RCA 数据的噪声「难度」审计（**不训练**）。

回答第二轮审查 §2 的三个问题，全部用**互不依赖的代理**交叉验证：

R0-1 数据资产与处理链      读 .mat、查形状/dtype/有限值/有效 mask、官方带 60:350 的固定投影
R0-2 有没有值得去除的噪声？ ①时间分块重复差 ②残余带能量 ③背景/血管分区 ④log 前后条件方差
                           ⑤空间 ACF/PSD、时间 ACF、深度趋势
R0-3 配对能不能当 N2N？     奇偶 vs 连续块；"共用基"投影；**噪声 ACF 从分块差分里估**（去掉共享信号）

关键设计（为什么这么做）
------------------------
1. **固定基**：所有分块 PD 都用**同一个**全帧 SVD 基（`[59,350)`，官方口径）投影，
   只改变"对哪些帧求和"。这样分块之间的差**只**来自时间上不共享的成分，
   不会把"每块各自估基"的自由度混进来（审查 R0-3 明确要求记录这一点）。
2. **噪声的空间相关必须从差分里估**：直接算 `corr(PD_A, PD_B)` 得到的是 0.9–0.95，
   里面主要是共享的血管与静态散斑（审查 0.8）。分块差分 `D = PD_i − PD_j` **消掉共享信号**，
   它的空间 ACF 就是噪声自己的 ACF。这是本轮对"噪声相关 ≠ 原图空间相关"的操作化。
3. **"可分离噪声"的定义写死在代码里**：把 T 帧分成 K 个等长连续块，第 k 块 PD 是块内平均功率。
   若瞬时噪声在时间上独立，则块间方差的期望 = 瞬时噪声方差 / n_block_frames。
   于是"单帧瞬时噪声方差" ≈ 块间方差 × n_block_frames；而最终 400 帧平均图上还剩
   块间方差 / K（4 块时）。两个量都报，避免"1% 是所有残余噪声"的误读。

用法
----
    python -u scripts/v3_round2_r0_audit.py --dataset SA64
    python -u scripts/v3_round2_r0_audit.py --dataset HSA64
产物
----
    reports/v3_round2/{dataset}_r0.json
    reports/v3_round2/fig_r0_{dataset}.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_svd import load_array, normalize_shape          # noqa: E402
from v3_phase5_real_compare import (                          # noqa: E402
    band_slice, svd_basis,
)

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
FRAME_RATE = {"SA64": 46.875, "HSA64": 46.875}
VOXEL_MM = 0.15


def block_pd(coeff, Q, idx, shape3d):
    """固定基下的块内平均功率。

    **口径**：系数 `coeff = C @ Q` 由**全部 T 帧**算出（投影算子与分块无关，审查 R0-3 要求的固定基），
    只把功率求和限制在块内的帧上：`filtered_t = coeff @ Q[idx].H`，`PD_block = mean_t |filtered_t|^2`。
    这样分块之间的差**只**来自"块内帧不同"，不会混入"每块各自估基"的自由度。
    """
    P = coeff @ Q[idx].conj().T
    return np.mean(np.abs(P) ** 2, axis=-1).reshape(shape3d).astype(np.float32)


def acf_axis(x, mask, axis, lags=8):
    """一维 ACF：沿 axis 平移，皮尔逊相关，掩膜内配对。"""
    out = []
    for lag in range(1, lags + 1):
        a = np.take(x, range(0, x.shape[axis] - lag), axis=axis)
        b = np.take(x, range(lag, x.shape[axis]), axis=axis)
        m = np.take(mask, range(0, mask.shape[axis] - lag), axis=axis) & \
            np.take(mask, range(lag, mask.shape[axis]), axis=axis)
        if m.sum() < 50:
            out.append(float("nan"))
            continue
        out.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    return out


def corr_len(acf, voxel=1.0):
    """相关长度：ACF 首次降到 1/e 的 lag（线性插值），单位=体素。"""
    e = 1 / np.e
    prev = 1.0
    for k, v in enumerate(acf, start=1):
        if not np.isfinite(v):
            continue
        if v <= e:
            t = (prev - e) / max(prev - v, 1e-12)
            return float((k - 1 + t) * voxel)
        prev = v
    return float(len(acf) * voxel)


def conditional_variance(sig, noise_var, bins=8, mask=None):
    """按信号分箱的噪声方差：`Var(noise | signal bin)`。

    `sig` 用块平均 log PD（信号代理），`noise_var` 用**逐体素**的块间方差。
    返回每个箱的中位信号、方差中位数、样本数、以及 log-log 斜率。
    """
    s = sig if mask is None else sig[mask]
    v = noise_var if mask is None else noise_var[mask]
    ok = np.isfinite(s) & np.isfinite(v) & (v > 0)
    s, v = s[ok], v[ok]
    if s.size < bins * 20:
        return {"bins": [], "slope": float("nan"), "n": int(s.size)}
    qs = np.quantile(s, np.linspace(0, 1, bins + 1))
    rows = []
    for i in range(bins):
        m = (s >= qs[i]) & (s <= qs[i + 1] if i == bins - 1 else s < qs[i + 1])
        if m.sum() < 20:
            continue
        rows.append({"sig_med": float(np.median(s[m])), "var_med": float(np.median(v[m])),
                     "n": int(m.sum())})
    if len(rows) >= 3:
        x = np.log(np.array([max(r["sig_med"], 1e-9) for r in rows]))
        y = np.log(np.array([max(r["var_med"], 1e-12) for r in rows]))
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float("nan")
    return {"bins": rows, "slope": slope, "n": int(s.size)}


def audit(dataset: str, n_blocks: int = 4, do_figure: bool = True) -> dict:
    t0 = time.time()
    rec: dict = {"dataset": dataset, "source": RAW[dataset],
                 "frame_rate_hz": FRAME_RATE[dataset], "voxel_mm": VOXEL_MM}

    x = normalize_shape(load_array(RAW[dataset], None), 0, 0)
    z, a, b, T = x.shape
    rec["r0_1_asset"] = {
        "shape_zyxt": [int(v) for v in x.shape], "dtype": str(x.dtype),
        "n_frames": int(T), "n_voxels": int(z * a * b),
        "finite": bool(np.isfinite(x).all()),
        "duration_s": round(T / FRAME_RATE[dataset], 3),
    }

    C = x.reshape(-1, T)
    evals, V = svd_basis(C)
    lo, hi = band_slice(T)                                    # 官方口径 [59,350) / [29,175) ...
    Q = V[:, lo:hi]
    rec["r0_1_asset"]["band_0based"] = [int(lo), int(hi)]
    rec["r0_1_asset"]["band_modes"] = int(hi - lo)
    energy = evals[:max(len(evals), 1)]
    tot = float(np.sum(np.abs(energy)))
    rec["r0_1_asset"]["energy_frac_in_band"] = float(np.sum(np.abs(energy[lo:hi])) / max(tot, 1e-12))
    rec["r0_1_asset"]["energy_frac_below_band"] = float(np.sum(np.abs(energy[:lo])) / max(tot, 1e-12))
    rec["r0_1_asset"]["energy_frac_above_band"] = float(np.sum(np.abs(energy[hi:])) / max(tot, 1e-12))

    shape3d = (z, a, b)
    # ---- 分块 PD（固定基）：系数由全帧算出，功率只按块求和
    coeff = C @ Q                                                        # (V, M)
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    PD = np.stack([block_pd(coeff, Q, idx, shape3d) for idx in blocks])   # (K, Z, A, B)
    pd_full = PD.mean(axis=0)
    valid = pd_full > 0
    odd = block_pd(coeff, Q, np.arange(0, T, 2), shape3d)
    even = block_pd(coeff, Q, np.arange(1, T, 2), shape3d)

    # ---- 血管 / 背景分区（阈值口径写明，不用每种方法自己的阈值）
    q_bg, q_vessel = 0.20, 0.99
    bg = valid & (pd_full <= np.quantile(pd_full[valid], q_bg))
    vessel = valid & (pd_full >= np.quantile(pd_full[valid], q_vessel))
    rec["r0_2_masks"] = {"background": "PD <= q20(valid)", "vessel": "PD >= q99(valid)",
                         "n_valid": int(valid.sum()), "n_bg": int(bg.sum()),
                         "n_vessel": int(vessel.sum())}

    # ---- 代理 ①：分块重复差（"有没有值得去除的噪声"）
    eps = 1e-12
    mean_pd = pd_full
    var_blocks = PD.var(axis=0, ddof=1)                     # 块间方差
    n_blk_frames = int(edges[1] - edges[0])
    # 瞬时（单帧）噪声方差 ≈ 块间方差 × 块内帧数；400 帧平均图上剩余 = 块间方差 / K
    inst_var = var_blocks * n_blk_frames
    rec["r0_2_block_repeat"] = {
        "n_blocks": int(n_blocks), "frames_per_block": n_blk_frames,
        "bg_rel_fluct_median": float(np.median(np.sqrt(var_blocks[bg]) / np.maximum(mean_pd[bg], eps))),
        "vessel_rel_fluct_median": float(np.median(np.sqrt(var_blocks[vessel]) /
                                                   np.maximum(mean_pd[vessel], eps))),
        "bg_noise_frac_of_mean_power": float(np.median(var_blocks[bg] / np.maximum(mean_pd[bg], eps) ** 2)),
        "full_image_residual_noise_var": float(np.median(var_blocks[bg] / n_blocks)),
        "instantaneous_noise_var_est": float(np.median(inst_var[bg])),
        "note": "块间方差 × 块内帧数 = 单帧瞬时噪声方差（假设瞬时噪声时间独立）；"
                "400 帧平均图上的剩余 = 块间方差 / K。两个量含义不同，不能混用",
    }

    # ---- 代理 ②：残余带能量（带外功率占比，代理杂波+噪声）
    pd_out = None
    Q_out = np.concatenate([V[:, :lo], V[:, hi:]], axis=1)
    pd_out = np.mean(np.abs((C @ Q_out) @ Q_out.conj().T) ** 2, axis=-1).reshape(shape3d)
    rec["r0_2_residual_band"] = {
        "out_band_modes": int(Q_out.shape[1]),
        "bg_out_over_in_ratio": float(np.median(pd_out[bg]) / max(float(np.median(pd_full[bg])), eps)),
        "vessel_out_over_in_ratio": float(np.median(pd_out[vessel]) /
                                           max(float(np.median(pd_full[vessel])), eps)),
        "note": "带外功率/带内功率；带外里既有被丢弃的杂波也有噪声，不能单独当噪声能量",
    }
    del pd_out, Q_out

    # ---- 代理 ③：噪声的空间相关（从**分块差分**里估，消掉共享信号）
    D = PD[0] - PD[1]
    Dlog = np.log(np.maximum(PD[0], eps)) - np.log(np.maximum(PD[1], eps))
    d_acf = {ax: acf_axis(Dlog, bg, i, 8) for i, ax in enumerate(("z", "y", "x"))}
    rec["r0_2_noise_acf_from_block_diff"] = {
        "log_domain": d_acf,
        "corr_len_voxels": {ax: corr_len(v, 1.0) for ax, v in d_acf.items()},
        "corr_len_mm": {ax: corr_len(v, VOXEL_MM) for ax, v in d_acf.items()},
        "note": "D = log PD_block0 − log PD_block1，共享的血管/静态散斑在差分里被消掉，"
                "剩下的就是噪声自身的空间相关；这是回应审查 0.8 的操作化定义",
    }
    # 对照：直接相关（含共享信号），只作为"为什么不能这么用"的反例
    rec["r0_2_naive_block_corr_bg"] = float(np.corrcoef(
        np.log(np.maximum(PD[0][bg], eps)), np.log(np.maximum(PD[1][bg], eps)))[0, 1])

    # ---- 代理 ④：log 前后的条件方差（B 的前提）
    sig_lin = PD.mean(axis=0)
    varlin = PD.var(axis=0, ddof=1)
    # log 域的块间方差：先在每块取 log 再算方差
    logPD = np.log(np.maximum(PD, eps))
    varlog = logPD.var(axis=0, ddof=1)
    sig_log = logPD.mean(axis=0)
    rec["r0_2_conditional_var"] = {
        "linear_vs_signal": conditional_variance(sig_lin, varlin, mask=bg),
        "log_vs_signal": conditional_variance(sig_log, varlog, mask=bg),
        "note": "斜率≈2 表示噪声方差 ∝ 信号²（乘性）；斜率≈0 表示与信号无关（log 后加性）。"
                "只用背景体素（PD<=q20），因此不把血流的时间变化算进噪声；"
                "但背景里若有缓慢运动，仍会被算进来，见报告 §限制",
    }

    # ---- 代理 ⑤：深度趋势 / 时间 ACF
    zprof = [{"z": int(k), "median_log": float(np.median(logPD.mean(axis=0)[k][valid[k]]))
              if valid[k].any() else None} for k in range(z)]
    # 时间 ACF：在背景体素上对每块求和前的逐帧功率做平均（用 |P_t|² 的时间相关）
    Pw = np.abs((C @ Q) @ Q.conj().T) ** 2
    ts = Pw.mean(axis=0)
    ts = ts - ts.mean()
    ac = np.correlate(ts, ts, mode="full")[len(ts) - 1:]
    ac = ac / max(ac[0], eps)
    t_acf = [float(v) for v in ac[1:21]]
    del Pw
    rec["r0_2_temporal"] = {"global_pd_temporal_acf_lag1_20": t_acf,
                            "frame_rate_hz": FRAME_RATE[dataset],
                            "z_profile_median_log_pd": zprof,
                            "note": "全卷平均功率的时间自相关；含血流与运动，不是纯噪声"},
    rec["r0_2_odd_even"] = {
        "bg_corr_log": float(np.corrcoef(np.log(np.maximum(odd[bg], eps)),
                                          np.log(np.maximum(even[bg], eps)))[0, 1]),
        "vessel_corr_log": float(np.corrcoef(np.log(np.maximum(odd[vessel], eps)),
                                              np.log(np.maximum(even[vessel], eps)))[0, 1]),
        "bg_odd_even_mse_over_var": float(np.mean((np.log(np.maximum(odd[bg], eps)) -
                                                    np.log(np.maximum(even[bg], eps))) ** 2) /
                                           max(np.var(np.log(np.maximum(odd[bg], eps))), eps)),
        "note": "奇偶两组帧互不重叠，但**共享同一份血管与静态散斑**，所以高相关不等于噪声独立",
    }
    # ---- R0-3：扣掉共同信号后，两半之间还剩多少相关（配对能不能当 N2N）
    # 四等分：g0/g1 当作"两次观测"，g2+g3 的 PD 当作**共享信号**的估计（与 g0、g1 都不共享帧）。
    g0, g1 = np.arange(0, T // 4), np.arange(T // 4, T // 2)
    g23 = np.arange(T // 2, T)
    pd_a = block_pd(coeff, Q, g0, shape3d)
    pd_b = block_pd(coeff, Q, g1, shape3d)
    pd_s = block_pd(coeff, Q, g23, shape3d)
    la, lb, ls = (np.log(np.maximum(v, eps)) for v in (pd_a, pd_b, pd_s))
    ra, rb = la - ls, lb - ls
    rec["r0_3_pairing"] = {
        "cross_half_residual_corr_bg": float(np.corrcoef(ra[bg], rb[bg])[0, 1]),
        "cross_half_residual_corr_vessel": float(np.corrcoef(ra[vessel], rb[vessel])[0, 1]),
        "half_abs_corr_bg": float(np.corrcoef(la[bg], lb[bg])[0, 1]),
        "shared_frac_bg": float(1.0 - 0.5 * np.mean((ra[bg] - rb[bg]) ** 2) /
                                max(np.var(la[bg]), eps)),
        "note": "g0、g1 是两段互不重叠的 100 帧；共享信号用 g2+g3 的 PD 估计（与 g0/g1 都不共享帧）。"
                "两者残差的相关 = 扣掉共同信号之后两次观测还剩多少共同成分（含未建模的运动、"
                "非线性、以及共用的固定基）。这是 pseudo-pair 能不能当 N2N 对的第一个直接读数",
    }
    # ---- R0-3b：两种**同积分长度**的二分法对比（奇偶 vs 连续前后半），G-R0-a 就靠这一对
    half_a = block_pd(coeff, Q, np.arange(0, T // 2), shape3d)
    half_b = block_pd(coeff, Q, np.arange(T // 2, T), shape3d)
    lha, lhb = np.log(np.maximum(half_a, eps)), np.log(np.maximum(half_b, eps))
    lo_, le_ = np.log(np.maximum(odd, eps)), np.log(np.maximum(even, eps))
    lfull = np.log(np.maximum(pd_full, eps))
    def _split_stats(a, b, m):
        var_tot = float(np.var(lfull[m]))
        mse = float(np.mean((a[m] - b[m]) ** 2))
        return {"mse_log": mse, "var_log_full": var_tot,
                "noise_frac_of_var": mse / 2.0 / max(var_tot, eps),
                "corr": float(np.corrcoef(a[m], b[m])[0, 1])}
    rec["r0_3_split_types"] = {
        "n_frames_per_half": int(T // 2),
        "bg": {"odd_even": _split_stats(lo_, le_, bg),
               "contiguous_halves": _split_stats(lha, lhb, bg)},
        "vessel": {"odd_even": _split_stats(lo_, le_, vessel),
                   "contiguous_halves": _split_stats(lha, lhb, vessel)},
        "note": "两种二分法**每半都是 200 帧**，所以噪声占比可直接比；差异只来自"
                "『交错拆分 vs 连续前后半』。连续前后半更可能混入慢运动/漂移，"
                "奇偶拆分把时间上相邻的帧分开，两者一致才说明拆分方式本身不是主导因素",
    }
    rec["meta"] = {"script": "scripts/v3_round2_r0_audit.py",
                   "elapsed_s": round(time.time() - t0, 1),
                   "convention": "固定全帧 SVD 基投影，块内平均功率；官方带 1-based 60:350 → 0-based [59,350)",
                   "caveat": "SA64/HSA64 是两次不同发射协议的成像，不是独立生物重复；"
                             "分块/奇偶都是同一次采集内的拆分，属于 pseudo-pair"}

    if do_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = ROOT / "reports/v3_round2"
        out.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].plot(range(z), [p["median_log"] if p["median_log"] is not None else np.nan
                                for p in zprof], marker="o", ms=3)
        axes[0].set_xlabel("z slice"); axes[0].set_ylabel("median log PD")
        axes[0].set_title(f"{dataset}: depth profile")
        for ax, name in zip(("z", "y", "x"), ("z", "y", "x")):
            axes[1].plot(range(1, 9), d_acf[ax], marker="o", ms=3, label=name)
        axes[1].axhline(1 / np.e, ls="--", c="k", lw=0.8)
        axes[1].set_xlabel("lag (voxel)"); axes[1].set_ylabel("ACF of block difference")
        axes[1].set_title("noise ACF (shared signal removed)"); axes[1].legend()
        for key, lab in (("linear_vs_signal", "linear PD"), ("log_vs_signal", "log PD")):
            rows = rec["r0_2_conditional_var"][key]["bins"]
            if rows:
                axes[2].loglog([r["sig_med"] for r in rows], [r["var_med"] for r in rows],
                               marker="o", ms=3, label=f"{lab} (slope "
                               f"{rec['r0_2_conditional_var'][key]['slope']:.2f})")
        axes[2].set_xlabel("signal (bin median)"); axes[2].set_ylabel("noise variance")
        axes[2].set_title("Var(noise | signal), background only"); axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"fig_r0_{dataset}.png", dpi=130)
        plt.close(fig)
        rec["figure"] = f"reports/v3_round2/fig_r0_{dataset}.png"

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    rec = audit(args.dataset, args.n_blocks, not args.no_figure)
    out = ROOT / "reports/v3_round2"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{args.dataset}_r0.json"
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"wrote {p}")
    print(json.dumps({k: rec[k] for k in ("r0_2_block_repeat", "r0_2_noise_acf_from_block_diff",
                                          "r0_2_naive_block_corr_bg")},
                     indent=1, ensure_ascii=False)[:2200])


if __name__ == "__main__":
    main()

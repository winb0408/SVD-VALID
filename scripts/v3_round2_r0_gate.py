"""G-R0：把第二轮审查 §2 的三条「真实数据够不够敏感」判据落成可复算的脚本。

判据（写死在跑之前，见 `reviews/V3_PR1_round2_plan.md` §3.1）：

* **G-R0-a**：两条相互独立的噪声代理给出的「可分离噪声相对能量」同量级（比值 ∈ [0.5, 2]）
    * 代理 1（时间）：`r0_2_block_repeat` 的块间差 —— 4 个 100 帧块，
      `noise_var = mse(log PD_0 - log PD_1)/2`；由 `scripts/v3_round2_r0_audit.py` 产出；
    * 代理 2（空间）：背景区按 **4×4×4 小块**分区，块内 log PD 方差取平均。
      静态散斑在空间上是平滑的（PSF σ≈3 体素），块内方差里既有噪声也有它的空间涟漪，
      所以这一条是**上界**；同时给一档"先减去块内 3D 高斯平滑（σ=1.5）再求方差"的版本，
      把慢涟漪削掉一些，作为更接近噪声的下界读法。两档都报。
* **G-R0-b**：背景区 `Var(noise|signal)` 在 log 域与线性域都要能测出显著斜率（|slope| 显著）；
* **G-R0-c**：奇偶拆分 vs 连续前后半拆分，在**同样 200 帧**下"扣掉共享信号后的残差相关"可分离。

不设单一通过阈值，三种结局（够敏感 / 不够敏感 / 数据不足）都写进 JSON。

用法
----
    python -u scripts/v3_round2_r0_gate.py --dataset SA64
输出 `reports/v3_round2/gate_r0_<dataset>.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_svd import load_array, normalize_shape            # noqa: E402
from v3_round2_r0_audit import band_slice, svd_basis, block_pd  # noqa: E402,F401

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
BG_Q = 0.20
EPS = 1e-12
PATCH = 4
PROXY_RATIO_RANGE = (0.5, 2.0)     # 写死在跑之前
SLOPE_MIN = 0.10                   # |slope| 小于此视为"测不出显著依赖"


def patch_stats(lp: np.ndarray, mask: np.ndarray, patch: int = PATCH) -> dict:
    """背景区 4×4×4 小块内的 log PD 方差（只统计整块都在背景内的块）。"""
    z, y, x = lp.shape
    nz, ny, nx = z // patch, y // patch, x // patch
    out, n_used = [], 0
    for iz, iy, ix in product(range(nz), range(ny), range(nx)):
        sl = (slice(iz * patch, (iz + 1) * patch),
              slice(iy * patch, (iy + 1) * patch),
              slice(ix * patch, (ix + 1) * patch))
        m = mask[sl]
        if not m.all():
            continue
        v = lp[sl]
        out.append(float(np.var(v)))
        n_used += 1
    out = np.asarray(out)
    return {"n_blocks": int(n_used), "patch": patch,
            "within_var_median": float(np.median(out)) if out.size else float("nan"),
            "within_var_mean": float(out.mean()) if out.size else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    a = ap.parse_args()
    D = a.dataset

    r0 = json.loads((ROOT / f"reports/v3_round2/{D}_r0.json").read_text())

    x = normalize_shape(load_array(RAW[D], None), 0, 0)
    z, a1, b1, T = x.shape
    shape3d = (z, a1, b1)
    C = x.reshape(-1, T)                     # (V, T)；svd_basis 要 2D
    evals, V = svd_basis(C)
    lo, hi = band_slice(T)
    Q = V[:, lo:hi]
    coeff = C @ Q
    pd_full = block_pd(coeff, Q, np.arange(T), shape3d)
    valid = pd_full > 0                      # 与 r0_audit 同口径
    lp = np.log(np.maximum(pd_full, EPS))
    bg = valid & (pd_full <= np.quantile(pd_full[valid], BG_Q))

    # ---- 代理 2（空间）----
    ps = patch_stats(lp, bg, PATCH)

    # ---- 代理 1（时间）：与 R0 JSON 同一口径，重算一遍确保同源 ----
    per = T // 4
    blocks = [block_pd(coeff, Q, np.arange(b * per, (b + 1) * per), shape3d) for b in range(4)]
    lbs = [np.log(np.maximum(b_, EPS)) for b_ in blocks]
    var_bg = float(np.var(lp[bg]))
    mse_pairs = [float(np.mean((lbs[i][bg] - lbs[j][bg]) ** 2))
                 for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    noise_var_temporal = float(np.median(mse_pairs)) / 2.0

    rel_temporal = noise_var_temporal / var_bg
    rel_spatial_med = ps["within_var_median"] / var_bg
    rel_spatial_mean = ps["within_var_mean"] / var_bg

    # ---- 关键：两条代理必须在**同一个场**上比，否则比值无意义 ----
    # 时间代理给的是 100 帧块级的噪声方差；空间代理直接作用在 400 帧的 log PD 上，
    # 那里的噪声方差是块级的 1/4，且还混着静态散斑的空间涟漪。所以先把空间代理搬到
    # **块间涨落场** F_b = log PD_b - mean_b 上：F_b 里静态散斑被块均值带走，只剩噪声，
    # 但它被去均值过，方差被压到 σ²·(1-1/K)，K=4 时乘 4/3 才是 σ² 的无偏估计。
    F = np.stack([lbs[b] - np.mean(lbs, axis=0) for b in range(4)])      # (4,z,y,x)
    K = F.shape[0]
    ps_noise_blocks = []
    for b in range(K):
        # corr_on 的产物是场本身，这里直接对 F_b 做块内方差
        ps_noise_blocks.append(patch_stats(F[b], bg, PATCH)["within_var_median"])
    ps_noise = float(np.median(ps_noise_blocks)) * K / (K - 1.0)          # 去均值偏差修正
    ratio_controlled = ps_noise / max(noise_var_temporal, EPS)
    ratio_naive = rel_spatial_med / max(rel_temporal, EPS)

    ga = {
        "temporal": {"noise_var": noise_var_temporal, "rel_energy": rel_temporal,
                     "mse_pairs": mse_pairs,
                     "note": "4 个 100 帧块的块间差，mse/2；块间不共享帧，静态散斑被块均值带走"},
        "spatial_on_noise_field": {
            "within_var_median_per_block": ps_noise_blocks,
            "within_var_demean_corrected": ps_noise, "patch": PATCH,
            "note": "块间涨落场 F_b 上的 4×4×4 块内方差，乘 K/(K-1) 修正去均值偏差；"
                    "与时间代理估计同一个量"},
        "spatial_on_full_log_pd": {**ps, "rel_energy_median": rel_spatial_med,
                                   "rel_energy_mean": rel_spatial_mean,
                                   "note": "直接对 400 帧 log PD 做块内方差：含静态散斑空间涟漪，"
                                           "只能当上界，不参与判定"},
        "var_log_pd_bg": var_bg,
        "ratio_controlled_noise_field": ratio_controlled,
        "ratio_naive_full_log_pd": ratio_naive,
        "ratio_range": list(PROXY_RATIO_RANGE),
        "pass": bool(PROXY_RATIO_RANGE[0] <= ratio_controlled <= PROXY_RATIO_RANGE[1]),
        "why_controlled_is_the_test": "两条代理只有在同一个场上才可比；全图 log PD 上的空间代理"
                                      "被静态散斑抬高（本例 rel 由噪声的 1/4 不到抬到 0.35），"
                                      "拿它去比会得出『空间噪声比时间噪声大 2.5 倍』的假结论",
    }

    # ---- G-R0-b：条件方差可测 ----
    lin_slope = r0["r0_2_conditional_var"]["linear_vs_signal"]["slope"]
    log_slope = r0["r0_2_conditional_var"]["log_vs_signal"]["slope"]
    gb = {"linear_slope": lin_slope, "log_slope": log_slope,
          "slope_min_abs": SLOPE_MIN,
          "pass": bool(abs(lin_slope) >= SLOPE_MIN and abs(log_slope) >= SLOPE_MIN),
          "status": "historical — 只当『斜率可测性检查』，不再当同方差化判据",
          "note": "线性域斜率≈2 ⇒ 噪声方差随信号成正比（乘性）；log 域斜率≈0 ⇒ 信号无关。"
                  "两条都要测得出，才能说 log 变换改变了噪声-信号依赖。"
                  "**Round3 审查 R3-P02 更正**：本判据只看『斜率是否不为零』，"
                  "它奖励的是『有斜率』，与『同方差化』的科学目标相反；"
                  "真正的判据见 G-R0-b2。"}

    # ---- G-R0-b2（Round3 新增，R3-P02）：log 域是否真的同方差 ----
    # 门槛**先写**（2026-10-28，在看 v2 数之前定）：
    #   ① 斜率落在等效零区间 |slope| ≤ 0.20；
    #   ② 8 个信号分位箱内，Var(log noise) 的 max/min ≤ 2.0（一个 2 倍以内）。
    # 依据：'同方差'若成立，方差沿信号轴应基本平坦；一个 2 倍的摆动已经是宽松上限。
    CV = r0["r0_2_conditional_var"]

    def _bin_ratio(entry):
        vs = [b.get("var_med") for b in (entry.get("bins") or []) if b.get("var_med")]
        return float(max(vs) / min(vs)) if len(vs) >= 3 else None

    slope_tol, bin_ratio_max = 0.20, 2.0
    log_ratio = _bin_ratio(CV["log_vs_signal"])
    lin_ratio = _bin_ratio(CV["linear_vs_signal"])
    dep = CV.get("log_vs_signal_deprecated_double_log") or {}
    gb2 = {
        "log_slope_same_x": log_slope,
        "log_slope_deprecated_double_log": dep.get("slope"),
        "slope_tol": slope_tol,
        "log_var_ratio_max_over_bins": log_ratio,
        "linear_var_ratio_max_over_bins": lin_ratio,
        "bin_ratio_max": bin_ratio_max,
        "slope_within_tol": bool(np.isfinite(log_slope) and abs(log_slope) <= slope_tol),
        "bins_within_ratio": bool(log_ratio is not None and log_ratio <= bin_ratio_max),
        "pass": bool(np.isfinite(log_slope) and abs(log_slope) <= slope_tol
                     and log_ratio is not None and log_ratio <= bin_ratio_max),
        "criteria": "①|log 域斜率| ≤ 0.20（同一 x）；②log 域方差跨 8 箱 max/min ≤ 2.0",
        "note": "**判据是『log 域方差与信号无关』，不是『测得出斜率』。** 旧 G-R0-b 的 "
                "abs(slope)≥0.10 恰好在奖励非零斜率，且它的 log 分支用的是 log(log PD)，"
                "已被 R3-P02 判定口径不一致；旧数原样保留在 JSON 里，不改不删。",
    }

    # ---- G-R0-d（Round3 新增，R3-P01）：残差相关里的"参考伪像" ----
    lsl = r0["r0_3_pairing"]
    dis = r0.get("r0_3_pairing_disjoint") or {}
    corr_shared = lsl["cross_half_residual_corr_bg"]
    corr_dis = dis.get("residual_corr_bg")
    # 门槛先写：独立噪声下共享参考应给 ≈1/3（解析值），互不共享参考应给 ≈0。
    # 判定分档（不看结果先定）：disjoint ≤ 0.15 → 旧数主要是伪像；≥ 0.30 → 数据侧确有关系；
    # 中间 → 只能说"部分伪像 + 部分未建模成分"，不得量化成"噪声相关 36%"。
    gd = {
        "corr_shared_reference_bg": corr_shared,
        "corr_disjoint_reference_bg": corr_dis,
        "independence_null_shared": 1.0 / 3.0,
        "independence_null_disjoint": 0.0,
        "excess_over_null_shared": (float(corr_shared) - 1.0 / 3.0
                                    if corr_shared is not None else None),
        "perm_corr_range": dis.get("perm_corr_range"),
        "bands": {"artifact_dominated_upper": 0.15, "data_dominated_lower": 0.30},
        "verdict": ("NOT_RUN" if corr_dis is None else
                    "artifact_dominated" if corr_dis <= 0.15 else
                    "data_side_component" if corr_dis >= 0.30 else "mixed"),
        "pass": None,
        "note": "本项**不设过门**：它是旧结论的更正项，只要求①两种口径都报；"
                "②旧的 0.357/0.374 加注『含共用参考伪像』；③不得再把 0.36 读成『噪声相关 36%』。",
    }

    # ---- G-R0-c：两种拆分的残差相关可分离 ----
    sp = r0["r0_3_split_types"]
    lsl = r0["r0_3_pairing"]
    d_bg = abs(sp["bg"]["odd_even"]["noise_frac_of_var"] - sp["bg"]["contiguous_halves"]["noise_frac_of_var"])
    d_ves = abs(sp["vessel"]["odd_even"]["noise_frac_of_var"] - sp["vessel"]["contiguous_halves"]["noise_frac_of_var"])
    gc = {"bg_noise_frac_odd_even": sp["bg"]["odd_even"]["noise_frac_of_var"],
          "bg_noise_frac_contiguous": sp["bg"]["contiguous_halves"]["noise_frac_of_var"],
          "vessel_noise_frac_odd_even": sp["vessel"]["odd_even"]["noise_frac_of_var"],
          "vessel_noise_frac_contiguous": sp["vessel"]["contiguous_halves"]["noise_frac_of_var"],
          "abs_diff_bg": d_bg, "abs_diff_vessel": d_ves,
          "residual_corr_bg": lsl["cross_half_residual_corr_bg"],
          "residual_corr_vessel": lsl["cross_half_residual_corr_vessel"],
          "pass": bool((d_bg > 0.02) or (d_ves > 0.02)),
          "note": "只要两种拆分的读数不同（背景差 2 个百分点以上或血管差 2 个百分点以上），"
                  "就说明配对方式本身是主导因素之一，不存在唯一可用的伪配对"}

    rec = {"dataset": D, "source": RAW[D],
           "G_R0_a": ga, "G_R0_b": gb, "G_R0_b2": gb2, "G_R0_c": gc, "G_R0_d": gd,
           "version": "v2 (2026-10-28, Round3 审查 R3-P01/P02 修订；旧判据原样保留在 G_R0_b)",
           "verdict": {
               "a": "pass" if ga["pass"] else "fail",
               "b": "pass" if gb["pass"] else "fail",
               "b2": "pass" if gb2["pass"] else "fail",
               "c": "pass" if gc["pass"] else "fail",
               "d": gd["verdict"]},
           "interpretation": ("三条全过 = 真实数据有可测、非伪影、且配对可分辨的残余噪声 ⇒ "
                              "『真实数据优先』这条路可行；任何一条不过则在报告里写清适用边界，"
                              "不外推")}

    out = ROOT / f"reports/v3_round2/gate_r0_{D}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))

    print(f"== {D}")
    print(f"  G-R0-a  时间={noise_var_temporal:.4e}  空间(噪声场)={ps_noise:.4e}  "
          f"ratio={ratio_controlled:.2f}  [全图口径的假比值 {ratio_naive:.2f}]  → "
          f"{'pass' if ga['pass'] else 'fail'}")
    print(f"  G-R0-b  线性斜率={lin_slope:.2f}  log 斜率={log_slope:.2f}  → {'pass' if gb['pass'] else 'fail'}"
          f"  （historical：只测『有没有斜率』，不是同方差判据）")
    print(f"  G-R0-b2 log 斜率={log_slope:.2f}（tol {slope_tol}）  方差跨箱 max/min={log_ratio}（上限 {bin_ratio_max}）"
          f"  → {'pass' if gb2['pass'] else 'fail'}")
    print(f"  G-R0-d  共享参考 {corr_shared:.3f}（null 0.333）  非共享参考 "
          f"{'未跑' if corr_dis is None else f'{corr_dis:.3f}'}（null 0.000）  → {gd['verdict']}")
    print(f"  G-R0-c  bg {sp['bg']['odd_even']['noise_frac_of_var']:.3f} vs "
          f"{sp['bg']['contiguous_halves']['noise_frac_of_var']:.3f} | "
          f"vessel {sp['vessel']['odd_even']['noise_frac_of_var']:.3f} vs "
          f"{sp['vessel']['contiguous_halves']['noise_frac_of_var']:.3f}  → "
          f"{'pass' if gc['pass'] else 'fail'}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

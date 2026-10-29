"""R06：仿真器正向模型 / 三类真值 / PSF 与血管真值的审计。

外部审查 R06 的原话（摘要）：仿真器的血流项不是"血管密度卷积 PSF"，且没有显式前向模型；
要求把前向模型写清楚、把"几何真值 / 无噪成像真值 / 有噪观测"三类真值分开、
并对 PSF 与血管真值做 sanity check。

本脚本**不产生新判据、不改任何既有数字**，只做三件事：
  1. 把前向模型逐项写成可核对的式子，并给出每一项在代码里的行号；
  2. 数值核验三类真值的分离是否成立（几何掩膜 / 仅血流 PD / 全项 PD）；
  3. PSF 与血管的 sanity check：背景 log-PD 的 ACF 对实测目标、血管剖面宽度对掩膜宽度、
     以及"真值不是输入的单调重标定"（否则实验 A 的 RMSE 比较就没意义）。

用法：
    python -u scripts/v3_r06_forward_audit.py
输出 `reports/v3_r06/forward_audit.json` 与两张图。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v3_simulator import (acf_vs_target, background_mask, load_target,  # noqa: E402
                          simulate, SimCfg, band_and_pd)

OUT = ROOT / "reports" / "v3_r06"


def radial_profile(pd, vessels, valid, rmax=6):
    """以血管掩膜的最近体素距离为横轴，统计 PD 的均值剖面（只算 valid 内）。"""
    dist = distance_transform_edt(~vessels)
    prof = {}
    for r in range(rmax):
        sel = valid & (np.abs(dist - r) < 0.5)
        if sel.sum() < 50:
            continue
        prof[int(r)] = {"mean": float(np.mean(pd[sel])),
                        "median": float(np.median(pd[sel])),
                        "n": int(sel.sum())}
    return prof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = load_target("fast")["SA64"]

    # ------------------------------------------------------------------ 1. 前向模型
    model = {
        "forward_model": [
            "X(z,y,x,t) = static(z,y,x)*u_static(t)",
            "           + dynamic_t(z,y,x)                        ~ CN(0, psf_sigma)",
            "           + sum_r c_r(z,y,x)*g_r(t)                   clutter, r=1..6",
            "           + flow_t(z,y,x) = G_t(z,y,x) * vessels(z,y,x)   ~ CN(0, psf_sigma)",
            "           + thermal_t(z,y,x)                          ~ CN(0, thermal_sigma)",
            "PD_obs  = sum_t | B{ X }_t |^2 / median(PD_obs[valid])",
            "PD_true = sum_t | B{ flow }_t |^2 / median(PD_obs[valid])",
            "B = SVD band-pass, official 1-based 60:350 -> 0-based [59,350) = 291 modes",
        ],
        "code_lines": {
            "static": "scripts/v3_simulator.py:150 static = s_spatial[...,None]*u_static[None,None,None,:]",
            "dynamic": "scripts/v3_simulator.py:153-155",
            "clutter": "scripts/v3_simulator.py:157-163 (rank 6, low-time g)",
            "flow": "scripts/v3_simulator.py:171-175 flow = gauss_field(psf)*vessels",
            "thermal": "scripts/v3_simulator.py:155",
            "band": "scripts/official_pipeline.py:55 svd_blood_band(x, 60, 350, faithful=True)",
            "pd": "scripts/v3_simulator.py:200 band_and_pd",
        },
        "three_truths": {
            "T1_geometric": "vessels (binary mask, radius 1.1 vox, 14 tubes)",
            "T2_noise_free_imaging": "gt_flow_pd = PD of the flow term only, through the same SVD band",
            "T3_noisy_observation": "pd_linear = PD of the full X, through the same SVD band",
        },
        "known_departure_from_a_real_imaging_chain": (
            "flow_t = G_t * vessels uses a HARD binary vessel mask multiplied by a PSF-correlated "
            "speckle field; a real chain would convolve a vessel DENSITY (partial volume) with a "
            "point-spread function and then apply the beamforming/vessel-geometry kernel. "
            "Consequences: (i) the vessel cross-section is the union of a 1-voxel mask and the PSF "
            "width, whichever is wider, not a partial-volume-weighted profile; (ii) the flow signal "
            "has no intra-voxel velocity distribution, so no spectral broadening; (iii) there is no "
            "sidelobe/clutter coupling INSIDE vessels. Numbers measured in this audit record how far "
            "these matter here; they are not grounds for claiming the simulator is realistic."
        ),
    }

    # ------------------------------------------------------------------ 2. 三类真值
    cfg = SimCfg(seed=1)
    d = simulate(cfg)
    pd, _ = band_and_pd(d["X"], d["valid"])
    gt, _ = band_and_pd(d["flow"], d["valid"])
    ves = d["vessels"].astype(bool)
    valid = d["valid"].astype(bool)
    s = float(np.median(pd[valid]))
    lo, lt = np.log1p(pd / s), np.log1p(gt / s)

    # (a) 几何真值是否真的对应"有血流"的位置：无噪成像真值在血管内外的分离度
    in_v, out_v = valid & ves, valid & ~ves
    sep = {
        "gt_mean_in_vessels": float(np.mean(lt[in_v])),
        "gt_mean_outside": float(np.mean(lt[out_v])),
        "gt_p99_outside": float(np.percentile(lt[out_v], 99)),
        "frac_outside_above_in_median": float(np.mean(lt[out_v] > np.median(lt[in_v]))),
        "obs_mean_in_vessels": float(np.mean(lo[in_v])),
        "obs_mean_outside": float(np.mean(lo[out_v])),
    }
    # (b) 观测是不是真值的单调重标定？否 ⇒ 实验 A 的 RMSE 比较有内容
    mono = {
        "pearson_log_obs_vs_log_truth_all_valid": float(np.corrcoef(lo[valid], lt[valid])[0, 1]),
        "pearson_log_obs_vs_log_truth_in_vessels": float(np.corrcoef(lo[in_v], lt[in_v])[0, 1]),
        "spearman_pooled": float(_spearman(lo[valid], lt[valid])),
        "note": "r 明显小于 1 且背景散斑占主导 ⇒ 观测不是真值的单调函数，去噪任务真实存在",
    }

    # (c) PSF sanity：背景 log-PD 的 ACF 对实测目标的 RMSE（标定时用的同一个量）
    acf = acf_vs_target(pd, valid, target, mode="fast")

    # (d) 血管剖面：无噪真值的沿距离剖面，与二值掩膜宽度的对比
    prof_gt = radial_profile(gt, ves, valid)
    prof_obs = radial_profile(pd, ves, valid)
    width = {
        "vessel_radius_cfg": cfg.vessel_radius,
        "psf_sigma_cfg": list(cfg.psf_sigma),
        "mean_in_radius_of_1voxel_mask": float(np.mean(distance_transform_edt(~ves)[ves])),
        "gt_profile_by_distance": prof_gt,
        "obs_profile_by_distance": prof_obs,
    }
    # 半高宽：以 r=0 的值为峰、最后一次越过峰值一半的距离作为 HWHM
    def hwhm(prof):
        if not prof:
            return float("nan")
        r0 = prof[min(prof)]["mean"]
        half = r0 / 2.0
        last = 0
        for r in sorted(prof):
            if prof[r]["mean"] >= half:
                last = r
        return float(last)
    width["gt_hwhm_voxels"] = hwhm(prof_gt)
    width["obs_hwhm_voxels"] = hwhm(prof_obs)

    # (e) 血管体素占比（决定"背景主导"程度）与背景散斑水平
    frac = {
        "vessel_voxels": int(in_v.sum()),
        "valid_voxels": int(valid.sum()),
        "vessel_fraction": float(in_v.sum() / valid.sum()),
        "bg_std_input": float(np.std(lo[out_v])),
        "bg_std_truth": float(np.std(lt[out_v])),
        "bg_mean_truth": float(np.mean(lt[out_v])),
        "gt_max_outside_vessels": float(np.max(lt[out_v])),
    }

    # (f) 退化控制：真值背景**恒为 0**，所以"输出全零"这个什么都不做的极端解，
    # 在全局 RMSE 上会拿到一个极低的分数。这是本仿真真值定义的一个直接后果，
    # 必须先量出来，否则任何"RMSE 优于输入"的说法都可能只是在描述背景被抹平。
    degenerate = {
        "rmse_if_output_all_zeros": float(np.sqrt(np.mean(lt[valid] ** 2))),
        "rmse_if_output_all_zeros_vessels_only": float(np.sqrt(np.mean(lt[in_v] ** 2))),
        "rmse_of_input": float(np.sqrt(np.mean((lo[valid] - lt[valid]) ** 2))),
        "note": "全零输出在 valid 上就能拿到上表的 RMSE；凡是据此宣称'去噪有效'的结论无效。"
                "实验 A 必须同时报血管内的保真度与背景压制，不能用全局 RMSE 单独下结论。",
    }

    rec = {"sim_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.__dict__.items()},
           "model": model, "separation": sep, "monotonicity": mono,
           "acf_vs_measured_target": acf, "vessel_profile": width,
           "scale_and_fractions": frac, "degenerate_controls": degenerate}
    (OUT / "forward_audit.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------------ 打印
    print("== 前向模型 ==")
    for line in model["forward_model"]:
        print("  " + line)
    print("\n== 三类真值的分离 ==")
    for k, v in sep.items():
        print(f"  {k:34s} {v:.4f}")
    print("\n== 观测 vs 真值（是否单调重标定）==")
    for k, v in mono.items():
        print(f"  {k:44s} {v if isinstance(v, str) else round(v, 4)}")
    print(f"\n== PSF sanity：背景 log-PD ACF vs 实测目标 SA64 ==")
    for ax in ("z", "y", "x"):
        print(f"  {ax}: rmse={acf[ax]['rmse']:.4f}  sim(lag1)={acf[ax]['sim'][0]:.3f} "
              f"target(lag1)={acf[ax]['target'][0]:.3f}")
    print(f"  overall_rmse={acf['overall_rmse']:.4f}")
    print(f"\n== 血管剖面（以到血管掩膜的距离为横轴）==")
    print(f"  gt HWHM={width['gt_hwhm_voxels']}  观察 HWHM={width['obs_hwhm_voxels']}  "
          f"cfg radius={cfg.vessel_radius}  psf_sigma={cfg.psf_sigma}")
    for r in sorted(prof_gt):
        g, o = prof_gt[r], prof_obs.get(r)
        print(f"  r={r}  gt={g['mean']:.4f}  obs={'nan' if o is None else round(o['mean'], 4)}")
    print(f"\n== 体素占比 ==")
    for k, v in frac.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    print("\n== 退化控制（真值背景恒为 0 的后果）==")
    for k, v in degenerate.items():
        print(f"  {k:44s} {v if isinstance(v, str) else round(v, 4)}")

    _figure(lo, lt, ves, valid, prof_gt, prof_obs, acf)


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic


def _figure(lo, lt, ves, valid, prof_gt, prof_obs, acf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (name, img) in zip(axes[:3], [("geometric truth: vessels (MIP)",
                                           ves.astype(float).max(axis=0)),
                                          ("noise-free imaging truth (MIP)", lt.max(axis=0)),
                                          ("noisy observation (MIP)", lo.max(axis=0))]):
        im = ax.imshow(img, cmap="magma", aspect="equal")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax = axes[3]
    rs = sorted(prof_gt)
    ax.plot(rs, [prof_gt[r]["mean"] for r in rs], "o-", label="noise-free truth")
    rr = sorted(prof_obs)
    ax.plot(rr, [prof_obs[r]["mean"] / max(prof_obs[min(rr)]["mean"], 1e-12) *
                 prof_gt[min(rs)]["mean"] for r in rr], "s--", label="observation (rescaled)")
    ax.set_xlabel("distance to vessel mask (voxels)")
    ax.set_ylabel("PD")
    ax.set_title(f"vessel profile; ACF rmse={acf['overall_rmse']:.3f}", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r06_truths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig_r06_truths.png")


if __name__ == "__main__":
    main()

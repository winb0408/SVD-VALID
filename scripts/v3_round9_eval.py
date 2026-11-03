#!/usr/bin/env python3
"""Round9 / Phase 1B 评价：在**线性 PD 真值域**对比 A / B 两臂与三条基线。

门槛与划分**全部来自** `reviews/V3_PR1_round9_prereg.md`，本脚本只负责算，不改门槛。

对比对象（同一场景、同一 ROI、同一显示标尺）：
    恒等（输入本身）
    固定高斯  sigma=0.60（校准选出，log1p 域）
    维纳      3D 立方窗 mysize=3
    A 臂      VALID-faithful random（--role-perm global）
    B 臂      固定 tet（pattern 5, role_order 0 1 3 2）
    oracle    频域维纳（用真值与第二次观测，**不可实现**，只作收益空间参照）

用法：
    python -u scripts/v3_round9_eval.py --runs runs/v3_round9/A runs/v3_round9/B \
        --split primary          # e902–e905
    python -u scripts/v3_round9_eval.py --split strict   # e906–e911
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener as scipy_wiener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = "/data2/wjb/anaconda3/envs/threePM/bin/python"
PATCH = [32, 64, 64]
OVERLAP = 0.1
GAUSS_SIGMA = 0.60          # 校准选出（reports/v3_round9/calibration.json）
OUT = ROOT / "reports/v3_round9"

SPLITS = {
    "primary": ("low6", ["e902", "e903", "e904", "e905"]),
    "strict": ("low_hold", ["e906", "e907", "e908", "e909", "e910", "e911"]),
    "calibration": ("low", ["e900", "e901"]),   # 只用于核对脚本自洽，不参与判定
}


# --------------------------------------------------------------------------- #
def load_scene(level, sid):
    p = ROOT / f"data/processed/v3_round2_s0/{level}/eval/s0_low_{sid}.npz"
    d = np.load(p)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s = float(d["s"])
    lg = lambda a: np.log1p(np.maximum(np.asarray(a, np.float32), 0.0) / max(s, 1e-12))
    obs, obs2, truth = lg(d["pd_linear"]), lg(d["pd2_linear"]), lg(d["gt_imaging_pd"])
    for a in (obs, obs2, truth):
        a[~valid] = 0.0
    return {
        "id": sid, "level": level, "s": s, "valid": valid, "rho": rho,
        "ves": d["vessels"].astype(bool),
        "obs": obs, "obs2": obs2, "truth": truth,
        "obs_pd": np.asarray(d["pd_linear"], np.float32),
        "truth_pd": np.asarray(d["gt_imaging_pd"], np.float32),
        "roi": valid & (rho > 0), "bg": valid & (rho <= 0),
        "sigma_eps": float(d["noise_sigma"]),
    }


def to_pd(y_log1p, s):
    return np.maximum(s * (np.expm1(np.asarray(y_log1p, np.float64))), 0.0).astype(np.float32)


def skeleton(ves):
    from skimage.morphology import skeletonize
    if ves.ndim == 3:
        try:
            from skimage.morphology import skeletonize_3d  # 旧版本
            return skeletonize_3d(ves)
        except Exception:
            return np.stack([skeletonize(f) for f in ves])
    return skeletonize(ves)


def masks(E):
    skel = skeleton(E["ves"])
    if not skel.any():
        return skel, skel
    q = np.quantile(E["rho"][skel], 0.33)
    return skel, skel & (E["rho"] <= q)


def core_metrics(pred_log1p, E, m_skel, m_weak):
    """所有指标：log1p 域 + 线性 PD 域 + 背景 + 分支/弱血管。"""
    v, roi, bg = E["valid"], E["roi"], E["bg"]
    pl = np.asarray(pred_log1p, np.float64)
    tl, ol = E["truth"].astype(np.float64), E["obs"].astype(np.float64)

    def rmse(a, b, m):
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")

    pd_p, pd_t = to_pd(pl, E["s"]), E["truth_pd"].astype(np.float64)
    peak = float(np.max(E["truth_pd"][v]))
    mse_pd = float(np.mean((pd_p[v] - pd_t[v]) ** 2))
    out = {
        "log1p_mse": float(np.mean((pl[v] - tl[v]) ** 2)),
        "log1p_rmse": rmse(pl, tl, v),
        "r_roi": float(np.corrcoef(pl[roi], tl[roi])[0, 1]) if roi.sum() > 2 else float("nan"),
        "pd_rmse": float(np.sqrt(mse_pd)),
        "pd_psnr": float(10 * np.log10(peak ** 2 / max(mse_pd, 1e-12))),
        "pd_peak": peak,
        "bg_mean": float(np.mean(pd_p[bg])) if bg.any() else float("nan"),
        "bg_std": float(np.std(pd_p[bg])) if bg.any() else float("nan"),
        "roi_mean": float(np.mean(pd_p[roi])) if roi.any() else float("nan"),
        "truth_roi_mean": float(np.mean(pd_t[roi])) if roi.any() else float("nan"),
        "branch_rmse": rmse(pd_p, pd_t, m_skel),
        "weak_rmse": rmse(pd_p, pd_t, m_weak),
        "weak_bias": (float(np.mean(pd_p[m_weak] - pd_t[m_weak]))
                      if m_weak.any() else float("nan")),
        "n_skel": int(m_skel.sum()), "n_weak": int(m_weak.sum()),
    }
    out["core_mean_ratio"] = (out["roi_mean"] / out["truth_roi_mean"]
                              if out["truth_roi_mean"] else float("nan"))
    return out


def run_infer(ckpt_dir: Path, npy: Path, gpu: int, tag: str) -> np.ndarray:
    dst = ROOT / "runs/_round9_infer" / f"{ckpt_dir.name}__{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "infer.py", "--volume", str(npy),
           "--checkpoint", str(ckpt_dir / "latest.pt"),
           "--patch", *[str(p) for p in PATCH], "--overlap", str(OVERLAP),
           "--device", "cuda", "--output", str(dst)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"infer.py 失败 {ckpt_dir}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return np.load(dst / "denoised.npy").astype(np.float32)


def oracle_freq_wiener(E):
    v = E["valid"].astype(np.float32)
    F = lambda a: np.fft.fftn(a.astype(np.float64))
    sxx = np.abs(F(E["truth"] * v)) ** 2
    snn = np.abs(F((E["obs"] - E["obs2"]) * v)) ** 2 / 2.0
    w = sxx / (sxx + snn + 1e-12)
    return np.real(np.fft.ifftn(F(E["obs"] * v) * w)).astype(np.float32)


def gaussian_curve(obs, truth, valid, sigmas):
    """高斯基线的强度曲线，用于回答"是否被平凡滤波完全支配"。"""
    out = []
    for sg in sigmas:
        sm = gaussian_filter(obs.astype(np.float64), sigma=float(sg), mode="nearest")
        mse = float(np.mean((sm[valid] - truth[valid]) ** 2))
        out.append({"sigma": float(sg), "log1p_mse": mse})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/v3_round9/A", "runs/v3_round9/B"])
    ap.add_argument("--split", default="primary", choices=list(SPLITS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    level, ids = SPLITS[args.split]
    tag = args.tag or args.split.replace("_", "")
    res = {"split": args.split, "level": level, "scenes": ids,
           "arms": [str(Path(r).name) for r in args.runs],
           "gauss_sigma": GAUSS_SIGMA, "patch": PATCH, "overlap": OVERLAP,
           "truth": "gt_imaging_pd（线性 PD 域主指标；log1p 域同时报）",
           "per_scene": {}, "timing": {}}
    tmp = ROOT / "runs/_round9_eval_npy"
    tmp.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for sid in ids:
        E = load_scene(level, sid)
        m_skel, m_weak = masks(E)
        npy = tmp / f"{level}_{sid}_log1p.npy"
        np.save(npy, E["obs"])

        preds = {
            "identity": E["obs"],
            f"fixed_gauss_s{GAUSS_SIGMA:.2f}":
                gaussian_filter(E["obs"].astype(np.float64), sigma=GAUSS_SIGMA,
                                mode="nearest").astype(np.float32),
            "fixed_wiener3d": scipy_wiener(E["obs"].astype(np.float64),
                                           mysize=3).astype(np.float32),
            "oracle_freq_wiener": oracle_freq_wiener(E),
        }
        for r in args.runs:
            cp = ROOT / r
            preds[Path(r).name] = run_infer(cp, npy, args.gpu, f"{level}_{sid}")

        per = {}
        for name, p in preds.items():
            p = np.asarray(p, np.float32).copy()
            p[~E["valid"]] = 0.0
            per[name] = core_metrics(p, E, m_skel, m_weak)
        per["_gaussian_curve"] = gaussian_curve(
            E["obs"], E["truth"], E["valid"], np.arange(0.10, 2.55, 0.05))
        res["per_scene"][sid] = per
        print(f"[{sid}] " + "  ".join(
            f"{k}={v['pd_psnr']:.2f}dB" for k, v in per.items() if not k.startswith("_")))

    res["timing"]["seconds"] = time.time() - t0

    # ---------- 聚合 + Gate ----------
    methods = list(res["per_scene"][ids[0]].keys())
    methods = [m for m in methods if not m.startswith("_")]
    agg = {}
    for m in methods:
        agg[m] = {k: float(np.nanmean([res["per_scene"][s][m][k] for s in ids]))
                  for k in ("pd_psnr", "pd_rmse", "log1p_mse", "r_roi",
                            "bg_mean", "bg_std", "branch_rmse", "weak_rmse",
                            "weak_bias", "core_mean_ratio")}
        # 逐场景胜负
        agg[m]["wins_vs_identity_psnr"] = int(sum(
            res["per_scene"][s][m]["pd_psnr"] > res["per_scene"][s]["identity"]["pd_psnr"]
            for s in ids))
        agg[m]["wins_vs_identity_weak"] = int(sum(
            res["per_scene"][s][m]["weak_rmse"] < res["per_scene"][s]["identity"]["weak_rmse"]
            for s in ids))
    res["aggregate"] = agg

    gates = {}
    for arm in [Path(r).name for r in args.runs]:
        g = agg[arm]
        idn = agg["identity"]
        gates[arm] = {
            "n_scenes": len(ids),
            "beats_identity_on_primary_error": g["wins_vs_identity_psnr"],
            "majority_beats_identity": bool(g["wins_vs_identity_psnr"] > len(ids) / 2),
            "weak_vessel_worse_than_identity_in": len(ids) - g["wins_vs_identity_weak"],
            "no_clear_weak_vessel_damage": bool(
                (len(ids) - g["wins_vs_identity_weak"]) < len(ids) / 2),
            "dominated_by_gaussian_or_wiener": bool(
                g["pd_psnr"] < max(agg[f"fixed_gauss_s{GAUSS_SIGMA:.2f}"]["pd_psnr"],
                                   agg["fixed_wiener3d"]["pd_psnr"])
                and g["branch_rmse"] > min(agg[f"fixed_gauss_s{GAUSS_SIGMA:.2f}"]["branch_rmse"],
                                           agg["fixed_wiener3d"]["branch_rmse"])),
            "delta_psnr_vs_identity": g["pd_psnr"] - idn["pd_psnr"],
            "delta_branch_rmse_vs_identity": g["branch_rmse"] - idn["branch_rmse"],
            "delta_weak_rmse_vs_identity": g["weak_rmse"] - idn["weak_rmse"],
        }
    res["gates"] = gates

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("\n%-22s %8s %8s %8s %8s %8s %9s" %
          ("method", "PSNR", "pdRMSE", "branch", "weak", "bg_mean", "core_ratio"))
    for m in methods:
        a = agg[m]
        print("%-22s %8.2f %8.4f %8.4f %8.4f %8.4f %9.3f" %
              (m, a["pd_psnr"], a["pd_rmse"], a["branch_rmse"], a["weak_rmse"],
               a["bg_mean"], a["core_mean_ratio"]))
    print("\ngates:", json.dumps(gates, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT / f'metrics_{tag}.json'}  ({res['timing']['seconds']:.1f}s)")


if __name__ == "__main__":
    main()

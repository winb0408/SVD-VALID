"""实验 B / R04 的评价：三种表示（linear / log1p / log(PD+ε)）在同一预算下的对照。

端点与门槛**先写**在 `reports/v3_expB/prereg.md`（写于任何 B 训练之前）。

跨表示比较的关键：**先反变换回线性 PD 域再算 n2n**
----------------------------------------------------
`n2n_ratio` 是"输出到第二观测的距离 / 输入到第二观测的距离"。如果三个臂各自在自己的表示域里算，
分母就不是同一个量：log 域的分母被压缩过，linear 域的分母由最亮的少数体素主导——两个比值**不可比**。
所以本脚本一律先反变换：

    linear : x -> x
    log1p  : x -> expm1(x)
    logeps : x -> exp(x) - eps

再在线性域（单位 = pd / s_glob，比值本身与 s_glob 无关）算：

    n2n_ratio_lin = mse(pred_lin, obs2_lin) / mse(obs1_lin, obs2_lin)     [valid]

主端点 = `n2n_ratio_lin`；辅端点 = `r_roi_log1p`（与实验 A 同口径，便于与 A 的数对照）、
`bg_std_ratio`；机制端点 = 条件残差（残差能量按输入电平十分位的分布 + 落在 ρ>0 的比例）。

用法
----
    python -u scripts/v3_expB_eval.py --runs runs/v3_expB_200k --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import ckpt_budget, infer, write_vol  # noqa: E402
from v3_expB_make_representations import REPS, inverse, transform  # noqa: E402
from v3_phase5_real_compare import wiener  # noqa: E402

REP_DIR = ROOT / "data/processed/v3_sim_v2/repB"
N_DECILE = 10
SPEARMAN_N = 20000


def scale() -> dict:
    return json.loads((REP_DIR / "scale.json").read_text())


def load_eval(npz: Path, s_glob: float) -> dict:
    """与实验 A 的 `load_eval` 同源，但额外保留线性域字段（B 的主端点在线性域算）。"""
    d = np.load(npz)
    pd = d["pd_linear"].astype(np.float64)
    pd2 = d["pd2_linear"].astype(np.float64)
    gt = d["gt_flow_pd"].astype(np.float64)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s_vol = float(d["s"])
    lin1, lin2, lin_gt = pd / s_glob, pd2 / s_glob, gt / s_glob
    E = {
        "pd": pd, "pd2": pd2, "gt": gt,
        "lin1": lin1, "lin2": lin2, "lin_gt": lin_gt,
        "s": s_vol, "s_glob": s_glob,
        "valid": valid, "rho": rho,
        "roi": valid & (rho > 0.0),
        "core": valid & (rho >= 0.5),
        "bg": valid & (rho <= 0.0),
        "npz": str(npz.relative_to(ROOT)),
        # 与 A 完全同口径的 log1p 域（逐体 s），只用于 r_roi_log1p 的可比性
        "truth_log1p": np.log1p(gt / s_vol),
        "vol_log1p": np.log1p(pd / s_vol),
    }
    E["ref_bg_std"] = float(np.std(E["vol_log1p"][E["bg"]]))
    return E


def rep_vol(E: dict, rep: str) -> np.ndarray:
    """该表示下网络的输入（无效区置 0，与 A 一致）。"""
    x = transform(E["pd"], E["s_glob"], rep).astype(np.float32)
    x[~E["valid"]] = 0.0
    return x


def metrics_lin(pred_lin: np.ndarray, E: dict) -> dict:
    """主端点：全部在线性域（pd/s_glob）算。"""
    v, roi, bg = E["valid"], E["roi"], E["bg"]
    o1, o2 = E["lin1"], E["lin2"]

    def mse(a, b, m):
        return float(np.mean((a[m] - b[m]) ** 2))

    den = mse(o1, o2, v)
    res = pred_lin - o2
    mass = res ** 2
    tot = float(mass[v].sum())
    return {
        "n2n_ratio_lin": mse(pred_lin, o2, v) / den if den else float("nan"),
        "n2n_ratio_lin_roi": (mse(pred_lin, o2, roi) / mse(o1, o2, roi)) if mse(o1, o2, roi) else float("nan"),
        "n2n_ratio_lin_bg": (mse(pred_lin, o2, bg) / mse(o1, o2, bg)) if mse(o1, o2, bg) else float("nan"),
        # 条件残差：训练到底把力气花在哪里
        "res_mass_frac_roi": float(mass[roi].sum() / tot) if tot else float("nan"),
        "voxel_frac_roi": float(roi.sum() / v.sum()),
        "res_mass_frac_core": float(mass[E["core"]].sum() / tot) if tot else float("nan"),
        "res_bias_roi": float(res[roi].mean()) if roi.any() else float("nan"),
        "res_bias_bg": float(res[bg].mean()) if bg.any() else float("nan"),
    }


def residuals_by_decile(pred_lin: np.ndarray, E: dict, seed_key: int = 0) -> dict:
    """残差能量按**输入电平**十分位的分布。若某个臂把力气全花在最暗的背景上，这里会直接看出来。"""
    v = E["valid"]
    x, res = E["lin1"][v], (pred_lin - E["lin2"])[v]
    qs = np.quantile(x, np.linspace(0, 1, N_DECILE + 1))
    qs[-1] += 1e-12
    idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, N_DECILE - 1)
    tot = float(np.sum(res ** 2))
    share, mean_res, lo_hi = [], [], []
    for k in range(N_DECILE):
        m = idx == k
        share.append(float(np.sum(res[m] ** 2) / tot) if tot else float("nan"))
        mean_res.append(float(res[m].mean()) if m.any() else float("nan"))
        lo_hi.append([float(qs[k]), float(qs[k + 1])])
    return {"share": share, "mean_res": mean_res, "bin_edges": lo_hi}


def spearman_abs_res_vs_input(pred_lin: np.ndarray, E: dict, rng: np.random.Generator) -> float:
    from scipy.stats import spearmanr
    v = E["valid"]
    x, r = E["lin1"][v], np.abs(pred_lin - E["lin2"])[v]
    n = x.size
    if n > SPEARMAN_N:
        sel = rng.choice(n, SPEARMAN_N, replace=False)
        x, r = x[sel], r[sel]
    rho = spearmanr(x, r).statistic
    return float(rho)


def r_roi_log1p(pred_lin: np.ndarray, E: dict) -> float:
    """与实验 A 同口径：把输出搬回 log1p(·/s_体) 再与 log1p 域真值求 ROI 内 Pearson。"""
    s = E["s"]
    pred = np.log1p(np.clip(pred_lin * E["s_glob"], 0.0, None) / s)
    roi, v = E["roi"], E["valid"]
    out = {
        "r_roi": float(np.corrcoef(pred[roi], E["truth_log1p"][roi])[0, 1]) if roi.sum() > 2 else float("nan"),
        "r_all": float(np.corrcoef(pred[v], E["truth_log1p"][v])[0, 1]),
        "bg_std_ratio": float(np.std(pred[E["bg"]])) / E["ref_bg_std"] if E["ref_bg_std"] else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/v3_expB_200k")
    ap.add_argument("--manifest", default="data/processed/v3_sim_v2/manifest.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/v3_expB/metrics.json")
    args = ap.parse_args()

    sc = scale()
    s_glob = float(sc["s_glob"])
    man = json.loads((ROOT / args.manifest).read_text())
    Es = [load_eval(ROOT / e["npz"], s_glob) for e in man["eval"]]
    rng = np.random.default_rng(0)
    print(f"评估体：{[Path(E['npz']).name for E in Es]}  s_glob={s_glob:.6g}")

    rec = {
        "meta": {
            "prereg": "reports/v3_expB/prereg.md（写于任何 B 训练之前）",
            "representation_scale": "s_glob（repB/scale.json），三表示共用",
            "eps": sc["eps"],
            "primary_endpoint": "n2n_ratio_lin（先反变换回线性域再算，跨表示可比）",
            "secondary": ["r_roi_log1p", "bg_std_ratio", "r_all"],
            "mechanism": ["res_mass_frac_roi", "residual_by_decile", "spearman_abs_res_vs_input"],
            "controls": ["input_linear", "zero", "wiener_w7", "oracle_log1p_lowpass"],
            "eval_volumes": [E["npz"] for E in Es],
            "seeds": args.seeds,
        },
        "controls": {}, "arms": {},
    }

    # ---------------- 对照（全部在线性域定义，再各自算端点） ----------------
    from scipy.ndimage import gaussian_filter
    from v3_simulator import SimCfg
    sig = SimCfg().psf_sigma
    controls = {
        "input_linear": lambda E: E["lin1"],
        "zero": lambda E: np.zeros_like(E["lin1"]),
        "wiener_w7": lambda E: inverse(wiener(E["vol_log1p"], 7, E["valid"])[0],
                                        E["s"], "log1p") * E["s"] / E["s_glob"],
        "oracle_lowpass_log1p": lambda E: inverse(
            gaussian_filter(E["vol_log1p"], sigma=sig, mode="nearest"), E["s"], "log1p") * E["s"] / E["s_glob"],
    }
    for tag, fn in controls.items():
        per = {}
        for E in Es:
            name = Path(E["npz"]).stem
            p = np.asarray(fn(E)).astype(np.float64)
            m = metrics_lin(p, E)
            m.update(r_roi_log1p(p, E))
            m["residual_by_decile"] = residuals_by_decile(p, E)
            m["spearman_abs_res_vs_input"] = spearman_abs_res_vs_input(p, E, rng)
            per[name] = m
        rec["controls"][tag] = per
        print(f"\n[{tag}] " + "  ".join(
            f"{k}: n2n_lin={m['n2n_ratio_lin']:.3f} r_roi={m['r_roi']:.4f}" for k, m in per.items()))

    # ---------------- 三个表示臂 ----------------
    for rep in REPS:
        for seed in args.seeds:
            ck = ROOT / args.runs / f"B_{rep}_s{seed}"
            tag = f"B_{rep}_s{seed}"
            if not (ck / "latest.pt").exists():
                print(f"[skip] {ck} 不存在")
                rec["arms"][tag] = {"status": "missing_checkpoint"}
                continue
            per = {}
            for E in Es:
                name = Path(E["npz"]).stem
                vp = write_vol(rep_vol(E, rep), ROOT / "runs" / "_infer_B" / "vols" / f"{name}_{rep}.npy")
                out = infer(ck, vp, args.gpu, f"{name}_{rep}")
                pred_lin = inverse(out, s_glob, rep)          # 回到线性域（单位 pd/s_glob）
                pred_lin[~E["valid"]] = E["lin1"][~E["valid"]]
                m = metrics_lin(pred_lin, E)
                m.update(r_roi_log1p(pred_lin, E))
                m["residual_by_decile"] = residuals_by_decile(pred_lin, E)
                m["spearman_abs_res_vs_input"] = spearman_abs_res_vs_input(pred_lin, E, rng)
                m["out_min_lin"] = float(pred_lin[E["valid"]].min())
                m["invert_nonfinite"] = int((~np.isfinite(pred_lin)).sum())
                per[name] = m
                print(f"  [{tag}] {name}: n2n_lin={m['n2n_ratio_lin']:.3f} r_roi={m['r_roi']:.4f} "
                      f"mass_roi={m['res_mass_frac_roi']:.3f} (vox_roi={m['voxel_frac_roi']:.3f})")
            rec["arms"][tag] = {"budget_checkpoints": ckpt_budget(ck / "latest.pt"), "per_volume": per}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}")
    print("门槛判读请对照 reports/v3_expB/prereg.md §3（不得事后调整）")


if __name__ == "__main__":
    main()

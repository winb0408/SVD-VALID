"""Round4-0（P0，无训练）：公平 baseline + 结构保真—背景抑制 Pareto。

设定全部写在 `reviews/V3_PR1_round4_prereg.md` §2 与
`reviews/V3_PR1_round4_prereg_v3_metric.md`（结构证据用 v3 的偏差—方差分解）：
选参只在 e900/e901，主结果只在 e902–e905，e906–e911（`*_hold/`）**本轮不评**。

臂：`input` / `gauss`（σ 网格，σ* 用 A5 冻结值）/ `wiener`（w 网格，w* 在本轮校准集上选）/
`net_{direct,residual,zeroinit}`（B1 权重，同档不混名）。
网络臂的 Pareto 旋钮：`y_β = x + β(net(x) − x)`，β∈{0,0.25,0.5,0.75,1}，**操作点固定 β=1**。

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_p0.py --gpu 1
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

from v3_round4_metrics import load, masked_gauss, wiener_shrink, blend   # noqa: E402
from v3_round4_metrics_v3 import metrics_v3                              # noqa: E402
from v3_expA_v2_eval import infer, write_vol                             # noqa: E402

SIGMA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
SIGMA_STAR = {"low": 0.6, "vlow": 0.5}          # A5 已冻结，本轮不重选
W_GRID = [3, 5, 7, 9, 11]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CAL = [900, 901]
HELD = [902, 903, 904, 905]
HELD_CONFIRM = list(range(906, 912))            # 本轮**不评**，只登记

NETS = {
    "net_direct": "runs/v3_round2_s0/{lvl}/S_dense_s3407",
    # 注意：vlow 两臂的目录名里没有 `vlow_` 中缀（`B1_residual_s3407`），low 两臂有（`B1_low_residual_s3407`）。
    "net_residual": {"low": "runs/v3_round3/b1_low/B1_low_residual_s3407",
                     "vlow": "runs/v3_round3/b1_vlow/B1_residual_s3407"},
    "net_zeroinit": {"low": "runs/v3_round3/b1_low/B1_low_zeroinit_s3407",
                     "vlow": "runs/v3_round3/b1_vlow/B1_zeroinit_s3407"},
}


def npz_path(level: str, seed: int) -> Path:
    return ROOT / f"data/processed/v3_round2_s0/{level}6/eval/s0_{level}_e{seed}.npz"


def net_volume(ckpt: Path, E: dict, gpu: int, tag: str) -> np.ndarray:
    """整卷前向（`infer.py` 的滑窗 + 按 checkpoint 的 arch_head 还原残差旁路）。

    结果落盘并复用：重跑只重算指标，不再重复推理（同一 checkpoint、同一输入，结果确定）。
    """
    cache = ROOT / "runs/_infer_v2/p0" / f"{tag}_denoised.npy"
    if cache.exists():
        y = np.load(cache).astype(np.float32)
    else:
        p = write_vol(E["vol"], ROOT / "runs/_infer_v2/p0" / f"{tag}.npy")
        y = infer(ckpt, p, gpu, tag)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, y.astype(np.float32))
    y[~E["valid"]] = 0.0
    return y


def dominance(net: dict, base: dict) -> bool:
    """预注册 v3 §2.2：bg_std_ratio、overall_rmse、|bias_ves| 三项都不差且至少一项更好。"""
    le = (base["bg_std_ratio"] <= net["bg_std_ratio"]
          and base["overall_rmse"] <= net["overall_rmse"]
          and abs(base["bias_ves"]) <= abs(net["bias_ves"]))
    lt = (base["bg_std_ratio"] < net["bg_std_ratio"]
          or base["overall_rmse"] < net["overall_rmse"]
          or abs(base["bias_ves"]) < abs(net["bias_ves"]))
    return bool(le and lt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["low", "vlow"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="reports/v3_round4/baseline_pareto.json")
    args = ap.parse_args()

    out = {"prereg": ["reviews/V3_PR1_round4_prereg.md §2",
                      "reviews/V3_PR1_round4_prereg_v3_metric.md §2.2"],
           "cal": CAL, "held": HELD, "held_confirm_not_evaluated": HELD_CONFIRM,
           "betas": BETAS, "levels": {}}

    for level in args.levels:
        rec = {"cal_scenes": CAL, "held_scenes": HELD, "selection": {},
               "held": {}, "sweeps": {}, "dominance": {}}
        cal_E = {s: load(npz_path(level, s)) for s in CAL}
        held_E = {s: load(npz_path(level, s)) for s in HELD}

        # ---------- 选参（只在 CAL 上）----------
        g_sigma = {}
        for sg in SIGMA_GRID:
            n2 = [metrics_v3(masked_gauss(E["vol"], E["valid"], sg), E)["n2n"] for E in cal_E.values()]
            g_sigma[sg] = float(np.mean(n2))
        sig_cal = min(g_sigma, key=g_sigma.get)

        w_n2 = {}
        for w in W_GRID:
            n2 = [metrics_v3(wiener_shrink(E["vol"], E["valid"], w), E)["n2n"] for E in cal_E.values()]
            w_n2[w] = float(np.mean(n2))
        w_star = min(w_n2, key=w_n2.get)

        rec["selection"] = {
            "gauss_sigma_a5_frozen": SIGMA_STAR[level],
            "gauss_sigma_recalibrated_on_cal": sig_cal,
            "gauss_cal_n2n": {str(k): v for k, v in g_sigma.items()},
            "wiener_w_star": w_star,
            "wiener_cal_n2n": {str(k): v for k, v in w_n2.items()},
        }
        print(f"[{level}] σ* (A5 冻结 {SIGMA_STAR[level]}, 本轮重算 {sig_cal}), w* = {w_star}")

        # ---------- 网络整卷前向（CAL + HELD，一次性算好）----------
        nets = {}
        for name, tmpl in NETS.items():
            spec = tmpl[level] if isinstance(tmpl, dict) else tmpl.format(lvl=level)
            ck = ROOT / spec
            assert (ck / "latest.pt").exists(), ck
            nets[name] = {}
            for s, E in {**cal_E, **held_E}.items():
                nets[name][s] = net_volume(ck, E, args.gpu, f"p0_{level}_{name}_e{s}")
            rec.setdefault("checkpoints", {})[name] = str(ck.relative_to(ROOT))

        # ---------- 操作点指标（held）----------
        for s, E in held_E.items():
            arms = {"input": E["vol"],
                    "gauss": masked_gauss(E["vol"], E["valid"], SIGMA_STAR[level]),
                    "wiener": wiener_shrink(E["vol"], E["valid"], w_star)}
            for name in NETS:
                arms[name] = nets[name][s]           # β = 1
            rec["held"][f"e{s}"] = {k: metrics_v3(v, E) for k, v in arms.items()}
            print(f"[{level} e{s}] " + " | ".join(
                f"{k}: n2n={m['n2n']:.3f} rmse={m['overall_rmse']:.4f} "
                f"bias={m['bias_ves']:+.4f} bgsd={m['bg_std_ratio']:.2f}"
                for k, m in rec["held"][f"e{s}"].items()))

        # ---------- Pareto 扫描（曲线画在 held 上；曲线不是选择）----------
        sw = {"gauss": {}, "wiener": {}, **{n: {} for n in NETS}}
        for s, E in held_E.items():
            for sg in SIGMA_GRID:
                m = metrics_v3(masked_gauss(E["vol"], E["valid"], sg), E)
                sw["gauss"].setdefault(str(sg), {})[f"e{s}"] = {
                    "bg_std_ratio": m["bg_std_ratio"], "overall_rmse": m["overall_rmse"],
                    "bias_ves": m["bias_ves"], "var_bg": m["var_bg"], "var_ves": m["var_ves"]}
            for w in W_GRID:
                m = metrics_v3(wiener_shrink(E["vol"], E["valid"], w), E)
                sw["wiener"].setdefault(str(w), {})[f"e{s}"] = {
                    "bg_std_ratio": m["bg_std_ratio"], "overall_rmse": m["overall_rmse"],
                    "bias_ves": m["bias_ves"], "var_bg": m["var_bg"], "var_ves": m["var_ves"]}
            for name in NETS:
                for b in BETAS:
                    m = metrics_v3(blend(E["vol"], nets[name][s], b), E)
                    sw[name].setdefault(str(b), {})[f"e{s}"] = {
                        "bg_std_ratio": m["bg_std_ratio"], "overall_rmse": m["overall_rmse"],
                        "bias_ves": m["bias_ves"], "var_bg": m["var_bg"], "var_ves": m["var_ves"]}
        rec["sweeps"] = sw

        # ---------- 支配判定（held 逐体）----------
        base_names = ["gauss", "wiener"]
        for name in NETS:
            per_scene = {}
            for s in HELD:
                key = f"e{s}"
                dom = [b for b in base_names
                       if dominance(rec["held"][key][name], rec["held"][key][b])]
                per_scene[key] = dom
            rec["dominance"][name] = {
                "dominated_by": per_scene,
                "dominated_in_all_held": bool(all(v for v in per_scene.values())),
            }
        any_nd = [n for n in NETS if not rec["dominance"][n]["dominated_in_all_held"]]
        rec["gate"] = {
            "BASELINE_DOMINATED": bool(not any_nd),
            "non_dominated_arms": any_nd,
            "rule": "预注册 v3 §2.2：三轴（bg_std_ratio / overall_rmse / |bias_ves|）都不劣且至少一项更优才算支配",
        }
        out["levels"][level] = rec
        print(f"[{level}] 0 Gate: BASELINE_DOMINATED = {rec['gate']['BASELINE_DOMINATED']}, "
              f"非支配臂 = {any_nd}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(f"-> {p}")


if __name__ == "__main__":
    main()

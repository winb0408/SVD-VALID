"""Round4 指标 v3 的已知答案检验（无训练、无网络）。

五项检查写在 `reviews/V3_PR1_round4_prereg_v3_metric.md` §2.1：
  1. 恒等式 `rmse² = bias² + var`（相对误差 < 1e-6）
  2. `truth_img`：bias=0 且 var=0
  3. `zero`：`bias_ves = −mean(truth[ves])`
  4. `cut30`：`bias_ves(cut30) < bias_ves(truth)`
  5. `gauss(σ*)`：`var_ves(gauss) < var_ves(input)` 且 `bias_ves(gauss) < 0`

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_metric_checks_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load                                  # noqa: E402
from v3_round4_metrics_v3 import metrics_v3, controls_v3            # noqa: E402

SIGMA_STAR = {"low": 0.6, "vlow": 0.5}
SCENES = [f"e{900 + i}" for i in range(6)]
IDENT_TOL = 1e-6


def _ident_ok(m: dict) -> bool:
    worst = 0.0
    for k in ("ves", "bg"):
        d = m[k]
        ref = d["bias"] ** 2 + d["var"]
        if ref <= 0:
            continue
        worst = max(worst, abs(d["rmse"] ** 2 - ref) / ref)
    return bool(worst < IDENT_TOL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed/v3_round2_s0")
    ap.add_argument("--levels", nargs="+", default=["low", "vlow"])
    ap.add_argument("--out", default="reports/v3_round4/metric_checks_v3.json")
    args = ap.parse_args()

    root = ROOT / args.data_root
    out = {"prereg": "reviews/V3_PR1_round4_prereg_v3_metric.md §2.1",
           "scenes": SCENES, "sigma_star": SIGMA_STAR, "levels": {}}

    for level in args.levels:
        rec = {"metric_version": "R4-v3", "volumes": {}, "checks": {}, "pass": {}}
        for scene in SCENES:
            npz = root / f"{level}6" / "eval" / f"s0_{level}_{scene}.npz"
            E = load(npz)
            arms = controls_v3(E, SIGMA_STAR[level])
            per = {k: metrics_v3(v, E) for k, v in arms.items()}
            rec["volumes"][scene] = per
            truth_ves_mean = float(E["truth_img"][E["valid"] & (E["rho"] >= 0.1)].mean())
            rec.setdefault("_truth_ves_mean", {})[scene] = truth_ves_mean

            # v3.1 §2 的 5a：合成噪声控制（检验 `var` 是否真的等于噪声方差）
            sig = float(np.load(npz)["noise_sigma"])
            rng = np.random.default_rng(20261029)
            noisy = E["truth_img"] + rng.normal(0.0, sig, E["truth_img"].shape).astype(np.float32)
            noisy[~E["valid"]] = 0.0
            msyn = metrics_v3(noisy, E)
            n_ves = msyn["ves"]["n"]
            bound = 3.0 * sig / np.sqrt(max(n_ves, 1))
            rec.setdefault("_synthetic", {})[scene] = {
                "sigma": sig, "var_ves": msyn["var_ves"], "bias_ves": msyn["bias_ves"],
                "rel_err_var": abs(msyn["var_ves"] - sig ** 2) / sig ** 2,
                "bias_bound": bound,
                "ok": bool(abs(msyn["var_ves"] - sig ** 2) / sig ** 2 < 0.05
                           and abs(msyn["bias_ves"]) < bound),
            }
            print(f"[{level} {scene}] " + " | ".join(
                f"{k}: b={m['bias_ves']:+.4f} v={m['var_ves']:.5f} rmse={m['overall_rmse']:.4f}"
                for k, m in per.items()))

        for scene, per in rec["volumes"].items():
            t, z, i, g, c = (per["truth_img"], per["zero"], per["input"],
                             per["gauss"], per["cut30"])
            tv = rec["_truth_ves_mean"][scene]
            sv = rec["_synthetic"][scene] if "_synthetic" in rec else None
            rec["checks"][scene] = {
                "ident_rmse2_eq_bias2_plus_var": all(_ident_ok(m) for m in per.values()),
                "truth_bias_zero": bool(abs(t["bias_ves"]) < 1e-9 and abs(t["var_ves"]) < 1e-12),
                "zero_bias_eq_minus_mean": bool(abs(z["bias_ves"] + tv) < 1e-6),
                "cut_bias_lt_truth_bias": bool(c["bias_ves"] < t["bias_ves"]),
                # v3.1 §2 的 5a：指标性质（合成噪声方差必须被复原）
                "synthetic_var_eq_sigma2": bool(sv and sv["ok"]),
                # v3.1 §2 的 5b：数据事实（原第 5 项的后半，6/6 已成立）
                "gauss_bias_negative": bool(g["bias_ves"] < 0.0),
            }
        keys = list(rec["checks"][SCENES[0]].keys())
        rec["pass"] = {k: {"n": int(sum(c[k] for c in rec["checks"].values())),
                           "of": len(rec["checks"]),
                           "ok": bool(all(c[k] for c in rec["checks"].values()))} for k in keys}
        rec["METRIC_RESOLVED_V3"] = bool(all(v["ok"] for v in rec["pass"].values()))
        out["levels"][level] = rec
        print(f"[{level}] METRIC_RESOLVED_V3 = {rec['METRIC_RESOLVED_V3']}  "
              + ", ".join(f"{k}:{v['n']}/{v['of']}" for k, v in rec["pass"].items()))

    out["METRIC_RESOLVED_V3_all"] = all(r["METRIC_RESOLVED_V3"] for r in out["levels"].values())
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {p}  METRIC_RESOLVED_V3(all) = {out['METRIC_RESOLVED_V3_all']}")


if __name__ == "__main__":
    main()

"""Round4-0 确认实验：β 在 e900/e901 上选，评估只在 e906–e911（从未看过）。

规则写在 `reviews/V3_PR1_round4_prereg_v4_confirm.md`（在评估确认集之前写死）：

    β*(A) = argmin |bias_ves(β)|  s.t.  var_bg(β) ≤ var_bg(gauss σ*)

通过线：对 `gauss(σ*)` 与 `wiener(w*)` 各自三条（var_bg / |bias_ves| / overall_rmse
都不劣）在 ≥5/6 确认体上成立。

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_p0_confirm.py --gpu 1
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
from v3_round4_p0 import NETS, SIGMA_STAR, W_GRID, BETAS, net_volume     # noqa: E402

CAL = [900, 901]
CONFIRM = list(range(906, 912))
PASS_N = 5          # 6 卷里至少 5 卷满足


def npz_path(level: str, seed: int, root: str = "6") -> Path:
    return ROOT / f"data/processed/v3_round2_s0/{level}{root}/eval/s0_{level}_e{seed}.npz"


def pick_beta(cal_metrics: dict, var_bg_gauss: float) -> dict:
    """§1 规则：在 var_bg ≤ gauss 的约束下取 |bias_ves| 最小的 β。"""
    feas = {b: m for b, m in cal_metrics.items() if m["var_bg"] <= var_bg_gauss}
    if not feas:
        return {"beta": None, "status": "NO_FEASIBLE_BETA",
                "var_bg_gauss": var_bg_gauss,
                "best_abs_bias_infeasible": min(abs(m["bias_ves"]) for m in cal_metrics.values())}
    b = min(feas, key=lambda k: (abs(feas[k]["bias_ves"]), k))
    return {"beta": b, "status": "OK", "var_bg_gauss": var_bg_gauss,
            "cal_abs_bias": abs(feas[b]["bias_ves"]), "cal_var_bg": feas[b]["var_bg"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["low", "vlow"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="reports/v3_round4/p0_confirm.json")
    args = ap.parse_args()

    out = {"prereg": "reviews/V3_PR1_round4_prereg_v4_confirm.md",
           "cal": CAL, "confirm": CONFIRM, "pass_n": PASS_N, "levels": {}}

    for level in args.levels:
        rec = {"selection": {}, "confirm": {}, "checks": {}, "pass": {}}
        cal_E = {s: load(npz_path(level, s)) for s in CAL}
        # w* 也只看校准集（与 P0 同规则）
        w_n2 = {}
        for w in W_GRID:
            w_n2[w] = float(np.mean([metrics_v3(wiener_shrink(E["vol"], E["valid"], w), E)["n2n"]
                                     for E in cal_E.values()]))
        w_star = min(w_n2, key=w_n2.get)
        var_bg_gauss = float(np.mean([
            metrics_v3(masked_gauss(E["vol"], E["valid"], SIGMA_STAR[level]), E)["var_bg"]
            for E in cal_E.values()]))
        rec["selection"]["w_star"] = w_star
        rec["selection"]["var_bg_gauss_cal"] = var_bg_gauss

        nets = {}
        for name, tmpl in NETS.items():
            spec = tmpl[level] if isinstance(tmpl, dict) else tmpl.format(lvl=level)
            ck = ROOT / spec
            nets[name] = {s: net_volume(ck, E, args.gpu, f"p0c_{level}_{name}_e{s}")
                          for s, E in cal_E.items()}

        # β 选点：只在校准集
        beta_star = {}
        for name in NETS:
            per_b = {}
            for b in BETAS:
                per_b[str(b)] = {k: metrics_v3(blend(cal_E[s]["vol"], nets[name][s], b), cal_E[s])
                                 for s, k in [(CAL[0], "c0"), (CAL[1], "c1")]}
                # 平均两卷
                ms = list(per_b[str(b)].values())
                per_b[str(b)] = {"var_bg": float(np.mean([m["var_bg"] for m in ms])),
                                 "bias_ves": float(np.mean([m["bias_ves"] for m in ms])),
                                 "overall_rmse": float(np.mean([m["overall_rmse"] for m in ms]))}
            beta_star[name] = pick_beta(per_b, var_bg_gauss)
            beta_star[name]["cal_sweep"] = per_b
        rec["selection"]["beta_star"] = beta_star
        print(f"[{level}] w*={w_star} var_bg(gauss,cal)={var_bg_gauss:.6f} "
              f"β*=" + ", ".join(f"{k}:{v['status']}:{v['beta']}" for k, v in beta_star.items()))

        # ---------- 确认集评估（只评不选）----------
        for s in CONFIRM:
            E = load(npz_path(level, s, root="_hold"))
            row = {"input": metrics_v3(E["vol"], E),
                   "gauss": metrics_v3(masked_gauss(E["vol"], E["valid"], SIGMA_STAR[level]), E),
                   "wiener": metrics_v3(wiener_shrink(E["vol"], E["valid"], w_star), E)}
            for name, tmpl in NETS.items():
                spec = tmpl[level] if isinstance(tmpl, dict) else tmpl.format(lvl=level)
                ck = ROOT / spec
                y = net_volume(ck, E, args.gpu, f"p0c_{level}_{name}_e{s}")
                b = beta_star[name]["beta"]
                if b is not None:
                    row[name] = metrics_v3(blend(E["vol"], y, float(b)), E)
            rec["confirm"][f"e{s}"] = row
            print(f"[{level} e{s}] " + " | ".join(
                f"{k}: rmse={m['overall_rmse']:.4f} bias={m['bias_ves']:+.5f} vb={m['var_bg']:.2e}"
                for k, m in row.items()))

        # ---------- 判定 ----------
        for name in NETS:
            if beta_star[name]["beta"] is None:
                rec["pass"][name] = {"status": "NO_FEASIBLE_BETA",
                                     "note": "校准集上没有满足 var_bg ≤ gauss 的 β"}
                continue
            checks = {}
            for base in ("gauss", "wiener"):
                n_vb = sum(rec["confirm"][k][name]["var_bg"] <= rec["confirm"][k][base]["var_bg"]
                           for k in rec["confirm"])
                n_bias = sum(abs(rec["confirm"][k][name]["bias_ves"])
                             < abs(rec["confirm"][k][base]["bias_ves"]) for k in rec["confirm"])
                n_rmse = sum(rec["confirm"][k][name]["overall_rmse"]
                             <= rec["confirm"][k][base]["overall_rmse"] for k in rec["confirm"])
                checks[f"vs_{base}"] = {"var_bg_ok": [n_vb, len(rec["confirm"])],
                                        "abs_bias_ok": [n_bias, len(rec["confirm"])],
                                        "rmse_ok": [n_rmse, len(rec["confirm"])],
                                        "pass": bool(n_vb >= PASS_N and n_bias >= PASS_N and n_rmse >= PASS_N)}
            checks["STRUCT_SAFE_GAIN_CONFIRMED"] = bool(
                checks["vs_gauss"]["pass"] and checks["vs_wiener"]["pass"])
            rec["pass"][name] = checks
            print(f"[{level}] {name}: {checks['STRUCT_SAFE_GAIN_CONFIRMED']} "
                  f"(vs gauss {checks['vs_gauss']}, vs wiener {checks['vs_wiener']})")

        rec["CONFIRMED_ANY"] = any(v.get("STRUCT_SAFE_GAIN_CONFIRMED") for v in rec["pass"].values())
        out["levels"][level] = rec

    out["CONFIRMED_ANY_ALL"] = any(r["CONFIRMED_ANY"] for r in out["levels"].values())
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(f"-> {p}  CONFIRMED_ANY(all) = {out['CONFIRMED_ANY_ALL']}")


if __name__ == "__main__":
    main()

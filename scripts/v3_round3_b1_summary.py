"""把 B1 三臂的 `a_probes_*.json` 汇总成一张表（不重跑任何推理）。

三臂（Round3 B1，审查 §3）：
    direct      : pred = net(x)                     旧基线，`runs/v3_round2_s0/vlow/S_dense_s3407`
    residual    : pred = x + net(x)                 `runs/v3_round3/b1_vlow/B1_residual_s3407`
    zeroinit    : pred = x + net(x)，net.out 置零    `runs/v3_round3/b1_vlow/B1_zeroinit_s3407`

输入（由 `scripts/v3_round3_a_probes.py` 逐臂产出）：
    reports/v3_round3/b1_direct_probe.json
    reports/v3_round3/b1_residual_probe.json
    reports/v3_round3/b1_zeroinit_probe.json

判读规则**写在跑之前**（`reviews/V3_PR1_round3_amended_plan.md` §4.2）：
    · 若 zeroinit 与 residual 的差距 < 场景间标准差，则收益来自**旁路**，不是初始化；
    · `G-B1`：6 场景 n2n 均值 ≤ 0.80，且至少 3/6 场景 r_roi 高于输入臂同一读数，才算通过
      （r_roi 是 Pearson，恒 ≤1；"r_roi > 1" 是早期定义笔误，见 plan §4.2 更正说明）；
    · 三臂都打不过 identity ⇒ 交失败证据，不追加复杂度。

用法：
    python -u scripts/v3_round3_b1_summary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v3_round3"
ARMS = ["direct", "residual", "zeroinit"]


def load(arm: str, tag: str) -> dict | None:
    p = OUT / f"b1_{arm}{tag}_probe.json"
    if not p.exists():
        print(f"[missing] {p}")
        return None
    return json.loads(p.read_text())


def ci95(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = sps.t.ppf(0.975, len(x) - 1)
    return (float(x.mean() - t * se), float(x.mean() + t * se))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="探针文件名后缀，如 _low")
    ap.add_argument("--gate", type=float, default=0.80)
    args = ap.parse_args()
    tag = args.tag

    rec, rows = {}, {}
    for arm in ARMS:
        d = load(arm, tag)
        if d is None:
            continue
        rec[arm] = {"checkpoint": d["meta"]["checkpoint"], "volumes": d["meta"]["volumes"]}
        per = {}
        for vol, a2 in d["A2"].items():
            a4 = d["A4"][vol]
            mm = d["metrics"][vol]
            full = a2["arms"]["trained"]["rows"][-1]          # α=1 ⇒ 纯模型输出
            per[vol] = {"n2n_model": full["n2n"], "r2": full["r2"],
                        "n2n_best_blend": a2["arms"]["trained"]["min_n2n"],
                        "alpha_best": a2["arms"]["trained"]["argmin_alpha"],
                        "bias": a4["valid"]["bias"], "mse": a4["valid"]["mse"],
                        "err_std": a4["valid"]["err_std"],
                        "r_roi": mm["model"]["r_roi"], "r_roi_input": mm["input"]["r_roi"],
                        "r_roi_gain": mm["model"]["r_roi"] - mm["input"]["r_roi"],
                        "bg_std_ratio": mm["model"]["bg_std_ratio"],
                        "low_frac": a4["bands"]["low"]["energy_frac"] / 2.0}
        rec[arm]["per_volume"] = per
        n2 = np.array([v["n2n_model"] for v in per.values()])
        r2 = np.array([v["r2"] for v in per.values()])
        blend = np.array([v["n2n_best_blend"] for v in per.values()])
        gain = np.array([v["r_roi_gain"] for v in per.values()])
        n_up = int((gain > 0).sum())
        rows[arm] = {
            "n2n_model_mean": float(n2.mean()), "n2n_model_min": float(n2.min()),
            "n2n_model_max": float(n2.max()),
            "r2_mean": float(r2.mean()), "r2_median": float(np.median(r2)),
            "n2n_best_blend_mean": float(blend.mean()), "alpha_mean": float(np.mean(
                [v["alpha_best"] for v in per.values()])),
            "n2n_model_ci95": ci95(n2),
            "n_roi_up": n_up, "n_volumes": len(per),
            "r_roi_gain_mean": float(gain.mean()),
            # 两条读数都留档：plan §4.2 登记的是"至少 1/6"（早期两卷版）以外的 3/6
            # 自我加严版（b_minimal_repair.md 规则 1）。两者不一致时**不挑赢家**，
            # 并列报出并把该臂标成"判据解读待裁"。
            "gate_n2n": bool(n2.mean() <= args.gate),
            "gate_r_roi_any": bool(n_up >= 1),
            "gate_r_roi_3of6": bool(n_up >= max(3, (len(per) + 1) // 2)),
        }
        rows[arm]["G_B1"] = ("PASS" if rows[arm]["gate_n2n"] and rows[arm]["gate_r_roi_any"]
                             and rows[arm]["gate_r_roi_3of6"] else
                             "AMBIGUOUS" if rows[arm]["gate_n2n"] and rows[arm]["gate_r_roi_any"]
                             else "fail")
    rec["summary"] = rows

    print(f"{'arm':10s} {'n2n(model)':>11s} {'min':>7s} {'max':>7s} {'r²':>7s} "
          f"{'best blend':>11s} {'r_roi↑':>7s} {'G-B1':>9s}")
    for arm, r in rows.items():
        print(f"{arm:10s} {r['n2n_model_mean']:11.3f} {r['n2n_model_min']:7.3f} "
              f"{r['n2n_model_max']:7.3f} {r['r2_mean']:7.3f} "
              f"{r['n2n_best_blend_mean']:11.3f} "
              f"{r['n_roi_up']:3d}/{r['n_volumes']:<3d} {r['G_B1']:>9s}")
    print("（r_roi↑ = `r_roi` 高于输入臂的场景数；G-B1 需 n2n 均值过门 + 3/6 场景 ROI 相关上升）")

    if {"residual", "zeroinit"} <= set(rows):
        # 逐场景配对比较：初始化带来的额外收益 vs 场景间波动
        vol = sorted(rec["residual"]["per_volume"])
        d = np.array([rec["zeroinit"]["per_volume"][v]["n2n_model"]
                      - rec["residual"]["per_volume"][v]["n2n_model"] for v in vol])
        sd = np.array([rec["residual"]["per_volume"][v]["n2n_model"] for v in vol]).std(ddof=1)
        rec["zeroinit_vs_residual"] = {
            "per_volume_diff": dict(zip(vol, d.tolist())),
            "mean_diff": float(d.mean()), "ci95": ci95(d),
            "scene_sd": float(sd),
            "verdict": ("初始化有独立贡献" if abs(d.mean()) > sd / 2 else
                        "初始化贡献小于场景间波动 ⇒ 收益主要来自旁路"),
        }
        z = rec["zeroinit_vs_residual"]
        print(f"\nzeroinit − residual: mean={z['mean_diff']:+.4f} "
              f"CI95=[{z['ci95'][0]:+.4f},{z['ci95'][1]:+.4f}]  场景 sd={z['scene_sd']:.3f}")
        print(f"判读：{z['verdict']}")

    out = OUT / f"b1_summary{tag}.json"
    (out).write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=float) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

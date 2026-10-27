"""R06/v2 的判据再标定：新真值下"哪些判据还退化"的**已知答案校准**。

方案 §0.5 要求任何新判据在开跑前必须有：①旧判据的证伪记录（R06 报告 §2.1 已有）；
②**已知答案校准**（本脚本）；③门槛先写（`reports/v3_expA_v2/prereg.md`）。

本脚本对每个 v2 体回答四件事，全部只用已知答案（不需要训练任何网络）：

  Q1 真值还退化吗？—— `gt_flow_pd` 在 valid 区里为 0 的比例、血管核心/半影/背景三档的
     真值中位数、连续密度 ρ∈(0,1) 的体素占比（v1 的对应值是 0.9903 / 0.0000）。
  Q2 全零输出还会赢吗？—— 对 v2 真值分别算全局 RMSE 与"核心区 RMSE"，
     比较 输入 / 全零 / 完美输出。这是 R06 §2.1 那两张表的 v2 版本。
  Q3 第二次观测还够不够"独立"？—— n2n 分母 `mse(obs1,obs2,valid)` 相对
     `var(log1p pd)` 的比例；分母越小，n2n 越容易被整体电平偏移污染。
  Q4 空间纹理对上了吗？—— 复用 `v3_simulator.acf_vs_target`，把 v2 背景 log-ACF 与
     SA64 实测逐轴比较（v1 的数字是 y 轴过相关 0.20，见 R06 报告 §3）。

用法
----
    python scripts/v3_r06_v2_audit.py

产物
----
    reports/v3_r06/v2_endpoint_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_simulator import acf_vs_target, background_mask, load_target  # noqa: E402


def audit_one(npz: Path, target: dict, mode: str) -> dict:
    d = np.load(npz)
    pd = d["pd_linear"].astype(np.float64)
    gt = d["gt_flow_pd"].astype(np.float64)
    rho = d["rho"].astype(np.float64) if "rho" in d.files else d["vessels"].astype(np.float64)
    valid = d["valid"].astype(bool)
    s = float(d["s"]) if "s" in d.files else float(np.median(pd[valid]))

    l1 = np.log1p(pd / s)
    lg = np.log1p(gt / s)
    core = valid & (rho >= 0.5)                      # 主 ROI：ρ≥0.5（核心 0.9 只有 10¹ 体素，不稳定）
    pen = valid & (rho > 0.0) & (rho < 0.5)          # 半影：PSF 展宽出来的那一圈
    deep = valid & (rho >= 0.9)                      # 深核：只用于看"最亮的血管"是否也非退化
    bgm = valid & (rho <= 0.0)

    out = {
        "npz": str(npz.relative_to(ROOT)),
        "s": s,
        # ---- Q1 真值退化程度 ----
        "q1": {
            "gt_zero_frac_valid": float(np.mean(gt[valid] <= 0.0)),
            "gt_zero_frac_all": float(np.mean(gt <= 0.0)),
            "frac_rho_partial": float(np.mean((rho > 0.0) & (rho < 1.0))),
            "n_core": int(core.sum()), "n_penumbra": int(pen.sum()), "n_bg": int(bgm.sum()),
            "n_deep": int(deep.sum()),
            "gt_log_median_deep": float(np.median(lg[deep])) if deep.any() else None,
            "gt_log_median_core": float(np.median(lg[core])) if core.any() else None,
            "gt_log_median_penumbra": float(np.median(lg[pen])) if pen.any() else None,
            "gt_log_median_bg": float(np.median(lg[bgm])) if bgm.any() else None,
            "obs_log_median_core": float(np.median(l1[core])) if core.any() else None,
            "obs_log_median_bg": float(np.median(l1[bgm])) if bgm.any() else None,
            "contrast_db_core_over_bg": float(
                10 * np.log10(max(np.median(gt[core]), 1e-30) / max(np.median(pd[bgm]), 1e-30))
            ) if core.any() and bgm.any() else None,
        },
        # ---- Q2 全零是否仍然最优（RMSE 类判据是否退化）----
        "q2_rmse": {
            "global_input": float(np.sqrt(np.mean((l1[valid] - lg[valid]) ** 2))),
            "global_zero": float(np.sqrt(np.mean(lg[valid] ** 2))),
            "global_perfect": 0.0,
        },
    }
    # 分区 RMSE：只在 core / penumbra 上算
    if core.any():
        out["q2_rmse"]["core_input"] = float(np.sqrt(np.mean((l1[core] - lg[core]) ** 2)))
        out["q2_rmse"]["core_zero"] = float(np.sqrt(np.mean(lg[core] ** 2)))
        out["q2_rmse"]["core_perfect"] = 0.0
    # 半影区（"血管变宽"这一维度真正可测的地方）
    if pen.any():
        out["q2_rmse"]["pen_input"] = float(np.sqrt(np.mean((l1[pen] - lg[pen]) ** 2)))
        out["q2_rmse"]["pen_zero"] = float(np.sqrt(np.mean(lg[pen] ** 2)))

    # ---- Q3 第二次观测的独立性（n2n 分母）----
    if "pd2_linear" in d.files:
        pd2 = d["pd2_linear"].astype(np.float64)
        l2 = np.log1p(pd2 / s)
        num = float(np.mean((l1[valid] - l2[valid]) ** 2))
        den = float(np.var(0.5 * (l1[valid] + l2[valid])))
        vf = float(np.var(l1[valid]))
        out["q3_n2n"] = {
            "denominator_mse_obs1_obs2": num,
            "var_log_pd": vf,
            "den_over_var": num / max(vf, 1e-30),
            "note": "den/var 越小 ⇒ n2n 的提升空间越窄、越易被电平偏移污染",
        }
        out["q3_n2n"]["noise_frac_2obs"] = num / max(2 * den, 1e-30)

    # ---- Q4 背景 log-ACF vs SA64 实测 ----
    e = acf_vs_target(pd.astype(np.float32), valid, target, mode=mode)
    out["q4_acf"] = {"overall_rmse": e["overall_rmse"],
                     "per_axis": {a: {"sim_lag1": e[a]["sim"][0], "real_lag1": e[a]["target"][0],
                                      "diff": e[a]["sim"][0] - e[a]["target"][0]}
                                  for a in ("z", "y", "x")}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="data/processed/v3_sim_v2/eval")
    ap.add_argument("--extra", nargs="*", default=None,
                    help="额外要审计的 npz（例如 v1 的 sim_seed0.npz 作对照）")
    ap.add_argument("--target-mode", default="fast", choices=["fast", "raw"])
    ap.add_argument("--seq", default="SA64")
    ap.add_argument("--out", default="reports/v3_r06/v2_endpoint_audit.json")
    args = ap.parse_args()

    target = load_target(args.target_mode)[args.seq]
    files = sorted((ROOT / args.eval_dir).glob("*.npz"))
    if args.extra:
        files = [ROOT / p for p in args.extra] + files
    if not files:
        raise SystemExit(f"没有 npz：{args.eval_dir}")

    res = {}
    for f in files:
        r = audit_one(f, target, args.target_mode)
        res[f.stem] = r
        q1, q2 = r["q1"], r["q2_rmse"]
        print(f"\n=== {f.stem} ===")
        print(f"  Q1 gt 为 0 的比例(valid)={q1['gt_zero_frac_valid']:.4f}  "
              f"ρ∈(0,1)={q1['frac_rho_partial']:.4f}  "
              f"核心/半影/背景体素={q1['n_core']}/{q1['n_penumbra']}/{q1['n_bg']}")
        fmt = lambda x: "—" if x is None else f"{x:.3f}"      # noqa: E731
        print(f"     真值 log 中位数 核心={fmt(q1['gt_log_median_core'])} "
              f"半影={fmt(q1['gt_log_median_penumbra'])} 背景={fmt(q1['gt_log_median_bg'])}"
              f"  带通后对比度={fmt(q1['contrast_db_core_over_bg'])} dB")
        print(f"  Q2 全局 RMSE 输入={q2['global_input']:.4f} 全零={q2['global_zero']:.4f} "
              f"完美={q2['global_perfect']:.4f}")
        if "core_input" in q2:
            print(f"     核心 RMSE 输入={q2['core_input']:.4f} 全零={q2['core_zero']:.4f}")
        if "pen_input" in q2:
            print(f"     半影 RMSE 输入={q2['pen_input']:.4f} 全零={q2['pen_zero']:.4f}")
        if "q3_n2n" in r:
            print(f"  Q3 n2n 分母={r['q3_n2n']['denominator_mse_obs1_obs2']:.4e} "
                  f"占 var={r['q3_n2n']['den_over_var']:.4f}")
        print(f"  Q4 ACF rmse={r['q4_acf']['overall_rmse']:.4f} "
              + " ".join(f"{a}:{r['q4_acf']['per_axis'][a]['diff']:+.3f}"
                         for a in ("z", "y", "x")))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

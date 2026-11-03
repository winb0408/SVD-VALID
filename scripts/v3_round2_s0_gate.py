"""S0 门的机械化判定（G-S0-1/2/3）。门槛照抄 `reviews/V3_PR1_round2_plan.md` §3.2，本脚本不改门槛。

为什么另起一道门（不改旧 T1/T2）
--------------------------------
第二轮审查 0.3：旧 T2 的冻结是**资源管理规则**，不是逻辑证明。旧判定留在
`reports/v3_expA_v2/gate_verdict.json`，此处是**新版本**，两者互不覆盖、互不蕴含。

判据（字面量，写在训练结果之前）
--------------------------------
G-S0-1  至少一档噪声上，两臂 `n2n_ratio < 1.0`
G-S0-2  同档上两臂 `r_roi > 输入 r_roi`（对 `gt_imaging_pd`）
G-S0-3  评分正反标定：除已记录为"端点定义问题"的 `r_roi:fake` 外，全部顺序关系成立
归因    若 G-S0-1/2 失败且 zero 仍退化（n2n 大于输入且不是 nan）→ 归因实现/目标/评价链

用法：
    python -u scripts/v3_round2_s0_gate.py --level low
产物：
    reports/v3_round2/gate_s0_<level>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- 门槛（字面量，抄自 reviews/V3_PR1_round2_plan.md §3.2，写于 2026-10-27 17:0x，训练启动之前）
THR = {"n2n_lt": 1.0, "r_roi_gt_input": True}
# 已记录为"端点定义问题"、不计入 G-S0-3 的检查（理由见 s0_calibration 脚本 CHECKS 的注释）
EXCLUDED_CHECKS = {("r_roi", "fake")}


def _cells(j: dict, arm: str) -> dict:
    return {vol: m for vol, m in j["arms"][arm]["per_volume"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mp = Path(args.metrics or f"reports/v3_round2/s0_{args.level}_metrics.json")
    cp = Path(args.calibration or f"reports/v3_round2/s0_calibration_{args.level}.json")
    rec: dict = {"level": args.level,
                 "prereg": "reviews/V3_PR1_round2_plan.md §3.2（写在训练启动之前）",
                 "thresholds_literal": THR,
                 "inputs": {"metrics": str(mp), "calibration": str(cp)},
                 "predates": {}}
    for p in (mp, cp):
        if not (ROOT / p).exists():
            rec.setdefault("missing", []).append(str(p))
    if rec.get("missing"):
        (ROOT / (args.out or f"reports/v3_round2/gate_s0_{args.level}.json")).write_text(
            json.dumps(rec, indent=1, ensure_ascii=False))
        print("缺输入，未判定：", rec["missing"])
        return

    m = json.loads((ROOT / mp).read_text())
    c = json.loads((ROOT / cp).read_text())

    # 每条臂的逐格 n2n / r_roi
    arms = {}
    for arm in m["arms"]:
        arms[arm] = {}
        for vol, cell in _cells(m, arm).items():
            inp = m["controls"]["input"][vol]
            arms[arm][vol] = {"n2n_ratio": cell["n2n_ratio"], "input_n2n": inp["n2n_ratio"],
                              "r_roi": cell["r_roi"], "input_r_roi": inp["r_roi"],
                              "pass_n2n": bool(cell["n2n_ratio"] < THR["n2n_lt"]),
                              "pass_r_roi": bool(cell["r_roi"] > inp["r_roi"])}
    rec["arms"] = arms

    def arm_mean(arm, key):
        vals = [v[key] for v in arms[arm].values() if not math.isnan(v[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    g1 = {arm: {"mean_n2n": arm_mean(arm, "n2n_ratio"),
                "all_cells_pass": all(v["pass_n2n"] for v in arms[arm].values())}
          for arm in arms}
    g2 = {arm: {"mean_r_roi": arm_mean(arm, "r_roi"),
                "all_cells_pass": all(v["pass_r_roi"] for v in arms[arm].values())}
          for arm in arms}

    rec["G_S0_1_n2n"] = {"per_arm": g1,
                         "pass_strict": all(v["all_cells_pass"] for v in g1.values()),
                         "pass_arm_mean": all(v["mean_n2n"] < THR["n2n_lt"] for v in g1.values())}
    rec["G_S0_2_r_roi"] = {"per_arm": g2,
                           "pass_strict": all(v["all_cells_pass"] for v in g2.values()),
                           "pass_arm_mean": all(
                               g2[a]["mean_r_roi"] > arm_mean_input(m) for a in g2)}

    checks = [ch for ch in c["checks"] if (ch["endpoint"], ch["worse"]) not in EXCLUDED_CHECKS]
    rec["G_S0_3_calibration"] = {
        "n_checks": len(checks), "n_failed": sum(1 for ch in checks if not ch["all_ok"]),
        "failed": [f"{ch['endpoint']}:{ch['better']}>{ch['worse']}"
                   for ch in checks if not ch["all_ok"]],
        "excluded": [f"{a}:{b}" for a, b in sorted(EXCLUDED_CHECKS)],
        "pass": all(ch["all_ok"] for ch in checks),
    }

    # zero 是否仍退化（归因规则的操作化）
    zeros = m["controls"]["zero"]
    rec["zero_control"] = {
        vol: {"n2n": z["n2n_ratio"],
              "still_degenerate": bool(z["n2n_ratio"] > m["controls"]["input"][vol]["n2n_ratio"])}
        for vol, z in zeros.items()}
    rec["zero_still_degenerate"] = all(v["still_degenerate"] for v in rec["zero_control"].values())

    passed = (rec["G_S0_1_n2n"]["pass_strict"] and rec["G_S0_2_r_roi"]["pass_strict"]
              and rec["G_S0_3_calibration"]["pass"])
    rec["verdict"] = "S0-通过" if passed else "S0-未通过"
    if not passed:
        rec["attribution"] = ("判据未退化（zero 仍远差于输入）→ 失败归因于实现/训练目标/评价链，"
                              "不得升级为 I1 科学证伪（审查 §7）"
                              if rec["zero_still_degenerate"] else
                              "zero 对照已退化 → 判据本身可疑，先修判据，不得归因于方法")
    out = ROOT / (args.out or f"reports/v3_round2/gate_s0_{args.level}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("verdict", "attribution") if k in rec}, ensure_ascii=False))
    print(f"wrote {out}")


def arm_mean_input(m):
    vals = [v["r_roi"] for v in m["controls"]["input"].values()]
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()

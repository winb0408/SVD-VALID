#!/usr/bin/env python3
"""Round10 判据：把预注册 §4 的规则**机械地**套到 `scores.json` 上，不掺判断。

规则（`reviews/V3_PR1_round10_prereg_real_train.md` §4，跑之前写死）：

    Δ = X_RMSE(本方法最佳臂) − X_RMSE(gauss 最佳)          （都在 α=1 的原始输出上）

    OURS-BETTER     两个数据集 × 两个方向**全部** Δ < −0.005
    BASELINE-BETTER 两个数据集 × 两个方向**全部** Δ > +0.005
    INCONCLUSIVE    其余

同一规则顺带判 vs wiener3d 与 vs sn2n 两栏。
「本方法最佳臂」= 在方向 1 上选出的那个臂（selections.json 的 `arms` 里 X 最小的那个），
方向 2 用同一配置的重训结果。

产物
----
    reports/v3_round10/judge.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v3_round10"
THRESH = 0.005
SEQS = ("SA64", "HSA64")
DIRS = ("odd2even", "even2odd")


def verdict(deltas: dict[str, float]) -> str:
    vals = list(deltas.values())
    if all(v < -THRESH for v in vals):
        return "OURS-BETTER"
    if all(v > THRESH for v in vals):
        return "BASELINE-BETTER"
    return "INCONCLUSIVE"


def main() -> int:
    scores = json.loads((OUT / "scores.json").read_text())
    sel = json.loads((OUT / "selections.json").read_text())

    table = {}
    for seq in SEQS:
        arm = min(sel[seq]["arms"], key=lambda a: sel[seq]["arms"][a]["x"])
        b = sel[seq]["arms"][arm]
        table[seq] = {"best_arm": arm, "config": {"domain": b["domain"], "epochs": b["epochs"]},
                      "alpha_used": b["alpha"], "compare": {}}
        for d in DIRS:
            run_name = b["run"] if d == "odd2even" else b["run"].replace("odd2even", "even2odd")
            r = scores["runs"].get(run_name)
            ref = scores["references"][f"{seq}_{d}"]
            if r is None:
                table[seq]["compare"][d] = {"missing": run_name}
                continue
            g = ref["gauss"]["grid"][ref["gauss"]["best_by_x"]]["x"]
            row = {"run": run_name, "x": r["x"], "x_in": r["x_in"], "x_ratio": r["x"] / r["x_in"],
                   "x_gain": r["x_gain"], "contrast": r["contrast"], "overfit": r["overfit"],
                   "gauss": g, "wiener3d": ref["wiener3d"]["x"],
                   "sn2n": ref.get("sn2n", {}).get("x"),
                   "delta_vs_gauss": r["x"] - g,
                   # 键名必须与下面 verdict 里 f"delta_vs_{opp}" 的拼接结果逐字一致。
                   # 2026-11-03 修：原来存成 "delta_vs_wiener"，而 opp 是 "wiener3d"，
                   # 查不到就落进 INCOMPLETE——是**字段名**的 bug，不是缺数据，阈值与判据未动。
                   "delta_vs_wiener3d": r["x"] - ref["wiener3d"]["x"],
                   "delta_vs_sn2n": (None if "sn2n" not in ref
                                     else r["x"] - ref["sn2n"]["x"])}
            table[seq]["compare"][d] = row

    # 判定：每个对手一栏
    out = {"threshold": THRESH, "table": table, "verdicts": {}}
    for opp in ("gauss", "wiener3d", "sn2n"):
        deltas = {}
        for seq in SEQS:
            for d in DIRS:
                row = table[seq]["compare"][d]
                if row.get("missing") or row.get(f"delta_vs_{opp}") is None:
                    deltas[f"{seq}_{d}"] = None
                    continue
                deltas[f"{seq}_{d}"] = row[f"delta_vs_{opp}"]
        if any(v is None for v in deltas.values()):
            out["verdicts"][f"vs_{opp}"] = {"verdict": "INCOMPLETE", "deltas": deltas}
        else:
            out["verdicts"][f"vs_{opp}"] = {"verdict": verdict(deltas), "deltas": deltas}

    (OUT / "judge.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out["verdicts"].items():
        print(f"{k}: {v['verdict']}")
        for kk, vv in v["deltas"].items():
            print(f"    {kk}: Δ={vv if vv is None else round(vv, 5)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Round4 结构指标 v2 的已知答案控制（无训练、无网络）。

判定写在 `reviews/V3_PR1_round4_prereg_v2_metric.md` §2.2，**先写死再跑**：

| 控制 | 期望 | 通过线 |
|---|---|---|
| 正 1 | 真值优于输入（中位段 CNR） | 6/6 卷 |
| 正 2 | 同上（最弱 10% 段） | 6/6 卷 |
| 正 3 | 真值不低于高斯（中位段 CNR） | 6/6 卷 |
| 负 A | `cut30` 的段召回 ≤0.75 且低于真值 | 6/6 卷 |
| 负 B | `fake` 的假段数高于真值 | 6/6 卷 |
| 正 4 | 全零图的中位段 CNR = 0 | 6/6 卷 |

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_metric_controls_v2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load                    # noqa: E402
from v3_round4_metrics_v2 import metrics_v2, controls_v2, segments   # noqa: E402

SIGMA_STAR = {"low": 0.6, "vlow": 0.5}
SCENES = [f"e{900 + i}" for i in range(6)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed/v3_round2_s0")
    ap.add_argument("--levels", nargs="+", default=["low", "vlow"])
    ap.add_argument("--out", default="reports/v3_round4/metric_controls_v2.json")
    args = ap.parse_args()

    root = ROOT / args.data_root
    out = {"prereg": "reviews/V3_PR1_round4_prereg_v2_metric.md §2.2",
           "scenes": SCENES, "sigma_star": SIGMA_STAR, "levels": {}}

    for level in args.levels:
        rec = {"metric_version": "R4-v2", "volumes": {}, "checks": {}, "pass": {}}
        for scene in SCENES:
            E = load(root / f"{level}6" / "eval" / f"s0_{level}_{scene}.npz")
            seg = segments(E)
            arms = controls_v2(E, SIGMA_STAR[level])
            per = {}
            for k, v in arms.items():
                m = metrics_v2(v, E)
                m.pop("seg_snr", None)                       # 每段一个数，太长，另存都会被截断
                per[k] = m
            rec["volumes"][scene] = per
            print(f"[{level} {scene}] N_seg={seg['n']} " + " | ".join(
                f"{k}: cnr={m['seg_cnr_median']:.2f} p10={m['seg_cnr_p10']:.2f} "
                f"rec2={m['seg_recall_z2']:.2f} fp={m['fp_segments']}"
                for k, m in per.items()))

        for scene, per in rec["volumes"].items():
            t, i, g, z, c, f = (per["truth_img"], per["input"], per["gauss"],
                                per["zero"], per["cut30"], per["fake"])
            rec["checks"][scene] = {
                "pos1_cnr_truth_gt_input": bool(t["seg_cnr_median"] > i["seg_cnr_median"]),
                "pos2_p10_truth_gt_input": bool(t["seg_cnr_p10"] > i["seg_cnr_p10"]),
                "pos3_cnr_truth_ge_gauss": bool(t["seg_cnr_median"] >= g["seg_cnr_median"]),
                "negA_cut_recall_le_075": bool(c["seg_recall_z2"] <= 0.75),
                "negA_cut_recall_lt_truth": bool(c["seg_recall_z2"] < t["seg_recall_z2"]),
                "negB_fake_fp_gt_truth": bool(f["fp_segments"] > t["fp_segments"]),
                "pos4_zero_cnr_is_zero": bool(abs(z["seg_cnr_median"]) < 1e-9),
            }
        keys = list(rec["checks"][SCENES[0]].keys())
        rec["pass"] = {k: {"n": int(sum(c[k] for c in rec["checks"].values())),
                           "of": len(rec["checks"]),
                           "ok": bool(all(c[k] for c in rec["checks"].values()))}
                       for k in keys}
        rec["METRIC_RESOLVED_V2"] = bool(all(v["ok"] for v in rec["pass"].values()))
        out["levels"][level] = rec
        print(f"[{level}] METRIC_RESOLVED_V2 = {rec['METRIC_RESOLVED_V2']}  "
              + ", ".join(f"{k}:{v['n']}/{v['of']}" for k, v in rec["pass"].items()))

    out["METRIC_RESOLVED_V2_all"] = all(r["METRIC_RESOLVED_V2"] for r in out["levels"].values())
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {p}  METRIC_RESOLVED_V2(all) = {out['METRIC_RESOLVED_V2_all']}")


if __name__ == "__main__":
    main()

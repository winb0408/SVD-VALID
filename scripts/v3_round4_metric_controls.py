"""Round4-0 前置闸门：结构指标的已知答案正负控制（无训练、无网络）。

判定写在 `reviews/V3_PR1_round4_prereg.md` §2.3，**在跑之前写死**：

| 控制 | 期望 | 通过线 |
|---|---|---|
| 正控制 1 | 无噪成像真值优于含噪输入 | `recall(truth_img) > recall(input)`，6/6 卷 |
| 正控制 2 | 同上（误差轴） | `rmse_img(truth_img) < rmse_img(input)`，6/6 卷 |
| 正控制 3 | 连通性单调 | `|ncomp_ratio(truth)−1| ≤ |ncomp_ratio(input)−1|`，6/6 卷 |
| 负控制 A | 断一枝必须被检出 | `recall(cut) < recall(truth)` 且 `ncomp_ratio(cut) < ncomp_ratio(truth)`，6/6 卷 |
| 负控制 B | 造一条假血管必须被计入假阳性 | `fp_blobs(fake) > fp_blobs(truth)`，6/6 卷 |

任一不满足 → `METRIC_UNRESOLVED`，此后只允许报 `n2n` / `rmse_img` / `bg_std_ratio`，
不得用结构指标宣称"结构安全"或"结构受损"。

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_metric_controls.py --out reports/v3_round4/metric_controls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load, metrics, known_answers   # noqa: E402

# A5 已冻结的 σ*（低/高噪声档）。本轮**不重选**，直接复用（预注册 §2.1）。
SIGMA_STAR = {"low": 0.6, "vlow": 0.5}
SCENES = [f"e{900 + i}" for i in range(6)]                  # e900–e905：校准 2 + 留出 4


def npz_for(root: Path, level: str, scene: str) -> Path:
    """`low6`/`vlow6` 目录里存的是 `eval/s0_<level>_<scene>.npz`。"""
    return root / f"{level}6" / "eval" / f"s0_{level}_{scene}.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed/v3_round2_s0")
    ap.add_argument("--levels", nargs="+", default=["low", "vlow"])
    ap.add_argument("--out", default="reports/v3_round4/metric_controls.json")
    args = ap.parse_args()

    root = ROOT / args.data_root
    out = {"prereg": "reviews/V3_PR1_round4_prereg.md §2.3",
           "scenes": SCENES, "sigma_star": SIGMA_STAR, "levels": {}}

    for level in args.levels:
        src = root
        rec = {"metric_version": "R4-v1", "volumes": {}, "checks": {}, "pass": {}}
        for scene in SCENES:
            E = load(npz_for(src, level, scene))
            arms = known_answers(E, SIGMA_STAR[level])
            rec["volumes"][scene] = {k: metrics(v, E) for k, v in arms.items()}
            print(f"[{level} {scene}] " + " | ".join(
                f"{k}: rec={m['recall_at_fpr5']:.3f} rmse={m['rmse_img']:.4f} "
                f"fp={m['fp_blobs']} ncr={m['ncomp_ratio']:.2f}"
                for k, m in rec["volumes"][scene].items()))

        for scene, per in rec["volumes"].items():
            t, i, c, f = per["truth_img"], per["input"], per["cut"], per["fake"]
            rec["checks"][scene] = {
                "pos1_recall_truth_gt_input": bool(t["recall_at_fpr5"] > i["recall_at_fpr5"]),
                "pos2_rmse_truth_lt_input": bool(t["rmse_img"] < i["rmse_img"]),
                "pos3_ncomp_truth_closer_to_1": bool(
                    abs(t["ncomp_ratio"] - 1.0) <= abs(i["ncomp_ratio"] - 1.0)),
                "negA_cut_recall_lt_truth": bool(c["recall_at_fpr5"] < t["recall_at_fpr5"]),
                "negA_cut_ncomp_lt_truth": bool(c["ncomp_ratio"] < t["ncomp_ratio"]),
                "negB_fake_fp_gt_truth": bool(f["fp_blobs"] > t["fp_blobs"]),
            }

        keys = list(rec["checks"][SCENES[0]].keys())
        rec["pass"] = {k: {"n": int(sum(c[k] for c in rec["checks"].values())),
                           "of": len(rec["checks"]), "ok": bool(all(c[k] for c in rec["checks"].values()))}
                       for k in keys}
        rec["METRIC_RESOLVED"] = bool(all(v["ok"] for v in rec["pass"].values()))
        out["levels"][level] = rec
        print(f"[{level}] METRIC_RESOLVED = {rec['METRIC_RESOLVED']}  "
              + ", ".join(f"{k}:{v['n']}/{v['of']}" for k, v in rec["pass"].items()))

    out["METRIC_RESOLVED_all"] = all(r["METRIC_RESOLVED"] for r in out["levels"].values())
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"-> {p}  METRIC_RESOLVED(all) = {out['METRIC_RESOLVED_all']}")


if __name__ == "__main__":
    main()

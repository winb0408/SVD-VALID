"""把 `v3_judge_real.py` 的两份结果画成一张对比图（读取 JSON，不重算）。

三列对应三道门，两行对应两个数据集。每根柱子按"这一道门是否通过"上色，
门槛用竖虚线标出。**图只呈现已标定的判据，不引入任何新判据。**
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from v3_phase5_real_compare import setup_cjk   # noqa: E402

GOOD, BAD = "#2e8b57", "#c0392b"


def main():
    plt = setup_cjk()
    dsets = ["SA64", "HSA64"]
    recs = {}
    for d in dsets:
        p = ROOT / f"reports/v3_judge/{d}_judge.json"
        recs[d] = json.loads(p.read_text())

    cols = [("n_rel", "N_rel（噪声，n2n）", recs[dsets[0]]["thr"]["n_rel"],
             "N 门槛 0.5", False),
            ("b_db", "B_db（暗区假阳性 / 血管能量）", recs[dsets[0]]["thr"]["b_db"],
             "B 门槛 −7.3 dB", False),
            ("s_rel", "S_rel（结构保留）", recs[dsets[0]]["thr"]["s_rel"],
             "S 门槛 0.95", True)]

    fig, axes = plt.subplots(2, 3, figsize=(21, 9.5))
    for ri, d in enumerate(dsets):
        rows = recs[d]["rows"]
        names = list(rows)
        short = [n.replace("②单纯 SVD（不处理）", "② 单纯 SVD（不处理）")
                 .replace("④网络 ", "④ 网络 ") for n in names]
        for ci, (field, label, thr, thrlab, higher_better) in enumerate(cols):
            ax = axes[ri, ci]
            vals = [rows[n][field] for n in names]
            okkey = {"n_rel": "ok_n", "b_db": "ok_b", "s_rel": "ok_s"}[field]
            colors = [GOOD if rows[n][okkey] else BAD for n in names]
            ypos = np.arange(len(names))
            ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.62)
            ax.set_yticks(ypos)
            ax.set_yticklabels(short if ci == 0 else [""] * len(short), fontsize=10)
            ax.invert_yaxis()
            ax.axvline(thr, color="k", ls="--", lw=1.5)
            ax.text(thr, -0.9, thrlab, fontsize=9.5, ha="center",
                    bbox=dict(fc="white", ec="none", alpha=0.8))
            ax.set_title(f"{d} — {label}", fontsize=12)
            ax.grid(axis="x", alpha=0.3)
            if field == "n_rel":
                ax.set_xlim(0, max(vals) * 1.12)
            for y, v in zip(ypos, vals):
                ax.text(v, y, f" {v:.3f}" if field == "s_rel" else f" {v:.2f}",
                        va="center", fontsize=9)
            if field == "n_rel":
                ax.set_xlabel("越低越好；不过门 = 噪声没去干净", fontsize=9.5)
            elif field == "b_db":
                ax.set_xlabel("越低越好；不过门 = 往背景添了东西", fontsize=9.5)
            else:
                ax.set_xlabel("越高越好；不过门 = 结构被削掉", fontsize=9.5)

    n_pass = {d: sum(r["pass"] for r in recs[d]["rows"].values()) for d in dsets}
    fig.suptitle(
        f"偏置感知判据 {recs['SA64']['judge_version']} 在真实数据上的结果（绿=该门通过，红=不过）"
        f"｜{dsets[0]}：{n_pass[dsets[0]]} 条通过；{dsets[1]}：{n_pass[dsets[1]]} 条通过"
        f"｜三道门全过才是'可接受'", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = ROOT / "reports/figures/v3_judge_real.png"
    fig.savefig(out, dpi=140)
    print(f"图 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

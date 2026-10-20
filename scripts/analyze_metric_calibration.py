#!/usr/bin/env python3
"""Aggregate the overlap sweep into the calibration the audit asked for.

Reads `metric_calibration_<seq>.json` and answers four things:

  1. do the two anchors hold -- overlap=0 reproduces the audit, overlap=1 gives
     exactly 0;
  2. what does the gate `rel_matched <= -0.10` actually test, expressed as the
     fraction of frames the two arms may share before it stops passing;
  3. is `rel_matched` a function of the true arm-to-arm noise correlation, or of
     something else;
  4. what is the best-case reading, and therefore whether the gate can be failed
     by a design that is wrong.

Writes a markdown summary next to the JSON plus one figure.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / max(np.sqrt((rx * rx).sum() * (ry * ry).sum()), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/phase2_calibration_2026-10-19")
    ap.add_argument("--gate", type=float, default=-0.10)
    args = ap.parse_args()

    root = Path(args.dir)
    data = json.loads((root / "metric_calibration.json").read_text())

    lines = ["# 残差 proxy 的标定：这把尺子到底量的是什么", ""]
    lines += ["数据：`metric_calibration_*.json`（overlap 扫描，真实折、真实秩规则、"
              "真实匹配算子）。", ""]

    summary = {}
    for seq, seq_out in data["sequences"].items():
        cells = seq_out["cells"]
        lines += [f"## {seq}", ""]

        # ---- anchor 1: overlap=1 must be exactly 0 -----------------------
        one = [c for c in cells if c["overlap"] == 1.0]
        worst = max(abs(c["rel_matched"]) for c in one) if one else None
        lines += [f"- **锚点 1（overlap=1 必须精确为 0）**：{len(one)} 个格点，"
                  f"最大 |rel_matched| = {worst:.2e}"
                  + (" ✅" if worst is not None and worst < 1e-9 else " ❌"), ""]

        # ---- anchor 2: overlap=0 vs the audit ----------------------------
        zero = [c for c in cells if c["overlap"] == 0.0]
        lines += ["- **锚点 2（overlap=0 复现审计）**：按 (variant, rule) 汇总的折均值 "
                  "rel_matched ——", ""]
        lines += ["  | variant | rule | 折数 | 均值 | 中位 | 最小 | 最大 |",
                  "  |---|---|---|---|---|---|---|"]
        agg0 = {}
        for v in sorted({c["variant"] for c in zero}):
            for r in sorted({c["rule"] for c in zero}):
                v_ = [c["rel_matched"] for c in zero if c["variant"] == v and c["rule"] == r]
                if not v_:
                    continue
                agg0[(v, r)] = float(np.mean(v_))
                lines += [f"  | {v} | {r} | {len(v_)} | {np.mean(v_):+.3f} | "
                          f"{np.median(v_):+.3f} | {np.min(v_):+.3f} | {np.max(v_):+.3f} |"]
        lines += [""]

        # ---- the curve ---------------------------------------------------
        lines += ["### rel_matched 随臂重叠的走势（折均值）", "",
                  "| variant | rule | split | N_A | " +
                  " | ".join(f"ov={o:g}" for o in sorted({c["overlap"] for c in cells})) + " |",
                  "|---|---|---|---|" + "---|" * len({c["overlap"] for c in cells})]
        overlaps = sorted({c["overlap"] for c in cells})
        curves = {}
        for v in sorted({c["variant"] for c in cells}):
            for r in sorted({c["rule"] for c in cells}):
                for sp in sorted({c["split"] for c in cells}):
                    sel = [c for c in cells if c["variant"] == v and c["rule"] == r
                           and c["split"] == sp]
                    if not sel:
                        continue
                    prof = []
                    for o in overlaps:
                        vals = [c["rel_matched"] for c in sel if c["overlap"] == o]
                        prof.append(float(np.mean(vals)) if vals else float("nan"))
                    curves[(v, r, sp)] = prof
                    n_a = sorted({c["n_a"] for c in sel if c["overlap"] == 0.0})
                    lines += [f"| {v} | {r} | {sp} | {n_a[0] if n_a else '-'} | " +
                              " | ".join(f"{x:+.3f}" for x in prof) + " |"]
        lines += [""]

        # ---- what the gate tests ----------------------------------------
        lines += [f"### 门槛 `rel <= {args.gate:.2f}` 实际在测什么", "",
                  "最大重叠比例仍能过门槛者（即门槛能容忍两臂共享多少帧）：", ""]
        lines += ["| variant | rule | split | 通过的 overlap 上界 | 说明 |", "|---|---|---|---|---|"]
        pass_max = {}
        for (v, r, sp), prof in curves.items():
            ok = [o for o, x in zip(overlaps, prof) if x <= args.gate]
            hi = max(ok) if ok else None
            pass_max[(v, r, sp)] = hi
            lines += [f"| {v} | {r} | {sp} | "
                      f"{('<= ' + format(hi, 'g')) if hi is not None else '无'} | "
                      f"{'几乎任何非完全相同的一对都能过' if hi is not None and hi >= 0.5 else ''} |"]
        lines += [""]

        # ---- is it a function of the true noise correlation? -------------
        rel = [c["rel_matched"] for c in cells]
        rho = [c["rho_eff"] for c in cells]
        mapr = [c["power_map_r"] for c in cells]
        lines += ["### rel_matched 到底跟着什么走", "",
                  f"- 与真实臂间噪声相关 rho_eff 的 Spearman：**{spearman(rho, rel):+.3f}**",
                  f"- 与两臂功率图相似度 power_map_r 的 Spearman：**{spearman(mapr, rel):+.3f}**",
                  f"- 与 rho_eff 的 Pearson：{np.corrcoef(rho, rel)[0, 1]:+.3f}；"
                  f"与 (1 - power_map_r) 的 Pearson："
                  f"{np.corrcoef(1.0 - np.asarray(mapr), rel)[0, 1]:+.3f}", ""]

        # within-design monotonicity in overlap
        mono = []
        for (v, r, sp), prof in curves.items():
            d = np.diff(prof)
            mono.append(int((d >= -1e-12).sum()))
        lines += [f"- 按 (variant, rule, split) 分组看，随 overlap 单调不降的组："
                  f"{sum(1 for m, prof in zip(mono, curves.values()) if m == len(prof) - 1)}"
                  f"/{len(mono)}（说明在**固定设计**内，尺子对'两臂共享多少帧'是单调的）", ""]

        summary[seq] = {
            "anchor_overlap1_max_abs": worst,
            "overlap0": {f"{k[0]}|{k[1]}": v for k, v in agg0.items()},
            "pass_max_overlap": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in pass_max.items()},
            "spearman_rho_eff": spearman(rho, rel),
            "spearman_power_map_r": spearman(mapr, rel),
            "best_case_rel": float(min(rel)),
        }

    (root / "calibration_summary.md").write_text("\n".join(lines) + "\n")
    (root / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {root / 'calibration_summary.md'}")

    make_figure(data, root, args.gate)


def make_figure(data, root, gate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    colors = {"sa64_int3": "C0", "sa64_int5": "C1", "sa64_cont3": "C2",
              "hsa64_int3": "C3"}

    ax = axes[0]
    for seq, seq_out in data["sequences"].items():
        cells = [c for c in seq_out["cells"] if c["variant"] == "f0_free"
                 and c["rule"] == "proportional"]
        for sp in sorted({c["split"] for c in cells}):
            ovs = sorted({c["overlap"] for c in cells})
            prof = [np.mean([c["rel_matched"] for c in cells
                             if c["split"] == sp and c["overlap"] == o]) for o in ovs]
            ax.plot(ovs, prof, "o-", color=colors.get(sp, "C7"), label=sp)
    ax.axhline(gate, color="k", ls="--", lw=0.8, label=f"gate {gate:g}")
    ax.set_xlabel("fraction of frames shared by the two arms")
    ax.set_ylabel("rel_matched")
    ax.set_title("the same metric, swept by arm overlap")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    for ax, key, xlabel in ((axes[1], "power_map_r", "correlation of the two arm power maps"),
                            (axes[2], "rho_eff", "true arm-to-arm noise correlation")):
        for seq, seq_out in data["sequences"].items():
            cells = [c for c in seq_out["cells"] if c["variant"] == "f0_free"]
            ax.scatter([c[key] for c in cells], [c["rel_matched"] for c in cells],
                       s=12, alpha=0.6, label=seq)
        ax.axhline(gate, color="k", ls="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rel_matched")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = root / "metric_calibration.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

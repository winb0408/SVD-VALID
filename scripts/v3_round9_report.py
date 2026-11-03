#!/usr/bin/env python3
"""Round9 / Phase 1B 对比图：同一标尺、同一 ROI、每行一个场景。

列：无噪成像真值 | 输入 | 固定高斯 σ0.60 | 维纳 3D | A(VALID-faithful) | B(固定 tet)
- 显示域：线性 PD 的 dB，`10*log10(pd / peak_ref)`，`peak_ref` = **该场景真值的 p99**
  （由真值定，与任何方法无关；**同一行所有列共用一个标尺**，绝不逐列拉伸）。
- 裁剪：`vessels ∪ (rho>0)` 的包围盒外扩 4 体素，同一行所有列同一裁剪。
- 每个场景取血管最多的那一帧（Z），所有列同一帧。

用法：python -u scripts/v3_round9_report.py [--split primary]
输出：reports/v3_round9/figs/*.png + reports/v3_round9/figures.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener as scipy_wiener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round9_eval import (GAUSS_SIGMA, SPLITS, load_scene, oracle_freq_wiener,
                            to_pd)  # noqa: E402

OUT = ROOT / "reports/v3_round9"
INF = ROOT / "runs/_round9_infer"
VMIN, VMAX = -30.0, 0.0


def preds_for(E, sid, level, arms):
    """取每列要画的预测（log1p 域）。网络列直接从 infer.py 的落盘结果读。"""
    d = {
        "truth": E["truth"],
        "input": E["obs"],
        f"gauss{GAUSS_SIGMA:.2f}": gaussian_filter(
            E["obs"].astype(np.float64), sigma=GAUSS_SIGMA, mode="nearest"),
        "wiener3d": scipy_wiener(E["obs"].astype(np.float64), mysize=3),
        "oracle": oracle_freq_wiener(E),
    }
    for arm in arms:
        p = INF / f"{arm}__{level}_{sid}" / "denoised.npy"
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}（先跑 scripts/v3_round9_eval.py）")
        d[arm] = np.load(p).astype(np.float32)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="primary", choices=list(SPLITS))
    ap.add_argument("--arms", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    level, ids = SPLITS[args.split]
    OUT.joinpath("figs").mkdir(parents=True, exist_ok=True)

    cols = [("truth", "truth (noise-free)"), ("input", "input"),
            (f"gauss{GAUSS_SIGMA:.2f}", f"fixed Gaussian σ{GAUSS_SIGMA:.2f}"),
            ("wiener3d", "Wiener 3D")]
    cols += [(a, f"{a}: VALID-faithful random" if a == "A" else f"{a}: fixed tet")
             for a in args.arms]
    cols += [("oracle", "oracle freq. Wiener\n(unrealisable reference)")]

    rows = []
    for sid in ids:
        E = load_scene(level, sid)
        P = preds_for(E, sid, level, args.arms)
        s = E["s"]
        peak_ref = float(np.percentile(E["truth_pd"][E["valid"]], 99))

        def to_db(y):
            pd = to_pd(y, s)
            return 10.0 * np.log10(np.maximum(pd, 1e-12) / max(peak_ref, 1e-12))

        # 血管最多的一帧（所有列同一帧）
        mir = E["ves"] | E["roi"]              # (z, y, x)
        z = int(np.argmax(mir.reshape(mir.shape[0], -1).sum(axis=1)))
        # 同一行所有列共用的裁剪：按 (y, x) 各自的投影取包围盒，外扩 4 体素
        y_idx = np.where(mir.any(axis=(0, 2)))[0]
        x_idx = np.where(mir.any(axis=(0, 1)))[0]
        y0, y1 = max(int(y_idx.min()) - 4, 0), min(int(y_idx.max()) + 5, mir.shape[1])
        x0, x1 = max(int(x_idx.min()) - 4, 0), min(int(x_idx.max()) + 5, mir.shape[2])

        fig, axes = plt.subplots(1, len(cols), figsize=(2.5 * len(cols), 3.1))
        for ax, (k, title) in zip(axes, cols):
            img = to_db(P[k])[z, y0:y1, x0:x1]
            im = ax.imshow(img, vmin=VMIN, vmax=VMAX, cmap="gray", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"S0 positive control · scene {sid} · z={z} · "
                     f"shared dB ref (truth p99 = {peak_ref:.3f}) · ROI crop "
                     f"[{y0}:{y1}, {x0}:{x1}]", fontsize=8)
        cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
        cb.set_label("dB re truth p99", fontsize=7)
        fp = OUT / "figs" / f"s0_{sid}_compare.png"
        fig.savefig(fp, dpi=170, bbox_inches="tight")
        plt.close(fig)
        rows.append({"scene": sid, "z": z, "peak_ref": peak_ref,
                     "crop": [int(y0), int(y1), int(x0), int(x1)],
                     "figure": str(fp.relative_to(ROOT))})
        print("wrote", fp)

    lines = ["# Round9 对比图索引\n",
             f"split `{args.split}`，场景 {', '.join(ids)}。\n",
             "显示：线性 PD 的 dB（`10log10(pd / peak_ref)`），**每行一个标尺**，"
             "标尺由该场景**真值的 p99** 定（与任何方法无关，不逐列拉伸）。"
             "裁剪与帧号对同一行所有列相同。\n",
             "| 场景 | 帧 z | 真值 p99（dB 参考） | 裁剪 [y0:y1, x0:x1] | 图 |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['z']} | {r['peak_ref']:.3f} | "
                     f"{r['crop']} | `{r['figure']}` |")
    (OUT / "figures.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "figures.md")


if __name__ == "__main__":
    main()

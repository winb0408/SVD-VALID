#!/usr/bin/env python3
"""Round8 产物：固定显示窗口的并排定性图 + 汇总表（Markdown）。

为什么另起一个脚本：`v3_round8_zero_train_full_infer.py`（审查方产物）只画了 7 种网络输出，
不含止损判定必需的固定 Gaussian/Wiener 对照。这里把对照一起放进**同一个 dB 窗口**，
否则"棋盘没了但背景更吵"这件事在图上看不出来。

显示约定（与审查要求一致，且不按列单独调）：
  * 统一参考：同一 SVD 输入锥内最大线性 PD（`refmax`，写在 analysis.json 里）
  * 统一窗口：vmin = -25 dB, vmax = 0 dB，所有列、所有行、两张序列相同
  * 两行：z 向 MIP / 中间 z 帧
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

D8 = ROOT / "data/processed/v3_round8_zero_train"
R8 = ROOT / "reports/v3_round8_zero_train"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())["logeps"]

COLUMNS = [
    ("svd_input", "SVD input"),
    ("fixed_gauss_s0.5", "Gauss σ0.5"),
    ("fixed_gauss_s1", "Gauss σ1.0"),
    ("fixed_wiener2d", "Wiener 2D"),
    ("legacy_8phase", "8-phase backfill (A)"),
    ("full_P0", "full P0 (B)"),
    ("full_P1", "full P1 (B)"),
    ("full_P2", "full P2 (B)"),
    ("full_P3", "full P3 (B)"),
    ("full_ensemble", "mean P0–P3 (C)"),
]
VMIN, VMAX = -25.0, 0.0


def to_pd(y, s):
    from v3_round4_b1_prep import inv
    return np.maximum(inv("logeps", y * NORM["std"] + NORM["mean"], s), 0.0)


def fig_one(seq, std, s, refmax, mask, tag="fixed_reference_grid",
            row_window_from_input=False):
    """并排图。`row_window_from_input=True` 时窗口按行取 SVD 输入的 p1/p99，
    同一行所有列共用 —— 仍然不逐列、不按输出调，只是把动态范围放对位置以便看清差异。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if row_window_from_input:
        dbin = 10 * np.log10(np.maximum(to_pd(std["svd_input"], s), 1e-12) / refmax)
        w = (float(np.percentile(dbin[mask], 1)), float(np.percentile(dbin[mask], 99)))
    else:
        w = (VMIN, VMAX)
    fig, ax = plt.subplots(2, len(COLUMNS), figsize=(2.4 * len(COLUMNS), 5.4))
    for i, (key, label) in enumerate(COLUMNS):
        db = 10 * np.log10(np.maximum(to_pd(std[key], s), 1e-12) / refmax)
        ax[0, i].imshow(db.max(axis=0), cmap="hot", vmin=w[0], vmax=w[1])
        ax[1, i].imshow(db[db.shape[0] // 2], cmap="hot", vmin=w[0], vmax=w[1])
        ax[0, i].set_title(label, fontsize=10)
    ax[0, 0].set_ylabel("X (z-MIP)", fontsize=9)
    ax[1, 0].set_ylabel("X (mid z)", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{seq}  Round8 zero-training inference — common dB reference, "
                 f"window {w[0]:.1f}…{w[1]:.1f} dB "
                 f"({'from SVD input p1/p99' if row_window_from_input else 'fixed -25…0'}); "
                 f"no per-column rescaling", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = R8 / f"{seq}_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def md_table(a, cols):
    hdr = "| method | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    rows = []
    for name, v in a["methods"].items():
        cells = [c[1](v) for c in cols]
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep, *rows])


def main():
    R8.mkdir(parents=True, exist_ok=True)
    out = ["# Round8 零训练推理：指标汇总", "",
           "所有数值由 `scripts/v3_round8_zero_train_analysis.py` 生成；"
           "`phase_bias` / `seam_y` 的定义直接 import 自审查方脚本，未改写。", ""]
    for seq in ("SA64", "HSA64"):
        a = json.loads((R8 / f"{seq}_analysis.json").read_text())
        meta = json.loads((ROOT / f"data/processed/v3_round6/{seq}/metadata.json").read_text())
        s = float(meta["s"])
        std = {f.name.replace("_logeps_standardized.npy", ""):
               np.load(f).astype(np.float64)
               for f in sorted(D8.glob(f"{seq}/*_logeps_standardized.npy"))}
        # 固定对照在 analysis 里只存了指标，图上要用像素：就地从输入重算一遍（同一固定参数）
        from v3_round8_zero_train_analysis import (gauss_blur, to_std, wiener2d)
        mask = np.load(ROOT / f"data/processed/official/{seq}/valid_mask.npy").astype(bool)
        mask = mask[:std["svd_input"].shape[0]]
        pd_in = to_pd(std["svd_input"], s)
        bright = mask & (pd_in >= np.quantile(pd_in[mask], 0.90))
        bg = mask & (pd_in <= np.quantile(pd_in[mask], 0.30))
        std["fixed_gauss_s0.5"] = gauss_blur(std["svd_input"], 0.5)
        std["fixed_gauss_s1"] = gauss_blur(std["svd_input"], 1.0)
        std["fixed_wiener2d"] = to_std(wiener2d(pd_in, float(np.var(pd_in[bg]))), s, mask)

        p = fig_one(seq, std, s, a["refmax_linear_pd"], mask)
        print("figure:", p)
        p2 = fig_one(seq, std, s, a["refmax_linear_pd"], mask,
                     tag="fixed_reference_grid_rowwindow", row_window_from_input=True)
        print("figure:", p2)

        out += [f"## {seq}", "",
                f"- 共用参考 `refmax` = {a['refmax_linear_pd']:.6g}（线性 PD，来自 SVD 输入锥内最大值）",
                f"- 锥内体素 {a['mask_voxels']}；亮区 ROI {int(bright.sum())} 体素"
                f"（输入 p90 以上）；背景 ROI {int(bg.sum())} 体素（输入 p30 以下）",
                "",
                "### 棋盘/接缝与背景噪声（核心表）", "",
                md_table(a, [
                    ("phase_bias_std ↓", lambda v: f"{v['phase_bias_std']:.4f}"),
                    ("phase_bias_range ↓", lambda v: f"{v['phase_bias_range']:.4f}"),
                    ("seamY lag1/(lag2/2) ↓", lambda v: f"{v['seam_y_db']['lag1_over_half_lag2']:.3f}"),
                    ("core_mean_ratio", lambda v: f"{v['core_mean_ratio']:.3f}"),
                    ("bg_std_ratio ↓", lambda v: f"{v['bg_std_ratio']:.2f}"),
                    ("bg_std_ratio(std域) ↓", lambda v: f"{v['bg_std_ratio_std_space']:.2f}"),
                    ("bg_MAD_ratio ↓", lambda v: f"{v['bg_mad_ratio']:.2f}"),
                    ("rough_all ↓", lambda v: f"{v['roughness_ratio_all']:.3f}"),
                    ("rough_bright", lambda v: f"{v['roughness_ratio_bright']:.3f}"),
                    ("rough_bg ↓", lambda v: f"{v['roughness_ratio_bg']:.3f}"),
                    ("corr(全掩膜) ↑", lambda v: f"{v['corr_vs_input_whole_mask']:.3f}"),
                    ("best shift", lambda v: str(v['registration_vs_input']['best_integer_shift'])),
                ]), "",
                "### 同强度（core 对齐）对比固定 Gaussian", "",
                "`σ*` 为 core_mean_ratio 最接近该网络输出的高斯尺度；`代价×` = 网络值 / 高斯值。",
                ""]
        ms = a["matched_strength_vs_fixed_gaussian"]
        out += ["| method | core_net | σ* | core_gauss | bg_std_net | bg_std_gauss | 代价× | "
                "rough_bg_net | rough_bg_gauss | 代价× |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in ms.items():
            out.append(f"| `{k}` | {v['core_mean_ratio_net']:.3f} | {v['matched_gauss_sigma']:.2f} | "
                       f"{v['core_mean_ratio_gauss']:.3f} | {v['bg_std_ratio_net']:.2f} | "
                       f"{v['bg_std_ratio_gauss']:.2f} | {v['bg_std_ratio_penalty']:.2f} | "
                       f"{v['roughness_ratio_bg_net']:.2f} | {v['roughness_ratio_bg_gauss']:.2f} | "
                       f"{v['roughness_bg_penalty']:.2f} |")
        out += [""]
    (R8 / "metrics.md").write_text("\n".join(out) + "\n")
    print("wrote", R8 / "metrics.md")


if __name__ == "__main__":
    main()

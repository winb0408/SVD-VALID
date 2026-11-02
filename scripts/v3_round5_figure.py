"""Round5 图：采样创新的**直接**定性 + 定量对比。

五条臂（`input` / `gauss(σ*)` / `S_log` 标准采样 / `O_log` 我们的采样 / `O_lin` 我们的采样-线性域），
上排 = held 四场景 `rmse_img` 柱状；下排 = e904 的同一切片（带血管的最弱区）。

数据全部来自 `reports/v3_round5/sampling_direct.json` 与
`runs/_infer_v2/v3_round5/<arm>_e<s>_pred_log1p.npy`，不重算指标、不改阈值。

用法
----
    python -u scripts/v3_round5_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load                        # noqa: E402

# 主判定用 own_pairing（固定角色臂的真实用法，覆盖 25%），fold_all 只作对照表
J = ROOT / "reports/v3_round5/sampling_direct_ownpair.json"
CACHE = ROOT / "runs/_infer_v2/v3_round5"
RECON = "own_pairing"
OUT = ROOT / "reports/figures/v3_round5_sampling_direct.png"

SERIES = [
    ("input", "input（不清洗）", "#7a7a7a"),
    ("gauss", "gauss σ*=0.6", "#1f77b4"),
    ("S_log", "标准采样 m=1（SN2N）", "#2ca02c"),
    ("O_log", "我们的采样 m=2 · logeps", "#d62728"),
    ("O_lin", "我们的采样 m=2 · linear", "#ff9f0a"),
]
SLICE_SCENE = 904


def cjk_font() -> None:
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                 "Source Han Sans CN", "SimHei", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    cjk_font()
    d = json.loads(J.read_text())
    held = d["held"]
    arms = d["arms"]

    fig = plt.figure(figsize=(19, 9.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.28)

    # ---------- 上排：四场景 rmse 柱状 ----------
    ax = fig.add_subplot(gs[0, :3])
    x = np.arange(len(held))
    w = 0.16
    for k, (key, label, color) in enumerate(SERIES):
        vals = []
        for s in held:
            if key in ("input", "gauss"):
                vals.append(d["baselines"][f"e{s}"][key]["overall_rmse"])
            elif key in arms and "held" in arms[key]:
                vals.append(arms[key]["held"][f"e{s}"]["overall_rmse"])
            else:
                vals.append(np.nan)
        ax.bar(x + (k - 2) * w, vals, w, label=label, color=color,
               edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + (k - 2) * w, vals):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.4f}", rotation=90, va="bottom", ha="center",
                        fontsize=6.5)
    ax.set_xticks(x, [f"e{s}" for s in held])
    ax.set_ylabel("rmse_img（PSF 匹配真值，越低越好）")
    ax.set_title("held 四场景：五条臂的定量对比（Round5 预注册 C1/C2/C3）", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ---------- 上排右：相对标准采样的变化 ----------
    ax2 = fig.add_subplot(gs[0, 3:])
    comp = d.get("criteria", {}).get("comparisons", {}).get("O_log_vs_S_log")
    if comp:
        rel = [100 * comp["rel_change"][f"e{s}"] for s in held]
        cols = ["#2ca02c" if v < 0 else "#d62728" for v in rel]
        ax2.bar([f"e{s}" for s in held], rel, 0.55, color=cols,
                edgecolor="black", linewidth=0.5)
        for i, v in enumerate(rel):
            ax2.text(i, v, f"{v:+.1f}%", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.axhline(-5, color="#2ca02c", ls="--", lw=1, label="C1 阈值 −5%")
        ax2.axhline(5, color="#d62728", ls="--", lw=1, label="C1 阈值 +5%")
        ax2.set_ylabel("我们的采样 − 标准采样（相对 %，负=更好）")
        ax2.set_title(f"预注册判据：{comp['verdict']} / {comp['direction']}", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)

    # ---------- 下排：e904 同切片 ----------
    E = load(ROOT / f"data/processed/v3_round2_s0/low6/eval/s0_low_e{SLICE_SCENE}.npz")
    ves = E["valid"] & (E["rho"] >= 0.1)
    zs, ys, xs = np.nonzero(ves)
    zc = int(np.median(zs)) if len(zs) else E["vol"].shape[0] // 2

    panels = [("truth", "真值 img（PSF 匹配）", E["truth_img"], "gray"),
              ("input", "input（不清洗）", E["vol"], "gray")]
    for key, label, _c in SERIES[1:]:
        p = CACHE / f"{key}_{RECON}_e{SLICE_SCENE}_pred_log1p.npy"
        if p.exists():
            panels.append((key, label, np.load(p), "gray"))
        elif key == "gauss":
            panels.append((key, label, None, "gauss"))

    if panels[-1][2] is None:                     # gauss 现算，不落缓存
        from v3_round4_metrics import masked_gauss
        panels[-1] = ("gauss", panels[-1][1],
                      masked_gauss(E["vol"], E["valid"], 0.6), "gray")

    # 我们的臂只有 25% 体素被填（固定角色对 + m³ 平移的几何上限），
    # 所以定性图也统一只显示这批体素，其余涂灰——否则会让人以为未填区域也是预测。
    from v3_round5_eval import own_pairing_mask, usable
    shape = usable(E["vol"].shape, 2)
    mask = own_pairing_mask(shape, 2, (3, 6))
    cmap_obj = plt.get_cmap("gray").copy()
    cmap_obj.set_bad("#3d3d3d")

    def show(arr_sl):
        a = arr_sl.astype(np.float32).copy()
        a[~mask[zc]] = np.nan
        return a

    vmin, vmax = np.percentile(E["truth_img"][E["valid"]], [1, 99])
    vmax = max(vmax, float(np.percentile(E["vol"][E["valid"]], 99)))
    for i, (key, label, arr, cmap) in enumerate(panels[:6]):
        a = fig.add_subplot(gs[1, i])
        sl = show(arr[zc]) if arr is not None else np.zeros(E["vol"].shape[1:])
        im = a.imshow(sl, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="equal")
        a.set_title(label, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        if key not in ("truth",) and arr is not None:
            m = arms.get(key, {}).get("held", {}).get(f"e{SLICE_SCENE}")
            if m:
                a.set_xlabel(f"rmse={m['overall_rmse']:.4f}", fontsize=8)
    fig.colorbar(im, ax=fig.axes[-6:], fraction=0.012, pad=0.01)
    fig.suptitle(f"e{SLICE_SCENE} 切片 z={zc}（同一片、同一灰度窗；灰=该臂无回填的体素，我们的臂固定角色对只能填 25%）", fontsize=11, y=0.005)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

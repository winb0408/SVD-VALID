"""Round5 图：采样几何的**直接**定性 + 定量对比。

五条臂
------
    input            原始输入（不清洗）
    gauss σ*=0.6     对数域高斯平滑基线
    D1               **对角配对，间距 1**（SN2N 原型做法）= 输入取 tetris 相位 T、
                     目标取补相位 T^c，两帧沿体素有 1 体素的对角位移
    BD-2 linear      **体对角固定配对，间距 2**（本项目采样方案）· linear 域
    BD-2 logeps      同上 · logeps 域

命名说明（2026-11-05 修正）
--------------------------
用户指出旧命名不可用：「我们的采样」不是方案名，「标准采样」把 SN2N 的对角配对
误标成了业界标准。现按**几何**命名，并保证与
`reviews/V3_PR1_round5_prereg_sampling_direct.md` 的 `S_log / O_log / O_lin`
一一对应：`S_log → D1`，`O_log → BD-2 logeps`，`O_lin → BD-2 linear`。
预注册文件是冻结文本，**不改**，只在上表建立映射。

数据来源（用户 2026-11-05 追问「用的是哪个数据」）
--------------------------------------------------
定量：`data/processed/v3_round2_s0/low6/eval/s0_low_e{900..905}.npz` 中的 **held = e902–e905**
      四个体数据（e900/e901 未进判定）。原始 37×64×64，按 `usable(shape, 2)` 裁成 36×64×64。
      指标 = `overall_rmse`，只在 `own_pairing_mask(shape, 2, (3, 6))` 的 **25% 共享体素**上评分，
      五条臂用同一掩膜，故可比。
定性：e904 一张切片，z 取血管（rho≥0.1）体素的中位平面。
数值：直接读 `reports/v3_round5/sampling_direct_ownpair.json`（`--recon own_pairing`），
      不重算指标、不改阈值；预测值读 `runs/_infer_v2/v3_round5/<arm>_e<s>_pred_log1p.npy`。

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

# (key, 图例全名, 轴内短名, 颜色)
SERIES = [
    ("input", "input（不清洗）", "input", "#7a7a7a"),
    ("gauss", "gauss 平滑 σ*=0.6", "gauss σ*", "#1f77b4"),
    ("S_log", "D1 · 对角配对（间距 1，SN2N 原型）", "D1（间距 1）", "#2ca02c"),
    ("O_lin", "BD-2 · 体对角固定配对（间距 2）· linear", "BD-2 linear", "#ff9f0a"),
    ("O_log", "BD-2 · 体对角固定配对（间距 2）· logeps", "BD-2 logeps", "#d62728"),
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

    def val(key: str, s: int) -> float:
        if key in ("input", "gauss"):
            return d["baselines"][f"e{s}"][key]["overall_rmse"]
        return arms[key]["held"][f"e{s}"]["overall_rmse"]

    fig = plt.figure(figsize=(20.0, 12.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.32], hspace=0.40,
                          left=0.055, right=0.985, top=0.845, bottom=0.055)

    # ---------- 上排：逐场景横向条形（对数轴，5 臂不重叠） ----------
    top = gs[0].subgridspec(1, len(held), wspace=0.16)
    handles = []
    for i, s in enumerate(held):
        ax = fig.add_subplot(top[0, i])
        ys = np.arange(len(SERIES))[::-1]              # 第 0 项画在最上面
        vals = [val(k, s) for k, _l, _sl, _c in SERIES]
        cols = [c for _k, _l, _sl, c in SERIES]
        bars = ax.barh(ys, vals, 0.70, color=cols, edgecolor="black", linewidth=0.5)
        for y, v in zip(ys, vals):
            ax.text(v * 1.10, y, f"{v:.4f}", va="center", ha="left", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(0.0075, 0.75)
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"], fontsize=9)
        ax.grid(axis="x", alpha=0.30, which="major")
        ax.set_ylim(-0.6, len(SERIES) - 0.4)
        ax.set_yticks(ys)
        if i == 0:
            ax.set_yticklabels([sl for _k, _l, sl, _c in SERIES], fontsize=10)
            ax.set_xlabel("rmse_img（PSF 匹配真值；对数轴，越低越好）", fontsize=10)
        else:
            ax.set_yticklabels([])
        ratio = val("O_log", s) / val("S_log", s)
        ax.set_title(f"e{s}    BD-2 logeps / D1 = {ratio:.1f}×", fontsize=11.5, pad=7)
        handles = bars

    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.5)
                for _k, _l, _sl, c in SERIES],
               [lb for _k, lb, _sl, _c in SERIES],
               loc="upper center", bbox_to_anchor=(0.5, 0.902), ncol=5,
               frameon=False, fontsize=11.5, handlelength=1.4, columnspacing=1.6)
    fig.text(0.5, 0.862,
             "预注册判据 C1（逐场景相对差 ≥5% 且方向一致 ≥3/4）= 成立；方向：BD-2 更差（4/4 场景）",
             ha="center", va="center", fontsize=10.5, color="#b22222")

    # ---------- 下排：e904 同切片 ----------
    E = load(ROOT / f"data/processed/v3_round2_s0/low6/eval/s0_low_e{SLICE_SCENE}.npz")
    ves = E["valid"] & (E["rho"] >= 0.1)
    zs, ys_, xs_ = np.nonzero(ves)
    zc = int(np.median(zs)) if len(zs) else E["vol"].shape[0] // 2

    panels = [("truth", "真值 img", E["truth_img"]),
              ("input", "input", E["vol"])]
    for key, _lb, short, _c in SERIES[1:]:
        p = CACHE / f"{key}_{RECON}_e{SLICE_SCENE}_pred_log1p.npy"
        if p.exists():
            panels.append((key, short, np.load(p)))
        elif key == "gauss":
            panels.append((key, short, None))
    if any(p[2] is None for p in panels):
        from v3_round4_metrics import masked_gauss
        panels = [(k, t, masked_gauss(E["vol"], E["valid"], 0.6) if a is None else a)
                  for k, t, a in panels]

    # BD-2 固定角色对 + m³ 平移的几何上限只覆盖 25% 体素，定性图统一只显示这批体素，
    # 其余涂灰——否则会让人以为未填区域也是预测。
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
    bot = gs[1].subgridspec(1, len(panels) + 1, wspace=0.07,
                            width_ratios=[1] * len(panels) + [0.07])
    im = None
    for i, (key, ttl, arr) in enumerate(panels[:7]):
        a = fig.add_subplot(bot[0, i])
        im = a.imshow(show(arr[zc]), cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="equal")
        a.set_title(ttl, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
        if key == "truth":
            a.set_xlabel("（参考）", fontsize=9)
        else:
            v = val(key, SLICE_SCENE)
            a.set_xlabel(f"rmse={v:.4f}", fontsize=9.5)
    cax = fig.add_subplot(bot[0, len(panels)])
    fig.colorbar(im, cax=cax)

    fig.suptitle("Round5：采样几何的直接对比——D1（对角配对，间距 1）vs BD-2（体对角固定配对，间距 2）",
                 fontsize=15, y=0.985)
    fig.text(0.5, 0.951,
             "定量 = held e902–e905（data/processed/v3_round2_s0/low6/eval，36×64×64），"
             f"只在五臂共享的 25% 体素掩膜上评分；定性 = e904 一张切片（z={zc}）",
             ha="center", va="center", fontsize=10.5)
    fig.text(0.5, 0.012,
             "灰 = 该臂不覆盖的体素（BD-2 固定角色对的几何只能填 25%，部署代价 = 8 倍推理次数）；"
             "D1 / gauss / input / 真值本身覆盖全图，此处同样只显示那 25% 以便逐像素对照",
             ha="center", va="center", fontsize=9.5, color="#444444")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

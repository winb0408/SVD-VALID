#!/usr/bin/env python3
"""Round6 作图：真实采集上**重新自监督训练**后的定性与定量对照。

口径（与 reviews/V3_PR1_round6_prereg_real_selfsup.md 一致）
--------------------------------------------------------------
- 全部模型都在真实半（37×64×64，200 帧）上从头自监督训练，不用任何仿真权重。
- 主图用**方向 2**（even 训练 → odd 参照）的权重：这条线完全不参与超参选择，是无偏的。
- `α` 是 1 参数增益校准，只在方向 1 上拟合，方向 2 沿用——所以"校准后"的数字仍然无偏。
- 每个数据集各用各的动态范围（两次独立采集，背景亮度差很多，共用窗口必然一边看不清）。

两张图
------
`reports/figures/v3_round6_real_selfsup_mip.png`
    2 行（SA64 / HSA64）× 6 列（input / gauss / N2N / D1 / BD-2 / colorbar）。
`reports/figures/v3_round6_real_selfsup_bars.png`
    `X/X_in` 分组柱状图：1.0 = 完全没去噪，0.707 = 完美去噪的理论上限。
    左 = 原始输出（α=1），右 = 增益校准后（每臂最好的样子）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import fwd, inv                            # noqa: E402
from v3_round4_metrics import masked_gauss                        # noqa: E402
from v3_round6_eval import (apply_gain, denoise, load_half,       # noqa: E402
                            usable)

NORM_PATH = ROOT / "reports/v3_round4/b1_norm.json"
SEL_PATH = ROOT / "reports/v3_round6/selections.json"
rows: dict = {}          # main() 里填，arm_alpha 要用
ARMS = ("gauss", "N2N", "D1", "BD2")
LABEL = {"input": "不去噪（input）", "gauss": "高斯平滑（最优 σ, α）",
         "N2N": "N2N dense（上界参照）", "D1": "D1 间距 1（SN2N 原型）",
         "BD2": "BD-2 间距 2（本项目）"}
NPASS = {"input": 0, "gauss": 0, "N2N": 8, "D1": 8, "BD2": 16}
COLOR = {"input": "0.55", "gauss": "0.80", "N2N": "#4a5568",
         "D1": "#2b6cb0", "BD2": "#c05621"}


def to_db(pd: np.ndarray, ref: float, support: np.ndarray | None) -> np.ndarray:
    """z-MIP（axis 0）相对 `ref` 的 dB；`support` 外的体素不参与 MIP。"""
    p = pd if support is None else np.where(support, pd, 0.0)
    mip = p.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(mip, 1e-30) / ref)
    return np.where(mip > 0, db, np.nan)


def denoise_with(seq: str, th: str, domain: str, geom: str, ckpt: Path):
    pd, valid, s = load_half(seq, th)
    shape = usable(pd.shape, 2)
    sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
    pd, valid = pd[sl], valid[sl]
    z = np.load(ROOT / f"data/processed/v3_round6_b1/{domain}/{seq}_{th}_x.npy")
    y, filled = denoise(ckpt, domain, z, geom)
    norm = json.loads(NORM_PATH.read_text())[domain]
    pd_hat = np.maximum(inv(domain, y * norm["std"] + norm["mean"], s), 0.0)
    return pd_hat, valid, s, filled


def pick_font(plt, font_manager) -> None:
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "Source Han Sans SC", "DejaVu Sans"):
        if any(x.name == f for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def arm_alpha(sel, scores, seq, geom, th, eh):
    """方向 1 定下的 α；方向 2 沿用它（不得重拟合）。gauss 的 α 来自参照网格。"""
    if geom == "gauss":
        return float(scores["references"][seq]["odd2even"]["gauss"]["alpha"])
    if geom == "N2N":
        nm = f"{seq}_N2N_log1p_e6_odd2even"
        return float(rows[nm]["alpha_fit"])          # N2N 不参与选型，α 直接取方向 1 的拟合值
    return float(sel[seq][geom]["best"]["alpha"])


def run_name(sel, seq, geom, th, eh):
    if geom == "N2N":
        return f"{seq}_N2N_log1p_e6_{th}2{eh}"
    b = sel[seq][geom]["best"]
    return f"{seq}_{geom}_{b['domain']}_e{b['epochs']}_{th}2{eh}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selections", default="reports/v3_round6/selections.json")
    ap.add_argument("--scores", default="reports/v3_round6/x_scores.json")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    pick_font(plt, font_manager)

    sel = json.loads((ROOT / a.selections).read_text())
    scores = json.loads((ROOT / a.scores).read_text())
    rows.update({r["run"]: r for r in scores["rows"]})
    series = ("input",) + ARMS

    # ---------- 图 1：MIP 对照（方向 2 权重 + 增益校准 = 每臂最好的样子）----------
    fig, axes = plt.subplots(2, 6, figsize=(30.0, 9.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.05]})
    for r, seq in enumerate(("SA64", "HSA64")):
        th, eh = "even", "odd"                      # 方向 2
        pd_ref, valid, s = load_half(seq, th)
        shape = usable(pd_ref.shape, 2)
        sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
        pd_ref, valid = pd_ref[sl], valid[sl]
        pd_eval, _, _ = load_half(seq, eh)
        pd_eval = pd_eval[sl]

        panels = {"input": (pd_ref, valid)}
        sig = float(scores["references"][seq][f"{th}2{eh}"]["gauss"]["best_sigma"])
        lg = fwd("log1p", pd_ref, s, valid)
        g = np.maximum(np.expm1(masked_gauss(lg, valid, sig)), 0.0) * s
        panels["gauss"] = (apply_gain(pd_ref, g, s, arm_alpha(sel, scores, seq, "gauss", th, eh)),
                           valid)
        for geom in ("N2N", "D1", "BD2"):
            nm = run_name(sel, seq, geom, th, eh)
            if nm not in rows:
                print(f"  [跳过] {nm} 尚无结果")
                continue
            dom = "log1p" if geom == "N2N" else sel[seq][geom]["best"]["domain"]
            ckpt = ROOT / "runs/v3_round6" / nm / "latest.pt"
            pd_hat, v2, _s2, filled = denoise_with(seq, th, dom, geom, ckpt)
            al = arm_alpha(sel, scores, seq, geom, th, eh)
            panels[geom] = (apply_gain(pd_ref, pd_hat, s, al),
                            v2 & filled if geom == "BD2" else v2)

        ref = float(np.percentile(pd_eval, 99.9))   # 参照半的 99.9 分位当 0 dB
        img_in = to_db(pd_ref, ref, valid)
        fin = img_in[np.isfinite(img_in)]
        lo, hi = float(np.percentile(fin, 2.0)), float(np.percentile(fin, 99.9))

        im = None
        for c, key in enumerate(series):
            if key not in panels:
                axes[r, c].axis("off")
                continue
            pd_i, sup_i = panels[key]
            img = to_db(pd_i, ref, sup_i)
            im = axes[r, c].imshow(img, cmap="inferno", vmin=lo, vmax=hi,
                                   interpolation="nearest")
            axes[r, c].set_title(f"{seq} · {LABEL[key]}", fontsize=13)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if NPASS[key]:
                cov = 100.0 * sup_i.sum() / valid.sum()
                axes[r, c].text(0.02, 0.02, f"覆盖 {cov:.0f}% · {NPASS[key]} 次前向",
                                transform=axes[r, c].transAxes, fontsize=10, color="w",
                                bbox=dict(fc="k", alpha=0.55, ec="none", pad=2))

        cb = fig.colorbar(im, cax=axes[r, 5])
        cb.set_label(f"{seq}：相对本行参照半 99.9 分位的 dB\n窗口 {lo:+.1f}…{hi:+.1f} dB",
                     fontsize=10)

    fig.suptitle("Round6 定性对照：真实采集上重新自监督训练（方向 2 = even 训练 → odd 参照，无偏；"
                 "已做增益校准）", fontsize=16, y=0.975)
    fig.text(0.5, 0.035,
             "全部模型在真实半（37×64×64，200 帧）上从头自监督训练，不使用任何仿真权重。"
             "BD-2 只显示其训练过的角色对直接覆盖的 1/4 体素（16 次前向），其余臂裁到同一批体素以便同面比较；"
             "BD-2 图上的横条纹来自覆盖几何：所选角色对 1→6、4→3 的目标相位在 y 方向只落在 2/4 行，"
             "z-MIP 后每 4 行有 2 行整列无数据（换成 4 对则每列都有数据、条纹消失）；"
             "两个数据集各用各的动态范围。\n"
             "注意：亮结构在所有臂上都被压暗（§4.4 带内 slope 0.79–0.93），本图只用于看纹理，不构成判优依据。",
             ha="center", fontsize=10.5, color="0.25")
    fig.subplots_adjust(left=0.02, right=0.97, top=0.90, bottom=0.10, wspace=0.04, hspace=0.10)
    p1 = ROOT / "reports/figures/v3_round6_real_selfsup_mip.png"
    fig.savefig(p1, dpi=130)
    print(f"saved {p1}")

    # ---------- 图 2：X/X_in 分组柱状图 ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(16.0, 6.8))
    for ax, mode in zip(axes2, ("原始输出（α=1）", "增益校准后（α=α*）")):
        x = np.arange(len(series) * 4)      # 5 臂 × 2 数据集 × 2 方向
        labels, vals, colors = [], [], []
        for seq in ("SA64", "HSA64"):
            for th, eh in (("odd", "even"), ("even", "odd")):
                ref = scores["references"][seq][f"{th}2{eh}"]
                x_in = ref["input"]
                d = {"input": x_in, "gauss": ref["gauss"]["x" if mode.startswith("原始")
                                                        else "x_gain"]}
                for geom in ("N2N", "D1", "BD2"):
                    nm = run_name(sel, seq, geom, th, eh)
                    rr = rows.get(nm)
                    d[geom] = np.nan if rr is None else (
                        rr["x_pair"] if mode.startswith("原始") else rr["x_gain"])
                for key in series:
                    labels.append(key)
                    vals.append(d[key] / x_in)
                    colors.append(COLOR[key])
        vals = np.array(vals, float)
        bars = ax.bar(x, np.where(np.isfinite(vals), vals, 0.0), width=0.72,
                      color=colors, edgecolor="k", linewidth=0.8)
        for b, v in zip(bars, vals):
            if not np.isfinite(v):
                b.set_hatch("//"); b.set_facecolor("white")
        ax.axhline(1.0, color="k", lw=1.2, ls="--")
        ax.axhline(2 ** -0.5, color="green", lw=1.2, ls=":")
        ax.text(len(x) - 0.5, 2 ** -0.5, " 理论上限 1/√2≈0.707", ha="right", va="bottom",
                fontsize=9, color="green")
        ax.text(5.0, 1.005, "SA64", fontsize=12, color="0.25", ha="center",
                transform=ax.get_xaxis_transform())
        ax.text(15.0, 1.005, "HSA64", fontsize=12, color="0.25", ha="center",
                transform=ax.get_xaxis_transform())
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.axvline(9.5, color="0.4", lw=1.0)
        ax.set_title(f"{mode}　（每臂两柱：左 = 方向 1 偏乐观，右 = 方向 2 无偏）", fontsize=12)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.set_ylim(0.6, max(12.0, float(np.nanmax(vals)) * 1.25))
        ax.set_ylabel("X_RMSE / X_RMSE_in（对数轴；1.0 = 完全没去噪）")
    fig2.suptitle("Round6 定量：交叉实现误差（两半噪声独立 ⇒ 方向 2 的排序无偏）", fontsize=14)
    fig2.text(0.5, 0.02,
              "真实数据可去除的 log-PD 方差上限：SA64 5.2%、HSA64 0.52%"
              "（reports/v3_r06/testretest_calibration.json）。"
              "注意这个分数偏袒低对比度输出：整幅方差被削掉的比例 gauss 9.85%/1.66%、"
              "D1 5.96%/1.04%、BD-2 4.44%/0.66%，超上限即说明削的是结构（见 real_selfsup.md §4.4）。"
              "斜纹柱 = 该组合缺结果。",
              ha="center", fontsize=9.5, color="0.25")
    fig2.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.22, wspace=0.10)
    p2 = ROOT / "reports/figures/v3_round6_real_selfsup_bars.png"
    fig2.savefig(p2, dpi=140)
    print(f"saved {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

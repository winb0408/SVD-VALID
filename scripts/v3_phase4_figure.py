"""Phase 4 方法对比图：输入 / ① / ② / 目标 + 误差图 + 指标柱状图。

## 这张图要回答的问题（L49）

"本项目的方法（② N2N 网络）到底有没有把噪声降下来，比不用训练的做法强在哪？"

图上每个量都在**同一个判据、同一片评价体素、同一张切片**上算：

    输入 obs1      恒等，n2n 定义上 = 1
    ①  Wiener 型收缩（v3_phase4_wiener.py，w=7，b≤1，训练无关，只用 obs1）
    ②  N2N 网络（跨场景训练 res_s3412，n2n 最好的那一颗种子）
    目标 obs2      同一物理场景的另一次独立实现，判据只用它
    b 图           ① 的逐体素收缩系数，看它在哪平滑、在哪不动

第三行柱状图把**所有试过的方法**放在同一把尺子上，包括失败的：
纯局部均值（b=0，n2n≈15，被截断）、SPW（场景 1 上 0.676 但不迁移）、
② 的两代训练（单场景 600 epoch 其实是背目标噪声）。

用法：
    python scripts/v3_phase4_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase3_eval import M_EVAL, usable, support_mask          # noqa: E402
from v3_phase4_scoreboard import score                            # noqa: E402
from v3_phase4_train_n2n import prep                              # noqa: E402

from matplotlib import font_manager as fm                          # noqa: E402

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_CJK)
plt.rcParams["font.family"] = [fm.FontProperties(fname=_CJK).get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "reports/figures/v3_phase4_method_comparison.png"
CKPT = ROOT / "runs/v3_n2n_multi/res_s3412/best.pt"


def collapse(x, w):
    mu = uniform_filter(x, w)
    lv = np.maximum(uniform_filter(x * x, w) - mu * mu, 0.0)
    return mu, lv


def predict_net(x, ckpt):
    import torch
    from models.network import NetworkCNR
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = ck["args"]
    model = NetworkCNR(features=cargs["features"], groups=cargs["groups"])
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        xin = torch.as_tensor(x, dtype=torch.float32)[None, None]
        h = model(xin)
        pred = (xin + h if cargs.get("residual") else h)[0, 0].numpy()
    return pred, cargs


def main():
    d1 = np.load(ROOT / "data/processed/v3_sim/sim_seed1.npz")
    shape = tuple((n // 32) * 32 for n in usable(d1["pd_linear"].shape, M_EVAL))
    crop = tuple(slice(0, n) for n in shape)

    x, valid_f = prep(ROOT / "data/processed/v3_sim/sim_seed1.npz")
    x = x.astype(np.float64)
    valid = valid_f.astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(d1["pd_linear"][crop].astype(np.float32)[valid]))
    from v3_phase4_scoreboard import to_log
    obs2 = to_log(np.load(ROOT / "data/processed/v3_sim/sim_seed1_n201_s1.npz")
                  ["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)
    xf = x.astype(np.float32)

    common = support_mask(shape, M_EVAL) & valid
    masks = {"common": common, "ves": vessels & common,
             "far": (common & (distance_transform_edt(~vessels) >= 4)
                     & (distance_transform_edt(valid) >= 2))}

    # ① Wiener（w=7，b≤1）
    mu7, lv7 = collapse(x, 7)
    sig2 = float(np.median(lv7[common]))
    b1 = np.clip(1.0 - sig2 / np.maximum(lv7, 1e-9), 0.0, 1.0)
    pred1 = (mu7 + b1 * (x - mu7)).astype(np.float32)

    # SPW（场景 1 上最优，留出不迁移）
    # 与 spw.py 的最优点一致：mu 的窗 7（r=3），lv 的窗 5，两者是**不同**的窗
    _, lv5 = collapse(x, 5)
    sig2_5 = float(np.median(lv5[common]))
    b_spw = np.clip(1.05 * (1.0 - sig2_5 / np.maximum(lv5, 1e-9)), 0.0, 1.05)
    pred_spw = (mu7 + b_spw * (x - mu7)).astype(np.float32)

    # ② 网络
    pred2, cargs = predict_net(xf, CKPT)

    m1 = score(pred1, xf, obs2, truth, masks)
    m2 = score(pred2.astype(np.float32), xf, obs2, truth, masks)
    mspw = score(pred_spw, xf, obs2, truth, masks)
    print(f"① {m1['n2n_ratio']:.4f}   ② {m2['n2n_ratio']:.4f}   "
          f"SPW {mspw['n2n_ratio']:.4f}   ② 血管 {m2['n2n_ratio_ves']:.3f}   ② 斜率 "
          f"{m2['slope_vs_input']:.3f}")

    # 切片：血管体素最多的 z
    zc = int(np.argmax([vessels[z].sum() for z in range(shape[0])]))
    sl = (zc, slice(None), slice(None))
    vmin, vmax = np.percentile(obs2[sl], [1, 99.5])
    emax = np.percentile(np.abs(xf - obs2)[sl], 99)

    fig = plt.figure(figsize=(17, 13.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.22, wspace=0.06)
    titles = [f"输入 obs1（判据里恒等 = 1.000）", f"① Wiener 收缩（w=7，n2n={m1['n2n_ratio']:.3f}）",
              f"② N2N 网络（跨场景，n2n={m2['n2n_ratio']:.3f}）", "目标 obs2（另一次独立实现）"]
    for j, (img, tt) in enumerate(zip([xf, pred1, pred2, obs2], titles)):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(img[sl], vmin=vmin, vmax=vmax, cmap="inferno", aspect="equal")
        ax.set_title(tt, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    for j, (img, lab) in enumerate([(xf, "误差 |输入 − 目标|"),
                                    (pred1, f"误差 |① − 目标|  血管区 n2n={m1['n2n_ratio_ves']:.2f}"),
                                    (pred2, f"误差 |② − 目标|  血管区 n2n={m2['n2n_ratio_ves']:.2f}"),
                                    (b1, "① 的收缩系数 b（0=完全平滑，1=不动）")]):
        ax = fig.add_subplot(gs[1, j])
        if j < 3:
            im = ax.imshow(np.abs(img - obs2)[sl], vmin=0, vmax=emax, cmap="magma",
                           aspect="equal")
        else:
            im = ax.imshow(img[sl], vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(lab, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # 柱状图：所有试过的方法
    ax = fig.add_subplot(gs[2, :])
    bars = [("纯局部均值\n(b=0)", 15.06, 1.0, "截断 15", "否"),
            ("② 单场景 600epoch\n(实为背噪声)", 0.875, 1.0, "否", "否"),
            ("② 单场景 120epoch", 0.794, 1.0, "否", "否"),
            ("② 跨场景 4 seed", 0.831, 0.073, "1/4", "否"),
            ("① Wiener w=7", 0.754, 0.0, "是", "是"),
            ("SPW（场景1拟合）", 0.676, 0.0, "4 留出场景\n3 个变差", "否"),
            ("天花板\n(3 实现均值)", 0.6359, 0.0, "参考", "—")]
    xs = np.arange(len(bars))
    vals = [min(b[1], 1.2) for b in bars]
    cols = ["#9e9e9e", "#9e9e9e", "#9e9e9e", "#9e9e9e", "#2e7d32", "#c62828", "#1565c0"]
    ax.bar(xs, vals, yerr=[b[2] for b in bars], color=cols, capsize=4, width=0.62)
    ax.axhline(1.0, color="k", lw=1.0, ls=":")
    ax.axhline(0.808, color="#e65100", lw=1.4, ls="--")
    ax.text(len(bars) - 0.45, 0.815, "预注册判定线 n2n≤0.808", color="#e65100",
            fontsize=10, ha="right")
    ax.axhline(0.6359, color="#1565c0", lw=1.0, ls=":")
    for i, b in enumerate(bars):
        ax.text(i, min(b[1], 1.2) + 0.045, f"{b[1]:.3f}", ha="center", fontsize=10)
        ax.text(i, 0.04, b[3], ha="center", fontsize=9, color="white"
                if cols[i] != "#9e9e9e" else "black")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylabel("n2n（越低越好；1.000 = 等于输入）", fontsize=11)
    ax.set_ylim(0, 1.32); ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_title("同一判据、同一片 15,360 个体素上的横向对比（绿=通过预注册门槛，红=不通过，灰=无论文价值）",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Phase 4 方法对比：训练无关的 ① 有效且稳定；② 与 SPW 都不成立（"
                 f"z={zc} 切片，n2n 在 common 掩膜上计算）", fontsize=13, y=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    return {"slice": zc, "wiener": m1, "n2n_net": m2, "spw": mspw,
            "cargs": {k: v for k, v in cargs.items()}}


if __name__ == "__main__":
    import json
    r = main()
    (ROOT / "reports/v3_phase4_2026-10-21/spw/figure_metrics.json").write_text(
        json.dumps(r, indent=1, ensure_ascii=False, default=str))

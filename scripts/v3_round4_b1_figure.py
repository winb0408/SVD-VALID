"""Round4-B1 图：三表示域 vs 高斯的逐场景数值 + 一张可视对比。

产物
----
    reports/figures/v3_round4_b1_domains.png    左：逐场景 rmse 柱状（e902–e905）
                                                中：bias_ves / var_bg 散点（越高越差）
                                                右：e904 的一层切片（输入/真值/高斯/三臂）
用法
----
    python -u scripts/v3_round4_b1_figure.py [--scene 904]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load, masked_gauss         # noqa: E402

HELD = [902, 903, 904, 905]
ARMS = ["linear", "log1p", "logeps"]
# Round4-B1 续训臂（预注册允许的唯一一条）：单独一色，用来把"多训更差"直接画出来
EXTRA = ["logeps40"]
COLOR = {"input": "#9e9e9e", "gauss": "#455a64", "linear": "#1e88e5",
         "log1p": "#43a047", "logeps": "#e53935", "logeps40": "#8e0000"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=904)
    args = ap.parse_args()

    d = json.loads((ROOT / "reports/v3_round4/b1_pilot.json").read_text())
    p40 = ROOT / "reports/v3_round4/b1_pilot_40ep.json"
    d40 = json.loads(p40.read_text()) if p40.exists() else None
    fig = plt.figure(figsize=(19, 5.4))

    # ---- 左：逐场景 rmse 柱状 ----
    ax = fig.add_subplot(1, 3, 1)
    series = {"input": [], "gauss": [], **{a: [] for a in ARMS}}
    for s in HELD:
        b = d["baselines"][f"e{s}"]
        series["input"].append(b["input"]["overall_rmse"])
        series["gauss"].append(b["gauss"]["overall_rmse"])
        for a in ARMS:
            series[a].append(d["arms"][a]["held"][f"e{s}"]["overall_rmse"])
    if d40 is not None:
        series["logeps40"] = [d40["arms"]["logeps"]["held"][f"e{s}"]["overall_rmse"] for s in HELD]
    x = np.arange(len(HELD))
    w = 0.16 if len(series) <= 5 else 0.135
    off = (len(series) - 1) / 2
    for k, name in enumerate(series):
        ax.bar(x + (k - off) * w, series[name], w, label=name, color=COLOR[name])
    ax.set_xticks(x, [f"e{s}" for s in HELD])
    ax.set_ylabel("rmse_img（越小越好）")
    ax.set_title("held-out 逐场景：三表示域 vs 高斯（含续训臂）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.25, axis="y")

    # ---- 中：bias_ves vs var_bg（越低越好）----
    ax = fig.add_subplot(1, 3, 2)
    for name, get in (
        ("gauss", lambda s: d["baselines"][f"e{s}"]["gauss"]),
        *[(a, lambda s, a=a: d["arms"][a]["held"][f"e{s}"]) for a in ARMS],
        *([("logeps40", lambda s: d40["arms"]["logeps"]["held"][f"e{s}"])] if d40 else []),
    ):
        xs = [get(s)["var_bg"] for s in HELD]
        ys = [abs(get(s)["bias_ves"]) for s in HELD]
        ax.scatter(xs, ys, s=46, color=COLOR[name], label=name)
    ax.set_xlabel("var_bg（背景方差，越小越好）")
    ax.set_ylabel("|bias_ves|（血管幅度偏差，越小越好）")
    ax.set_title("结构代价 vs 背景抑制：左下角最好")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)

    # ---- 右：一层切片 ----
    s = args.scene
    E = load(ROOT / f"data/processed/v3_round2_s0/low6/eval/s0_low_e{s}.npz")
    vol = E["vol"]
    rho = E["rho"]
    # 选血管最多的一层
    z = int(np.argmax((rho > 0.5).sum(axis=(1, 2))))
    panels = [("输入（含噪）", vol[z]), ("真值（PSF 匹配）", E["truth_img"][z]),
              ("gauss(0.6)", masked_gauss(vol, E["valid"], 0.6)[z])]
    for a in ARMS + EXTRA:
        p = ROOT / (f"runs/_infer_v2/b1/b1_{a}_e{s}_pred_log1p.npy" if a in ARMS
                    else f"runs/_infer_v2/b1/b1_logeps_40ep_e{s}_pred_log1p.npy")
        if p.exists():
            panels.append((f"{a} 臂", np.load(p)[z]))
    ax = fig.add_subplot(1, 3, 3)
    cat = np.concatenate([p.reshape(-1) for _, p in panels])
    vmin, vmax = np.percentile(cat, [2, 98])
    n = len(panels)
    grid = np.concatenate([np.pad(p, ((2, 2), (2, 2)), constant_values=vmax) for _, p in panels], axis=1)
    ax.imshow(grid, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xticks([i * (vol.shape[2] + 4) + vol.shape[2] / 2 for i in range(n)])
    ax.set_xticklabels([t for t, _ in panels], fontsize=7, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title(f"e{s} 第 {z} 层（同一窗宽窗位）")
    fig.tight_layout()
    out = ROOT / "reports/figures/v3_round4_b1_domains.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

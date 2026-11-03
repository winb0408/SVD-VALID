"""Round4-0 图：Pareto（噪声抑制 vs 血管幅度偏差）+ 确认集配对差。

左两栏：`(var_bg, |bias_ves|)`，每臂一条曲线，旋钮标在点上（gauss=σ、wiener=w、网络=β）。
右两栏：确认集 e906–e911 上 `net(β*)` 与 `gauss(σ*)` 的配对差（相对值），逐体一个点。

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round4_figure.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P0 = json.load(open(ROOT / "reports/v3_round4/baseline_pareto.json"))
CF = json.load(open(ROOT / "reports/v3_round4/p0_confirm.json"))

plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"gauss": "#1f77b4", "wiener": "#ff7f0e", "net_direct": "#d62728",
          "net_residual": "#2ca02c", "net_zeroinit": "#9467bd"}
LABEL = {"gauss": "高斯模糊", "wiener": "Wiener 收缩", "net_direct": "网络 direct",
         "net_residual": "网络 residual", "net_zeroinit": "网络 zeroinit"}


def mean_of(per: dict, key: str) -> float:
    return float(np.mean([v[key] for v in per.values()]))


def main() -> None:
    levels = [l for l in ("low", "vlow") if l in P0["levels"]]
    fig, axes = plt.subplots(2, len(levels) + 1, figsize=(5.2 * (len(levels) + 1), 8.6))

    for j, lv in enumerate(levels):
        ax = axes[0][j]
        for arm, knobs in P0["levels"][lv]["sweeps"].items():
            xs = [mean_of(p, "var_bg") for p in knobs.values()]
            ys = [abs(mean_of(p, "bias_ves")) for p in knobs.values()]
            ks = list(knobs.keys())
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]; ks = np.array(ks)[order]
            ax.plot(xs, ys, "-o", ms=5, lw=1.4, color=COLORS.get(arm, "gray"),
                    label=LABEL.get(arm, arm))
            for x, y, k in zip(xs, ys, ks):
                if arm == "gauss" or (arm.startswith("net") and k in ("0.5", "1")):
                    ax.annotate(k, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("背景残余方差 var_bg（越小=噪声压得越净）")
        ax.set_ylabel("|血管幅度偏差 bias_ves|（越小=结构越保真）")
        ax.set_title(f"{lv} 档：留出体 e902–e905 的 Pareto")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

        # 确认集配对差
        ax2 = axes[1][j]
        cf = {k: v for k, v in CF["levels"][lv]["confirm"].items()}
        if cf:
            scenes = sorted(cf)
            for arm in ("net_residual", "net_zeroinit"):
                if arm not in next(iter(cf.values())):
                    continue
                d = [cf[s][arm]["bias_ves"] / cf[s]["gauss"]["bias_ves"] for s in scenes]
                ax2.plot(range(len(scenes)), d, "-o", ms=5,
                         color=COLORS[arm], label=f"{LABEL[arm]} β*")
            ax2.axhline(1.0, color="k", lw=1, ls="--")
            ax2.set_xticks(range(len(scenes)))
            ax2.set_xticklabels([s.replace("e", "") for s in scenes], fontsize=8)
            ax2.set_ylabel("bias_ves(网络) / bias_ves(高斯)")
            ax2.set_title(f"{lv} 确认集 e906–e911：血管偏差比值（<1 更好）")
            ax2.grid(alpha=0.3)
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, "该档没有可行 β*", ha="center", va="center", fontsize=10)
            ax2.set_axis_off()

    axes[0][-1].axis("off")
    axes[1][-1].axis("off")
    axes[0][-1].text(0.02, 0.95,
                     "读图要点\n\n"
                     "• 横轴：背景残余方差（相对真值的方差，越小噪声剩得越少）\n"
                     "• 纵轴：血管区系统偏差绝对值（越小结构越保真）\n"
                     "• 左下角 = 既降噪又不伤血管，右下角 = 降噪但把血管压暗\n\n"
                     "结论（确认集已通过）\n"
                     "• vlow 档：网络 residual / zeroinit 在 β*=0.5 处，\n"
                     "  噪声抑制与冻结高斯同档，血管偏差小 10–20 倍\n"
                     "• low 档（噪声更大）：没有任何 β 能把噪声压到\n"
                     "  高斯的水平 → 该档记 NO_FEASIBLE_BETA\n"
                     "• 旧 Gate（n2n≤0.80、必须用满强度输出）与本结论不矛盾，\n"
                     "  它测的是另一件事：满强度输出会过度平滑",
                     fontsize=9.5, va="top")

    fig.suptitle("Round4-0：结构保真—背景抑制 Pareto（仿真，含无噪成像真值）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dst = ROOT / "reports/figures/v3_round4_pareto.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=150)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()

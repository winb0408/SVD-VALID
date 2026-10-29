"""R06/v2 的对照图：v1 与 v2 的"真值"到底差在哪。

四列，同一张画布上摆 v1 与 v2 的对应量（都在 log1p(·/s) 域，MIP 投影）：

  ① 观测（v1 与 v2 用同一个背景实现，血流项不同）
  ② 真值（v1：血管外恒为 0 的硬掩膜；v2：带晕的连续密度卷积）
  ③ 血管剖面：到血管掩膜距离 → PD 均值（v1 一条断崖，v2 一条斜坡）
  ④ 血管内 hub：真值 vs 观测的联合分布（v1 是两条竖线，v2 是散点云）

用法：python scripts/v3_r06_v2_figure.py
产物：reports/v3_r06/fig_v1_vs_v2.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy.ndimage import distance_transform_edt                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v3_r06"
OUT.mkdir(parents=True, exist_ok=True)


def load(npz: Path):
    d = np.load(npz)
    pd = d["pd_linear"].astype(np.float64)
    gt = d["gt_flow_pd"].astype(np.float64)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float64) if "rho" in d.files else d["vessels"].astype(np.float64)
    s = float(d["s"]) if "s" in d.files else float(np.median(pd[valid]))
    return {"obs": np.log1p(pd / s), "gt": np.log1p(gt / s), "valid": valid, "rho": rho}


def profile(E: dict, rmax=6) -> dict:
    """到血管（ρ≥0.5）距离 → 真值/观测的 PD 均值（沿 z 的 MIP 不参与，逐体素统计）。"""
    ves = (E["rho"] >= 0.5) & E["valid"]
    dist = distance_transform_edt(~ves)
    out = {}
    for name in ("gt", "obs"):
        out[name] = [float(np.mean(E[name][E["valid"] & (np.round(dist) == r)]))
                     if np.any(E["valid"] & (np.round(dist) == r)) else np.nan
                     for r in range(rmax + 1)]
    return out


def main():
    v1 = load(ROOT / "data/processed/v3_sim/sim_seed0.npz")
    v2 = load(sorted((ROOT / "data/processed/v3_sim_v2/eval").glob("*.npz"))[0])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for row, (tag, E) in enumerate([("v1 (old forward model)", v1), ("v2 (this fix)", v2)]):
        for col, (name, key) in enumerate([("observation (MIP)", "obs"),
                                           ("truth (MIP)", "gt")]):
            ax = axes[row, col]
            im = ax.imshow(E[key].max(axis=0), cmap="magma", aspect="equal")
            ax.set_title(f"{tag}\n{name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.72)
        # 剖面
        ax = axes[row, 2]
        pr = profile(E)
        r = np.arange(len(pr["gt"]))
        ax.plot(r, pr["gt"], "o-", label="truth")
        ax.plot(r, pr["obs"], "s--", label="observation")
        ax.set_xlabel("distance to vessel (voxels)", fontsize=8)
        ax.set_ylabel("log1p PD", fontsize=8)
        ax.set_title(f"{tag}: vessel profile", fontsize=9)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        gt0 = float(np.mean(np.expm1(E["gt"][E["valid"]]) <= 0.0))
        ax.text(0.98, 0.06, f"truth==0: {gt0*100:.2f}%", transform=ax.transAxes,
                ha="right", fontsize=8, color="w" if row else "k",
                bbox=dict(fc="k" if row else "w", alpha=0.35, ec="none"))
    fig.suptitle("R06: v1 truth is identically 0 outside the mask; v2 truth is a blurred density",
                 fontsize=10)
    fig.tight_layout()
    p = OUT / "fig_v1_vs_v2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

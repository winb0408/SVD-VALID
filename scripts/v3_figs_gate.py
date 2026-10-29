"""V3 Phase 1-2 结论图：采样几何 / 泄漏与信号两侧 / 仿真器真值。

英文标签（服务器无中文点阵字体，中文会变豆腐块）。三张图：
  fig1_sampling_geometry.png  覆盖率是怎么掉的、网络实际看到什么
  fig2_leak_and_signal.png    位移类的泄漏热图 + 泄漏 vs 信号散点
  fig3_simulator_truth.png    仿真器的真值 / 观测 / 噪声（血流项已修）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402

FIG = Path("reports/figs")
FIG.mkdir(parents=True, exist_ok=True)
PD = np.load("data/processed/official/SA64/pd_linear.npy")
VALID = np.load("data/processed/official/SA64/valid_mask.npy")


def logimg(a):
    return np.log10(np.maximum(a, np.percentile(a[a > 0], 1)))


# ------------------------------------------------------------------ fig 1
def fig_geometry():
    m = 2
    z0 = 18
    a, y = 0, 7                      # 体对角角色对
    ax_idx = [np.arange(int(m * o), n, 2 * m) for n, o in zip(PD.shape, off(a))]
    ay_idx = [np.arange(int(m * o), n, 2 * m) for n, o in zip(PD.shape, off(y))]

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 0.8, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    im = logimg(PD[z0])
    ax.imshow(im, cmap="magma", origin="lower")
    yy, xx = np.meshgrid(ax_idx[1], ax_idx[2], indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=4, c="#39ff14", label=f"input role {a} (m={m})")
    yy, xx = np.meshgrid(ay_idx[1], ay_idx[2], indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=4, c="#00d0ff", label=f"target role {y}")
    ax.set_title(f"PD slice z={z0}: one phase = 16x16 of 64x64\ngreen=8 phases cover 12.5% of voxels",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.4)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1])
    z = z0
    sub = PD[z, 22:38, 22:38]
    ax.imshow(logimg(sub), cmap="magma", origin="lower", interpolation="nearest",
              extent=(-0.5, sub.shape[1] - 0.5, -0.5, sub.shape[0] - 0.5))
    for dz, dy, dx in [off(i) for i in PATS[0]]:
        ax.plot(dx * 2 + 4, dy * 2 + 4, "o", ms=14, mfc="none", mec="#39ff14", mew=2)
    for dz, dy, dx in [off(i) for i in range(8) if i not in PATS[0]]:
        ax.plot(dx * 2 + 4, dy * 2 + 4, "o", ms=14, mfc="none", mec="#00d0ff", mew=2, ls="--")
    ax.add_patch(plt.Rectangle((4 - 2, 4 - 2), 4, 4, fill=False, ec="w", lw=1.2))
    ax.set_title(f"zoom: one (2m)^3 = 4^3 block, 8 corners,\npattern 0 takes 4 (green), target 4 (cyan)",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2])
    ms = [1, 2, 3]
    cov = [100 * 8 / (2 * mm) ** 3 for mm in ms]
    bars = ax.bar([str(mm) for mm in ms], cov, color=["#4c72b0", "#dd8452", "#c44e52"])
    for b, c in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, c * 1.25, f"{c:.1f}%", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1, 400)
    ax.set_xlabel("view spacing m"); ax.set_ylabel("% of volume sampled (8 phases, log)")
    ax.set_title("coverage identity 8/(2m)^3", fontsize=9)

    ax = fig.add_subplot(gs[3])
    views = {mm: [int(PD.shape[0] // (2 * mm) * 2 * mm / (2 * mm)), int(64 // (2 * mm))] for mm in ms}
    rows = [[f"m={mm}", f"{PD.shape[0]//(2*mm)}x{64//(2*mm)}x{64//(2*mm)}",
             f"{mm**3}", f"{cov[i]:.1f}%"] for i, mm in enumerate(ms)]
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["m", "view size", "lattices\nfor full grid", "coverage"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    ax.set_title("cost of increasing m", fontsize=9)

    fig.suptitle("Fig 1  Sampling geometry at m=2: coverage cost and what the network actually sees",
                 fontsize=11)
    fig.savefig(FIG / "fig1_sampling_geometry.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig_leak_signal():
    J = json.loads(Path("data/processed/v3_role_order_m2.json").read_text())
    T = json.loads(Path("data/processed/v3_signal_transfer.json").read_text())

    L = np.zeros((3, 3, 3))
    n = np.zeros((3, 3, 3))
    for seq in ("SA64", "HSA64"):
        for r in J["seq"][seq]["rows"]:
            i = tuple(np.array(r["shift"]) + 1)
            L[i] += r["L"] / 2
            n[i] += 0.5
    L[n == 0] = np.nan

    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.06, 1.9], wspace=0.25)
    for k, dz in enumerate((-1, 0, 1)):
        ax = fig.add_subplot(gs[k])
        im = ax.imshow(L[dz + 1], cmap="viridis", vmin=0, vmax=0.55, origin="lower")
        for (i, j), v in np.ndenumerate(L[dz + 1]):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v < 0.35 else "k", fontsize=9)
        ax.set_title(f"shift dz={dz} (x2 voxels)", fontsize=9)
        ax.set_xticks([0, 1, 2], ["-2", "0", "+2"]); ax.set_yticks([0, 1, 2], ["-2", "0", "+2"])
        ax.set_xlabel("dx (vox)"); ax.set_ylabel("dy (vox)")
    cax = fig.add_subplot(gs[3]); fig.colorbar(im, cax=cax, label="leak L")

    ax = fig.add_subplot(gs[4])
    fam = {"diag": ("#c44e52", "o"), "face": ("#dd8452", "s"), "axis": ("#4c72b0", "^")}
    for r in T["rows"]:
        nz = np.count_nonzero(np.abs(r["vox"]))
        key = {3: "diag", 2: "face", 1: "axis"}[nz]
        c, mk = fam[key]
        ax.scatter(r["L_SA64"], r["r_sig"], s=70, c=c, marker=mk, alpha=0.85,
                   edgecolor="k", linewidth=0.4)
    ax.axvline(0.25, ls="--", c="grey", lw=1); ax.axhline(0.25, ls="--", c="grey", lw=1)
    ax.add_patch(plt.Rectangle((0, 0.25), 0.25, 0.45, fc="green", alpha=0.07, ec="green", ls=":"))
    for lbl, xy, dx, dy in [("diagonal\n(L=0.017, r=0.29)", (0.017, 0.289), 0.03, -0.07),
                            ("y axis\n(L=0.082, r=0.63)", (0.082, 0.631), 0.03, 0.02),
                            ("x axis (banned)\n(L=0.536, r=0.62)", (0.536, 0.618), -0.42, -0.09)]:
        ax.annotate(lbl, xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("noise leak L (real data, m=2)")
    ax.set_ylabel("signal transfer r_sig (simulator)")
    ax.set_title("Fig 2  Two sides disagree: cleanest pair is also the weakest signal path\n"
                 "candidate set = L<=0.25 and r_sig>=0.25 (green box); diagonals/faces/axes by colour",
                 fontsize=9)
    ax.legend(handles=[plt.Line2D([], [], marker=v[1], ls="", c=v[0], label=k)
                       for k, v in fam.items()], fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "fig2_leak_and_signal.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_simulator():
    from v3_simulator import SimCfg, simulate, band_and_pd
    d = simulate(SimCfg(seed=0))
    flow = d["flow"].astype(np.complex64)
    obs = d["X"].astype(np.complex64)
    pd_sig = band_and_pd(flow, d["valid"])[0]
    pd_obs = band_and_pd(obs, d["valid"])[0]
    pd_noise = band_and_pd(obs - flow, d["valid"])[0]
    z = 18
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, img, ttl in zip(axs[:3], (pd_obs, pd_sig, pd_noise),
                            ("observed PD (what we denoise)", "ground truth: blood only",
                             "noise only (static+dynamic+clutter+thermal)")):
        vmin = np.percentile(img[img > 0], 1) if (img > 0).any() else 1
        ax.imshow(np.log10(np.maximum(img[z], vmin)), cmap="magma", origin="lower")
        ax.set_title(ttl, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = axs[3]
    err = 20 * np.log10(np.maximum(pd_obs[z], 1e-9) / np.maximum(pd_sig[z], 1e-9))
    im = ax.imshow(np.clip(err, -20, 20), cmap="coolwarm", origin="lower", vmin=-20, vmax=20)
    ax.set_title("observed vs truth, dB\n(what a perfect denoiser would remove)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="dB")
    fig.suptitle("Fig 3  Simulator after the flow fix: blood signal now actually survives the SVD band filter",
                 fontsize=11)
    fig.savefig(FIG / "fig3_simulator_truth.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_geometry()
    fig_leak_signal()
    fig_simulator()
    print("wrote", *[str(p) for p in sorted(FIG.glob("*.png"))], sep="\n  ")

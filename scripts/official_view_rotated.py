#!/usr/bin/env python3
"""Rotated MIP views + adaptive colorbars, per user feedback.

Changes vs official_view.py:
  - YZ and XZ views rotated: depth axis Z on the HORIZONTAL axis (wider),
    Y / X on the vertical axis.
  - Per-row ADAPTIVE colorbar: each volume gets its own dB window
    [p1 .. p99.99] instead of the fixed official [-20, 0].
    (The fixed window is kept for the official reference row.)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from official_view import (
    AXIS_X, AXIS_Y, AXIS_Z, load_sa64, band_power_doppler_db, log01_to_db,
)


def mips(vol_db):
    """Return the three rotated MIPs: XY, ZY (Z horizontal), ZX (Z horizontal)."""
    return {
        "XY": vol_db.max(axis=0).T,          # rows=Y, cols=X
        "ZY": vol_db.max(axis=1).T,          # rows=Y, cols=Z  (Z horizontal)
        "ZX": vol_db.max(axis=2).T,          # rows=X, cols=Z  (Z horizontal)
    }


def render(rows, out_png, suptitle, adaptive=True, official_clim=(-20.0, 0.0)):
    """rows: list of (title, vol_db). One row per volume, 3 rotated MIP panels."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for ax_row, (title, vol) in zip(axes, rows):
        views = mips(vol)
        extents = {
            "XY": [AXIS_X[0], AXIS_X[-1], AXIS_Y[0], AXIS_Y[-1]],
            "ZY": [AXIS_Z[0], AXIS_Z[-1], AXIS_Y[0], AXIS_Y[-1]],
            "ZX": [AXIS_Z[0], AXIS_Z[-1], AXIS_X[0], AXIS_X[-1]],
        }
        labels = {
            "XY": ("X (mm)", "Y (mm)"),
            "ZY": ("Z (mm)", "Y (mm)"),
            "ZX": ("Z (mm)", "X (mm)"),
        }
        if adaptive:
            vmin, vmax = float(np.percentile(vol, 1.0)), float(np.percentile(vol, 99.99))
            if vmax - vmin < 1e-6:
                vmin, vmax = vol.min(), vol.max() + 1e-6
        else:
            vmin, vmax = official_clim
        for ax, (key, img) in zip(ax_row, views.items()):
            im = ax.imshow(img, origin="lower", cmap="hot", vmin=vmin, vmax=vmax,
                           extent=extents[key], aspect="equal", interpolation="nearest")
            ax.set_xlabel(labels[key][0])
            ax.set_ylabel(labels[key][1])
            ax.set_title(f"{key} (MIP)", fontsize=8)
        ax_row[0].set_title(f"{title}\nXY (MIP) | adaptive [{vmin:.1f}, {vmax:.1f}] dB"
                            if adaptive else f"{title}\nXY (MIP)", fontsize=9)
        fig.colorbar(im, ax=ax_row.tolist(), label="dB", shrink=0.85, pad=0.015)

    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    raw = load_sa64("/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat")
    z, x, y, t = raw.shape
    C = raw.reshape(-1, t).astype(np.complex64)
    G = (C.conj().T @ C).astype(np.complex64)
    evals, V = np.linalg.eigh(G)
    V = V[:, np.argsort(evals)[::-1]]

    pd_blood = band_power_doppler_db(C, V, 60, 350).reshape(z, x, y)
    lo = float(np.percentile(pd_blood, 99.9)) - 60.0

    # 1) Official reference, rotated views, official fixed clim.
    render([("Official band 60-350 (reference)", pd_blood)],
           "runs/official_mip/figure_official_rotated.png",
           "SA64 Power Doppler MIP - rotated views (Z horizontal), official clim [-20,0] dB",
           adaptive=False)

    # 2) Input / B2 / C2 with adaptive per-row colorbars, rotated views.
    def to_db(p):
        db = log01_to_db(np.load(p).astype(np.float32), lo)
        return db - db.max()

    rows = [
        ("Input: official band 60-350", pd_blood),
        ("B2 denoised (random Tetris)", to_db("runs/B2_sa64_blood_random/infer/denoised.npy")),
        ("C2 denoised (SVD-guided Tetris)", to_db("runs/C2_sa64_blood_guided/infer/denoised.npy")),
    ]
    render(rows, "runs/official_mip/figure_denoised_B2C2_adaptive.png",
           "B2 vs C2 - rotated MIP views (Z horizontal), ADAPTIVE per-row colorbars")

    # Also a fixed-clim version with rotated views for strict comparison.
    render(rows, "runs/official_mip/figure_denoised_B2C2_rotated.png",
           "B2 vs C2 - rotated MIP views (Z horizontal), official clim [-20,0] dB",
           adaptive=False)


if __name__ == "__main__":
    main()

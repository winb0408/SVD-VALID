#!/usr/bin/env python3
"""Phase 1 QC figure: what the constructed observation pairs look like.

Renders one fold's P0 / P1 / P2 volumes as axial maximum-intensity projections in
the official display convention (dB relative to the pair maximum, clim [-20, 0],
axes in mm).  This is supporting evidence only -- the project rule is that an
image looking right is never the acceptance criterion, and the numeric QC in
`pair_qc.json` is what Phase 1 is judged on.  The point of the figure is the
opposite one: to make it visible when a pair does *not* look like blood, so a
purely numeric agreement score cannot quietly pass a pair of noise fields.

Usage:
  python scripts/phase1_pair_figure.py --sequence SA64 \
      --fold sa64_int3_c0a1b2 --rule proportional
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOXEL = 0.15
CLIM = (-20.0, 0.0)
SHAPE = (37, 64, 64)
AXIS_X = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Y = np.arange(-4.725, 4.725 + 1e-9, VOXEL)


def to_db(d, vmax):
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(d, 1e-30) / vmax)


def mip_z(d3):
    """Axial MIP over Z -> a 64 x 64 image."""
    return d3.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="SA64")
    ap.add_argument("--fold", default="sa64_int3_c0a1b2")
    ap.add_argument("--rule", default="proportional")
    ap.add_argument("--root", default="data/processed/crossfit")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.root) / args.sequence / args.fold
    valid = np.load(Path("data/processed/official") / args.sequence / "valid_mask.npy").reshape(-1)

    vols = {}
    for name in ("P0_A", "P0_B", f"P1_{args.rule}_A", f"P1_{args.rule}_B",
                 f"P2_{args.rule}_A", f"P2_{args.rule}_B"):
        vols[name] = np.load(d / f"{name}.npy").astype(np.float64).reshape(SHAPE)

    rows = [("P0 (no SVD)", "P0_A", "P0_B"),
            ("P1 (shared P_C)", f"P1_{args.rule}_A", f"P1_{args.rule}_B"),
            ("P2 (own basis each)", f"P2_{args.rule}_A", f"P2_{args.rule}_B")]

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.4 * len(rows)))
    for i, (label, ka, kb) in enumerate(rows):
        pooled = np.concatenate([vols[ka][valid.reshape(SHAPE)],
                                 vols[kb][valid.reshape(SHAPE)]])
        vmax = np.percentile(pooled, 99.9)
        ia = to_db(mip_z(vols[ka]), vmax)
        ib = to_db(mip_z(vols[kb]), vmax)
        for ax, img, tag in ((axes[i, 0], ia, f"A  {label}"),
                             (axes[i, 1], ib, f"B  {label}")):
            ax.imshow(img, cmap="hot", clim=CLIM, origin="lower",
                      extent=[AXIS_X[0], AXIS_X[-1], AXIS_Y[0], AXIS_Y[-1]])
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("X (mm)", fontsize=7)
            ax.set_ylabel("Y (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
        diff = ia - ib
        im = axes[i, 2].imshow(diff, cmap="coolwarm", clim=(-6, 6), origin="lower",
                               extent=[AXIS_X[0], AXIS_X[-1], AXIS_Y[0], AXIS_Y[-1]])
        axes[i, 2].set_title(f"A - B (dB), {label}", fontsize=9)
        axes[i, 2].set_xlabel("X (mm)", fontsize=7)
        axes[i, 2].tick_params(labelsize=6)
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    fig.suptitle(f"{args.sequence}  {args.fold}  rule={args.rule}\n"
                 f"axial MIP, dB re pooled 99.9th pct of valid voxels, clim {CLIM}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else (
        Path("reports/figures") / f"phase1_pair_{args.sequence}_{args.fold}_{args.rule}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

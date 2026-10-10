#!/usr/bin/env python3
"""Same-scale input/denoised comparison figures (L22 fix).

Supersedes the per-row-normalised, per-row-adaptive pseudo-dB figures
(official_view_rotated.py) for the contrast-remap pipeline. Their defect was
that every row had its own baseline and its own percentile window, so a vessel
could look dimmer or thicker purely because of the display. Here:

  * every row uses ONE fixed window -- no per-row max subtraction, no per-row
    percentiles, one shared colorbar;
  * both mid-slices and MIPs are shown per row (MIP is dominated by a few
    brightest voxels and hides whether thin vessels stayed connected);
  * two domains are produced:
      remap  : the [0, 1] domain the network actually sees, vmin=0, vmax=1
      db     : physical dB re the input max, fixed clim (-20, 0), i.e. the
               official display window. Requires the prepare_svd metadata
               (remap_db_range) to invert the quantile remap.

Honest caveat baked into the dB figure: the contrast remap maps the p1
percentile to 0, so every voxel weaker than the remap floor collapses onto one
value. The floor is annotated in the figure -- the dark-region plateau is a
representation limit, not a denoising result.

Usage
-----
  python scripts/visualize_remap.py --sequence sa64 \
      --processed data/processed/sa64_blood_remap \
      --models input=data/processed/sa64_blood_remap/power_doppler.npy \
               B3=runs/B3_sa64_remap_random/infer/denoised.npy \
               C3=runs/C3_sa64_remap_guided/infer/denoised.npy \
      --out-dir runs/official_mip
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from official_view import AXIS_X, AXIS_Y, AXIS_Z  # noqa: E402

DB_CLIM = (-20.0, 0.0)
REMAP_CLIM = (0.0, 1.0)


def mid_slices(vol):
    """Three orthogonal mid-plane slices, Z horizontal where Z is an axis."""
    z, x, y = vol.shape
    return {
        "XY": (vol[z // 2], ("X (mm)", "Y (mm)")),
        "ZY": (vol[:, x // 2, :].T, ("Z (mm)", "Y (mm)")),
        "ZX": (vol[:, :, y // 2].T, ("Z (mm)", "X (mm)")),
    }


def mips(vol):
    return {
        "XY": (vol.max(axis=0).T, ("X (mm)", "Y (mm)")),
        "ZY": (vol.max(axis=1).T, ("Z (mm)", "Y (mm)")),
        "ZX": (vol.max(axis=2).T, ("Z (mm)", "X (mm)")),
    }


def extents_for(key):
    return {
        "XY": (AXIS_X[0], AXIS_X[-1], AXIS_Y[0], AXIS_Y[-1]),
        "ZY": (AXIS_Z[0], AXIS_Z[-1], AXIS_Y[0], AXIS_Y[-1]),
        "ZX": (AXIS_Z[0], AXIS_Z[-1], AXIS_X[0], AXIS_X[-1]),
    }[key]


def render(rows, out_png, suptitle, vmin, vmax, cbar_label, zline=None):
    """rows: list of (label, volume-in-display-domain). Six panels per row."""
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(19, 2.9 * n))
    if n == 1:
        axes = axes[None, :]
    groups = [("mid-slice", mid_slices), ("MIP", mips)]
    im = None
    for ax_row, (label, vol) in zip(axes, rows):
        col = 0
        for gname, fn in groups:
            views = fn(vol)
            for key, (img, (xl, yl)) in views.items():
                ax = ax_row[col]
                im = ax.imshow(img, origin="lower", cmap="hot", vmin=vmin,
                               vmax=vmax, extent=extents_for(key),
                               aspect="equal", interpolation="nearest")
                if zline is not None:
                    for axx in (ax,):
                        pass
                ax.set_xlabel(xl, fontsize=7)
                ax.set_ylabel(yl, fontsize=7)
                ax.set_title(f"{key} {gname}", fontsize=7)
                ax.tick_params(labelsize=6)
                col += 1
        ax_row[0].text(-0.32, 0.5, label, transform=ax_row[0].transAxes,
                       rotation=90, va="center", ha="center", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=cbar_label,
                 shrink=0.85, pad=0.01, fraction=0.02)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def to_physical_db(vol, p_lo, p_hi, ref_max_db):
    """Invert the contrast remap (clipped monotone linear) and go dB re ref.

    Remapped values in [0, 1] correspond to db in [p_lo, p_hi]. Structural zeros
    stay at their original value (there is no information to invert them), and
    values outside [0, 1] (raw model output is unconstrained) extrapolate
    linearly -- they are shown as such rather than clipped, so out-of-range
    output remains visible.
    """
    db = p_lo + vol.astype(np.float64) * (p_hi - p_lo)
    return db - ref_max_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True, help="Label, e.g. sa64 / hsa64.")
    ap.add_argument("--processed", required=True,
                    help="prepare_svd output dir (for metadata remap_db_range).")
    ap.add_argument("--models", nargs="+", required=True, help="NAME=path.npy")
    ap.add_argument("--out-dir", default="runs/official_mip")
    ap.add_argument("--tag", default="remap")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.processed) / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    p_lo, p_hi = meta["remap_db_range"]
    remap = bool(meta.get("contrast_remap", False))
    if not remap:
        raise SystemExit(f"{meta_path} has contrast_remap=false; this script "
                         f"assumes the contrast-remap pipeline.")

    rows = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"--models entries must be NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        vol = np.load(path).astype(np.float32)
        rows.append((name, vol))
        print(f"{name:>6}: min {vol.min():+.4f}  max {vol.max():+.4f}  "
              f"oob {(np.abs(vol - np.clip(vol, 0, 1)) > 0).mean():.3%}  "
              f"({path})")

    ref_max_db = p_lo + float(rows[0][1].max()) * (p_hi - p_lo)

    # --- Figure 1: the domain the network sees, one shared fixed window -------
    render(rows, out_dir / f"figure_{args.tag}_{args.sequence}_01.png",
           f"{args.sequence.upper()} - input vs denoised, network domain [0,1] "
           f"(FIXED window for every row and panel; mid-slice + MIP)",
           REMAP_CLIM[0], REMAP_CLIM[1], "remap value [0,1] (shared)")

    # --- Figure 2: physical dB, official [-20,0] window, same for all rows ----
    db_rows = [(n, to_physical_db(v, p_lo, p_hi, ref_max_db)) for n, v in rows]
    floor_db = p_lo - ref_max_db
    render(db_rows, out_dir / f"figure_{args.tag}_{args.sequence}_02_db.png",
           f"{args.sequence.upper()} - same comparison in physical dB re input max "
           f"(FIXED clim {DB_CLIM[0]:.0f}..{DB_CLIM[1]:.0f} dB, official window)\n"
           f"remap floor = {floor_db:.2f} dB: everything weaker sits on one plateau "
           f"(representation limit, not a denoising result)",
           DB_CLIM[0], DB_CLIM[1], "Power Doppler (dB re input max)")


if __name__ == "__main__":
    main()

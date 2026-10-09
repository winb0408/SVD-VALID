#!/usr/bin/env python3
"""Faithful Python port of the official Dryad visualization code
(main_supplementary_figure_2.m) for the SA64 acquisition.

Official recipe (per the .m file):
  - blood band SVDcut1_400=60 .. SVDcut2_400=350 (modes 0-60 = tissue/clutter,
    350-400 = noise, in 0-based indexing 60..350 exclusive end 350)
  - PowerDoppler = sum over frames of |filtered|^2
  - dB = 10*log10(PD / max(PD))
  - display: maximum intensity projections along Z/X/Y, colormap hot,
    clim [-20, 0], equal aspect, axes in mm (voxel 0.15 mm).

This script additionally renders our pipeline volumes (band 10-80 input,
full-band input, and the A/B/C denoised outputs) in the SAME official style
so the appearances can be compared side by side.

Outputs (runs/official_mip/):
  figure_official.png     : official band 60-350, exactly the official view
  figure_inputs.png       : official band vs our band 10-80 vs full band
  figure_denoised_ABC.png : A/B/C denoised outputs in official style
  official_pd_db.npy      : official power doppler volume in dB
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VOXEL = 0.15  # mm
AXIS_X = np.arange(-4.725, 4.725 + 1e-9, VOXEL)   # 64
AXIS_Y = np.arange(-4.725, 4.725 + 1e-9, VOXEL)   # 64
AXIS_Z = np.arange(2.025, 7.4525 + 1e-9, VOXEL)   # 37
CLIM = (-20.0, 0.0)


def load_sa64(path):
    """Load Data_SA64.mat (v7.3 HDF5, [T,X,Y,Z] reversed, compound complex)."""
    with h5py.File(path, "r") as f:
        key = [k for k in f.keys() if not k.startswith("__")][0]
        raw = f[key][:]
    raw = raw["real"] + 1j * raw["imag"]
    raw = np.transpose(raw, axes=tuple(range(raw.ndim))[::-1])  # -> [Z,X,Y,T]
    return raw


def band_power_doppler_db(C, V, lo, hi):
    """Official: project onto band [lo,hi), sum |.|^2 over frames, dB re max."""
    Q = V[:, lo:hi]
    filtered = (C @ Q) @ Q.conj().T
    pd = np.sum(np.abs(filtered) ** 2, axis=-1)
    return 10.0 * np.log10(np.maximum(pd, 1e-12) / pd.max())


def log01_to_db(x, lo_db):
    """Invert prepare_svd's log compression: db = x*60 + lo (clip domain)."""
    return x * 60.0 + lo_db


def mip_panels(ax_row, vol_db, title):
    """Three MIP views per the official layout: XY (max over Z), YZ (max over X), XZ (max over Y)."""
    z, x, y = vol_db.shape
    assert (z, x, y) == (len(AXIS_Z), len(AXIS_X), len(AXIS_Y)), vol_db.shape
    views = [
        (vol_db.max(axis=0).T, AXIS_Y, AXIS_X, "XY (MIP over Z)"),
        (vol_db.max(axis=1).T, AXIS_Y, AXIS_Z, "YZ (MIP over X)"),
        (vol_db.max(axis=2).T, AXIS_X, AXIS_Z, "XZ (MIP over Y)"),
    ]
    for ax, (img, ax_h, ax_v, label) in zip(ax_row, views):
        im = ax.imshow(img, origin="lower", cmap="hot", vmin=CLIM[0], vmax=CLIM[1],
                       extent=[ax_h[0], ax_h[-1], ax_v[0], ax_v[-1]],
                       aspect="equal", interpolation="nearest")
        ax.set_xlabel(f"{label.split()[0][0]} (mm)")
        ax.set_ylabel(f"{label.split()[0][1]} (mm)")
        ax.set_title(label, fontsize=8)
    ax_row[0].set_ylabel("X (mm)")
    ax_row[0].set_title(f"{title}\nXY (MIP over Z)", fontsize=9)
    ax_row[1].set_ylabel("Z (mm)")
    ax_row[2].set_ylabel("Z (mm)")
    return im


def render(rows, out_png, suptitle):
    """rows: list of (title, vol_db). One figure row per volume."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 3.1 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for ax_row, (title, vol) in zip(axes, rows):
        mip_panels(ax_row, vol, title)
    im = axes[-1][-1].images[-1]
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Power Doppler (dB)",
                 shrink=0.9, pad=0.02)
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat")
    ap.add_argument("--out", default="runs/official_mip")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_sa64(args.input)
    z, x, y, t = raw.shape
    C = raw.reshape(-1, t).astype(np.complex64)
    G = (C.conj().T @ C).astype(np.complex64)
    evals, V = np.linalg.eigh(G)
    V = V[:, np.argsort(evals)[::-1]]

    # Official band 60-350 (0-based, exclusive end) and our bands.
    pd_official = band_power_doppler_db(C, V, 60, 350).reshape(z, x, y)
    pd_ourband = band_power_doppler_db(C, V, 10, 80).reshape(z, x, y)
    pd_full = band_power_doppler_db(C, V, 0, t).reshape(z, x, y)
    np.save(out / "official_pd_db.npy", pd_official.astype(np.float32))

    # Figure 1: exactly the official view.
    render([("Official band 60-350", pd_official)], out / "figure_official.png",
           "SA64 Power Doppler MIP - official visualization (band 60-350, clim [-20,0] dB)")

    # Figure 2: official band vs the bands our pipeline used.
    render([
        ("Official band 60-350 (reference)", pd_official),
        ("Our band 10-80 (B/C input)", pd_ourband),
        ("Full band 0-400 (A input)", pd_full),
    ], out / "figure_inputs.png",
        "Input volume comparison - official recipe applied to each rank band")

    # Figure 3: our A/B/C denoised outputs in the official style.
    # Denoised volumes are in prepare_svd's log [0,1] domain; invert with the
    # same lo = ref(99.9 pct) - 60 dB used at preparation time (recomputed here).
    def lo_db(pd_db):
        return float(np.percentile(pd_db, 99.9)) - 60.0

    rows = []
    for name, pd_ref, denoised_path in [
        ("A denoised (input: full band)", pd_full, "runs/A_sa64_full_random/infer/denoised.npy"),
        ("B denoised (input: band 10-80)", pd_ourband, "runs/B_sa64_svd_random/infer/denoised.npy"),
        ("C denoised (input: band 10-80)", pd_ourband, "runs/C_sa64_svd_guided/infer/denoised.npy"),
    ]:
        den = np.load(denoised_path).astype(np.float32)
        vol_db = log01_to_db(den, lo_db(pd_ref))
        vol_db -= vol_db.max()  # dB re own max, like the official display
        rows.append((name, vol_db))
    render(rows, out / "figure_denoised_ABC.png",
           "A/B/C denoised outputs - official MIP style (dB re own max, clim [-20,0] dB)")


if __name__ == "__main__":
    main()

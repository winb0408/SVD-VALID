#!/usr/bin/env python3
"""Phase 0 / T0-2: numpy port of the official Dryad processing chain.

`main_supplementary_figure_2.m` (shipped with the Dryad dataset) is MATLAB and
no MATLAB or Octave is available on this machine, so this is a line-by-line
port, not a reimplementation.  The official recipe is:

    Casorati = reshape(Data, [V, T])              % all T frames
    Cov      = Casorati' * Casorati               % T x T temporal Gram
    [U, S]   = eig(Cov);  U = fliplr(U)           % descending eigen-order
    M        = Casorati * U
    filtered = M(:, c1:c2) * U(:, c1:c2)'         % 1-BASED, inclusive: 60:350
    PD       = sum(abs(filtered).^2, 4)           % SUM over frames, not mean
    PD_dB    = 10*log10(PD / max(PD(:)))          % global-max normalisation

Known deviations of this repository's older pipeline (`scripts/prepare_svd.py`)
that this script exists to quantify:

  1. mode count -- the official keeps 1-based 60:350 = 291 modes; the older
     pipeline used the 0-based slice V[:, 60:350], i.e. 290 modes.  This script
     uses the official convention when `--faithful` (default).
  2. frame reduction -- official `sum`, older pipeline `mean`.  Invariant after
     global-max normalisation of the dB volume, but NOT invariant between two
     observation groups of different length (see Project_plan.md 1.3 item 6).
  3. the "350-400 residual band" is a local construct for noise estimation; the
     official script has no such band and simply discards everything above 350.

Outputs per sequence (float32, shape [Z, X, Y]):
    pd_linear.npy  summed blood-band power, arbitrary units
    pd_db.npy      10*log10(PD / max(PD)), i.e. 0 dB at the global maximum
    pd_remap.npy   quantile remap of the non-zero dB values onto [0, 1]
                   (project convention for network input; NOT official)
and metadata.json with the singular spectrum and band-energy fractions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.iq_io import load_iq  # noqa: E402

# Per-sequence official settings, from main_supplementary_figure_2.m.
SEQUENCE_SETTINGS = {
    "SA64":   {"variable": "Data_SA64",   "cut1": 60,  "cut2": 350, "sum_arrays": False},
    "HSA64":  {"variable": "Data_HSA64",  "cut1": 60,  "cut2": 350, "sum_arrays": False},
    "PW3232": {"variable": "Data_PW3232", "cut1": 60,  "cut2": 350, "sum_arrays": True},
    "PW1616": {"variable": "Data_PW1616", "cut1": 120, "cut2": 750, "sum_arrays": True},
}


def svd_blood_band(x, cut1, cut2, faithful=True):
    """Official blood-band SVD filter.

    `x` is [Z, X, Y, T] complex.  Returns the filtered volume and the descending
    singular-value spectrum (energies).  `cut1`/`cut2` are the official 1-based
    inclusive MATLAB indices; with `faithful=False` they are used as the 0-based
    half-open slice instead, reproducing the older pipeline for comparison.
    """
    z, a, b, t = x.shape
    casorati = x.reshape(-1, t).astype(np.complex64, copy=False)

    gram = casorati.conj().T @ casorati
    evals, vecs = np.linalg.eigh(gram)          # ascending
    order = np.argsort(evals)[::-1]
    evals, vecs = evals[order], vecs[:, order]  # descending, = MATLAB fliplr

    lo, hi = (cut1 - 1, cut2) if faithful else (cut1, cut2)
    if not 0 <= lo < hi <= t:
        raise ValueError(f"Invalid blood band [{lo},{hi}) for T={t}")
    modes = vecs[:, lo:hi]

    m = casorati @ modes
    filtered = m @ modes.conj().T
    return filtered.reshape(z, a, b, t), evals


def remap_valid(db, valid, floor_pct=1.0, ceil_pct=99.9):
    """Quantile remap of the *valid* dB values onto [0, 1]; dead voxels stay 0."""
    out = np.zeros_like(db, dtype=np.float32)
    if not valid.any():
        return out
    vals = db[valid]
    lo = float(np.percentile(vals, floor_pct))
    hi = float(np.percentile(vals, ceil_pct))
    out[valid] = np.clip((vals - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return out, (lo, hi)


def run_one(path, name, out_root, faithful=True):
    settings = SEQUENCE_SETTINGS[name]
    x = load_iq(path, settings["variable"])
    if settings["sum_arrays"] and x.ndim == 5:
        # Official: squeeze(sum(Data, 5)) -- coherent sum over emitting arrays.
        x = np.squeeze(x.sum(axis=-1))
    if x.ndim != 4:
        raise ValueError(f"{name}: expected [Z,X,Y,T], got {x.shape}")

    filtered, evals = svd_blood_band(x, settings["cut1"], settings["cut2"], faithful)
    frames = int(x.shape[3])
    del x

    linear = np.sum(np.abs(filtered) ** 2, axis=3)
    del filtered

    # Dead voxels: Casorati rows that are identically zero because they fall
    # outside the beamformed sector (near-field Z slices).  They are zero in the
    # RAW IQ too -- this is acquisition geometry, not an artifact of filtering.
    # Being identically zero in every frame group, leaving them in any
    # correlation / agreement metric inflates it for free; all pairwise
    # statistics must be restricted to `valid`.
    valid = linear > 0

    peak = float(linear.max())
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(linear / peak)   # -inf at dead voxels, as in MATLAB
    remap, (lo_pct, hi_pct) = remap_valid(db, valid)

    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pd_linear.npy", linear.astype(np.float32))
    np.save(out / "pd_db.npy", db.astype(np.float32))
    np.save(out / "pd_remap.npy", remap)
    np.save(out / "valid_mask.npy", valid)

    total = float(evals.sum())
    lo, hi = (settings["cut1"] - 1, settings["cut2"]) if faithful else (settings["cut1"], settings["cut2"])
    kept = float(evals[lo:hi].sum())
    top = float(evals[:lo].sum())          # clutter / tissue modes above the band
    tail = float(evals[hi:].sum())         # everything the official script discards
    meta = {
        "sequence": name,
        "source": str(path),
        "variable": settings["variable"],
        "faithful_official_indices": bool(faithful),
        "blood_band_1based": [settings["cut1"], settings["cut2"]],
        "blood_band_0based_slice": [lo, hi],
        "n_modes_kept": int(hi - lo),
        "shape_zyx": [int(v) for v in linear.shape],
        "frames_used": frames,
        "reduce": "sum over frames",
        "db_normalisation": "10*log10(PD / max(PD))",
        "display_clim_db": [-20, 0],
        "energy_fraction": {
            "clutter_above_band": top / total,
            "blood_band_kept": kept / total,
            "discarded_below_band": tail / total,
            "blood_band_percent": 100.0 * kept / total,
        },
        "spectrum_top5": [float(v) for v in evals[:5]],
        "valid_mask": {
            "n_valid": int(valid.sum()),
            "n_dead": int((~valid).sum()),
            "dead_fraction": float((~valid).mean()),
            "origin": "acquisition geometry (near-field Z outside the beamformed sector); "
                      "identical in raw IQ and in the filtered volume",
            "required_by": "all pairwise / agreement statistics must use this mask",
        },
        "remap": {"floor_pct": 1.0, "ceil_pct": 99.9,
                  "db_range_valid": [lo_pct, hi_pct]},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"{name}: kept {hi - lo} modes = {meta['energy_fraction']['blood_band_percent']:.4f}% "
          f"of total energy (clutter {100 * top / total:.4f}%) -> {out}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ssd1/wjb/Data/Dyrad/data")
    ap.add_argument("--sequences", nargs="*", default=["SA64", "HSA64"])
    ap.add_argument("--out", default="data/processed/official")
    ap.add_argument("--not-faithful", action="store_true",
                    help="Use the 0-based half-open band (older pipeline) instead "
                         "of the official 1-based inclusive band, for comparison.")
    args = ap.parse_args()

    root = Path(args.data_root)
    for name in args.sequences:
        run_one(root / name / f"Data_{name}.mat", name, args.out,
                faithful=not args.not_faithful)


if __name__ == "__main__":
    main()

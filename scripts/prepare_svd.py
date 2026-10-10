#!/usr/bin/env python3
"""Prepare a volumetric ultrasound acquisition for SVD-VALID.

Expected input layout (MATLAB convention, as documented by the Dryad
RCA/CMUT dataset): [Z, X, Y, T] or [Z, X, Y, T, EmittingArray].

MAT files of both v7.2-and-below (scipy) and v7.3/HDF5 (h5py) are
supported. Note that h5py returns MATLAB arrays with reversed axis
order and complex numbers as compound fields, both of which are
handled here.

The SVD is computed economically through the T x T temporal Gram
matrix, so the full spatial U matrix is never materialised.

Outputs (all float32, shape [Z, X, Y]):
  power_doppler.npy : log-compressed SVD-filtered power-Doppler volume in [0, 1]
  signal_map.npy    : RMS amplitude of the retained (blood) rank band
  noise_map.npy     : RMS amplitude of the high-rank residual band
  metadata.json     : parameters and shapes
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _mat_to_numpy(raw):
    """Convert an h5py-read MATLAB array (reversed axes, compound complex) to numpy."""
    if raw.dtype.names is not None:
        fields = set(raw.dtype.names)
        if not {"real", "imag"} <= fields:
            raise ValueError(f"Unsupported compound dtype fields: {raw.dtype.names}")
        raw = raw["real"] + 1j * raw["imag"]
    return raw


def load_array(path, variable=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        z = np.load(path)
        if variable is None:
            variable = list(z.keys())[0]
        return z[variable]
    if suffix == ".mat":
        try:
            mat = loadmat(str(path))
            if variable is None:
                keys = [k for k in mat.keys() if not k.startswith("__")]
                if len(keys) != 1:
                    raise ValueError(f"Pass --variable. Found variables: {keys}")
                variable = keys[0]
            return np.asarray(mat[variable])
        except NotImplementedError:
            # MATLAB v7.3 files are HDF5; read with h5py instead.
            import h5py

            with h5py.File(str(path), "r") as f:
                if variable is None:
                    keys = [k for k in f.keys() if not k.startswith("__")]
                    if len(keys) != 1:
                        raise ValueError(f"Pass --variable. Found variables: {keys}")
                    variable = keys[0]
                raw = f[variable][:]
            raw = _mat_to_numpy(np.asarray(raw))
            # h5py reverses MATLAB axis order.
            return np.transpose(raw, axes=tuple(range(raw.ndim))[::-1])
    raise ValueError(f"Unsupported input: {path}")


def normalize_shape(x, emitting_array, frames, frame_start=0, frame_step=1):
    x = np.asarray(x)
    if x.ndim == 5:
        x = x[..., emitting_array]
    if x.ndim != 4:
        raise ValueError(f"Expected [Z,X,Y,T], got {x.shape}")
    if frame_start or frame_step != 1:
        x = x[..., frame_start::frame_step]
    if frames > 0:
        x = x[..., :frames]
    return x


def project_band(C, V, lo, hi):
    lo = max(0, int(lo))
    hi = min(V.shape[1], int(hi))
    if lo >= hi:
        raise ValueError(f"Invalid rank band [{lo},{hi})")
    Q = V[:, lo:hi]
    return (C @ Q) @ Q.conj().T


def rms_volume(C, shape):
    return np.sqrt(np.mean(np.abs(C) ** 2, axis=-1)).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--variable")
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--frame-start", type=int, default=0,
                    help="First frame index (after loading, before --frames cap).")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="Frame stride; 2 with --frame-start 0/1 selects even/odd "
                         "frames for temporally disjoint input/reference splits.")
    ap.add_argument("--rank-low", type=int, default=10,
                    help="First retained (blood) singular mode index.")
    ap.add_argument("--rank-high", type=int, default=80,
                    help="Exclusive end of the retained (blood) band.")
    ap.add_argument("--noise-high", type=int, default=-1,
                    help="Exclusive end of the residual/noise band. "
                         "-1 (default) uses everything above --rank-high.")
    ap.add_argument("--dynamic-range-db", type=float, default=60.0)
    ap.add_argument("--contrast-remap", action="store_true",
                    help="Remap non-zero voxels' dB distribution onto [0,1] via "
                         "quantile mapping (Plan 1: keep structural zeros at 0, "
                         "give vessel contrast the full dynamic range).")
    ap.add_argument("--remap-floor", type=float, default=1.0,
                    help="Percentile of the non-zero dB distribution mapped to 0.")
    ap.add_argument("--remap-ceil", type=float, default=99.9,
                    help="Percentile of the non-zero dB distribution mapped to 1.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = normalize_shape(
        load_array(args.input, args.variable),
        args.emitting_array,
        args.frames,
        args.frame_start,
        args.frame_step,
    )

    z, x, y, t = raw.shape
    dtype = np.complex64 if np.iscomplexobj(raw) else np.float32
    C = raw.reshape(-1, t).astype(dtype, copy=False)

    # Economic SVD via the temporal Gram matrix.
    G = (C.conj().T @ C).astype(np.complex64 if np.iscomplexobj(C) else np.float32)
    evals, V = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, V = evals[order], V[:, order]

    rank_low = args.rank_low
    rank_high = min(args.rank_high, t)
    if rank_low >= rank_high:
        raise ValueError(f"Retained band [{rank_low},{rank_high}) is empty; "
                         f"check --rank-low/--rank-high (T={t}).")
    if args.noise_high < 0:
        noise_low, noise_high = rank_high, t
    else:
        noise_low, noise_high = rank_high, min(args.noise_high, t)

    signal_C = project_band(C, V, rank_low, rank_high)
    if noise_low < noise_high:
        noise_C = project_band(C, V, noise_low, noise_high)
    else:
        noise_C = np.zeros_like(signal_C)

    shape3d = (z, x, y)
    signal_map = rms_volume(signal_C, shape3d)
    noise_map = rms_volume(noise_C, shape3d)
    power = np.mean(np.abs(signal_C) ** 2, axis=-1).reshape(shape3d)

    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    if args.contrast_remap:
        # Plan 1: quantile-remap the dB distribution of NON-ZERO voxels onto [0,1].
        # Structural zeros (no blood band energy) stay at 0; vessel contrast
        # occupies the full dynamic range instead of the top ~0.1 of it.
        zero_mask = power <= 0.0
        db_nz = db[~zero_mask]
        p_lo = float(np.percentile(db_nz, args.remap_floor))
        p_hi = float(np.percentile(db_nz, args.remap_ceil))
        log_power = np.zeros_like(db, dtype=np.float32)
        log_power[~zero_mask] = np.clip(
            (db[~zero_mask] - p_lo) / max(p_hi - p_lo, 1e-12), 0.0, 1.0
        )
        lo, ref = p_lo, p_hi
    else:
        ref = float(np.percentile(db, 99.9))
        lo = ref - args.dynamic_range_db
        log_power = np.clip((db - lo) / args.dynamic_range_db, 0.0, 1.0).astype(np.float32)

    np.save(out / "power_doppler.npy", log_power)
    np.save(out / "signal_map.npy", signal_map.astype(np.float32))
    np.save(out / "noise_map.npy", noise_map.astype(np.float32))

    meta = {
        "input": str(Path(args.input)),
        "variable": args.variable,
        "emitting_array": int(args.emitting_array),
        "frame_start": int(args.frame_start),
        "frame_step": int(args.frame_step),
        "shape_zyxt": [int(z), int(x), int(y), int(t)],
        "rank_low": int(rank_low),
        "rank_high": int(rank_high),
        "noise_band": [int(noise_low), int(noise_high)],
        "log_dynamic_range_db": float(args.dynamic_range_db),
        "contrast_remap": bool(args.contrast_remap),
        "remap_db_range": [float(lo), float(ref)],
        "eigenvalue_top5": [float(v) for v in evals[:5]],
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

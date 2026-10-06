#!/usr/bin/env python3
"""Prepare a volumetric ultrasound acquisition for SVD-VALID.

Expected input:
  [Z, X, Y, T, A] or [Z, X, Y, T]

The implementation uses the T x T temporal Gram matrix instead of forming
the full spatial U matrix. This is intentionally economical for first tests.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_array(path, variable=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        if variable is None:
            variable = list(z.keys())[0]
        return z[variable]
    if path.suffix.lower() == ".mat":
        mat = loadmat(path)
        if variable is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            if len(keys) != 1:
                raise ValueError(f"Pass --variable. Found variables: {keys}")
            variable = keys[0]
        return mat[variable]
    raise ValueError(f"Unsupported input: {path}")


def normalize_shape(x, emitting_array, frames):
    x = np.asarray(x)
    if x.ndim == 5:
        x = x[..., emitting_array]
    if x.ndim != 4:
        raise ValueError(f"Expected [Z,X,Y,T], got {x.shape}")
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


def rms_volume(C):
    return np.sqrt(np.mean(np.abs(C) ** 2, axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--variable")
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--rank-low", type=int, default=10)
    ap.add_argument("--rank-high", type=int, default=80)
    ap.add_argument("--noise-high", type=int, default=-1)
    ap.add_argument("--dynamic-range-db", type=float, default=60.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = normalize_shape(
        load_array(args.input, args.variable),
        args.emitting_array,
        args.frames,
    )

    z, x, y, t = raw.shape
    dtype = np.complex64 if np.iscomplexobj(raw) else np.float32
    C = raw.reshape(-1, t).astype(dtype, copy=False)

    # Econonomic SVD via the temporal Gram matrix.
    G = C.conj().T @ C
    evals, V = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    V = V[:, order]

    rank_high = min(args.rank_high, t)
    noise_high = t if args.noise_high < 0 else min(args.noise_high, t)

    signal_C = project_band(C, V, args.rank_low, rank_high)
    noise_C = project_band(C, V, noise_high, t) if noise_high > args.rank_low else np.zeros_like(C)

    signal_map = rms_volume(signal_C).reshape(z, x, y)
    noise_map = rms_volume(noise_C).reshape(z, x, y)
    power = np.sqrt(np.mean(np.abs(signal_C) ** 2, axis=-1)).reshape(z, x, y)

    db = 20.0 * np.log10(np.maximum(power, 1e-8))
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
        "shape_zyxt": [int(z), int(x), int(y), int(t)],
        "rank_low": int(args.rank_low),
        "rank_high": int(rank_high),
        "noise_high": int(noise_high),
        "log_dynamic_range_db": float(args.dynamic_range_db),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

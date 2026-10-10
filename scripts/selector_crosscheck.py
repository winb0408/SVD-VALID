#!/usr/bin/env python3
"""Selector cross-check: is the guided choice a property of the noise field, or
of the particular frames the signal/noise maps were computed from?

The SVD-guided selector runs on prepared signal/noise maps, i.e. on band-energy
volumes averaged over whatever frames the dataset was built from. If the
selector's per-patch choice is really detecting spatial noise structure, then a
selector fitted on the EVEN frames should agree with one fitted on the ODD
frames, and the pattern it picks should still decorrelate the noise band on the
frames it never saw.

Procedure (SA64, contrast-remap pipeline, even/odd = --frame-step 2):
  1. Build band-power volumes separately from even frames and from odd frames.
  2. On a window grid, select the pattern from the even maps and from the odd
     maps independently (same selector as training).
  3. Report agreement, and evaluate rho_noise / rho_signal on the HELD-OUT
     frame half's band power for the crossed-over choice.

Outputs: <out>/selector_crosscheck.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_stats import (  # noqa: E402
    band_power_volume,
    block_bootstrap_ci,
    patch_stats,
    sign_flip_permutation_test,
    summarize_independent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.sampling import select_guided_pattern  # noqa: E402


def slices_on_grid(shape, win, stride):
    z, x, y = shape
    zs = list(range(0, z - win[0] + 1, stride[0]))
    xs = list(range(0, x - win[1] + 1, stride[1]))
    ys = list(range(0, y - win[2] + 1, stride[2]))
    return [(slice(z0, z0 + win[0]), slice(x0, x0 + win[1]), slice(y0, y0 + win[2]))
            for z0 in zs for x0 in xs for y0 in ys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat")
    ap.add_argument("--even", default="data/processed/sa64_even_remap")
    ap.add_argument("--odd", default="data/processed/sa64_odd_remap")
    ap.add_argument("--rank-low", type=int, default=60)
    ap.add_argument("--rank-high", type=int, default=350)
    ap.add_argument("--noise-low", type=int, default=350)
    ap.add_argument("--noise-high", type=int, default=400)
    ap.add_argument("--window", nargs=3, type=int, default=[16, 16, 16])
    ap.add_argument("--stride", nargs=3, type=int, default=[16, 16, 16],
                    help="Non-overlapping by default: units must be independent.")
    ap.add_argument("--detrend-size", type=int, default=9)
    ap.add_argument("--out", default="runs/selector_crosscheck")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from official_view import load_sa64

    print("loading + SVD (full / even / odd) ...")
    raw = load_sa64(args.input)                     # [z,x,y,t]
    z, x, y, t = raw.shape
    C = raw.reshape(-1, t).astype(np.complex64)
    G = (C.conj().T @ C).astype(np.complex64)
    evals, V = np.linalg.eigh(G)
    V = V[:, np.argsort(evals)[::-1]]
    del G

    halves = {}
    for tag, sl in [("even", slice(0, None, 2)), ("odd", slice(1, None, 2))]:
        # Same band subspace as training (V estimated on all frames) but only the
        # rows belonging to this half, so the band-power volume is built purely
        # from this half's frames. Rows of an orthonormal V are orthonormal, so a
        # subset of rows is still an isometry -- the band definition is unchanged.
        Ch = np.ascontiguousarray(raw.reshape(-1, t)[:, sl]).astype(np.complex64)
        Vs = np.ascontiguousarray(V[sl, :])
        sig = band_power_volume(Ch, Vs, args.rank_low, args.rank_high, (z, x, y))
        noi = band_power_volume(Ch, Vs, args.noise_low, args.noise_high, (z, x, y))
        halves[tag] = dict(
            sig=sig, noi=noi,
            sig_det=(sig - uniform_filter(sig, size=args.detrend_size)).astype(np.float32),
            noi_det=(noi - uniform_filter(noi, size=args.detrend_size)).astype(np.float32),
        )
        del Ch
    del raw, C, V

    e_map = (np.load(Path(args.even) / "signal_map.npy"),
             np.load(Path(args.even) / "noise_map.npy"))
    o_map = (np.load(Path(args.odd) / "signal_map.npy"),
             np.load(Path(args.odd) / "noise_map.npy"))

    grid = slices_on_grid((z, x, y), args.window, args.stride)
    print(f"window grid: {len(grid)} units of {args.window} at stride {args.stride}")

    p_even = np.array([select_guided_pattern(e_map[0][sl], e_map[1][sl])[0] for sl in grid])
    p_odd = np.array([select_guided_pattern(o_map[0][sl], o_map[1][sl])[0] for sl in grid])

    agree = float((p_even == p_odd).mean())
    print(f"selector agreement even vs odd: {agree:.3f} "
          f"({int((p_even == p_odd).sum())}/{len(grid)})")
    print(f"even-side picks: {[int(v) for v in p_even]}")
    print(f"odd-side  picks: {[int(v) for v in p_odd]}")

    results = dict(window=args.window, stride=args.stride, n_units=len(grid),
                   agreement=agree,
                   picks_even=[int(v) for v in p_even],
                   picks_odd=[int(v) for v in p_odd])

    # Crossed evaluation: choice made on one half, rho measured on the other.
    for domain in ["raw", "detrended"]:
        for sel_tag, held_tag in [("even", "odd"), ("odd", "even")]:
            picks = p_even if sel_tag == "even" else p_odd
            h = halves[held_tag]
            sig_v = h["sig"] if domain == "raw" else h["sig_det"]
            noi_v = h["noi"] if domain == "raw" else h["noi_det"]
            rows = []
            for sl, pid in zip(grid, picks):
                rs, rn = patch_stats(sig_v[sl], noi_v[sl])
                rows.append((rn[int(pid)], rn.mean(), rn.min(), rs[int(pid)], rs.mean()))
            key = f"{domain}/select-{sel_tag}_eval-{held_tag}"
            results[key] = summarize_independent(key, rows)

    with open(out / "selector_crosscheck.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out / 'selector_crosscheck.json'}")


if __name__ == "__main__":
    main()

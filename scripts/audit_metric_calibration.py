#!/usr/bin/env python3
"""Gate D follow-up: give the residual proxy a scale, on data with a known answer.

The audit verdict (`reports/phase2_audit_verdict.md`) left one question open:

    the residual-correlation proxy reports -0.16..-0.19 for the SA64 arm pair once
    the spatial operator is matched, but a random split of the same frames reports
    the same.  So the number has no scale -- we cannot say whether it means "the
    noise is independent" or "the arms came from different frames".

This script builds that scale without a noise model.  Instead of generating
surrogates and hoping they are faithful, it manipulates the real data: the two
arms are made to share a controlled FRACTION of their frames,

    A = A0[n_shared:] + B0[:n_shared],      B = B0,      n_shared = round(overlap * |A0|)

so `overlap=0` is exactly the audited design and `overlap=1` is a pair of
identical arms.  Everything else is untouched: the real splits from the fold
manifests, the real calibration set, the real rank rules, the matched Tetris
operator, the real block statistic.

The sweep carries both ends of its own calibration:

  * overlap=0 must reproduce the audit's matched numbers.  That is the anchor to
    the real design, and it is an end-to-end check of this script against code
    that was already validated.
  * overlap=1 makes the two arms the same map, so the matched operator turns them
    into the two complementary halves of that map -- which is exactly the Tetris
    control.  Hence rel_matched must be 0 there, exactly.  If it is not, the
    implementation is wrong, not the metric.

Why not a surrogate.  Four generators were built and all four failed in ways only
measurement revealed:

  * temporal phase randomisation per voxel -- keeping |E_v(w)| gives every frame
    the same per-voxel power, so both arms collapse onto the fixed envelope
    (arm-to-arm r = 0.998) even with white noise;
  * complex AR(1) noise -- averaging over N frames again leaves the envelope to
    within 1/sqrt(N), and it discards the burstiness of real frames;
  * the real residual frames as a pool -- `e` has its 400-frame mean removed, so
    two arms drawn without replacement from it are forced to anti-correlate; the
    measured arm correlation read -0.10 at N=133 and +0.88 at N=20;
  * spatially phase-randomised frames -- each frame's power map becomes its total
    power times the average spectral density, so the per-frame spatial
    heterogeneity that carries most of the independent variation is gone.

In all four `rel_matched` came out flat in the true arm correlation to three
decimals.  Rather than keep tuning a generator until it produces the answer, this
manipulation changes nothing about the data, and the true arm-to-arm noise
correlation is still MEASURED (complex coefficient, `arm_noise_corr`) instead of
assumed.  Correlating magnitudes is not used: |mean_A| and |mean_B| both scale
with the voxel's own fluctuation power, so that estimator reads 0.9998 for
perfectly independent white noise.

Usage:
    python scripts/audit_metric_calibration.py --sequences SA64 HSA64 \
        --overlaps 0 0.25 0.5 0.75 0.9 1.0 \
        --out reports/phase2_calibration_2026-10-19
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import evaluate_pair_statistics as E                                    # noqa: E402
from build_crossfit_pairs import eigh_desc, rule_edges                   # noqa: E402
from audit_pair_statistics import PATTERN_A, PATTERN_B, tetris_operator  # noqa: E402

FRAME0 = 0


def basis_for(rule, cfg, evals, vecs, n_cap):
    lo, hi = rule_edges(rule, cfg, evals)
    hi = min(hi, n_cap)
    if hi <= lo:
        return None, (lo, hi)
    ev = np.maximum(evals[lo:hi], 0.0)
    if ev[0] <= 0.0:
        return None, (lo, hi)
    return vecs[:, lo:hi] / np.sqrt(ev)[None, :], (lo, hi)


def block_r(a_map, b_map, shape3d, cshape, valid3d):
    vals, _ = E.block_residual_diffs(
        E.to_cropped(a_map, shape3d, cshape),
        E.to_cropped(b_map, shape3d, cshape), valid3d[:cshape[0]])
    return float(vals.mean())


def with_overlap(a0, b0, overlap):
    """Make A share `overlap` of its frames with B, keeping |A| fixed."""
    n_sh = min(max(int(round(overlap * len(a0))), 0), len(a0))
    return np.concatenate([a0[n_sh:], b0[:n_sh]])


def arm_noise_corr(e, a, b, valid):
    """Complex correlation between the two arm-mean noise fields."""
    ma = e[:, a].mean(axis=1)[valid]
    mb = e[:, b].mean(axis=1)[valid]
    den = np.sqrt(np.sum(np.abs(ma) ** 2) * np.sum(np.abs(mb) ** 2))
    return float(np.real(np.sum(ma.conj() * mb)) / max(den, 1e-30))


def power_map_corr(pa, pb, valid):
    x = pa[valid] - pa[valid].mean()
    y = pb[valid] - pb[valid].mean()
    return float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-30))


def evaluate(x2d, gram, c_idx, basis, a_idx, b_idx, shape3d, cshape, valid3d,
             validbool):
    """One cell: the two arms, the same-frame-set Tetris control, and the gate."""
    pa = E.projected_power_map(x2d, gram, c_idx, basis, tgt=a_idx) / a_idx.size
    pb = E.projected_power_map(x2d, gram, c_idx, basis, tgt=b_idx) / b_idx.size
    punion = (pa * a_idx.size + pb * b_idx.size) / (a_idx.size + b_idx.size)
    tet = E.tetris_pair(punion, shape3d, valid3d)
    r_tet = block_r(tet[0], tet[1], shape3d, cshape, valid3d)
    r_as = block_r(pa, pb, shape3d, cshape, valid3d)
    r_mat = block_r(tetris_operator(pa, PATTERN_A, shape3d, valid3d),
                    tetris_operator(pb, PATTERN_B, shape3d, valid3d),
                    shape3d, cshape, valid3d)
    return {"power_map_r": power_map_corr(pa, pb, validbool),
            "r_p1": r_as, "r_tetris": r_tet, "r_matched": r_mat,
            "rel_as_impl": float((r_as - r_tet) / max(abs(r_tet), 1e-12)),
            "rel_matched": float((r_mat - r_tet) / max(abs(r_tet), 1e-12)),
            "passes_10pct": bool((r_mat - r_tet) / max(abs(r_tet), 1e-12) <= -0.10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--pairs-root", default="data/processed/crossfit")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--overlaps", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--rules", nargs="+", default=["proportional", "clutter_knee"])
    ap.add_argument("--n-trims", nargs="+", type=int, default=[0],
                    help="extra arm-budget axis: truncate both arms to this many "
                         "frames after the overlap is applied (0 = no truncation). "
                         "C and the rank band are untouched, so this isolates how "
                         "much the arm budget matters.")
    ap.add_argument("--variants", nargs="+", default=["as_impl", "f0_free"],
                    help="as_impl = frame 0 as manifested; f0_free = frame 0 "
                         "removed from C and from whichever arm holds it, rank "
                         "rule re-derived (the audit's honest variant).")
    ap.add_argument("--out", default="reports/phase2_calibration_2026-10-19")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path(args.pairs_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    result = {
        "note": ("overlap sweep on real frames through the real fold manifests. "
                 "The arms share a controlled fraction of their frames; C and the "
                 "rank band are fixed inside each fold.  rho_eff is the MEASURED "
                 "complex correlation between the two arm-mean noise fields, "
                 "power_map_r the correlation of the two arm power maps. "
                 "overlap=0 must reproduce the audit; overlap=1 must give "
                 "rel_matched = 0 exactly."),
        "sequences": {}}

    for seq in args.sequences:
        print(f"\n=== {seq} ===", flush=True)
        t0 = time.time()
        iq4d = E.load_sequence_iq(seq, base)
        shape3d = tuple(iq4d.shape[:3])
        cshape = E.cropped_shape(shape3d)
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        valid = np.load(Path("data/processed/official") / seq / "valid_mask.npy").ravel()
        valid3d = valid.reshape(shape3d)
        validbool = valid.astype(bool)
        e = x2d - x2d.mean(axis=1)[:, None]
        gram = x2d.conj().T @ x2d
        print(f"loaded {x2d.shape}, gram {gram.shape} ({time.time()-t0:.1f}s)", flush=True)

        seq_out = {"cells": []}
        for name in [s["name"] for s in cfg["sequences"][seq]["splits"]]:
            man = json.load(open(pairs_root / "splits" / f"{name}.json"))
            for fold in man["folds"]:
                fid = fold["fold_id"]
                c_idx = np.asarray(fold["calibration"], dtype=int)
                a0_all = np.asarray(fold["A"], dtype=int)
                b0_all = np.asarray(fold["B"], dtype=int)
                for variant in args.variants:
                    if variant == "as_impl":
                        c_use, a0, b0 = c_idx, a0_all, b0_all
                    else:
                        c_use = c_idx[c_idx != FRAME0]
                        a0 = a0_all[a0_all != FRAME0]
                        b0 = b0_all[b0_all != FRAME0]
                    # equal-size arms are required, not cosmetic: `with_overlap`
                    # reaches overlap=1 as A = B0[:|A0|], which coincides with B0
                    # only when the two arms are the same length.  Frame 0 sits in
                    # one arm, so removing it shrinks that arm by one and the
                    # anchor at overlap=1 would silently become a strict subset.
                    n_eq = min(a0.size, b0.size)
                    a0, b0 = a0[:n_eq], b0[:n_eq]
                    ev_c, vec_c = eigh_desc(gram[np.ix_(c_use, c_use)])
                    for rule in args.rules:
                        basis, band = basis_for(rule, cfg, ev_c, vec_c, c_use.size)
                        if basis is None:
                            continue
                        for overlap in args.overlaps:
                          a_ov = with_overlap(a0, b0, overlap)
                          for n_trim in args.n_trims:
                            a = a_ov[:n_trim] if n_trim else a_ov
                            b = b0[:n_trim] if n_trim else b0
                            cell = {
                                "sequence": seq, "split": name, "fold_id": fid,
                                "variant": variant, "rule": rule,
                                "band_0based_half_open": list(band),
                                "overlap": overlap, "n_a": int(a.size),
                                "n_trim": int(n_trim),
                                "n_shared": int(round(overlap * a0.size)),
                                "rho_eff": arm_noise_corr(e, a, b, validbool),
                            }
                            cell.update(evaluate(x2d, gram, c_use, basis, a, b,
                                                 shape3d, cshape, valid3d, validbool))
                            seq_out["cells"].append(cell)
                    # progress line for the last overlap only, to keep logs short
                    last = seq_out["cells"][-1]
                    print(f"  {name:<12}{fid:<26}{variant:<9}{rule:<13}"
                          f"ov {last['overlap']:>4.2f}  rho_eff {last['rho_eff']:+.3f}"
                          f"  mapR {last['power_map_r']:+.3f}"
                          f"  rel {last['rel_matched']:+.3f}", flush=True)
        seq_out["elapsed_s"] = round(time.time() - t0, 1)
        result["sequences"][seq] = seq_out
        (outdir / f"metric_calibration_{seq}.json").write_text(
            json.dumps(seq_out, indent=1))
        print(f"  wrote {outdir / f'metric_calibration_{seq}.json'} "
              f"({time.time()-t0:.0f}s)", flush=True)

    (outdir / "metric_calibration.json").write_text(json.dumps(result, indent=1))
    print(f"\nwrote {outdir / 'metric_calibration.json'}")


if __name__ == "__main__":
    main()

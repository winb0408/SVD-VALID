#!/usr/bin/env python3
"""Audit Gate C: remove frame 0 from the pipeline entirely, not just from A/B.

The Phase 2 report called the folds whose A and B exclude frame 0 the "clean
folds".  That wording is wrong on two counts: frame 0 is an SA64 acquisition
artifact, and in the folds where it sits in C it still enters the projector
P_C that is applied to both arms.  So "frame 0 not in A or B" is not "frame 0
absent".

This script builds the version that *is* absent, for every fold, by exact
recomputation (not by re-running the whole Phase 1 pipeline):

  all three variants share one code path -- project the frames of interest with
  the calibration basis, average the per-frame powers, then run the identical
  block-residual statistic.  The only thing that changes between variants is
  which frames and which calibration set go in:

    as_impl     C as manifested, A and B as manifested
    variant_a   C as manifested, frame 0 dropped from whichever arm holds it
    variant_b   frame 0 dropped from C as well, with the rank rule re-derived
                from the shorter calibration spectrum (same rule, no retuning)

  The Tetris control is rebuilt under each variant from the same frame set, so
  the gate difference stays matched.

Convention note, because it silently bites: `projected_power_map` returns the
SUM of per-frame powers over the target frames, while the arm volumes Phase 1
saves are MEANS.  Everything here is computed from the sum and divided by the
frame count, so the two never mix.

Usage:
    python scripts/audit_frame0_exclusion.py \
        --out reports/phase2_audit_rerun_2026-10-19/frame0_exclusion.json
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

FRAME0 = 0


def basis_for(rule, cfg, evals, vecs, n_cap):
    lo, hi = rule_edges(rule, cfg, evals)
    hi = min(hi, n_cap)
    if hi <= lo:
        return None, (lo, hi)
    return (vecs[:, lo:hi] / np.sqrt(evals[lo:hi])[None, :]).astype(np.complex64), (lo, hi)


def gate_r(p1_a, p1_b, tet, shape3d, cshape, valid3d):
    """Block-mean residual r for P1 and the Tetris control, on one grid."""
    v3c = valid3d[:cshape[0]]

    def mean_r(a_map, b_map):
        vals, _ = E.block_residual_diffs(E.to_cropped(a_map, shape3d, cshape),
                                         E.to_cropped(b_map, shape3d, cshape), v3c)
        return float(vals.mean()), int(vals.size)

    r_p1, n = mean_r(p1_a, p1_b)
    r_tet, _ = mean_r(*tet)
    return {"p1_mean_r": r_p1, "tetris_mean_r": r_tet, "n_units": n,
            "relative_reduction": float((r_p1 - r_tet) / max(abs(r_tet), 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--rules", nargs="+",
                    default=["proportional", "clutter_knee", "cum_energy", "spectral"])
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19/"
                                    "frame0_exclusion.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path(cfg["output"]["pairs_root"])
    out = {"frame0": FRAME0,
           "variants": {
               "as_impl": "C, A, B as manifested",
               "variant_a": "frame 0 dropped from the arm that holds it; C unchanged",
               "variant_b": "frame 0 dropped from C as well; rank rule re-derived "
                            "on the shorter calibration spectrum"},
           "convention": "projected_power_map sums over frames; arm volumes are "
                         "means; everything below divides the sum by the frame "
                         "count, so the two never mix.",
           "sequences": {}}

    for seq in args.sequences:
        print(f"\n=== {seq} ===")
        t0 = time.time()
        iq4d = E.load_sequence_iq(seq, base)
        shape3d = tuple(iq4d.shape[:3])
        cshape = E.cropped_shape(shape3d)
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        valid = np.load(Path("data/processed/official") / seq / "valid_mask.npy").ravel()
        valid3d = valid.reshape(shape3d)
        gram_all = np.load(pairs_root / seq / "gram_all.npy")
        print(f"loaded IQ {x2d.shape} + gram {gram_all.shape} ({time.time() - t0:.1f}s)")

        split_names = args.splits or [s["name"] for s in cfg["sequences"][seq]["splits"]]
        folds = []
        for name in split_names:
            man = json.load(open(pairs_root / "splits" / f"{name}.json"))
            for fold in man["folds"]:
                fold = dict(fold)
                fold["split"] = name
                folds.append(fold)

        seq_out = {"shape3d": list(shape3d), "cropped_shape3d": list(cshape),
                   "folds": []}
        for fold in folds:
            fid = fold["fold_id"]
            c_idx = np.asarray(fold["calibration"], dtype=int)
            a_idx = np.asarray(fold["A"], dtype=int)
            b_idx = np.asarray(fold["B"], dtype=int)
            role = ("C" if FRAME0 in c_idx else
                    "A" if FRAME0 in a_idx else "B")
            row = {"fold_id": fid, "split": fold["split"], "frame0_role": role,
                   "T_C": int(c_idx.size), "T_A": int(a_idx.size), "rules": {}}

            ev_c, vec_c = eigh_desc(gram_all[np.ix_(c_idx, c_idx)])
            has_f0_c = FRAME0 in c_idx
            if has_f0_c:
                assert c_idx[0] == FRAME0, "frame 0 must be C's first index"
                ev_c0, vec_c0 = eigh_desc(gram_all[np.ix_(c_idx[1:], c_idx[1:])])
            else:
                ev_c0 = vec_c0 = None

            for rule in args.rules:
                basis, band = basis_for(rule, cfg, ev_c, vec_c, c_idx.size)
                if basis is None:
                    continue
                rec = {}

                # --- as implemented ------------------------------------
                p1a = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=a_idx) / a_idx.size
                p1b = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=b_idx) / b_idx.size
                tf = np.concatenate([a_idx, b_idx])
                pdt = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=tf)
                tet = tuple(m.ravel() for m in E.tetris_pair(pdt, shape3d, valid3d))
                rec["as_impl"] = gate_r(p1a, p1b, tet, shape3d, cshape, valid3d)
                rec["as_impl"]["band_0based_half_open"] = list(band)

                # --- variant A: frame 0 out of the arm that holds it -----
                if role in ("A", "B"):
                    p0 = E.projected_power_map(x2d, gram_all, c_idx, basis,
                                               tgt=np.array([FRAME0]))
                    a_use = a_idx[a_idx != FRAME0]
                    b_use = b_idx[b_idx != FRAME0]
                    va = (E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=a_use)
                          / a_use.size)
                    vb = (E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=b_use)
                          / b_use.size)
                    pdt_a = pdt - p0
                    tet_a = tuple(m.ravel() for m in E.tetris_pair(pdt_a, shape3d, valid3d))
                    rec["variant_a"] = gate_r(va, vb, tet_a, shape3d, cshape, valid3d)
                    rec["variant_a"]["band_0based_half_open"] = list(band)
                    # how much of the affected arm's power is frame 0 alone --
                    # the number that decides whether this variant can matter
                    p_arm = E.projected_power_map(x2d, gram_all, c_idx, basis,
                                                  tgt=a_idx if role == "A" else b_idx)
                    rec["variant_a"]["frame0_share_of_arm"] = float(
                        p0.sum() / max(p_arm.sum(), 1e-30))
                else:
                    rec["variant_a"] = dict(rec["as_impl"])
                    rec["variant_a"]["note"] = "frame 0 not in A/B"

                # --- variant B: frame 0 out of C too ---------------------
                if has_f0_c:
                    basis0, band0 = basis_for(rule, cfg, ev_c0, vec_c0, c_idx[1:].size)
                    if basis0 is None:
                        rec["variant_b"] = dict(rec["variant_a"])
                        rec["variant_b"]["note"] = "band empty without frame 0"
                    else:
                        src0 = c_idx[1:]
                        b0a = (E.projected_power_map(x2d, gram_all, src0, basis0, tgt=a_idx)
                               / a_idx.size)
                        b0b = (E.projected_power_map(x2d, gram_all, src0, basis0, tgt=b_idx)
                               / b_idx.size)
                        pdt0 = E.projected_power_map(x2d, gram_all, src0, basis0, tgt=tf)
                        if FRAME0 in tf:
                            pdt0 = pdt0 - E.projected_power_map(
                                x2d, gram_all, src0, basis0, tgt=np.array([FRAME0]))
                        tet0 = tuple(m.ravel() for m in E.tetris_pair(pdt0, shape3d, valid3d))
                        rec["variant_b"] = gate_r(b0a, b0b, tet0, shape3d, cshape, valid3d)
                        rec["variant_b"]["band_0based_half_open"] = list(band0)
                        rec["variant_b"]["rank_change"] = int(band0[1] - band0[0]) - int(band[1] - band[0])
                else:
                    rec["variant_b"] = dict(rec["variant_a"])
                    rec["variant_b"]["note"] = "frame 0 not in C; same as variant A"

                row["rules"][rule] = rec
            seq_out["folds"].append(row)
            print(f"  {fid:<26} f0={role} " + " ".join(
                f"{r}: {row['rules'][r]['as_impl']['relative_reduction']:+.3f}"
                f"->{row['rules'][r]['variant_b']['relative_reduction']:+.3f}"
                for r in row["rules"]))
        out["sequences"][seq] = seq_out

        # ---- aggregate ---------------------------------------------------
        print(f"\n-- {seq} aggregate over {len(seq_out['folds'])} folds")
        agg = {}
        for rule in args.rules:
            agg[rule] = {}
            for tag in ("as_impl", "variant_a", "variant_b"):
                cells = [f["rules"][rule][tag] for f in seq_out["folds"]
                         if rule in f["rules"]]
                if not cells:
                    continue
                rel = np.asarray([c["relative_reduction"] for c in cells])
                p1 = np.asarray([c["p1_mean_r"] for c in cells])
                tet = np.asarray([c["tetris_mean_r"] for c in cells])
                agg[rule][tag] = {
                    "n_folds": int(rel.size),
                    "p1_mean_r": float(p1.mean()), "tetris_mean_r": float(tet.mean()),
                    "relative_reduction_mean": float(rel.mean()),
                    "relative_reduction_min": float(rel.min()),
                    "relative_reduction_max": float(rel.max()),
                    "n_folds_negative": int((rel < 0).sum()),
                    "n_folds_passing_10pct": int((rel <= -0.10).sum()),
                }
                print(f"   {rule:<13} {tag:<10} P1 {p1.mean():.3f} TET {tet.mean():.3f} "
                      f"rel {rel.mean():+.3f} [{rel.min():+.3f},{rel.max():+.3f}] "
                      f"neg {int((rel < 0).sum())}/{rel.size}")
        seq_out["aggregate"] = agg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

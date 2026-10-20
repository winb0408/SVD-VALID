#!/usr/bin/env python3
"""Gate C, second half: does the *matched* verdict survive removing frame 0?

`audit_frame0_exclusion.py` showed that the P1-vs-Tetris gap on SA64 is mostly
frame 0.  `audit_pair_statistics.py --task variants` showed that matching the
spatial operator removes most of the remaining gap.  Both were run separately,
so neither answers the question the Gate D verdict actually needs:

    with the operator matched AND frame 0 absent, is the SA64 arm pair still
    distinguishable from a Tetris pair?

This script evaluates the matched operator on two arm configurations that share
one code path:

  as_impl    C, A, B exactly as manifested
  f0_free    frame 0 removed from C (rank rule re-derived from the shorter
             calibration spectrum, same rule -- no retuning) and from whichever
             arm holds it, so frame 0 enters nothing at all

For each configuration the gate statistic is reported for three operators:

  p1     the arm pair as built                       (the Phase 2 number)
  tet    tetris_pair on the same frame set           (the control)
  mat    the SAME frame set through the Tetris spatial operator, so the arms
         and the control differ only in *which frames* they came from, not in
         how the power map was spatially processed

`mat` is the number the audit's Gate D is about.  `rel_matched = (mat - tet)/tet`
is the frame-0-free version of the matched relative reduction.

Convention note: `projected_power_map` SUMS over target frames; every arm here
is divided by its frame count before the statistic sees it.

Usage:
    python scripts/audit_frame0_matched.py --sequences SA64 HSA64 \
        --out reports/phase2_audit_rerun_2026-10-19/frame0_matched.json
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
    return (vecs[:, lo:hi] / np.sqrt(evals[lo:hi])[None, :]).astype(np.complex64), (lo, hi)


def stats(a_map, b_map, band_map, shape3d, cshape, valid3d):
    """p1 / tetris / matched block-mean r on one arm pair, plus the gate.

    `band_map` is the MEAN power map over the union of the two arm frame sets --
    passed in rather than reconstructed as (a+b)/2, because removing frame 0
    from one arm only makes |A| != |B| and the two would then disagree.
    """
    v3c = valid3d[:cshape[0]]
    mat_a = tetris_operator(a_map, PATTERN_A, shape3d, valid3d)
    mat_b = tetris_operator(b_map, PATTERN_B, shape3d, valid3d)
    tet_a, tet_b = E.tetris_pair(band_map, shape3d, valid3d)

    def mean_r(x, y):
        vals, _ = E.block_residual_diffs(E.to_cropped(x, shape3d, cshape),
                                         E.to_cropped(y, shape3d, cshape), v3c)
        return float(vals.mean()), int(vals.size)

    r_p1, n = mean_r(a_map, b_map)
    r_tet, _ = mean_r(tet_a, tet_b)
    r_mat, _ = mean_r(mat_a, mat_b)
    return {"p1": r_p1, "tet": r_tet, "mat": r_mat, "n_units": n,
            "rel_as_impl": float((r_p1 - r_tet) / max(abs(r_tet), 1e-12)),
            "rel_matched": float((r_mat - r_tet) / max(abs(r_tet), 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--rules", nargs="+", default=["proportional", "clutter_knee"])
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19/"
                                    "frame0_matched.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path(cfg["output_root"]) if "output_root" in cfg else Path("data/processed/crossfit")
    if not pairs_root.exists():
        pairs_root = Path("data/processed/crossfit")

    out = {"note": ("matched spatial operator (Tetris operator on the P1 arms) "
                    "evaluated with frame 0 present vs absent.  `mat` is the "
                    "audit's Gate D quantity; `tet` is the same-frame-set "
                    "control, so the two differ only in frame membership."),
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

        seq_out = {"shape3d": list(shape3d), "cropped_shape3d": list(cshape), "folds": []}
        for fold in folds:
            fid = fold["fold_id"]
            c_idx = np.asarray(fold["calibration"], dtype=int)
            a_idx = np.asarray(fold["A"], dtype=int)
            b_idx = np.asarray(fold["B"], dtype=int)
            role = ("C" if FRAME0 in c_idx else "A" if FRAME0 in a_idx else "B")
            row = {"fold_id": fid, "split": fold["split"], "frame0_role": role, "rules": {}}

            ev_c, vec_c = eigh_desc(gram_all[np.ix_(c_idx, c_idx)])
            src0 = c_idx[c_idx != FRAME0]
            ev_c0, vec_c0 = eigh_desc(gram_all[np.ix_(src0, src0)])

            for rule in args.rules:
                basis, band = basis_for(rule, cfg, ev_c, vec_c, c_idx.size)
                basis0, band0 = basis_for(rule, cfg, ev_c0, vec_c0, src0.size)
                if basis is None or basis0 is None:
                    continue

                # --- frame 0 present, exactly as Phase 2 built it ---------
                tf = np.concatenate([a_idx, b_idx])
                p1a = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=a_idx) / a_idx.size
                p1b = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=b_idx) / b_idx.size
                band_map = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=tf) / tf.size
                rec = {"as_impl": stats(p1a, p1b, band_map, shape3d, cshape, valid3d),
                       "band_0based_half_open": list(band)}

                # --- frame 0 absent from the projector and both arms ------
                a_use, b_use = a_idx[a_idx != FRAME0], b_idx[b_idx != FRAME0]
                tf0 = np.concatenate([a_use, b_use])
                f0a = (E.projected_power_map(x2d, gram_all, src0, basis0, tgt=a_use)
                       / a_use.size)
                f0b = (E.projected_power_map(x2d, gram_all, src0, basis0, tgt=b_use)
                       / b_use.size)
                band0_map = (E.projected_power_map(x2d, gram_all, src0, basis0, tgt=tf0)
                             / tf0.size)
                rec["f0_free"] = stats(f0a, f0b, band0_map, shape3d, cshape, valid3d)
                rec["f0_free"]["band_0based_half_open"] = list(band0)
                rec["f0_free"]["rank_change"] = int(band0[1] - band0[0]) - int(band[1] - band[0])
                rec["f0_free"]["T_A"], rec["f0_free"]["T_B"] = int(a_use.size), int(b_use.size)

                row["rules"][rule] = rec
            seq_out["folds"].append(row)
            print(f"  {fid:<26} f0={role} " + " ".join(
                f"{r}: as {row['rules'][r]['as_impl']['rel_matched']:+.3f}"
                f" f0free {row['rules'][r]['f0_free']['rel_matched']:+.3f}"
                for r in row["rules"]), flush=True)
        out["sequences"][seq] = seq_out

        print(f"\n-- {seq} aggregate over {len(seq_out['folds'])} folds")
        agg = {}
        for rule in args.rules:
            agg[rule] = {}
            for tag in ("as_impl", "f0_free"):
                cells = [f["rules"][rule][tag] for f in seq_out["folds"] if rule in f["rules"]]
                if not cells:
                    continue
                rel = np.asarray([c["rel_matched"] for c in cells])
                rel_as = np.asarray([c["rel_as_impl"] for c in cells])
                entry = {
                    "n_folds": int(rel.size),
                    "p1": float(np.mean([c["p1"] for c in cells])),
                    "tet": float(np.mean([c["tet"] for c in cells])),
                    "mat": float(np.mean([c["mat"] for c in cells])),
                    "rel_matched_mean": float(rel.mean()),
                    "rel_matched_min": float(rel.min()),
                    "rel_matched_max": float(rel.max()),
                    "rel_matched_median": float(np.median(rel)),
                    "n_folds_matched_negative": int((rel < 0).sum()),
                    "n_folds_matched_passing_10pct": int((rel <= -0.10).sum()),
                    "rel_as_impl_mean": float(rel_as.mean()),
                }
                # the same aggregate split by where frame 0 lived, because on
                # SA64 that split is what carries the as-implemented verdict
                by_role = {}
                for rname in ("A", "B", "C"):
                    sub = [f["rules"][rule][tag]["rel_matched"]
                           for f in seq_out["folds"]
                           if f["frame0_role"] == rname and rule in f["rules"]]
                    if sub:
                        by_role[rname] = {"n": len(sub), "rel_matched_mean": float(np.mean(sub))}
                entry["rel_matched_by_frame0_role"] = by_role
                agg[rule][tag] = entry
                print(f"   {rule:<13} {tag:<8} P1 {entry['p1']:.3f} TET {entry['tet']:.3f} "
                      f"MAT {entry['mat']:.3f} relM {rel.mean():+.3f} "
                      f"[{rel.min():+.3f},{rel.max():+.3f}] "
                      f"pass10 {entry['n_folds_matched_passing_10pct']}/{rel.size}  "
                      + " ".join(f"f0={k}:{v['rel_matched_mean']:+.3f}(n={v['n']})"
                                 for k, v in by_role.items()), flush=True)
        seq_out["aggregate"] = agg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

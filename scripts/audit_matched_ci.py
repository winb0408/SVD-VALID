#!/usr/bin/env python3
"""Gate C: paired CI on the *matched* difference, and a recomputation check.

`audit_frame0_matched.py` reports a point estimate of the matched relative
reduction.  The pre-registered gate is a paired test (rel <= -0.10 AND the CI
excludes 0), so a point estimate alone cannot decide it.  This script adds the
two things the verdict still needs:

  1. consistency  -- the arms rebuilt here from IQ + gram must reproduce the
                     Phase 1 volumes on disk (`P1_<rule>_A.npy`), otherwise the
                     f0_free recomputation is not comparable to them;
  2. paired CI    -- block bootstrap on (matched - tetris) per fold, under
                     as_impl and under f0_free, with the same block unit the
                     Phase 2 gate uses.

Usage:
    python scripts/audit_matched_ci.py --sequences SA64 HSA64 \
        --out reports/phase2_audit_rerun_2026-10-19/matched_ci.json
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
from scripts.phase3_stats import block_bootstrap_ci                      # noqa: E402

FRAME0 = 0
N_BOOT = 4000


def basis_for(rule, cfg, evals, vecs, n_cap):
    lo, hi = rule_edges(rule, cfg, evals)
    hi = min(hi, n_cap)
    if hi <= lo:
        return None, (lo, hi)
    return (vecs[:, lo:hi] / np.sqrt(evals[lo:hi])[None, :]).astype(np.complex64), (lo, hi)


def units(a_map, b_map, shape3d, cshape, valid3d):
    """Block-residual values for the arm pair and the two operator variants."""
    v3c = valid3d[:cshape[0]]
    mat = E.block_residual_diffs(
        E.to_cropped(tetris_operator(a_map, PATTERN_A, shape3d, valid3d), shape3d, cshape),
        E.to_cropped(tetris_operator(b_map, PATTERN_B, shape3d, valid3d), shape3d, cshape), v3c)
    return mat[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--rules", nargs="+", default=["proportional", "clutter_knee"])
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19/matched_ci.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path("data/processed/crossfit")

    out = {"note": ("paired block bootstrap on (matched operator - tetris control), "
                    "under frame-0-present and frame-0-free arm construction.  "
                    "`consistency_max_rel_diff` compares the rebuilt arms against "
                    "the Phase 1 volumes on disk."),
           "sequences": {}}

    for seq in args.sequences:
        print(f"\n=== {seq} ===", flush=True)
        t0 = time.time()
        iq4d = E.load_sequence_iq(seq, base)
        shape3d = tuple(iq4d.shape[:3])
        cshape = E.cropped_shape(shape3d)
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        valid3d = np.load(Path("data/processed/official") / seq / "valid_mask.npy").ravel().reshape(shape3d)
        gram_all = np.load(pairs_root / seq / "gram_all.npy")
        print(f"loaded IQ {x2d.shape} + gram {gram_all.shape} ({time.time() - t0:.1f}s)", flush=True)

        folds = []
        for s in cfg["sequences"][seq]["splits"]:
            man = json.load(open(pairs_root / "splits" / f"{s['name']}.json"))
            for fold in man["folds"]:
                fold = dict(fold)
                fold["split"] = s["name"]
                folds.append(fold)

        seq_out = {"folds": []}
        for fold in folds:
            fid = fold["fold_id"]
            c_idx = np.asarray(fold["calibration"], dtype=int)
            a_idx = np.asarray(fold["A"], dtype=int)
            b_idx = np.asarray(fold["B"], dtype=int)
            role = ("C" if FRAME0 in c_idx else "A" if FRAME0 in a_idx else "B")
            ev_c, vec_c = eigh_desc(gram_all[np.ix_(c_idx, c_idx)])
            src0 = c_idx[c_idx != FRAME0]
            ev_c0, vec_c0 = eigh_desc(gram_all[np.ix_(src0, src0)])
            row = {"fold_id": fid, "split": fold["split"], "frame0_role": role, "rules": {}}

            for rule in args.rules:
                basis, band = basis_for(rule, cfg, ev_c, vec_c, c_idx.size)
                basis0, band0 = basis_for(rule, cfg, ev_c0, vec_c0, src0.size)
                if basis is None or basis0 is None:
                    continue
                rec = {}

                for tag, src, bas, ai, bi in (
                        ("as_impl", c_idx, basis, a_idx, b_idx),
                        ("f0_free", src0, basis0,
                         a_idx[a_idx != FRAME0], b_idx[b_idx != FRAME0])):
                    amap = E.projected_power_map(x2d, gram_all, src, bas, tgt=ai) / ai.size
                    bmap = E.projected_power_map(x2d, gram_all, src, bas, tgt=bi) / bi.size
                    tf = np.concatenate([ai, bi])
                    band_map = E.projected_power_map(x2d, gram_all, src, bas, tgt=tf) / tf.size
                    tet = tuple(m.ravel() for m in E.tetris_pair(band_map, shape3d, valid3d))
                    v_tet = E.block_residual_diffs(
                        E.to_cropped(tet[0], shape3d, cshape),
                        E.to_cropped(tet[1], shape3d, cshape), valid3d[:cshape[0]])[0]
                    v_mat = units(amap, bmap, shape3d, cshape, valid3d)
                    n = min(v_mat.size, v_tet.size)
                    diff = v_mat[:n] - v_tet[:n]
                    lo, hi = block_bootstrap_ci(diff, n_boot=N_BOOT, seed=7)
                    rel = float((v_mat[:n].mean() - v_tet[:n].mean())
                                / max(abs(v_tet[:n].mean()), 1e-12))
                    rec[tag] = {"mat": float(v_mat[:n].mean()), "tet": float(v_tet[:n].mean()),
                                "diff_mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
                                "relative_reduction": rel, "n_units": int(n),
                                "passes_10pct_reduction": bool(rel <= -0.10 and hi < 0)}

                    if tag == "as_impl":
                        # consistency anchor: compare against the saved Phase 1 arms
                        fd = pairs_root / seq / fid
                        try:
                            sa = np.load(fd / f"P1_{rule}_A.npy")
                            sb = np.load(fd / f"P1_{rule}_B.npy")
                            d = max(float(np.abs(amap - sa).max() / max(np.abs(sa).max(), 1e-30)),
                                    float(np.abs(bmap - sb).max() / max(np.abs(sb).max(), 1e-30)))
                            rec["consistency_max_rel_diff"] = d
                        except FileNotFoundError:
                            rec["consistency_max_rel_diff"] = None
                row["rules"][rule] = rec
            seq_out["folds"].append(row)
            print(f"  {fid:<26} f0={role} " + " ".join(
                f"{r}: as {row['rules'][r]['as_impl']['relative_reduction']:+.3f}"
                f" f0free {row['rules'][r]['f0_free']['relative_reduction']:+.3f}"
                for r in row["rules"]), flush=True)
        out["sequences"][seq] = seq_out

        print(f"\n-- {seq} aggregate", flush=True)
        agg = {}
        for rule in args.rules:
            agg[rule] = {}
            for tag in ("as_impl", "f0_free"):
                cells = [f["rules"][rule][tag] for f in seq_out["folds"] if rule in f["rules"]]
                rel = np.asarray([c["relative_reduction"] for c in cells])
                ci = np.asarray([c["ci95"] for c in cells])
                agg[rule][tag] = {
                    "n_folds": int(rel.size),
                    "mat": float(np.mean([c["mat"] for c in cells])),
                    "tet": float(np.mean([c["tet"] for c in cells])),
                    "relative_reduction_mean": float(rel.mean()),
                    "relative_reduction_range": [float(rel.min()), float(rel.max())],
                    "ci95_mean": [float(ci[:, 0].mean()), float(ci[:, 1].mean())],
                    "n_folds_ci_excludes_0": int(((ci[:, 0] > 0) | (ci[:, 1] < 0)).sum()),
                    "n_folds_passing_10pct": int((rel <= -0.10).sum()),
                }
                print(f"   {rule:<13}{tag:<8} MAT {agg[rule][tag]['mat']:.3f} "
                      f"TET {agg[rule][tag]['tet']:.3f} rel {rel.mean():+.3f} "
                      f"ci [{ci[:, 0].mean():+.4f},{ci[:, 1].mean():+.4f}] "
                      f"excl0 {agg[rule][tag]['n_folds_ci_excludes_0']}/{rel.size} "
                      f"pass10 {agg[rule][tag]['n_folds_passing_10pct']}/{rel.size}", flush=True)
        cons = [f["rules"][r]["consistency_max_rel_diff"] for f in seq_out["folds"]
                for r in f["rules"] if f["rules"][r].get("consistency_max_rel_diff") is not None]
        seq_out["aggregate"] = agg
        seq_out["consistency_max_rel_diff"] = float(np.max(cons)) if cons else None
        print(f"   consistency vs saved arms: max rel diff "
              f"{seq_out['consistency_max_rel_diff']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

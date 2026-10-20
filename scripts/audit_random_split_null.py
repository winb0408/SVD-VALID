#!/usr/bin/env python3
"""Gate D, construct validity: does the matched gate reward the C/A/B design?

`audit_frame0_matched.py` shows that once the spatial operator is matched, the
SA64 arm pair sits ~16-19% below the Tetris control, i.e. it clears the
pre-registered -10% margin.  Before that can be read as "the noise is
independent", one control is missing:

    is that margin a property of the *CrossFit design* (disjoint C/A/B frame
    groups, basis from C, same projector on both arms), or would ANY split of
    the same 400 frames into two disjoint halves produce it?

If a random split produces the same margin, the matched gate is measuring
"the two arms came from different frames" -- which is true of every split by
construction -- and says nothing about the SVD cross-fitting apparatus.  If
CrossFit's margin is clearly separated from the random-split null, the design
is doing work.

Design: every pair in a fold is scored against the SAME Tetris control, built
from the union power map pm(A u B), so the comparison is paired at the level of
the exact same control volume:

    crossfit      op(pm(A), PATTERN_A) vs op(pm(B), PATTERN_B)
    random_<i>    op(pm(R1), PATTERN_A) vs op(pm(R2), PATTERN_B),
                  R1 u R2 = A u B, |R1| = |A|, drawn without replacement

Both use the fold's own projector, so the only thing that changes is which
frames went into which arm.

Usage:
    python scripts/audit_random_split_null.py --sequences SA64 HSA64 \
        --out reports/phase2_audit_rerun_2026-10-19/random_split_null.json
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


def matched_r(a_map, b_map, shape3d, cshape, valid3d):
    v3c = valid3d[:cshape[0]]
    vals, _ = E.block_residual_diffs(
        E.to_cropped(tetris_operator(a_map, PATTERN_A, shape3d, valid3d), shape3d, cshape),
        E.to_cropped(tetris_operator(b_map, PATTERN_B, shape3d, valid3d), shape3d, cshape), v3c)
    return float(vals.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--rules", nargs="+", default=["proportional", "clutter_knee"])
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19/"
                                    "random_split_null.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path("data/processed/crossfit")
    rng = np.random.default_rng(args.seed)

    out = {"note": ("matched-operator margin for the designed C/A/B arm split vs "
                    "random disjoint splits of the same A u B frame pool, all "
                    "scored against the same per-fold Tetris control."),
           "n_random": args.n_random, "seed": args.seed, "sequences": {}}

    for seq in args.sequences:
        print(f"\n=== {seq} ===", flush=True)
        iq4d = E.load_sequence_iq(seq, base)
        shape3d = tuple(iq4d.shape[:3])
        cshape = E.cropped_shape(shape3d)
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        valid3d = np.load(Path("data/processed/official") / seq / "valid_mask.npy").ravel().reshape(shape3d)
        gram_all = np.load(pairs_root / seq / "gram_all.npy")

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
            ev_c, vec_c = eigh_desc(gram_all[np.ix_(c_idx, c_idx)])
            pool = np.sort(np.concatenate([a_idx, b_idx]))
            n_a = a_idx.size
            row = {"fold_id": fid, "split": fold["split"], "n_pool": int(pool.size), "rules": {}}

            for rule in args.rules:
                basis, band = basis_for(rule, cfg, ev_c, vec_c, c_idx.size)
                if basis is None:
                    continue
                # one union map -> one Tetris control, shared by every pair below
                pd = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=pool) / pool.size
                tet = tuple(m.ravel() for m in E.tetris_pair(pd, shape3d, valid3d))
                v_tet = E.block_residual_diffs(
                    E.to_cropped(tet[0], shape3d, cshape),
                    E.to_cropped(tet[1], shape3d, cshape), valid3d[:cshape[0]])[0]
                r_tet = float(v_tet.mean())

                amap = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=a_idx) / a_idx.size
                bmap = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=b_idx) / b_idx.size
                r_cf = matched_r(amap, bmap, shape3d, cshape, valid3d)

                r_rand = []
                for _ in range(args.n_random):
                    pick = rng.permutation(pool.size)
                    r1, r2 = pool[pick[:n_a]], pool[pick[n_a:]]
                    m1 = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=r1) / r1.size
                    m2 = E.projected_power_map(x2d, gram_all, c_idx, basis, tgt=r2) / r2.size
                    r_rand.append(matched_r(m1, m2, shape3d, cshape, valid3d))
                r_rand = np.asarray(r_rand)

                cf_rel = (r_cf - r_tet) / max(abs(r_tet), 1e-12)
                rand_rel = (r_rand - r_tet) / max(abs(r_tet), 1e-12)
                row["rules"][rule] = {
                    "tetris_r": r_tet,
                    "crossfit_r": r_cf, "crossfit_rel": float(cf_rel),
                    "random_r_mean": float(r_rand.mean()), "random_r_std": float(r_rand.std()),
                    "random_rel_mean": float(rand_rel.mean()),
                    "random_rel_p05": float(np.percentile(rand_rel, 5)),
                    "random_rel_p95": float(np.percentile(rand_rel, 95)),
                    "crossfit_percentile_vs_random": float((rand_rel <= cf_rel).mean() * 100.0),
                }
                print(f"  {fid:<26} {rule:<13} TET {r_tet:.3f} CF {r_cf:.3f} ({cf_rel:+.3f}) "
                      f"rand {r_rand.mean():.3f}±{r_rand.std():.3f} ({rand_rel.mean():+.3f}) "
                      f"CF pct {row['rules'][rule]['crossfit_percentile_vs_random']:.0f}", flush=True)
            seq_out["folds"].append(row)
        out["sequences"][seq] = seq_out

        print(f"\n-- {seq} aggregate", flush=True)
        agg = {}
        for rule in args.rules:
            cells = [f["rules"][rule] for f in seq_out["folds"] if rule in f["rules"]]
            if not cells:
                continue
            agg[rule] = {
                "n_folds": len(cells),
                "crossfit_rel_mean": float(np.mean([c["crossfit_rel"] for c in cells])),
                "random_rel_mean": float(np.mean([c["random_rel_mean"] for c in cells])),
                "random_rel_p95_mean": float(np.mean([c["random_rel_p95"] for c in cells])),
                "crossfit_percentile_mean": float(np.mean(
                    [c["crossfit_percentile_vs_random"] for c in cells])),
                "n_folds_crossfit_below_random_p05": int(sum(
                    c["crossfit_percentile_vs_random"] < 5.0 for c in cells)),
            }
            a = agg[rule]
            print(f"   {rule:<13} CF {a['crossfit_rel_mean']:+.3f}  "
                  f"random {a['random_rel_mean']:+.3f} (p95 {a['random_rel_p95_mean']:+.3f})  "
                  f"CF percentile {a['crossfit_percentile_mean']:.0f}  "
                  f"below-random-p05 {a['n_folds_crossfit_below_random_p05']}/{a['n_folds']}", flush=True)
        seq_out["aggregate"] = agg

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

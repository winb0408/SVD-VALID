#!/usr/bin/env python3
"""Audit Gate C: frame-0 sensitivity of the residual-correlation gate.

Frame 0 of SA64 is an acquisition artifact: inside the official band
[59, 350) it carries 0.002x the median frame power, while its *total* Gram
energy is 1.15x the median.  In the cross-fit design it lands in a different
role in every fold, so "the 8 clean folds" phrasing in the Phase 2 report is
wrong on two counts:

  * frame 0 is never absent from the pipeline -- when it is in C it still
    shapes the projector P_C that is applied to both A and B;
  * the folds it is absent from A and B are not a "clean" subset of a
    contaminated whole; they are a different fold population with a different
    T_A (133 vs 80), and they must be reported as such.

This script recomputes the gate under three frame-0 treatments and prints
them side by side, so the reader can see whether the verdict moves:

  all          every pre-registered fold
  f0_not_in_AB folds whose A and B exclude frame 0
  f0_in_AB     folds whose A or B include frame 0

It reads only ``pair_statistics.json`` -- no volumes are touched -- so it is
cheap and cannot disagree with the main run about what the arms were.

Both block-r estimators are reported, because the Phase 2 report mixed them:
the per-fold LEVELS in its section 4.1 table are mean-over-folds of the
per-fold *median* block r, while its relative-reduction column is
mean-over-folds of the per-fold relative reduction computed from the *mean*
block r.  A single table row therefore quotes a ratio of two different
estimators.  They are separated here.

Usage:
    python scripts/audit_frame0_sensitivity.py \
        --stats data/processed/crossfit/pair_statistics.json \
        --out reports/phase2_audit_rerun_2026-10-19/frame0_sensitivity.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FRAME0 = 0


def frame0_role(splits_root, split_name, fold_id):
    """The role frame 0 plays in this fold: 'C', 'A', 'B' or 'trimmed'."""
    man = json.loads((Path(splits_root) / f"{split_name}.json").read_text())
    for row in man["per_frame_roles"]:
        if row["fold_id"] == fold_id:
            return row["per_frame_role"][FRAME0]
    raise KeyError(f"{split_name}: no per_frame_roles row for {fold_id}")


def summarise(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_folds": 0}
    lo, hi = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"n_folds": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "ci95_t": [lo, hi],
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/processed/crossfit/pair_statistics.json")
    ap.add_argument("--splits-root", default="data/processed/crossfit/splits")
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19/"
                                    "frame0_sensitivity.json")
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    out = {"frame0": FRAME0,
           "note": ("frame 0 is an SA64 acquisition artifact; HSA64 has no "
                    "anomalous frame, so it is reported for completeness only. "
                    "'f0_in_C' folds are NOT contamination-free: C feeds the "
                    "projector applied to both arms."),
           "sequences": {}}

    for seq, spec in stats["sequences"].items():
        folds = spec["folds"]
        rows = []
        for f in folds:
            role = frame0_role(args.splits_root, f["split"], f["fold_id"])
            row = {"fold_id": f["fold_id"], "split": f["split"],
                   "frame0_role": role, "T_A": f["T_A"], "T_C": f["T_C"],
                   "rules": {}}
            for rule, r in f["rules"].items():
                g = r.get("gate_noise_vs_tetris")
                u = r.get("residual_units", {})
                if not g or "P1" not in u or "TETRIS" not in u:
                    continue
                p1m, tem = u["P1"]["mean_r"], u["TETRIS"]["mean_r"]
                p1d, ted = u["P1"]["median_r"], u["TETRIS"]["median_r"]
                row["rules"][rule] = {
                    # as the JSON stores it: mean-block-r ratio
                    "p1_mean_r": float(p1m), "tetris_mean_r": float(tem),
                    "rel_from_mean_r": float(g["relative_reduction"]),
                    # the other estimator, for the levels the report quotes
                    "p1_median_r": float(p1d), "tetris_median_r": float(ted),
                    "rel_from_median_r": float(
                        (p1d - ted) / max(abs(ted), 1e-12)),
                    "sign_flip_p": float(g["sign_flip_p"]),
                    "paired_diff_ci95": [float(x) for x in g["paired_diff_ci95"]],
                    "passes_10pct": bool(g["passes_10pct_reduction"]),
                }
            rows.append(row)

        seq_out = {"n_folds_total": len(rows), "treatments": {},
                   "per_fold": rows}
        populations = {
            "all": lambda r: True,
            "f0_not_in_AB": lambda r: r["frame0_role"] not in ("A", "B"),
            "f0_in_AB": lambda r: r["frame0_role"] in ("A", "B"),
            "f0_in_C": lambda r: r["frame0_role"] == "C",
        }
        rules = sorted({k for r in rows for k in r["rules"]})
        for name, keep in populations.items():
            sub = [r for r in rows if keep(r)]
            entry = {"n_folds": len(sub),
                     "fold_ids": [r["fold_id"] for r in sub], "rules": {}}
            for rule in rules:
                cell = [r["rules"][rule] for r in sub if rule in r["rules"]]
                if not cell:
                    continue
                entry["rules"][rule] = {
                    "p1_mean_r": summarise([c["p1_mean_r"] for c in cell]),
                    "tetris_mean_r": summarise([c["tetris_mean_r"] for c in cell]),
                    "rel_from_mean_r": summarise([c["rel_from_mean_r"] for c in cell]),
                    "rel_from_median_r": summarise(
                        [c["rel_from_median_r"] for c in cell]),
                    "n_folds_passing_10pct": int(
                        sum(c["passes_10pct"] for c in cell)),
                }
            seq_out["treatments"][name] = entry
        out["sequences"][seq] = seq_out

        # ---- console table ------------------------------------------------
        print(f"\n=== {seq}  ({len(rows)} folds) ===")
        for name, entry in seq_out["treatments"].items():
            print(f"-- {name}  ({entry['n_folds']} folds)")
            for rule, v in entry["rules"].items():
                rm, rd = v["rel_from_mean_r"], v["rel_from_median_r"]
                print(f"   {rule:<13} "
                      f"P1 {v['p1_mean_r']['mean']:+.3f} "
                      f"TET {v['tetris_mean_r']['mean']:+.3f} | "
                      f"rel(mean) {rm['mean']:+.3f} "
                      f"[{rm['min']:+.3f},{rm['max']:+.3f}] | "
                      f"rel(med) {rd['mean']:+.3f} | "
                      f"pass {v['n_folds_passing_10pct']}/{rm['n_folds']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

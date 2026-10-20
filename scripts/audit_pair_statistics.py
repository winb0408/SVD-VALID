"""Audit recomputation for the Phase 2 residual-correlation gate.

This is the audit companion to `scripts/evaluate_pair_statistics.py`.  It does
NOT overwrite any Phase 2 artifact and it does NOT change the pre-registered
gate.  It answers one question the Phase 2 report could not:

    P1's arms carry the SVD projection straight into the residual statistic,
    while the TETRIS control is first block-averaged over each 2x2x2 cell and
    then repeated back.  How much of the reported "P1 is 50-111% worse than
    Tetris" is the observation construction, and how much is the spatial
    operator?

Two things are separated here that the Phase 2 report conflated:

  1. the *smoothing* (block-average + backfill), and
  2. the *phase split* (arm A keeps phases [0,1,2,4], arm B keeps [3,5,6,7]).

Variants, all applied to the SAME P1 arms so nothing else moves:

  as_impl       P1 arms untouched
  v2_same       both arms through PATTERN_A       -> adds smoothing only
  v1_compl      A through PATTERN_A, B through PATTERN_B  -> smoothing + split
  v3_full       both arms through all 8 phases    -> maximal smoothing
  tetris        the actual control (tetris_pair on the A u B band map)

`v1_compl` reproduces `tetris`'s operator exactly, so `v1_compl ~ tetris` is a
correctness check on this module, not a scientific result -- see
`tests/test_phase2_statistics.py::test_complementary_operator_matches_tetris_pair`.

It also runs the three sensitivity analyses the audit asked for:

  --task all        variants + block-size + kernel-size + ROI (one run)
  --task variants   only the operator decomposition
  --task blocks     only the spatial bootstrap block size sweep
  --task kernel     only the detrending kernel size sweep
  --task frames     only the frame-budget comparison (needs IQ; slow)

Usage:

    python scripts/audit_pair_statistics.py --task all \\
        --out reports/phase2_audit_rerun_2026-10-19

Every table it prints is also written as JSON next to the report so a reviewer
can recompute the aggregates without re-reading the volumes.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_pair_statistics import (                                    # noqa: E402
    BLOCK, PATTERN_A, PATTERN_B, block_residual_diffs, cropped_shape,
    normalised_box, pearson, residual_correlation, tetris_pair, to_cropped,
)
from datasets.sampling import one_phase                                   # noqa: E402

RULES = ("proportional", "clutter_knee", "spectral", "cum_energy")
ALL_PHASES = tuple(range(8))


# --------------------------------------------------------------- the operator ---
def tetris_operator(power_map, pattern, shape3d, valid3d):
    """`tetris_pair()`'s per-arm operator, on an arbitrary phase subset.

    Kept as a separate function so the audit can apply the *same* operator with
    different phase sets.  It is line-for-line the body of `tetris_pair()`; the
    test suite asserts that PATTERN_A/PATTERN_B reproduce `tetris_pair()`
    bit-for-bit, so a drift here fails loudly instead of silently changing the
    audit's conclusion.
    """
    pd3 = power_map.reshape(shape3d)
    phases = [one_phase(pd3, i) for i in range(8)]
    vph = [one_phase(valid3d.astype(np.float32), i) for i in range(8)]
    num = np.zeros_like(phases[0], dtype=np.float64)
    den = np.zeros_like(phases[0], dtype=np.float64)
    for i in pattern:
        num += phases[i].astype(np.float64) * vph[i]
        den += vph[i]
    block = num / np.maximum(den, 1e-12)
    block = block * (den > 0)
    return np.repeat(np.repeat(np.repeat(block, BLOCK, 0), BLOCK, 1), BLOCK, 2).ravel()


def block_mean_r(a, b, shape3d, cshape, v3c, size=4):
    vals, _ = block_residual_diffs(to_cropped(a, shape3d, cshape),
                                   to_cropped(b, shape3d, cshape), v3c, size=size)
    return float(vals.mean()), float(np.median(vals))


def load_seq(seq):
    valid = np.load(f"data/processed/official/{seq}/valid_mask.npy").ravel()
    shape3d = (37, 64, 64)
    v3 = valid.reshape(shape3d)
    cshape = cropped_shape(shape3d)
    return shape3d, v3, cshape, v3[:cshape[0]]


def fold_dirs(seq):
    return sorted(p.rsplit("/", 1)[0] for p in
                  glob.glob(f"data/processed/crossfit/{seq}/*/P1_proportional_A.npy"))


def load_arms(fd, rule):
    try:
        return np.load(f"{fd}/P1_{rule}_A.npy"), np.load(f"{fd}/P1_{rule}_B.npy")
    except FileNotFoundError:
        return None, None


# ------------------------------------------------------------------- variants ---
def task_variants(out_path):
    result = {"note": ("all variants applied to the same P1 arms; tetris is the "
                       "actual control and is a consistency anchor, not a result"),
              "sequences": {}}
    for seq in ("SA64", "HSA64"):
        shape3d, v3, cshape, v3c = load_seq(seq)
        result["sequences"][seq] = {}
        for fd in fold_dirs(seq):
            fid = fd.rsplit("/", 1)[1]
            rec = {}
            for rule in RULES:
                pa, pb = load_arms(fd, rule)
                if pa is None:
                    continue
                ta, tb = tetris_pair(pa, shape3d, v3)
                vals = {
                    "as_impl": block_mean_r(pa, pb, shape3d, cshape, v3c),
                    "tetris": block_mean_r(ta, tb, shape3d, cshape, v3c),
                    "v1_compl": block_mean_r(
                        tetris_operator(pa, PATTERN_A, shape3d, v3),
                        tetris_operator(pb, PATTERN_B, shape3d, v3), shape3d, cshape, v3c),
                    "v2_same": block_mean_r(
                        tetris_operator(pa, PATTERN_A, shape3d, v3),
                        tetris_operator(pb, PATTERN_A, shape3d, v3), shape3d, cshape, v3c),
                    "v3_full": block_mean_r(
                        tetris_operator(pa, ALL_PHASES, shape3d, v3),
                        tetris_operator(pb, ALL_PHASES, shape3d, v3), shape3d, cshape, v3c),
                }
                rec[rule] = {k: {"mean": v[0], "median": v[1]}
                             for k, v in vals.items()}
            result["sequences"][seq][fid] = rec
            print(f"  [{seq}] {fid}", flush=True)
    _dump(result, out_path / "variants.json")
    return result


# ------------------------------------------------------------------ block size ---
def task_blocks(out_path, n_folds=6):
    from scripts.phase3_stats import block_bootstrap_ci
    result = {"note": "as_impl and matched gate vs spatial bootstrap block size",
              "sequences": {}}
    for seq in ("SA64", "HSA64"):
        shape3d, v3, cshape, v3c = load_seq(seq)
        folds = fold_dirs(seq)[:n_folds]
        result["sequences"][seq] = {}
        print(f"== {seq} block size sweep ({len(folds)} folds)", flush=True)
        for size in (4, 6, 8, 12):
            acc = {"as": [], "tet": [], "mat": []}
            ci = []
            for fd in folds:
                pa, pb = load_arms(fd, "proportional")
                ta, tb = tetris_pair(pa, shape3d, v3)
                ma = tetris_operator(pa, PATTERN_A, shape3d, v3)
                mb = tetris_operator(pb, PATTERN_B, shape3d, v3)
                va, _ = block_residual_diffs(to_cropped(pa, shape3d, cshape),
                                             to_cropped(pb, shape3d, cshape), v3c, size=size)
                vt, _ = block_residual_diffs(to_cropped(ta, shape3d, cshape),
                                             to_cropped(tb, shape3d, cshape), v3c, size=size)
                vm, _ = block_residual_diffs(to_cropped(ma, shape3d, cshape),
                                             to_cropped(mb, shape3d, cshape), v3c, size=size)
                n = min(va.size, vt.size)
                lo, hi = block_bootstrap_ci(va[:n] - vt[:n], n_boot=4000, seed=7)
                acc["as"].append(va.mean()); acc["tet"].append(vt.mean())
                acc["mat"].append(vm.mean()); ci.append((lo, hi, int(va.size)))
            a, t, m = (float(np.mean(acc[k])) for k in ("as", "tet", "mat"))
            entry = {"n_units": ci[0][2], "as_impl": a, "tetris": t, "matched": m,
                     "rel_as_impl": (a - t) / abs(t), "rel_matched": (m - t) / abs(t),
                     "paired_diff_ci95_mean": [float(np.mean([c[0] for c in ci])),
                                               float(np.mean([c[1] for c in ci]))]}
            result["sequences"][seq][size] = entry
            print(f"  size {size:2d} units {entry['n_units']:5d} as {a:.3f} tet {t:.3f} "
                  f"mat {m:.3f} rel_as {entry['rel_as_impl']:+.3f} "
                  f"rel_m {entry['rel_matched']:+.3f}", flush=True)
    _dump(result, out_path / "blocksize.json")
    return result


# --------------------------------------------------------------- kernel + ROI ---
def task_kernel(out_path, n_folds=6):
    result = {"note": "detrending kernel size and ROI, as_impl vs matched",
              "sequences": {}}
    for seq in ("SA64", "HSA64"):
        shape3d, v3, cshape, v3c = load_seq(seq)
        valid = v3.ravel()
        ref = np.load(f"data/processed/official/{seq}/pd_linear.npy").ravel()
        tb, tv = np.percentile(ref[valid], (50.0, 99.0))
        rois = {
            "background": ((ref < tb) & valid).reshape(shape3d)[:cshape[0]],
            "mid": ((ref >= tb) & (ref < tv) & valid).reshape(shape3d)[:cshape[0]],
            "vessel": ((ref >= tv) & valid).reshape(shape3d)[:cshape[0]],
            "all_valid": v3c.copy(),
        }
        folds = fold_dirs(seq)[:n_folds]
        result["sequences"][seq] = {"kernel": {}, "roi": {}}

        def pair_maps(fd):
            pa, pb = load_arms(fd, "proportional")
            ta, tb2 = tetris_pair(pa, shape3d, v3)
            ma = tetris_operator(pa, PATTERN_A, shape3d, v3)
            mb = tetris_operator(pb, PATTERN_B, shape3d, v3)
            return pa, pb, ta, tb2, ma, mb

        for det in (5, 7, 9, 13, 17):
            acc = {"as": [], "tet": [], "mat": []}
            for fd in folds:
                pa, pb, ta, tb2, ma, mb = pair_maps(fd)

                def r_corr(a, b):
                    a3 = to_cropped(a, shape3d, cshape)
                    b3 = to_cropped(b, shape3d, cshape)
                    aa = a3 - normalised_box(a3, v3c, size=det)
                    bb = b3 - normalised_box(b3, v3c, size=det)
                    return pearson(aa[v3c], bb[v3c])

                acc["as"].append(r_corr(pa, pb)); acc["tet"].append(r_corr(ta, tb2))
                acc["mat"].append(r_corr(ma, mb))
            a, t, m = (float(np.mean(acc[k])) for k in ("as", "tet", "mat"))
            result["sequences"][seq]["kernel"][det] = {
                "as_impl": a, "tetris": t, "matched": m,
                "rel_as_impl": (a - t) / abs(t), "rel_matched": (m - t) / abs(t)}
            print(f"{seq} kernel {det:3d} as {a:.3f} tet {t:.3f} mat {m:.3f} "
                  f"rel_as {(a - t) / abs(t):+.3f} rel_m {(m - t) / abs(t):+.3f}", flush=True)

        for rn in rois:
            acc = {"as": [], "tet": [], "mat": []}
            for fd in folds:
                pa, pb, ta, tb2, ma, mb = pair_maps(fd)

                def rr(a, b):
                    a3 = to_cropped(a, shape3d, cshape)
                    b3 = to_cropped(b, shape3d, cshape)
                    return residual_correlation(a3, b3, v3c, rois, True)[rn]["r"]

                acc["as"].append(rr(pa, pb)); acc["tet"].append(rr(ta, tb2))
                acc["mat"].append(rr(ma, mb))
            a, t, m = (float(np.mean(acc[k])) for k in ("as", "tet", "mat"))
            result["sequences"][seq]["roi"][rn] = {
                "as_impl": a, "tetris": t, "matched": m,
                "rel_as_impl": (a - t) / abs(t), "rel_matched": (m - t) / abs(t)}
            print(f"{seq} roi {rn:11s} as {a:.3f} tet {t:.3f} mat {m:.3f} "
                  f"rel_as {(a - t) / abs(t):+.3f} rel_m {(m - t) / abs(t):+.3f}", flush=True)
    _dump(result, out_path / "kernel_roi.json")
    return result


# ---------------------------------------------------------------- frame budget ---
def task_frames(out_path):
    """Task C: does the Tetris control's 2x frame budget explain its low r?

    Builds the control from A-only (same budget as a P1 arm) and from A u B
    (the current setting).  If the two agree, the frame asymmetry is not the
    driver.  Needs the IQ volume, so it is the slow task.
    """
    import yaml
    from evaluate_pair_statistics import load_sequence_iq, projected_power_map
    from build_crossfit_pairs import eigh_desc, rule_edges

    base = yaml.safe_load(open("configs/phase0_baseline.yaml"))
    cfg = yaml.safe_load(open("configs/crossfit_pairs.yaml"))
    result = {"note": "Tetris control built from A-only vs A u B frames", "sequences": {}}
    for seq in ("HSA64",):
        shape3d, v3, cshape, v3c = load_seq(seq)
        iq4d = load_sequence_iq(seq, base)
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        gram = np.load(f"data/processed/crossfit/{seq}/gram_all.npy")
        fold = json.load(open("data/processed/crossfit/splits/hsa64_int3.json"))["folds"][0]
        c_idx = np.asarray(fold["calibration"], dtype=int)
        A = np.asarray(fold["A"], dtype=int)
        B = np.asarray(fold["B"], dtype=int)
        ev_c, vec_c = eigh_desc(gram[np.ix_(c_idx, c_idx)])
        result["sequences"][seq] = {"fold": fold["fold_id"], "rules": {}}
        for rule in ("proportional", "clutter_knee"):
            lo, hi = rule_edges(rule, cfg, ev_c)
            hi = min(hi, len(c_idx))
            basis = (vec_c[:, lo:hi] / np.sqrt(ev_c[lo:hi])[None, :]).astype(np.complex64)
            pa = projected_power_map(x2d, gram, c_idx, basis, tgt=A)
            pb = projected_power_map(x2d, gram, c_idx, basis, tgt=B)
            pab = projected_power_map(x2d, gram, c_idx, basis,
                                      tgt=np.concatenate([A, B]))
            entries = {"P1_native": block_mean_r(pa, pb, shape3d, cshape, v3c)[0]}
            for label, p in (("A_u_B", pab), ("A_only", pa), ("B_only", pb)):
                ta, tb = tetris_pair(p, shape3d, v3)
                entries[label] = block_mean_r(ta, tb, shape3d, cshape, v3c)[0]
            result["sequences"][seq]["rules"][rule] = entries
            print(f"{seq} {rule}: " + "  ".join(f"{k} {v:.3f}" for k, v in entries.items()),
                  flush=True)
    _dump(result, out_path / "frame_budget.json")
    return result


def _dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print(f"  wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all", "variants", "blocks", "kernel", "frames"])
    ap.add_argument("--out", default="reports/phase2_audit_rerun_2026-10-19")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.task in ("all", "variants"):
        print("== operator variants", flush=True); task_variants(out)
    if args.task in ("all", "blocks"):
        print("== block size", flush=True); task_blocks(out)
    if args.task in ("all", "kernel"):
        print("== kernel / ROI", flush=True); task_kernel(out)
    if args.task in ("all", "frames"):
        print("== frame budget", flush=True); task_frames(out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()

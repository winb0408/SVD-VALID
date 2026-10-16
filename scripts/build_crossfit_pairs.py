#!/usr/bin/env python3
"""Phase 1 / T1-2: build the cross-fitted observation pairs (P0 / P1 / P2).

The volume of a pair is a power Doppler image, i.e. one number per voxel.  What
differs between the three pair kinds is *which frames were allowed to influence
the spatial filtering*:

  P0  no filtering at all -- D is the raw IQ power of the group's own frames.
      The floor case: how much blood contrast survives with zero SVD.
  P1  the proposal -- both groups are filtered by the SAME projector
      P_C = U_{C,R} U_{C,R}^H, where U_{C,R} is the retained spatial basis of the
      calibration group C.  A and B contribute nothing to P_C.
  P2  the leaky reference -- each group is filtered by its OWN retained basis.
      No calibration group is involved.  This is what a per-image SVD does, and
      the gap between P2 and P1 is the price of not letting each observation
      choose its own filter.

Implementation note -- why the projector is never materialised.  With the
economy SVD ``X_basis = U S V^H``, the projector is

    P = U_R U_R^H = X_basis V_R S_R^-2 V_R^H X_basis^H

so ``Y = P X_target = X_basis (V_R S_R^-2 V_R^H) (X_basis^H X_target)``.  The
rightmost factor is a small ``T_basis x T_target`` Gram block, which we obtain by
slicing a Gram matrix computed ONCE for all 400 frames.  That turns a chain of
V x V = 151552^2 operations into T x T ones, and -- more importantly -- it makes
the cross-fitting explicit: the only calibration-derived objects are ``V_R`` and
``S_R``, both read off ``X_C^H X_C``.

Everything is done in float64/complex128.  The rank rules read the noise floor of
the spectrum, where the ratio between the largest and smallest eigenvalues runs
past 10^8, and complex64 accumulation would blur exactly the region that decides
the upper edge.

All QC statistics are restricted to the Phase 0 valid mask (17.96% of voxels are
identically zero because of the acquisition geometry; including them would
inflate any agreement number for free).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.iq_io import load_iq  # noqa: E402


# --------------------------------------------------------------------------
# rank rules -- each is a pure function of a descending eigenvalue spectrum
# --------------------------------------------------------------------------

def _noise_floor(evals, pct):
    n = max(3, int(round(len(evals) * pct / 100.0)))
    return float(np.median(evals[-n:]))


def rule_edges(name, cfg, evals):
    """(lo, hi) as a 0-based half-open mode band for a descending spectrum."""
    t = len(evals)
    spec = cfg["rank_rules"][name]
    kind = spec["kind"]

    if kind == "proportional_mapping":
        ref = spec["reference_frames"]
        n_band = int(round((spec["cut2_1based"] - spec["cut1_1based"] + 1) * t / ref))
        lo = int(round((spec["cut1_1based"] - 1) * t / ref))
        return max(0, lo), min(max(0, lo) + n_band, t)

    if kind == "threshold":
        below = evals <= spec["clutter_drop"] * evals[0]
        lo = int(np.argmax(below)) if below.any() else 0
        floor = _noise_floor(evals, spec["noise_floor_pct"])
        above = np.nonzero(evals > spec["snr_gate"] * floor)[0]
        hi = int(above[-1] + 1) if len(above) else 0
        return lo, hi

    if kind == "cumulative_energy":
        cum = np.cumsum(evals) / evals.sum()
        lo = int(np.searchsorted(cum, spec["clutter_energy_frac"]) + 1)
        tail = np.cumsum(evals[::-1])[::-1] / evals.sum()
        hi = int(np.searchsorted(-tail, -spec["discarded_energy_frac"]))
        return min(lo, t), max(min(hi, t), 0)

    if kind == "knee":
        # Where the steep clutter decay meets the flat tail.
        #
        # Not a max-curvature search: mode 1 alone carries ~93% of the energy, so
        # the second difference of log-eigenvalue peaks there, not at the bend.
        # Instead, fit the tail line over the last `tail_frac` of the spectrum and
        # take the first mode that is within `margin_db` of that line.
        log_ev = np.log10(evals)
        start = int(t * spec["tail_frac"])
        slope, intercept = np.polyfit(np.arange(start, t), log_ev[start:], 1)
        limit = slope * np.arange(t) + intercept + spec["margin_db"] / 10.0
        below = log_ev <= limit
        lo = int(np.argmax(below)) if below.any() else t
        return lo, t

    raise ValueError(f"unknown rank rule kind {kind!r}")


def rule_params(name, cfg):
    spec = {k: v for k, v in cfg["rank_rules"][name].items() if k != "kind"}
    return spec


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def eigh_desc(gram):
    """Eigen-decomposition of a Hermitian Gram matrix, descending by eigenvalue."""
    evals, vecs = np.linalg.eigh(gram)          # ascending
    return evals[::-1].copy(), vecs[:, ::-1].copy()


def retained_basis(x_basis, evals, vecs, lo, hi):
    """The retained spatial basis U_R of ``x_basis``, with orthonormal columns.

    U_R = X_basis V_R S_R^-1.  Forming U_R costs V x R complex numbers (a few
    hundred MB) but keeps the projection well conditioned: the algebraically
    equivalent ``X_basis (V_R S_R^-2 V_R^H) X_basis^H`` squares the condition
    number of the band, and on HSA64 that pushed the idempotence check to a
    relative error of 1.5e-6 -- small, but the fingerprint of a formulation that
    would silently lose precision in the flat tail where the rank rule lives.
    """
    return (x_basis @ vecs[:, lo:hi]) / np.sqrt(evals[lo:hi])


def projected_power(x_target, u_r, x_source=None, check_projector=False):
    """Power Doppler of ``x_target`` after projection onto span(U_R)."""
    y = u_r @ (u_r.conj().T @ x_target)         # V x T_target
    d = np.mean(np.abs(y) ** 2, axis=1).real.astype(np.float32)

    err = None
    if check_projector:
        # P is Hermitian and idempotent, so <P x, x> == ||P x||^2.  This catches
        # a wrong conjugation or a transposed basis, which are otherwise silent
        # -- the image still looks like a power Doppler either way.
        lhs = np.vdot(y, x_source if x_source is not None else x_target)
        rhs = float(np.sum(np.abs(y) ** 2))
        err = float(abs(lhs - rhs) / max(rhs, 1e-300))
    return d, err


def raw_power(x):
    return np.mean(np.abs(x) ** 2, axis=1).real.astype(np.float32)


# --------------------------------------------------------------------------
# QC
# --------------------------------------------------------------------------

def volume_qc(d, valid, lo_q, hi_q):
    v = d[valid]
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p1": float(np.percentile(v, 1)),
        "p50": float(np.percentile(v, 50)),
        "p999": float(np.percentile(v, 99.9)),
        "max": float(v.max()),
        "zero_fraction_in_valid": float(np.mean(v == 0)),
        "clamped_low_fraction": float(np.mean(v <= lo_q)),
        "clamped_high_fraction": float(np.mean(v >= hi_q)),
    }


def shared_mapping_bounds(da, db, valid, floor_pct=1.0, ceil_pct=99.9):
    """Intensity-mapping bounds from the POOLED valid voxels of A and B.

    Both volumes of a pair must be mapped with the same bounds taken from the
    same population; deriving them per-volume would quantile-normalise away
    exactly the differences Phase 2 is trying to measure.
    """
    pooled = np.concatenate([da[valid], db[valid]])
    return float(np.percentile(pooled, floor_pct)), float(np.percentile(pooled, ceil_pct))


def principal_angles(entry_i, entry_j, gram_all):
    """Cosines of the principal angles between two retained subspaces.

    U_i^H U_j = S_i^-1 V_i^H (X_i^H X_j) V_j S_j^-1, and the Gram block X_i^H X_j
    is a slice of the global Gram.  Subspace comparison therefore costs a few
    small matmuls -- no V x V work, and U never has to be stored.
    """
    g = gram_all[np.ix_(entry_i["C"], entry_j["C"])]
    vi = entry_i["vecs"][:, entry_i["lo"]:entry_i["hi"]]
    vj = entry_j["vecs"][:, entry_j["lo"]:entry_j["hi"]]
    si = np.sqrt(entry_i["evals"][entry_i["lo"]:entry_i["hi"]])
    sj = np.sqrt(entry_j["evals"][entry_j["lo"]:entry_j["hi"]])
    m = (vi.conj().T / si[:, None]) @ g @ (vj / sj[None, :])
    cos = np.linalg.svd(m, compute_uv=False)
    return float(cos.min()), float(cos.mean())


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def cross_fold_angles(entries, gram_all, min_pairs=1):
    """Subspace angles between every pair of C-subspaces in one sequence.

    Two questions, both answerable here and both cheap:
      * within a split -- does the retained band survive a different frame
        partition (stability of the rule)?
      * across splits -- two folds whose calibration frames are *different
        frames* still give nearly the same subspace.  If that happens, C is not
        doing any work and cross-fitting is ceremony; worth knowing before
        Phase 2 rather than after.
    """
    out = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            ei, ej = entries[i], entries[j]
            cos_min, cos_mean = principal_angles(ei, ej, gram_all)
            ci, cj = set(ei["C"].tolist()), set(ej["C"].tolist())
            out.append({
                "fold_i": ei["fold_id"], "fold_j": ej["fold_id"],
                "split_i": ei["split"], "split_j": ej["split"],
                "same_split": ei["split"] == ej["split"],
                "calibration_frames_shared": len(ci & cj),
                "rank_i": int(ei["hi"] - ei["lo"]), "rank_j": int(ej["hi"] - ej["lo"]),
                "cos_min": cos_min, "cos_mean": cos_mean,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--splits-root", default="data/processed/crossfit/splits")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--validation-only", action="store_true",
                    help="Report each rule's band on the full 400-frame spectrum and stop.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.baseline).read_text(encoding="utf-8"))
    rules = args.rules or list(cfg["rank_rules"])
    kinds = args.kinds or list(cfg["pair_kinds"])
    out_root = Path(cfg["output"]["pairs_root"])

    summary = {"config": args.config, "rules": rules, "kinds": kinds,
               "sequences": {}, "folds": [], "subspace_angles": []}

    for seq, spec in cfg["sequences"].items():
        if args.sequences and seq not in args.sequences:
            continue
        f = base["data"]["files"][seq]
        iq_path = str(Path(base["data"]["root"]) / f["path"])
        variable = "Data_" + seq
        valid = np.load(Path("data/processed/official") / seq / "valid_mask.npy").reshape(-1)
        print(f"\n=== {seq} ===  valid {valid.sum()}/{valid.size} "
              f"({100 * (1 - valid.mean()):.2f}% dead)")

        t0 = time.time()
        x = load_iq(iq_path, variable=variable)
        assert x.shape[:3] == tuple(f["shape_zyxt"][:3]), x.shape
        frames = int(x.shape[3])
        x_full = x.reshape(-1, frames).astype(np.complex128)
        del x
        print(f"loaded IQ {x_full.shape} ({time.time() - t0:.1f}s)")

        t0 = time.time()
        gram_all = x_full.conj().T @ x_full
        print(f"global Gram {gram_all.shape} ({time.time() - t0:.1f}s)")

        seq_registry = []

        # ---- validate the rank rules on the full 400-frame movie ----------
        validation = {}
        evals_full, _ = eigh_desc(gram_all)
        ref = cfg["rule_validation"]["reference_band_1based"]
        for name in rules:
            lo, hi = rule_edges(name, cfg, evals_full)
            validation[name] = {
                "band_0based_half_open": [lo, hi],
                "band_1based_inclusive": [lo + 1, hi],
                "n_modes": hi - lo,
                "official_band_1based_inclusive": ref,
                "official_n_modes": ref[1] - ref[0] + 1,
                "retained_energy_fraction": float(evals_full[lo:hi].sum() / evals_full.sum()),
            }
            print(f"  rule {name:<12} full-movie [{lo + 1},{hi}] = {hi - lo} modes "
                  f"vs official {ref} = {ref[1] - ref[0] + 1}")

        # The spectrum's shape decides which rank rules are even meaningful, so
        # it is reported at fixed probe points rather than assumed.
        probes = [0, 1, 5, 10, 20, 40, 59, 60, 100, 200, 290, 349, 350, 399]
        print("  spectrum (1-based mode: eigenvalue, dB rel. dominant):")
        for i in probes:
            print(f"    m{i + 1:<4} {evals_full[i]:.6e}  "
                  f"{10 * np.log10(evals_full[i] / evals_full[0]):8.2f} dB  "
                  f"cumTop={evals_full[:i + 1].sum() / evals_full.sum():.6f}  "
                  f"tail={evals_full[i:].sum() / evals_full.sum():.6e}")
        validation["spectrum_probe"] = {
            f"mode_{i + 1}": {
                "eval": float(evals_full[i]),
                "db_rel_dominant": float(10 * np.log10(evals_full[i] / evals_full[0])),
                "cum_from_top": float(evals_full[:i + 1].sum() / evals_full.sum()),
                "cum_from_bottom": float(evals_full[i:].sum() / evals_full.sum()),
            } for i in probes}

        (out_root / seq).mkdir(parents=True, exist_ok=True)
        np.save(out_root / seq / "spectrum_full.npy", evals_full)

        if args.validation_only:
            summary["sequences"][seq] = {"iq": iq_path, "frames": frames,
                                         "valid_voxels": int(valid.sum()),
                                         "rule_validation_full_movie": validation}
            del x_full, gram_all
            continue

        # ---- folds --------------------------------------------------------
        for split_file in sorted(Path(args.splits_root).glob(f"{seq.lower()}_*.json")):
            manifest = json.loads(split_file.read_text(encoding="utf-8"))
            print(f"\n-- {manifest['name']} ({len(manifest['folds'])} folds)")

            for fold in manifest["folds"]:
                fid = fold["fold_id"]
                c_idx = np.asarray(fold["calibration"], dtype=np.int64)
                a_idx = np.asarray(fold["A"], dtype=np.int64)
                b_idx = np.asarray(fold["B"], dtype=np.int64)

                # Re-assert the contract at the point of use, not just at build.
                assert not set(fold["calibration"]) & set(fold["A"])
                assert not set(fold["calibration"]) & set(fold["B"])
                assert not set(fold["A"]) & set(fold["B"])
                assert len(a_idx) == len(b_idx) > 0

                out_dir = out_root / seq / fid
                out_dir.mkdir(parents=True, exist_ok=True)

                x_c, x_a, x_b = x_full[:, c_idx], x_full[:, a_idx], x_full[:, b_idx]
                gram_c = gram_all[np.ix_(c_idx, c_idx)]
                gram_a = gram_all[np.ix_(a_idx, a_idx)]
                gram_b = gram_all[np.ix_(b_idx, b_idx)]

                rec = {"fold_id": fid, "sequence": seq, "split": manifest["name"],
                       "strategy": manifest["strategy"],
                       "T_C": int(len(c_idx)), "T_A": int(len(a_idx)), "T_B": int(len(b_idx)),
                       "trimmed_for_equal_length": fold["trimmed_for_equal_length"],
                       "rules": {}}

                # ---- P0 ---------------------------------------------------
                if "P0" in kinds:
                    da0, db0 = raw_power(x_a), raw_power(x_b)
                    lo_q, hi_q = shared_mapping_bounds(da0, db0, valid)
                    rec["rules"]["P0"] = {
                        "status": "OK", "rank": None, "band_0based_half_open": None,
                        "mapping_bounds": [lo_q, hi_q],
                        "A": volume_qc(da0, valid, lo_q, hi_q),
                        "B": volume_qc(db0, valid, lo_q, hi_q),
                    }
                    np.save(out_dir / "P0_A.npy", da0)
                    np.save(out_dir / "P0_B.npy", db0)

                evals_c, vecs_c = eigh_desc(gram_c)
                evals_a, vecs_a = eigh_desc(gram_a)
                evals_b, vecs_b = eigh_desc(gram_b)
                np.save(out_dir / "spectrum_C.npy", evals_c)

                for rule in rules:
                    lo, hi = rule_edges(rule, cfg, evals_c)
                    if hi <= lo:
                        rec["rules"][rule] = {"status": "EMPTY_BAND",
                                              "band_0based_half_open": [lo, hi]}
                        print(f"   {fid} {rule:<12} EMPTY BAND [{lo},{hi}) -- skipped")
                        continue

                    # P1: one shared calibration basis, applied to both groups.
                    d = {}
                    p1_err = {}
                    u_c = retained_basis(x_c, evals_c, vecs_c, lo, hi)
                    for side, tgt in (("A", x_a), ("B", x_b)):
                        dd, err = projected_power(tgt, u_c, x_source=tgt,
                                                  check_projector=True)
                        d[("P1", side)] = dd
                        p1_err[side] = err
                        if err > 1e-8:
                            raise AssertionError(
                                f"{fid} {rule} P1{side}: projector identity check failed "
                                f"(rel err {err:g}) -- not an orthogonal projection")
                    del u_c
                    # P2: each group's own projector.
                    for side, tgt, ev, vv in (("A", x_a, evals_a, vecs_a),
                                              ("B", x_b, evals_b, vecs_b)):
                        lo_s, hi_s = rule_edges(rule, cfg, ev)
                        if hi_s <= lo_s:
                            d[("P2", side)] = np.zeros(x_c.shape[0], dtype=np.float32)
                            continue
                        u_own = retained_basis(tgt, ev, vv, lo_s, hi_s)
                        dd, _ = projected_power(tgt, u_own, x_source=tgt)
                        d[("P2", side)] = dd
                        del u_own

                    qc, bounds = {}, {}
                    for k in ("P1", "P2"):
                        lo_q, hi_q = shared_mapping_bounds(d[(k, "A")], d[(k, "B")], valid)
                        bounds[k] = [lo_q, hi_q]
                        qc[k] = {"A": volume_qc(d[(k, "A")], valid, lo_q, hi_q),
                                 "B": volume_qc(d[(k, "B")], valid, lo_q, hi_q)}
                        np.save(out_dir / f"{k}_{rule}_A.npy", d[(k, "A")])
                        np.save(out_dir / f"{k}_{rule}_B.npy", d[(k, "B")])

                    # Subspaces: C's retained basis vs each group's own choice.
                    own = {}
                    for side, ev, vv, idx in (("A", evals_a, vecs_a, a_idx),
                                              ("B", evals_b, vecs_b, b_idx)):
                        lo_s, hi_s = rule_edges(rule, cfg, ev)
                        if hi_s <= lo_s:
                            own[side] = None
                            continue
                        e_side = {"C": c_idx, "evals": evals_c, "vecs": vecs_c,
                                  "lo": lo, "hi": hi}
                        e_own = {"C": idx, "evals": ev, "vecs": vv, "lo": lo_s, "hi": hi_s}
                        cos_min, cos_mean = principal_angles(e_side, e_own, gram_all)
                        own[side] = {"cos_min": cos_min, "cos_mean": cos_mean,
                                     "rank": int(hi_s - lo_s),
                                     "band_0based_half_open": [lo_s, hi_s]}
                        seq_registry.append({"sequence": seq, "split": manifest["name"],
                                         "fold_id": fid, "rule": rule, "role": f"own_{side}",
                                         "C": idx, "evals": ev, "lo": lo_s, "hi": hi_s,
                                         "vecs": vv})

                    seq_registry.append({"sequence": seq, "split": manifest["name"],
                                     "fold_id": fid, "rule": rule, "role": "C",
                                     "C": c_idx, "evals": evals_c,
                                     "lo": lo, "hi": hi, "vecs": vecs_c})

                    base_mean = float(rec["rules"]["P0"]["A"]["mean"]) if "P0" in kinds else None
                    rec["rules"][rule] = {
                        "status": "OK",
                        "kind": cfg["rank_rules"][rule]["kind"],
                        "params": rule_params(rule, cfg),
                        "rank": int(hi - lo),
                        "band_0based_half_open": [lo, hi],
                        "band_1based_inclusive": [lo + 1, hi],
                        "retained_energy_fraction": float(evals_c[lo:hi].sum() / evals_c.sum()),
                        "noise_floor_eval": _noise_floor(evals_c, 5),
                        "projector_check_rel_err": p1_err,
                        "own_rank_vs_C": own,
                        "mapping_bounds": bounds,
                        "qc": qc,
                        "mean_ratio_to_P0_A": (
                            float(d[("P1", "A")][valid].mean() / max(base_mean, 1e-30))
                            if base_mean else None),
                    }
                    print(f"   {fid} {rule:<12} R={hi - lo:<4} "
                          f"E={evals_c[lo:hi].sum() / evals_c.sum():.4f}  "
                          f"cos(C,ownA)={own['A']['cos_min'] if own['A'] else float('nan'):.3f}  "
                          f"cos(C,ownB)={own['B']['cos_min'] if own['B'] else float('nan'):.3f}")

                (out_dir / "pair_qc.json").write_text(json.dumps(rec, indent=2),
                                                      encoding="utf-8")
                summary["folds"].append(rec)
                del x_c, x_a, x_b

        # ---- cross-fold subspace angles (gram_all still in scope) ---------
        by_rule = {}
        for e in seq_registry:
            if e["role"] == "C":
                by_rule.setdefault(e["rule"], []).append(e)
        for rule, entries in sorted(by_rule.items()):
            angles = cross_fold_angles(entries, gram_all)
            for a in angles:
                a["sequence"], a["rule"] = seq, rule
            summary["subspace_angles"].extend(angles)
            same = [a["cos_mean"] for a in angles if a["same_split"]]
            diff = [a["cos_mean"] for a in angles if not a["same_split"]]
            print(f"  subspace angle[{rule}] same-split mean cos={np.mean(same):.4f} "
                  f"({len(same)} pairs), cross-split mean cos="
                  f"{np.mean(diff):.4f} ({len(diff)} pairs)" if diff else
                  f"  subspace angle[{rule}] same-split mean cos={np.mean(same):.4f}")

        summary["sequences"][seq] = {"iq": iq_path, "variable": variable, "frames": frames,
                                     "valid_voxels": int(valid.sum()),
                                     "rule_validation_full_movie": validation}
        del x_full, gram_all

    (out_root / "phase1_qc_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(f"\nwrote {out_root / 'phase1_qc_summary.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 2, item 1: is there temporally independent content, and does frame
disjointness make the pair's noise independent?

This is the cheapest measurement that can decide the whole project, so it is
done first and on its own.  The premise of the CrossFit scheme is:

    frames in A and frames in B are disjoint  ==>  the noise in D_A and D_B
    is independent  ==>  (D_A, D_B) is a usable Noise2Noise pair.

Nothing about SVD, VALID or training is needed to test the middle step.  What
is needed is the temporal autocorrelation of the data's frame-to-frame
fluctuation, and the autocorrelation of the per-frame power estimate.

WHY THIS NEEDS NO BIG-DATA PASS
-------------------------------
Let G[t, s] = <x(t), x(s)> = sum_v conj(x[v, t]) x[v, s] be the T x T temporal
Gram, G = W diag(lam) W^H.  The spatial basis is U_k = X w_k / sqrt(lam_k), so
the coefficient of frame t on mode k is

    c_k(t) = <U_k, x(t)> = w_k^H G[:, t] / sqrt(lam_k)
           = sqrt(lam_k) * conj(w_k[t])          (because w_k^H G = lam_k w_k^H)

i.e. the coefficient time series is just the scaled eigenvector.  Every
autocorrelation below is therefore a function of the Gram alone.  The identity

    sum_k |c_k(t)|^2 == G[t, t]                   (asserted below)

reconstructs the per-frame energy exactly and would fail loudly under a wrong
conjugation.  The `conj` in c_k(t) is not cosmetic: it fixes the SIGN of
Im rho(lag), so it is verified against the data explicitly (check_convention).

The Gram is cached so that later Phase 2 work needs no IQ load at all.

WHY PHYSICAL LAG, NOT GROUP INDEX
---------------------------------
For an interleaved 3-way split, A = {1, 4, 7, ...}, so "lag 1 within A" is
physical lag 3 and lag 1 can never be measured inside a group.  But the
premise is about pairs of frames from DIFFERENT groups, and for an interleaved
split every A frame is physically adjacent to a B frame.  So lags are counted
in physical frames and the cross-group pairing (t in A, t+lag in B) is a
first-class measurement, not a by-product.

WHY A PERMUTATION NULL
----------------------
An eigenvector of an estimated Gram is not a random unit vector: for the
lowest-variance modes the estimate is biased toward whatever the sample happens
to contain.  So the null for rho is not exactly 0 +/- 1/sqrt(n).  Both the
analytic 1/sqrt(n) and an empirical null from randomly permuting the frame
order are reported, and the larger of the two is treated as the null scale.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
It does not label any band "the noise band".  Phase 1 established that the
eigenspectrum is flat across the official band, so there is no defensible way
to point at a band and call it noise.  Instead the autocorrelation is reported
as a PROFILE over mode index (and over normalized mode index k/T, which is the
only way to compare bases estimated from different frame counts).  If no region
of the profile is white, the premise fails regardless of band choice; if some
region is white, the next question is whether that region carries any power --
which is why the power levels are reported in the same table and never
separately.

All statistics are restricted to the valid-voxel mask.  Dead voxels are
identically zero in the raw IQ, so they contribute exactly 0 to every inner
product and therefore to the Gram; the mask is applied explicitly wherever a
per-voxel quantity is formed, and is asserted to be a no-op for the Gram.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.iq_io import load_iq  # noqa: E402
from scripts.build_crossfit_pairs import eigh_desc, rule_edges  # noqa: E402
from scripts.make_crossfit_splits import assert_fold  # noqa: E402

# Physical frame lags to probe.  Dense at short lag (where independence lives),
# sparse later (where a slow drift would show up).
LAGS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)

# Short physical lags used for the cross-group premise test.  Lags that have no
# pairs for a given split (e.g. within a 5-way interleaved group) come back
# empty from masked_pairs and are simply skipped, so one list serves every
# split without special-casing.
SMALL_LAGS = (1, 2, 3, 5)

# Mode bins, as 1-based inclusive ranges over an ABSOLUTE mode index.  They are
# chosen to straddle every band any Phase 1 rule proposed, so that "is the
# residual band white?" can be read off without re-running.
MODE_BINS = ((1, 1), (2, 5), (6, 20), (21, 60), (61, 100), (101, 200),
             (201, 290), (291, 350), (351, 400))


def gram_path(root: Path, seq: str) -> Path:
    return root / seq / "gram_all.npy"


def load_or_build_gram(seq, base, root, valid, rebuild=False):
    """400x400 temporal Gram, cached to disk.  One IQ load total."""
    p = gram_path(root, seq)
    if p.exists() and not rebuild:
        print(f"  [gram] cached {p}")
        return np.load(p)

    f = base["data"]["files"][seq]
    iq_path = str(Path(base["data"]["root"]) / f["path"])
    t0 = time.time()
    x = load_iq(iq_path, variable="Data_" + seq)
    assert x.shape[:3] == tuple(f["shape_zyxt"][:3]), x.shape
    frames = int(x.shape[3])
    x_full = x.reshape(-1, frames).astype(np.complex128)
    del x
    print(f"  [gram] loaded IQ {x_full.shape} ({time.time() - t0:.1f}s)")

    # The mask is a no-op for the Gram: dead voxels are identically zero in the
    # raw IQ, so dropping their rows cannot change any inner product.  Asserting
    # it here means a later change that makes dead voxels nonzero (e.g. a
    # fill/interpolation step) cannot silently leak into the Gram.
    g_all = x_full.conj().T @ x_full
    if not valid.all():
        g_valid = x_full[valid].conj().T @ x_full[valid]
        rel = np.abs(g_all - g_valid).max() / np.abs(g_all).max()
        assert rel < 1e-12, (
            f"valid mask is NOT a no-op for the Gram (rel {rel:.3e}); dead "
            f"voxels are no longer zero, so the Gram needs the mask applied")
        print(f"  [gram] mask is a no-op for the Gram (rel {rel:.1e})")
    del x_full

    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, g_all)
    print(f"  [gram] computed {g_all.shape} ({time.time() - t0:.1f}s) -> {p}")
    return g_all


def check_convention(x_path, variable, gram, evals, vecs, n_modes=3):
    """Verify c_k(t) = sqrt(lam_k) conj(v_k[t]) against the data directly.

    The `conj` fixes the sign of Im rho(lag), and a wrong sign there is exactly
    the kind of silent error that still produces a plausible-looking plot.  So
    it is checked against an explicitly materialized basis rather than trusted
    from the algebra.
    """
    x = load_iq(x_path, variable=variable)
    frames = int(x.shape[3])
    x_full = x.reshape(-1, frames).astype(np.complex128)
    del x
    u = (x_full @ vecs[:, :n_modes]) / np.sqrt(evals[:n_modes])
    c_data = u.conj().T @ x_full                                  # n_modes x T
    c_formula = np.sqrt(evals[:n_modes])[:, None] * vecs[:, :n_modes].conj().T
    del x_full, u
    denom = np.abs(c_data).max()
    rel = float(np.abs(c_data - c_formula).max() / denom)
    assert rel < 1e-10, (
        f"coefficient convention check FAILED (rel {rel:.3e}); c_k(t) is not "
        f"sqrt(lam_k) conj(v_k[t])")
    return rel


def coeffs_from_basis(gram, idx_source, evals_s, vecs_s, idx_all):
    """c_k(t) = w_k^H G[source, t] / sqrt(lam_k) for every t in idx_all.

    Returns (n_modes, T_all) complex.  Only Gram slices are used, so applying a
    calibration-only basis to A and to B costs nothing extra.
    """
    sub = gram[np.ix_(idx_source, idx_all)]                       # Ts x Tall
    return (vecs_s.conj().T @ sub) / np.sqrt(evals_s)[:, None]


def corr_complex(a, b, null_draws=0, rng=None):
    """Complex Pearson correlation of two equal-length series, plus null scale.

    rho = sum_matched conj(a) b / sqrt(sum|a|^2 sum|b|^2).  This matched-pair
    normalization (rather than division by the total power) keeps |rho| <= 1
    even when the matched subset is short.  The null scale is max(analytic,
    empirical from `null_draws` frame-order permutations).
    """
    if a.size < 4:
        return None
    denom = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    if denom <= 0:
        return None
    rho = np.sum(np.conj(a) * b) / denom
    null = 1.0 / np.sqrt(a.size)
    if null_draws and rng is not None:
        perm = np.empty(null_draws)
        for i in range(null_draws):
            p = rng.permutation(b.size)
            perm[i] = abs(np.sum(np.conj(a) * b[p]) / denom)
        null = max(null, float(np.percentile(perm, 95)))
    return {"re": float(rho.real), "im": float(rho.imag),
            "abs": float(abs(rho)), "n": int(a.size), "null": float(null)}


def corr_real(a, b, null_draws=0, rng=None):
    """Real-series Pearson correlation with the same null treatment."""
    if a.size < 4:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom <= 0:
        return None
    r = float(np.sum(a * b) / denom)
    null = 1.0 / np.sqrt(a.size)
    if null_draws and rng is not None:
        perm = np.empty(null_draws)
        for i in range(null_draws):
            perm[i] = abs(np.sum(a * b[rng.permutation(b.size)]) / denom)
        null = max(null, float(np.percentile(perm, 95)))
    return {"r": r, "n": int(a.size), "null": float(null),
            "exceeds_null": bool(abs(r) > null)}


def masked_pairs(idx_left, idx_right, lag):
    """Positions t where left[t] and right[t + lag] are both available.

    idx_left / idx_right are sorted physical frame indices.  Returns (i, j) as
    positions INTO those arrays, so the caller indexes its own coefficient array
    and no assumption is made about either group being contiguous.
    """
    right_at = {int(f): k for k, f in enumerate(idx_right)}
    i, j = [], []
    for k, f in enumerate(idx_left):
        m = right_at.get(int(f) + lag)
        if m is not None:
            i.append(k)
            j.append(m)
    return np.asarray(i, dtype=int), np.asarray(j, dtype=int)


def available_cross_lags(idx_left, idx_right, max_lags=4, min_pairs=8):
    """The dominant physical separations between two groups of frames.

    Hardcoding a lag list for the cross-group test would be wrong for the
    contiguous split: there, A and B are whole blocks separated by the entire
    calibration block, so the characteristic separation is the block offset
    (~134 frames), not 1.  The contiguous split therefore CANNOT test
    adjacent-frame independence at all, and this function is what makes that
    visible instead of silently reporting the wrong lag.

    Candidates are ranked by HOW MANY pairs they carry, not by being smallest.
    That distinction matters: two equal-length blocks also overlap in a
    triangular fashion, so separations of 9, 10, 11, ... each carry a handful
    of pairs clustering at the block boundary.  Those are real frame pairs but
    they describe the boundary, not the separation between the groups.  The
    dominant separation carries ~T pairs and is selected here.

    Returns a list of {lag, n_pairs}, sorted by ascending lag.
    """
    left = np.asarray(idx_left, dtype=int)
    right = np.asarray(idx_right, dtype=int)
    diffs = (right[:, None] - left[None, :]).ravel()
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return []
    vals, counts = np.unique(diffs, return_counts=True)
    keep = counts >= min_pairs
    if not keep.any():
        return []
    vals, counts = vals[keep], counts[keep]
    order = np.lexsort((vals, -counts))          # most pairs first, then lag
    chosen = np.sort(vals[order[:max_lags]])
    return [{"lag": int(v), "n_pairs": int(counts[np.searchsorted(vals, v)])}
            for v in chosen]


def group_period(manifest):
    """Physical frame period at which within-group neighbours occur.

    For an interleaved n_groups split the groups are residue classes mod
    n_groups, so the smallest physical lag at which two frames of the SAME
    group appear is n_groups -- lag 1 simply does not exist inside such a
    group.  For a contiguous split it is 1.  This number is reported (and used
    to sanity-check the pair counts) but it is NOT used to label lags: every
    lag in this script is a physical frame difference, resolved by
    ``masked_pairs``.  Labelling group-index lag as "lag 1" would silently
    relabel physical lag 3 (or 5) as 1 and make the interleaved and contiguous
    splits look identical.
    """
    if manifest["strategy"] == "interleaved":
        return int(manifest["n_groups"]), int(manifest["n_groups"])
    return 1, int(manifest["n_groups"])


def analyse_sequence(seq, cfg, base, args, valid):
    """Return the full per-sequence result dict."""
    out_root = Path(cfg["output"]["pairs_root"])
    gram = load_or_build_gram(seq, base, out_root, valid, rebuild=args.rebuild_gram)
    frames = gram.shape[0]
    rng = np.random.default_rng(20260215)

    evals_full, vecs_full = eigh_desc(gram)
    print(f"  [basis] full-movie spectrum: lam1={evals_full[0]:.4e} "
          f"lam1/trace={evals_full[0] / evals_full.sum():.4f}")

    # ---- exact reconstruction identity (guards the conjugation) ------------
    c_full = np.sqrt(evals_full)[:, None] * vecs_full.conj().T
    per_frame = np.sum(np.abs(c_full) ** 2, axis=0)
    diag = np.real(np.diag(gram))
    rel = float(np.abs(per_frame - diag).max() / diag.max())
    assert rel < 1e-9, f"sum_k |c_k(t)|^2 != G[t,t] (rel {rel:.3e})"
    print(f"  [check] sum_k |c_k(t)|^2 == G[t,t]  (rel {rel:.2e})")

    result = {
        "sequence": seq,
        "frames": frames,
        "lam1_fraction": float(evals_full[0] / evals_full.sum()),
        "reconstruction_rel_err": rel,
        "trace": float(np.real(gram.trace())),
        "full_movie": {},
        "folds": [],
    }

    # ---- filter-free control: per-frame total energy ------------------------
    # G[t,t] = ||x(t)||^2 is the raw (no SVD) per-frame power, i.e. the P0
    # observation's per-frame series.  Its autocorrelation is the baseline that
    # every mode-resolved number below has to be read against: if the raw total
    # power is already strongly autocorrelated, then a correlated residual band
    # is not evidence about the noise, it is inherited from the acquisition.
    raw_energy = np.real(np.diag(gram))
    result["raw_per_frame_energy"] = {
        "mean": float(raw_energy.mean()),
        "std_over_mean": float(raw_energy.std() / raw_energy.mean()),
        "autocorr": {str(l): corr_real(raw_energy[:-l], raw_energy[l:],
                                       null_draws=args.null_draws, rng=rng)
                     for l in LAGS if raw_energy.size > l + 4},
    }

    # ---- full-movie basis: mode-resolved autocorrelation, physical lags -----
    full_prof = {}
    for name, (lo, hi) in {f"{a}-{b}": (a - 1, b) for a, b in MODE_BINS}.items():
        lo = max(0, lo)
        hi = min(frames, hi)
        if hi <= lo:
            continue
        row = {}
        for lag in LAGS:
            rho_k = []
            for k in range(lo, hi):
                c = c_full[k]
                # The empirical null costs null_draws correlations, so it is
                # taken on a capped subsample per bin; the analytic 1/sqrt(n)
                # covers the rest.
                r = corr_complex(c[:-lag], c[lag:],
                                 null_draws=args.null_draws if k < lo + 30 else 0,
                                 rng=rng)
                if r is not None:
                    rho_k.append(r)
            if not rho_k:
                continue
            row[str(lag)] = {
                "mean_re": float(np.mean([r["re"] for r in rho_k])),
                "mean_abs": float(np.mean([r["abs"] for r in rho_k])),
                "mean_im_abs": float(np.mean([np.abs(r["im"]) for r in rho_k])),
                "frac_exceed_null": float(np.mean([r["abs"] > r["null"] for r in rho_k])),
                "null": float(rho_k[0]["null"]),
                "n_modes": len(rho_k),
            }
        row["power_fraction"] = float(
            evals_full[lo:hi].sum() / evals_full.sum())
        row["mode_1based"] = [lo + 1, hi]
        full_prof[name] = row
    result["full_movie"]["mode_bins"] = full_prof

    # ---- per-fold: C-only basis applied to A and B -------------------------
    splits_root = Path(args.splits_root)
    manifests = []
    for spec in cfg["sequences"][seq]["splits"]:
        m = json.loads((splits_root / f"{spec['name']}.json").read_text())
        manifests.append((spec, m))

    for spec, manifest in manifests:
        period, n_groups = group_period(manifest)
        print(f"  [{manifest['name']}] strategy={manifest['strategy']} "
              f"n_groups={n_groups} within-group period={period} "
              f"folds={len(manifest['folds'])}")

        for fold in manifest["folds"]:
            assert_fold(fold)
            c_idx = np.asarray(fold["calibration"], dtype=int)
            a_idx = np.asarray(fold["A"], dtype=int)
            b_idx = np.asarray(fold["B"], dtype=int)
            all_idx = np.arange(frames)

            evals_c, vecs_c = eigh_desc(gram[np.ix_(c_idx, c_idx)])
            coh = coeffs_from_basis(gram, c_idx, evals_c, vecs_c, all_idx)

            rec = {
                "fold_id": fold["fold_id"],
                "split": manifest["name"],
                "strategy": manifest["strategy"],
                "group_period": period,
                "n_groups": n_groups,
                "T_C": int(c_idx.size), "T_A": int(a_idx.size), "T_B": int(b_idx.size),
                "rank_C": int(evals_c.size),
                "rules": {},
                "cross_group": {},
            }

            # (i) cross-group complex autocorrelation: the premise test itself.
            # t in one group, t+lag in another.  Under "frame disjoint => noise
            # independent" this must be ~0 at every available physical lag.
            # All lags here are physical frame differences, so the interleaved
            # and contiguous splits are directly comparable.
            for name_l, idx_l, name_r, idx_r in (
                    ("A", a_idx, "B", b_idx), ("A", a_idx, "C", c_idx),
                    ("B", b_idx, "C", c_idx), ("C", c_idx, "A", a_idx)):
                for spec_lag in available_cross_lags(idx_l, idx_r):
                    lag, n_possible = spec_lag["lag"], spec_lag["n_pairs"]
                    i, j = masked_pairs(idx_l, idx_r, lag)
                    if i.size < 4:
                        continue
                    # masked_pairs returns POSITIONS within each group, so they
                    # must be mapped back to physical frames before indexing the
                    # coefficient array (which is over all 400 frames).  Using
                    # the positions directly silently compares the wrong frames
                    # and, for a group whose frames start at an offset, produced
                    # a spurious rho of exactly 1.
                    fi, fj = idx_l[i], idx_r[j]
                    # average over the modes that carry most of the power and
                    # over a mid-rank slice, reported separately
                    for tag, klo, khi in (("topR", 0, min(20, evals_c.size)),
                                          ("mid", int(0.5 * evals_c.size),
                                           int(0.75 * evals_c.size)),
                                          ("bottom", max(0, evals_c.size - 20),
                                           evals_c.size)):
                        sub = coh[klo:khi]
                        if sub.size == 0:
                            continue
                        rho = np.mean([
                            np.sum(np.conj(sub[k, fi]) * sub[k, fj])
                            / max(np.sqrt(np.sum(np.abs(sub[k, fi]) ** 2)
                                          * np.sum(np.abs(sub[k, fj]) ** 2)), 1e-300)
                            for k in range(sub.shape[0])])
                        rec["cross_group"][f"{name_l}->{name_r}_lag{lag}_{tag}"] = {
                            "mean_re": float(np.real(rho)),
                            "mean_abs": float(abs(rho)),
                            "n_pairs": int(i.size),
                            "n_pairs_possible": int(n_possible),
                            "null": float(1.0 / np.sqrt(i.size)),
                        }

            # (ii) per-rule band: power-level autocorrelation and coefficient
            # autocorrelation, inside A, inside B, and across A->B.
            for rule in cfg["rank_rules"]:
                lo, hi = rule_edges(rule, cfg, evals_c)
                if hi <= lo:
                    rec["rules"][rule] = {"status": "EMPTY_BAND"}
                    continue
                p_all = np.sum(np.abs(coh[lo:hi]) ** 2, axis=0)     # per-frame PD
                band = {
                    "status": "OK",
                    "band_0based_half_open": [int(lo), int(hi)],
                    "band_1based_inclusive": [int(lo + 1), int(hi)],
                    "rank": int(hi - lo),
                    "power_fraction_of_trace": float(
                        evals_c[lo:hi].sum() / evals_c.sum()),
                    "power": {},
                }

                # Absolute level, so "no autocorrelation" cannot be confused
                # with "no power there".
                for grp, idx in (("C", c_idx), ("A", a_idx), ("B", b_idx)):
                    band["power"][grp] = {
                        "mean": float(p_all[idx].mean()),
                        "std": float(p_all[idx].std()),
                        "p50": float(np.percentile(p_all[idx], 50)),
                        "max": float(p_all[idx].max()),
                    }
                band["power_ratio_vs_raw_iq"] = float(
                    p_all.mean() / np.real(np.diag(gram)).mean())

                # Coefficient autocorrelation across A->B (the premise) at the
                # separations this split actually offers.
                coh_ab = {}
                for spec_lag in available_cross_lags(a_idx, b_idx):
                    lag = spec_lag["lag"]
                    i, j = masked_pairs(a_idx, b_idx, lag)
                    if i.size >= 4:
                        fi, fj = a_idx[i], b_idx[j]
                        sub = coh[lo:hi]
                        vals = [np.sum(np.conj(sub[k, fi]) * sub[k, fj])
                                / max(np.sqrt(np.sum(np.abs(sub[k, fi]) ** 2)
                                              * np.sum(np.abs(sub[k, fj]) ** 2)), 1e-300)
                                for k in range(sub.shape[0])]
                        coh_ab[f"A->B_lag{lag}"] = {
                            "mean_re": float(np.real(np.mean(vals))),
                            "mean_abs": float(np.abs(np.mean(vals))),
                            "n_pairs": int(i.size),
                            "n_pairs_possible": int(spec_lag["n_pairs"]),
                            "null": float(1.0 / np.sqrt(i.size)),
                        }
                band["coeff_cross_A_B"] = coh_ab

                # Power autocorrelation, which is what the pair actually
                # averages over frames.  `p_all` is indexed by PHYSICAL frame,
                # so pairs are resolved with masked_pairs at a physical lag;
                # slicing the group-indexed series instead would relabel
                # physical lag n_groups as lag 1.
                for grp, idx in (("A", a_idx), ("B", b_idx), ("C", c_idx)):
                    rows = {}
                    # The full lag list, not just the short one: inside an
                    # interleaved group only multiples of the group period have
                    # any pairs at all, so the decay has to be followed out to
                    # several periods to see where the correlation dies.  That
                    # decay length is the one actionable design knob here --
                    # it says how wide an interleave the acquisition needs.
                    for lag in LAGS:
                        i, j = masked_pairs(idx, idx, lag)
                        r = corr_real(p_all[idx][i], p_all[idx][j],
                                      null_draws=args.null_draws, rng=rng)
                        if r is not None:
                            r["physical_lag"] = int(lag)
                            rows[str(lag)] = r
                    band["power"][f"autocorr_{grp}"] = {
                        "note": "lag keys are PHYSICAL frame differences",
                        "lags": rows,
                    }
                rec["rules"][rule] = band

            result["folds"].append(rec)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--splits-root", default="data/processed/crossfit/splits")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--null-draws", type=int, default=200,
                    help="frame-order permutations for the empirical null")
    ap.add_argument("--rebuild-gram", action="store_true")
    ap.add_argument("--check-convention", action="store_true",
                    help="verify c_k(t)=sqrt(lam_k)conj(v_k[t]) against the IQ")
    ap.add_argument("--out", default="data/processed/crossfit/phase2_autocorr.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.baseline).read_text(encoding="utf-8"))

    allowed = set(cfg["sequences"])  # noqa: F841  (kept for the guard below)
    results = {}
    for seq in cfg["sequences"]:
        if args.sequences and seq not in args.sequences:
            continue
        print(f"\n=== {seq} ===")
        valid = np.load(Path("data/processed/official") / seq
                        / "valid_mask.npy").reshape(-1)
        if args.check_convention:
            f = base["data"]["files"][seq]
            g = load_or_build_gram(seq, base,
                                   Path(cfg["output"]["pairs_root"]), valid)
            evals, vecs = eigh_desc(g)
            rel = check_convention(str(Path(base["data"]["root"]) / f["path"]),
                                   "Data_" + seq, g, evals, vecs)
            print(f"  [check] coefficient convention verified on 3 modes "
                  f"(rel {rel:.2e})")
            continue
        results[seq] = analyse_sequence(seq, cfg, base, args, valid)

    # A verification or filter-only invocation must never clobber a finished
    # measurement: an earlier run of `--check-convention` overwrote the full
    # JSON with `{}`, which then looked like a legitimate empty result.
    if args.check_convention:
        if args.out != ap.get_default("out"):
            print("\n[warning] --out is ignored by --check-convention: this run "
                  "measures nothing and writes nothing")
        print("\n[check-convention] no measurement performed; output not written")
        return
    if not results:
        raise SystemExit(
            f"refusing to write {args.out}: no sequence was analysed "
            "(did --sequences filter everything out?)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 2, item 1 (second half): the grouped residual report that plan v0.3
makes mandatory before any conclusion about item 1 may be believed.

Why this file exists
--------------------
Plan v0.3 §5 item 1 says, in effect: measuring only "is the residual band
temporally autocorrelated?" cannot distinguish

    (i)  the residual contains no noise                       -> the pair is dead
    (ii) the residual contains noise, but the projection
         already removed it (together with the frame-specific
         structure that makes up the wide-band subspace)     -> projection too strong

and those two imply opposite next moves (stop, versus narrow the band).  So the
residual must be reported as a TRIPLE -- {autocorrelation, absolute power,
power relative to the unfiltered P0} -- and it must be reported for the wide
and the narrow band construction separately.  A conclusion missing the last two
entries is explicitly not admissible.

Why it is still cheap
---------------------
No IQ pass is needed for the global statistics.  Let G = X^H X be the cached
400x400 temporal Gram and C the calibration frame indices.  Every ingredient
here is a slice of G:

  * C's spectrum is the eigen-decomposition of the SUBMATRIX G[C, C], which is
    exactly X_C^H X_C;
  * the coefficients of any frame t on C's modes are
    c_k(t) = w_k^H G[C, t] / sqrt(lam_k)          (w_k from that submatrix eig);
  * the unfiltered P0 power of frame t is G[t, t].

So a calibration-only basis can be applied to A and B without ever forming X_C.
The identity sum_k |c_k(t)|^2 == G[t,t] is asserted, exactly as in
phase2_noise_autocorr.py, and it is what makes the shortcut checkable rather
than merely plausible.

What "the residual band" means here
-----------------------------------
The official band is 1-based 60:350 inclusive out of 400 frames.  Phase 1
established that the band's upper edge cannot be identified from the spectrum,
and that the proportional rule reproduces the official boundary as a fixed
fraction of the calibration frame count.  So the noise-proxy band is defined as

    proxy = [hi_proportional, T_C)     hi_proportional = rule_edges("proportional")

i.e. everything above the official band's upper edge, mapped into C's basis by
the same rule the project already froze.  It is a single, rule-independent
definition, so the wide/narrow comparison below is not smuggled in through the
band choice.

Two quantities are reported for every fold, because they answer different
questions and the plan's 判读 clause needs the second:

  * `proxy`     -- the noise-proxy band above the official edge.  This is the
                   band that may be called a noise proxy on the ORIGINAL
                   reasoning.  Does it carry power, and is it white?
  * `discarded` -- everything the rule's projection throws away, i.e. the
                   complement of the retained band.  This is what actually
                   leaves the observation, so it is what "the residual" means
                   when asking whether the projector removed the noise.

Deliberately not done here
--------------------------
No band is labelled "the noise band" as a fact; the report states what the
proxy band carries and whether it is white, and leaves the labelling to the
interpretation.  Depth/ROI-resolved versions are produced by the same reduction
applied to depth-slab Grams; those slabs are built only when --depth-slabs > 0,
because a slab Gram cannot be sliced out of the full Gram.
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
from scripts.phase2_noise_autocorr import (  # noqa: E402
    LAGS,
    available_cross_lags,
    coeffs_from_basis,
    corr_real,
    gram_path,
    group_period,
    masked_pairs,
)

WIDE_RULE = "proportional"
NARROW_RULE = "spectral"

# Physical lags reported for the residual band.  Dense at short lag, where the
# N2N premise lives, and including 6/15/30 because inside an interleaved group
# of period 3 or 5 only multiples of the period have pairs at all -- without
# them the interleaved splits show a single point and no decay.
RESIDUAL_LAGS = (1, 2, 3, 5, 6, 10, 15, 20, 30)


# --------------------------------------------------------------------------
# Grams (one IQ load, optionally with depth slabs)
# --------------------------------------------------------------------------
def build_grams(seq, base, out_root, valid, n_slabs=0, rebuild=False):
    """Return (gram_full, [(label, gram_slab), ...]).

    A depth-slab Gram is NOT a slice of the full Gram -- the inner product is
    taken over voxels, so every slab needs its own product.  All of them are
    formed from a single load of the IQ volume.
    """
    p = gram_path(out_root, seq)
    slab_dir = out_root / seq
    slab_paths = [slab_dir / f"gram_{lab}.npy" for lab in slab_labels(seq, base, n_slabs)]
    have_slabs = all(sp.exists() for sp in slab_paths)

    if p.exists() and have_slabs and not rebuild:
        print(f"  [gram] cached {p}" + (f" (+{len(slab_paths)} slabs)" if n_slabs else ""))
        return (np.load(p),
                [(lab, np.load(sp)) for lab, sp in zip(slab_labels(seq, base, n_slabs),
                                                       slab_paths)])

    f = base["data"]["files"][seq]
    iq_path = str(Path(base["data"]["root"]) / f["path"])
    t0 = time.time()
    x = load_iq(iq_path, variable="Data_" + seq)
    z, xx, yy, frames = x.shape
    x2 = x.reshape(z, xx * yy, frames).astype(np.complex128)
    del x
    print(f"  [gram] loaded IQ {(z, xx * yy, frames)} ({time.time() - t0:.1f}s)")

    flat = x2.reshape(z * xx * yy, frames)
    g_full = flat.conj().T @ flat
    if not valid.all():
        gv = flat[valid].conj().T @ flat[valid]
        rel = np.abs(g_full - gv).max() / np.abs(g_full).max()
        assert rel < 1e-12, (
            f"valid mask is NOT a no-op for the Gram (rel {rel:.3e}); dead "
            f"voxels are no longer zero")
        print(f"  [gram] mask is a no-op for the Gram (rel {rel:.1e})")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, g_full)

    slabs = []
    for lab, z0, z1 in slab_ranges(seq, base, n_slabs):
        sub = x2[z0:z1].reshape(-1, frames)
        g_s = sub.conj().T @ sub
        sp = slab_dir / f"gram_{lab}.npy"
        np.save(sp, g_s)
        slabs.append((lab, g_s))
        print(f"  [gram] slab {lab} z=[{z0},{z1}) -> {sp}")
    print(f"  [gram] done ({time.time() - t0:.1f}s)")
    return g_full, slabs


def slab_labels(seq, base, n_slabs):
    return [lab for lab, _, _ in slab_ranges(seq, base, n_slabs)]


def slab_ranges(seq, base, n_slabs):
    if n_slabs <= 0:
        return []
    z = int(base["data"]["files"][seq]["shape_zyxt"][0])
    edges = np.linspace(0, z, n_slabs + 1).round().astype(int)
    return [(f"z{z0}_{z1}", int(z0), int(z1))
            for z0, z1 in zip(edges[:-1], edges[1:]) if z1 > z0]


# --------------------------------------------------------------------------
# The triple
# --------------------------------------------------------------------------
def residual_triple(gram, c_idx, a_idx, b_idx, evals_c, vecs_c, rule, cfg, rng,
                    null_draws):
    """{autocorrelation, absolute power, power / P0} for one band, one fold."""
    lo, hi = rule_edges(rule, cfg, evals_c)
    if hi <= lo:
        return None
    coh = coeffs_from_basis(gram, c_idx, evals_c, vecs_c, np.arange(gram.shape[0]))
    p0 = np.real(np.diag(gram))                       # unfiltered per-frame power

    bands = {
        # Above the official band's upper edge.  Rule-independent, and the only
        # one of the three that the original reasoning could call a noise proxy.
        "proxy": [(rule_edges(WIDE_RULE, cfg, evals_c)[1], evals_c.size)],
        # What the rule's projection removes, split in two, because lumping
        # them together makes the question unanswerable: mode 1 alone is >93%
        # of the trace, so a combined "discarded" band is >99.8% clutter and
        # reports the clutter's autocorrelation no matter what the noise does.
        #   below -- the low-rank clutter the projection is designed to remove
        #   above -- the high-rank tail it removes as a side effect; THIS is the
        #            part that may or may not contain usable noise
        "discarded_below": [(0, lo)],
        "discarded_above": [(hi, evals_c.size)],
    }

    out = {}
    for name, parts in bands.items():
        parts = [pp for pp in parts if pp[1] > pp[0]]
        n_modes = int(sum(b - a for a, b in parts))
        if n_modes == 0:
            out[name] = {"status": "EMPTY"}
            continue
        p_band = np.zeros(gram.shape[0])
        for a, b in parts:
            p_band += np.sum(np.abs(coh[a:b]) ** 2, axis=0)

        rec = {"status": "OK", "n_modes": n_modes,
               "bands_0based_half_open": [[int(a), int(b)] for a, b in parts],
               "power_fraction_of_trace": float(
                   sum(evals_c[a:b].sum() for a, b in parts) / evals_c.sum())}

        # --- absolute power and the ratio the plan requires ---------------
        for grp, idx in (("A", a_idx), ("B", b_idx)):
            rec[f"power_{grp}"] = float(p_band[idx].mean())
            rec[f"p0_{grp}"] = float(p0[idx].mean())
            rec[f"power_over_P0_{grp}"] = float(
                p_band[idx].mean() / p0[idx].mean()) if p0[idx].mean() else None

        # --- temporal autocorrelation of the band power -------------------
        rec["autocorr_A"] = {}
        for lag in RESIDUAL_LAGS:
            i, j = masked_pairs(a_idx, a_idx, lag)
            r = corr_real(p_band[a_idx][i], p_band[a_idx][j],
                          null_draws=null_draws, rng=rng)
            if r is not None:
                rec["autocorr_A"][str(lag)] = r
        # --- and across the pair, at the separations this split offers ----
        rec["autocorr_A_to_B"] = {}
        for spec_lag in available_cross_lags(a_idx, b_idx):
            lag = spec_lag["lag"]
            i, j = masked_pairs(a_idx, b_idx, lag)
            r = corr_real(p_band[a_idx][i], p_band[b_idx][j],
                          null_draws=null_draws, rng=rng)
            if r is not None:
                rec["autocorr_A_to_B"][str(lag)] = r
        # --- and inside the calibration group, which is the only group that
        # was allowed to shape the projection (a useful upper reference) ----
        rec["autocorr_C"] = {}
        for lag in RESIDUAL_LAGS:
            i, j = masked_pairs(c_idx, c_idx, lag)
            r = corr_real(p_band[c_idx][i], p_band[c_idx][j],
                          null_draws=null_draws, rng=rng)
            if r is not None:
                rec["autocorr_C"][str(lag)] = r

        # The band-power series and the mean of the per-mode correlations are
        # NOT the same statistic, and they can even disagree in sign: the band
        # power weights each mode by its power, the mean over modes weights
        # them equally.  Reporting only the first invites reading a band-level
        # negative as "no correlation" when individual modes are correlated.
        # The per-mode version is included so the two can be compared.
        rec["mode_mean_r"] = {}
        for lag in RESIDUAL_LAGS:
            i, j = masked_pairs(a_idx, a_idx, lag)
            if i.size < 4:
                continue
            per_mode = []
            for a, b in parts:
                for k in range(a, b):
                    vals = np.abs(coh[k, a_idx]) ** 2
                    rr = corr_real(vals[i], vals[j])
                    if rr is not None:
                        per_mode.append(rr["r"])
            if per_mode:
                rec["mode_mean_r"][str(lag)] = {
                    "mean_r": float(np.mean(per_mode)), "n_modes": len(per_mode)}
        out[name] = rec
    return out


def analyse(seq, label, gram, cfg, base, args, valid):
    frames = gram.shape[0]
    rng = np.random.default_rng(20260215)
    evals_f, _ = eigh_desc(gram)
    res = {"sequence": seq, "volume": label, "frames": frames,
           "lam1_fraction": float(evals_f[0] / evals_f.sum()), "folds": []}

    splits_root = Path(args.splits_root)
    for spec in cfg["sequences"][seq]["splits"]:
        manifest = json.loads((splits_root / f"{spec['name']}.json").read_text())
        period, n_groups = group_period(manifest)
        for fold in manifest["folds"]:
            assert_fold(fold)
            c_idx = np.asarray(fold["calibration"], dtype=int)
            a_idx = np.asarray(fold["A"], dtype=int)
            b_idx = np.asarray(fold["B"], dtype=int)
            evals_c, vecs_c = eigh_desc(gram[np.ix_(c_idx, c_idx)])

            # same guard as the autocorrelation script: a wrong conjugation
            # would still produce a plausible-looking number.  It can only be
            # exact on C's OWN frames -- for t outside C, summing over C's
            # modes reconstructs P_C x(t), not x(t), so the shortfall is not an
            # error but the cross-fit capture ratio reported just below.
            coh = coeffs_from_basis(gram, c_idx, evals_c, vecs_c, np.arange(frames))
            per_frame = np.sum(np.abs(coh) ** 2, axis=0)
            diag = np.real(np.diag(gram))
            rel = float(np.abs(per_frame[c_idx] - diag[c_idx]).max() / diag[c_idx].max())
            assert rel < 1e-9, f"{fold['fold_id']}: identity fails (rel {rel:.3e})"

            rec = {"fold_id": fold["fold_id"], "split": manifest["name"],
                   "strategy": manifest["strategy"], "group_period": period,
                   "n_groups": n_groups, "T_C": int(c_idx.size),
                   "T_A": int(a_idx.size), "T_B": int(b_idx.size),
                   "identity_rel_err_on_C": rel,
                   # How much of an OUTSIDE frame's energy lies inside
                   # span(X_C) at all.  This is the "frame-specific structure"
                   # the plan's v0.3 note worries about, in the form the Gram
                   # can measure directly.  It is computed with ALL of C's
                   # modes, i.e. before any band is chosen.
                   "capture_fraction": {
                       grp: float(per_frame[idx].sum() / diag[idx].sum())
                       for grp, idx in (("A", a_idx), ("B", b_idx))},
                   "rules": {}}
            for rule in (WIDE_RULE, NARROW_RULE):
                triple = residual_triple(gram, c_idx, a_idx, b_idx, evals_c,
                                         vecs_c, rule, cfg, rng, args.null_draws)
                if triple is not None:
                    rec["rules"][rule] = triple
            res["folds"].append(rec)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--splits-root", default="data/processed/crossfit/splits")
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--null-draws", type=int, default=50)
    ap.add_argument("--depth-slabs", type=int, default=0,
                    help="also report depth-slab Grams (needs one IQ load)")
    ap.add_argument("--rebuild-gram", action="store_true")
    ap.add_argument("--out", default="data/processed/crossfit/phase2_residual_proxy.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.baseline).read_text(encoding="utf-8"))
    out_root = Path(cfg["output"]["pairs_root"])

    results, checked = {}, 0
    for seq in cfg["sequences"]:
        if args.sequences and seq not in args.sequences:
            continue
        print(f"\n=== {seq} ===")
        valid = np.load(Path("data/processed/official") / seq
                        / "valid_mask.npy").reshape(-1)
        g_full, slabs = build_grams(seq, base, out_root, valid,
                                    n_slabs=args.depth_slabs,
                                    rebuild=args.rebuild_gram)
        results[seq] = {"whole_volume": analyse(seq, "whole_volume", g_full, cfg,
                                                base, args, valid)}
        checked += len(results[seq]["whole_volume"]["folds"])
        for lab, g_s in slabs:
            r = analyse(seq, lab, g_s, cfg, base, args, valid)
            r["z_range"] = lab
            results[seq][lab] = r
            checked += len(r["folds"])
        print(f"  [done] {seq}: {len(results[seq]) - 1} depths + whole volume")

    if not results:
        raise SystemExit("refusing to write: no sequence analysed")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}  ({checked} fold-volumes)")


if __name__ == "__main__":
    main()

"""Print the Phase 2 item-1 report tables from the autocorrelation JSON.

This exists so that every number quoted in `reports/phase2_noise_autocorrelation.md`
can be regenerated from `data/processed/crossfit/phase2_autocorr.json` instead of
being transcribed by hand.  No measurement happens here: it is pure aggregation
of `phase2_noise_autocorr.py`'s output.

Usage:
    python scripts/phase2_summarize.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_JSON = Path("data/processed/crossfit/phase2_autocorr.json")
DEFAULT_RESIDUAL = Path("data/processed/crossfit/phase2_residual_proxy.json")
DEFAULT_GRAM_ROOT = Path("data/processed/crossfit")

# The lag that actually separates two frames of the same A group, per split.
# For an interleaved split with period p, frames of one group sit p apart;
# for a contiguous split the neighbouring frames sit 1 apart, which is exactly
# the comparison the interleave is meant to avoid -- that is the point of the
# table, so both are printed on their own terms.
PERIOD = {"sa64_int3": 3, "hsa64_int3": 3, "sa64_int5": 5, "sa64_cont3": 1}


def h(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def by_split(doc):
    out = defaultdict(list)
    for fold in doc["folds"]:
        out[fold["split"]].append(fold)
    return out


def table_raw_energy(doc):
    h("1. Raw per-frame total IQ energy (NO filtering at all)")
    print("This is the control: it needs no basis and no band, so it cannot be")
    print("an artefact of the SVD.  If even this is autocorrelated, then frame")
    print("disjointness alone does not buy independence.")
    print()
    print(f"{'seq':8s} {'mean energy':>16s} {'std/mean':>9s} "
          + " ".join(f"r({L})" for L in (1, 2, 3, 5, 10, 20, 50)) + "   null(1)")
    for seq, s in doc.items():
        rpe = s["raw_per_frame_energy"]
        ac = rpe["autocorr"]
        vals = " ".join(f"{ac[str(L)]['r']:+7.3f}" for L in (1, 2, 3, 5, 10, 20, 50))
        print(f"{seq:8s} {rpe['mean']:16.4g} {rpe['std_over_mean']:9.4f} {vals}   "
              f"{ac['1']['null']:.4f}")


def table_within_group(doc):
    h("2. Band-power correlation WITHIN one observation group (A), at the lag")
    print("   that separates two frames of the SAME group.  This is the quantity")
    print("   an N2N pair needs to be small: it is how much of A's band power is")
    print("   reproducible from B, i.e. signal the network may copy.")
    print()
    print(f"{'split':12s} {'strat':11s} {'p':>2s} {'rule':13s} {'n_fold':>6s} "
          f"{'r(A)@p':>8s} {'range':>16s} {'null':>7s} {'exceeds':>7s}")
    for seq, s in doc.items():
        for split, folds in sorted(by_split(s).items()):
            p = PERIOD[split]
            for rule in ("proportional", "spectral"):
                rs, nuls, exc = [], [], 0
                for f in folds:
                    lag = f["rules"][rule]["power"][f"autocorr_{'A'}"]["lags"].get(str(p))
                    if lag is None:
                        continue
                    rs.append(lag["r"])
                    nuls.append(lag["null"])
                    exc += bool(lag["exceeds_null"])
                if not rs:
                    continue
                print(f"{split:12s} {folds[0]['strategy']:11s} {p:2d} {rule:13s} "
                      f"{len(rs):6d} {mean(rs):+8.3f} "
                      f"[{min(rs):+.3f},{max(rs):+.3f}] {mean(nuls):7.3f} "
                      f"{exc}/{len(rs)}")


def table_contiguous_curve(doc):
    h("3. Within ONE contiguous window: the decay of band power with lag")
    print("   (SA64 sa64_cont3 fold c0a1b2, proportional band).  This is the best")
    print("   available estimate of the correlation LENGTH of the band power, and")
    print("   it is what an interleave has to beat.  Compare with table 2: the real")
    print("   interleaved groups sit BELOW this curve.")
    print()
    fold = next(f for f in doc["SA64"]["folds"] if f["fold_id"] == "sa64_cont3_c0a1b2")
    for rule in ("proportional", "spectral"):
        lags = fold["rules"][rule]["power"]["autocorr_A"]["lags"]
        print(f"  {rule:13s} " + "  ".join(
            f"{L}:{lags[L]['r']:+.3f}" for L in sorted(lags, key=int)))


def table_power_levels(doc):
    h("4. POWER AFTER projection onto C's basis, per group")
    print("   CAREFUL: this is NOT a drift diagnostic and must not be read as one.")
    print("   These are the P1 powers -- each group's retained-band power as seen")
    print("   through C's own basis -- so they measure CROSS-FIT ENERGY RETENTION")
    print("   (how well C's wide-band subspace covers that group), which is Phase")
    print("   1's territory, not how far apart the groups sit in the drift.")
    print("   The split mean is anyway unfair to the contiguous split: its three")
    print("   folds are cyclic relabellings of the same blocks, so a block-level")
    print("   effect cancels in the mean and survives only per fold -- printed too.")
    print("   Table 4b below is the actual drift diagnostic.")
    print()
    print(f"{'split':12s} {'rule':13s} {'rank':>4s} {'C':>11s} {'A':>11s} {'B':>11s} "
          f"{'band/raw':>9s} {'band/trace':>10s}   per-fold B/A")
    for seq, s in doc.items():
        for split, folds in sorted(by_split(s).items()):
            for rule in ("proportional", "spectral"):
                Cs, As, Bs, ratios, per_fold, rank = [], [], [], [], [], None
                for f in folds:
                    r = f["rules"][rule]
                    if r.get("status") != "OK":
                        continue
                    Cs.append(r["power"]["C"]["mean"])
                    As.append(r["power"]["A"]["mean"])
                    Bs.append(r["power"]["B"]["mean"])
                    ratios.append(r["power_ratio_vs_raw_iq"])
                    a, b = r["power"]["A"]["mean"], r["power"]["B"]["mean"]
                    per_fold.append(b / a if a else float("nan"))
                    rank = r["rank"]
                if not Cs:
                    continue
                fr = mean([f["rules"][rule]["power_fraction_of_trace"]
                           for f in folds if f["rules"][rule].get("status") == "OK"])
                spread = " ".join(f"{v:5.2f}" for v in per_fold)
                print(f"{split:12s} {rule:13s} {rank:4d} {mean(Cs):11.4g} "
                      f"{mean(As):11.4g} {mean(Bs):11.4g} "
                      f"{mean(ratios):9.2e} {fr:10.2e}   {spread}")


def table_unprojected_group_power(doc, gram_root):
    h("4b. POWER BEFORE any projection, per group (the real drift diagnostic)")
    print("   Band power of each frame computed on the FULL-MOVIE basis and then")
    print("   averaged inside each group.  If the drift put the groups in")
    print("   different regimes, these means would separate -- and for the")
    print("   contiguous split, whose groups ARE contiguous thirds of the movie,")
    print("   that is the strongest form of the test.")
    print()
    print("   What actually happens: they do NOT separate, for ANY split.  The")
    print("   band's own frame-to-frame fluctuation (rel-std 0.33-0.50) swamps a")
    print("   slow drift of a few percent.  Drift lives in the total energy, which")
    print("   is >90% mode 1 -- see table 1 -- not in this band's level.")
    print()
    for seq, s in doc.items():
        gpath = Path(gram_root) / seq / "gram_all.npy"
        if not gpath.exists():
            print(f"  [{seq}] no cached gram at {gpath}; skipped")
            continue
        G = np.load(gpath)
        evals, vecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        evals, vecs = evals[order], vecs[:, order]
        # |c_k(t)|^2 = lam_k |v_k[t]|^2, summed over the proportional band
        pow_frame = (np.abs(vecs) ** 2 * evals[None, :])[:, 20:117].sum(1).real
        base = pow_frame.mean()
        print(f"  {seq}: band power per frame mean {base:.4g}  "
              f"rel-std {pow_frame.std() / base:.3f}  "
              f"loudest frames "
              + " ".join(f"f{i}({pow_frame[i] / base:.1f}x)"
                         for i in np.argsort(pow_frame)[::-1][:3]))
        for split, folds in sorted(by_split(s).items()):
            p = folds[0].get("group_period")
            if folds[0]["strategy"] == "contiguous":
                parts = [("block0", np.arange(0, 133)),
                         ("block1", np.arange(133, 266)),
                         ("block2", np.arange(266, 400))]
            else:
                parts = [(f"g{g}", np.arange(g, 400, p)) for g in range(p)]
            txt = "  ".join(f"{nm}={pow_frame[ix].mean() / base:.3f}x"
                            for nm, ix in parts)
            print(f"     {split:12s} n={len(folds)}  {txt}")


def table_mode_resolved(doc):
    h("5. Mode-resolved lag-1 autocorrelation of frame coefficients (full movie)")
    print("   're' is the signed mean over modes in the bin -- the sign is the")
    print("   physics.  NOTE the inflated null for the 1-1 bin: mode 1 carries")
    print("   >90% of the energy and is nearly constant, so permuting frame order")
    print("   barely perturbs a coefficient series dominated by its own mean.")
    print("   That null is therefore not the right yardstick for that bin.")
    print()
    for seq, s in doc.items():
        print(f"-- {seq}")
        print(f"   {'mode bin':>9s} {'re(lag1)':>9s} {'abs':>7s} {'im_abs':>7s} "
              f"{'null':>7s} {'frac exc':>8s} {'n_modes':>7s}")
        for binname, lags in s["full_movie"]["mode_bins"].items():
            L = lags.get("1")
            if L is None:
                continue
            print(f"   {binname:>9s} {L['mean_re']:+9.3f} {L['mean_abs']:7.3f} "
                  f"{L['mean_im_abs']:7.3f} {L['null']:7.3f} "
                  f"{L['frac_exceed_null']:8.2f} {L['n_modes']:7d}")
        print()


def table_cross_group(doc):
    h("6. Cross-group coefficient correlation, A->B, by mode slice (lag 1)")
    print("   topR = the top of the retained range, bottom = just past the band,")
    print("   mid = its middle.  All are positive and far above null.")
    print()
    print(f"{'split':12s} {'slice':8s} {'re':>8s} {'abs':>7s} {'null':>7s} {'n_pairs':>8s}")
    for seq, s in doc.items():
        for split, folds in sorted(by_split(s).items()):
            for sl in ("topR", "mid", "bottom"):
                key = f"A->B_lag1_{sl}"
                res = [f["cross_group"][key] for f in folds if key in f["cross_group"]]
                if not res:
                    continue
                print(f"{split:12s} {sl:8s} {mean([r['mean_re'] for r in res]):+8.3f} "
                      f"{mean([r['mean_abs'] for r in res]):7.3f} "
                      f"{mean([r['null'] for r in res]):7.3f} "
                      f"{mean([r['n_pairs'] for r in res]):8.1f}")


def iter_splits(doc, volume):
    """(seq, split, strategy, folds) over one volume of a residual-proxy doc."""
    for seq, vols in doc.items():
        if volume not in vols:
            continue
        for split, folds in sorted(by_split(vols[volume]).items()):
            yield seq, split, folds[0]["strategy"], folds


def table_capture(doc):
    h("7. Cross-fit CAPTURE: how much of an outside frame lies in span(X_C)")
    print("   Computed with ALL of C's modes, before any band is chosen.  Near 1")
    print("   means a frame is almost entirely inside the span of DISJOINT frames")
    print("   -- which is expected here because the data is very low-dimensional")
    print("   (mode 1 alone is >93%).  So 'cross-fit loses almost no span' is")
    print("   compatible with Phase 1's 'the BAND subspaces are near-orthogonal':")
    print("   the loss is in which part of the span is KEPT, not in the span.")
    print()
    print(f"{'seq':8s} {'split':12s} {'strat':11s} {'capture A':>10s} {'capture B':>10s}")
    for seq, split, strat, folds in iter_splits(doc, "whole_volume"):
        A = mean(f["capture_fraction"]["A"] for f in folds)
        B = mean(f["capture_fraction"]["B"] for f in folds)
        print(f"{seq:8s} {split:12s} {strat:11s} {A:10.5f} {B:10.5f}")


def table_residual(doc, volume="whole_volume"):
    h(f"8. The mandatory triple, residual bands, {volume}")
    print("   proxy      = above the official band's upper edge (rule-independent).")
    print("                This is the band the original reasoning called a noise")
    print("                proxy.  power/P0 is its share of the UNFILTERED power.")
    print("   disc_above = the high-rank tail THIS rule's projector removes.")
    print("   disc_below = the low-rank clutter it removes; >99% of the trace, so")
    print("                its autocorrelation is the clutter's, not the noise's.")
    print("   'r(A)' is the band-power autocorrelation inside A at the smallest")
    print("   physical lag the split can measure (1 contiguous, 3 for p=3, 5 for")
    print("   p=5) -- an interleaved split simply cannot measure adjacent frames.")
    print("   'per-mode' is the mean of the per-mode correlations, a different")
    print("   statistic that weights modes equally instead of by power.")
    print()
    for seq, split, strat, folds in iter_splits(doc, volume):
        p = folds[0]["group_period"] if strat == "interleaved" else 1
        print(f"  {seq} / {split} ({strat})")
        for rule in ("proportional", "spectral"):
            for band, tag in (("proxy", "proxy     "),
                              ("discarded_above", "disc_above"),
                              ("discarded_below", "disc_below")):
                rows = [f["rules"][rule][band] for f in folds
                        if f["rules"].get(rule, {}).get(band, {}).get("status") == "OK"]
                if not rows:
                    continue
                lag = str(p)
                rs = [r["autocorr_A"][lag]["r"] for r in rows
                      if lag in r["autocorr_A"]]
                exc = sum(r["autocorr_A"][lag]["exceeds_null"] for r in rows
                          if lag in r["autocorr_A"])
                nuls = [r["autocorr_A"][lag]["null"] for r in rows
                        if lag in r["autocorr_A"]]
                pm = [r["mode_mean_r"][lag]["mean_r"] for r in rows
                      if lag in r.get("mode_mean_r", {})]
                print(f"     {rule:13s} {tag} n={rows[0]['n_modes']:3d} "
                      f"pow/P0={mean(r['power_over_P0_A'] for r in rows):9.2e} "
                      f"frac_trace={mean(r['power_fraction_of_trace'] for r in rows):9.2e} "
                      f"r(A)@{lag}={mean(rs):+7.3f} (null {mean(nuls):.3f}, "
                      f"{exc}/{len(rs)})"
                      + (f" per-mode={mean(pm):+7.3f}" if pm else ""))
        print()


def table_residual_decay(doc, volume="whole_volume"):
    h(f"9. Residual (proxy band) autocorrelation decay inside A, {volume}")
    print("   Does the proxy band decorrelate?  Compare with the band the pair")
    print("   actually averages over -- table 2 -- which is the same question for")
    print("   the retained band.  'lag' is physical; '.' means no pairs exist.")
    print()
    for seq, split, strat, folds in iter_splits(doc, volume):
        rows = [f["rules"]["proportional"]["proxy"] for f in folds
                if f["rules"]["proportional"]["proxy"].get("status") == "OK"]
        if not rows:
            continue
        lags = sorted({int(L) for r in rows for L in r["autocorr_A"]})
        cells = []
        for L in lags:
            vals = [r["autocorr_A"][str(L)]["r"] for r in rows
                    if str(L) in r["autocorr_A"]]
            cells.append(f"{L}:{mean(vals):+.3f}" if vals else f"{L}:    .")
        print(f"  {seq} {split:12s} ({strat:11s})  " + "  ".join(cells))


def table_residual_depth(doc):
    h("10. Depth dependence of the proxy band (plan asks for 分深度)")
    print("   Same estimator, Gram recomputed on each depth slab.  Reported at")
    print("   the smallest lag the split can measure, so the columns are")
    print("   comparable down a slab but not across splits.")
    print()
    for seq, vols in doc.items():
        depths = [v for v in vols if v != "whole_volume"]
        if not depths:
            continue
        print(f"  {seq}")
        for split in sorted({f["split"] for f in vols["whole_volume"]["folds"]}):
            p = next(f["group_period"] for f in vols["whole_volume"]["folds"]
                     if f["split"] == split)
            lag = str(p if p > 1 else 1)
            cells = []
            for vol in ["whole_volume"] + sorted(depths):
                rows = [f["rules"]["proportional"]["proxy"]
                        for f in vols[vol]["folds"] if f["split"] == split]
                rows = [r for r in rows if r.get("status") == "OK"
                        and lag in r["autocorr_A"]]
                if rows:
                    cells.append(f"{vol.replace('z', 'z'):>9s}="
                                 f"{mean(r['autocorr_A'][lag]['r'] for r in rows):+.3f}")
            print(f"     {split:12s} r(A)@{lag}:  " + "  ".join(cells))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--gram-root", type=Path, default=DEFAULT_GRAM_ROOT,
                    help="where the cached 400x400 Grams live (table 4b)")
    ap.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL,
                    help="phase2_residual_proxy.json (tables 7-10)")
    args = ap.parse_args()
    doc = json.loads(args.json.read_text())

    print(f"source: {args.json}")
    for seq, s in doc.items():
        print(f"  {seq}: frames={s['frames']} lam1_frac={s['lam1_fraction']:.5f} "
              f"recon_rel_err={s['reconstruction_rel_err']:.2e}")

    table_raw_energy(doc)
    table_within_group(doc)
    table_contiguous_curve(doc)
    table_power_levels(doc)
    table_unprojected_group_power(doc, args.gram_root)
    table_mode_resolved(doc)
    table_cross_group(doc)

    if args.residual.exists():
        resid = json.loads(args.residual.read_text())
        table_capture(resid)
        table_residual(resid, "whole_volume")
        table_residual_decay(resid, "whole_volume")
        table_residual_depth(resid)
    else:
        print(f"\n[skip] {args.residual} not found (tables 7-10)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 / 前置诊断: does a calibration group's retained SVD subspace transfer?

Phase 1 recorded one number for this question -- ``cos_mean(C, X)`` = 0.05-0.18 for
the wide band -- and the plan's reading of it was "the wide band is held up by
frame-specific directions".  This script asks the next question, which is the one
that actually decides the gate: **what does that do to the image?**

It is deliberately not a new estimator of the pair's quality.  Every measurement
here is a diagnostic of the *projector*, not of (D_A, D_B), and the whole script
can be read as an audit of the single step

    D = |P_basis  x_target|^2 ,      P_basis built from a frame group

by varying (basis source, target data) independently while holding the band rule,
the grid, the valid mask and the dB relativisation fixed.

Five controls, because each of the cheap explanations for "the cross image looks
wrong" has to be killed separately:

  A. basis x target 3x3  -- is it a labelling/implementation bug?  (all six cross
     directions fail symmetrically, so no)
  B. random subspace at matched rank *and* matched mode range -- is the band's
     content doing anything a random choice would not?  (for the wide band, no)
  C. frame-overlap titration -- is what transfers the shared physiology or the
     shared data realisation?  (linear in overlap, so the latter)
  D. lower-edge scan -- is *any* band usable?  (yes: deeper bands transfer
     structurally, which is the one piece of good news here)
  E. independent-group ceiling -- the full-sequence reference shares 1/3 of its
     frames with A, so correlating against it cannot separate "same anatomy" from
     "same noise realisation".  Comparing C->A against *B's own* image can.

Outputs: ``data/processed/crossfit/basis_transfer.json`` (gitignored) and
``reports/figures/phase2_basis_transfer.png``.

Usage:
    python scripts/diagnose_basis_transfer.py --sequences SA64 HSA64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_crossfit_pairs import eigh_desc, rule_edges, principal_angles  # noqa: E402
from datasets.iq_io import load_iq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LO_SCAN = (2, 5, 8, 10, 12, 14, 17, 20, 24, 28, 32, 38, 45, 55, 70, 90, 110)


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def load_gram(seq, root):
    return np.load(root / seq / "gram_all.npy")


def eigs(G, idx):
    return eigh_desc(G[np.ix_(idx, idx)])


def coeffs(G, src_idx, ev, vecs, tgt_idx):
    """<u_k, x_t> for every retained-able mode k of the basis group and frame t."""
    return (vecs.conj().T @ G[np.ix_(src_idx, tgt_idx)]) / np.sqrt(ev)[:, None]


def image(X, G, src_idx, tgt_idx, ev, vecs, lo, hi, sel=None):
    """|P x_tgt|^2 for P the projector onto span{ U_src[lo:hi] }, optionally
    further restricted to the subspace ``sel`` (an isometry, columns orthonormal)."""
    hi = min(hi, len(src_idx))
    Vb = vecs[:, lo:hi] / np.sqrt(ev[lo:hi])[None, :]
    W = coeffs(G, src_idx, ev, vecs, tgt_idx)[lo:hi]
    if sel is not None:
        Vb, W = Vb @ sel, sel.conj().T @ W
    return np.mean(np.abs(X[:, src_idx] @ (Vb @ W)) ** 2, axis=1).real


def to_db(v, valid):
    """Own-p99.9-relativised dB.  Scale-free on purpose: the structure question is
    about shape, and each image is allowed to be arbitrarily weak.  The *absolute*
    level is reported separately, never folded into this number."""
    return 10 * np.log10(np.maximum(v[valid], 1e-30) / np.percentile(v[valid], 99.9))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def dice(a_db, b_db, thr=-4.0):
    m, r = a_db > thr, b_db > thr
    return float(2 * np.sum(m & r) / (m.sum() + r.sum()))


def same_band_energy(G, src_idx, tgt_idx, ev, vecs, band):
    """||P_src x_tgt||_F^2 for the band, i.e. the target energy the projector keeps."""
    lo, hi = band
    hi = min(hi, len(src_idx))
    return float(np.sum(np.abs(coeffs(G, src_idx, ev, vecs, tgt_idx)[lo:hi]) ** 2))


# --------------------------------------------------------------------------
# A. basis x target, all nine directions, with the full-sequence reference
# --------------------------------------------------------------------------

def part_a(cfg, X, G, idx, valid, ref_db, rules_seq):
    out = {}
    for rule in rules_seq:
        tab = {}
        for bas in ("C", "A", "B"):
            ev_b, v_b = eigs(G, idx[bas])
            lo, hi = rule_edges(rule, cfg, ev_b)
            for tgt in ("C", "A", "B"):
                if bas == tgt:
                    continue
                d = image(X, G, idx[bas], idx[tgt], ev_b, v_b, lo, hi)
                tab[f"{bas}2{tgt}"] = {
                    "band": [int(lo), int(hi)], "rank": int(hi - lo),
                    "corr_db_ref": corr(to_db(d, valid), ref_db),
                    "power": float(d[valid].mean()),
                }
        for own in ("C", "A", "B"):
            ev_b, v_b = eigs(G, idx[own])
            lo, hi = rule_edges(rule, cfg, ev_b)
            d = image(X, G, idx[own], idx[own], ev_b, v_b, lo, hi)
            tab[f"{own}2{own}"] = {
                "band": [int(lo), int(hi)], "rank": int(hi - lo),
                "corr_db_ref": corr(to_db(d, valid), ref_db),
                "power": float(d[valid].mean()),
            }
        out[rule] = tab
    return out


# --------------------------------------------------------------------------
# B. random subspaces at matched rank and matched mode range
# --------------------------------------------------------------------------

def part_b(G, idx, ev, vecs, src, tgt, band, ambient_lo, rng, draws=32):
    """Is the band's retained subspace doing anything a random choice would not?

    The null is a random ``rank(band)``-dimensional subspace of the calibration
    group's whole post-clutter span ``[ambient_lo, T_C)``.  Both the ambient and the
    rank have to be matched to the band, and the rank must be strictly smaller than
    the ambient -- drawing an orthonormal basis of the *entire* band span gives back
    the same projector and a ratio of exactly 1, which is a false negative, not a
    control.
    """
    lo, hi = band
    hi = min(hi, len(idx[src]))
    amb_hi = len(idx[src])
    amb = amb_hi - ambient_lo
    rank = hi - lo
    assert rank < amb, (rank, amb)
    W = coeffs(G, idx[src], ev, vecs, idx[tgt])
    det = float(np.sum(np.abs(W[lo:hi]) ** 2))
    vals = []
    for _ in range(draws):
        Q, _ = np.linalg.qr(rng.standard_normal((amb, rank)))
        vals.append(float(np.sum(np.abs(Q.conj().T @ W[ambient_lo:amb_hi]) ** 2)))
    return {"band": [int(lo), int(hi)], "rank": int(rank),
            "ambient": [int(ambient_lo), int(amb_hi)], "ambient_dim": int(amb),
            "deterministic": det,
            "random_mean": float(np.mean(vals)),
            "random_min": float(np.min(vals)), "random_max": float(np.max(vals)),
            "ratio": det / float(np.mean(vals))}


def part_b_image(X, G, idx, ev_c, v_c, valid, ref_db, rng, draws=8):
    """The same null, but scored by image structure instead of captured energy.

    The two scorings disagree in sign, which is the point: the wide band is at the
    top of the random envelope by *energy* while being *worse* than a random subspace
    of the same rank by *structure*.  Energy transfer is therefore not a valid proxy
    for "does the calibration basis carry the blood".
    """
    out = {}
    amb_lo, amb_hi = 20, len(idx["C"])
    for lo, hi, med in ((20, 117, 97), (40, 134, 64), (70, 134, 64)):
        hi = min(hi, len(idx["C"]))
        det = image(X, G, idx["C"], idx["A"], ev_c, v_c, lo, hi)
        crs = [image(X, G, idx["C"], idx["A"], ev_c, v_c, amb_lo, amb_hi,
                     sel=np.linalg.qr(rng.standard_normal((amb_hi - amb_lo, med)))[0])
               for _ in range(draws)]
        out[f"[{lo},{hi})"] = {
            "rank": hi - lo,
            "deterministic_corr_ref": corr(to_db(det, valid), ref_db),
            "deterministic_power": float(det[valid].mean()),
            "random_rank_corr_ref_mean": float(np.mean([corr(to_db(d, valid), ref_db) for d in crs])),
            "random_rank_power_mean": float(np.mean([d[valid].mean() for d in crs])),
        }
    return out


# --------------------------------------------------------------------------
# C. frame-overlap titration
# --------------------------------------------------------------------------

def part_c(cfg, X, G, c_idx, a_idx, valid, ref_db, rule, fractions, rng):
    T = len(c_idx)
    ev_a, v_a = eigs(G, a_idx)
    lo, hi = rule_edges(rule, cfg, ev_a)
    own = float(np.sum(np.abs(coeffs(G, a_idx, ev_a, v_a, a_idx)[lo:hi]) ** 2))
    rows = []
    for fr in fractions:
        k = int(round(fr * T))
        cs = np.concatenate([a_idx[:k], c_idx[:T - k]]) if k < T else a_idx[:T].copy()
        if len(set(cs.tolist())) != T:
            continue
        ev_c, v_c = eigs(G, cs)
        l2, h2 = rule_edges(rule, cfg, ev_c)
        crs = float(np.sum(np.abs(coeffs(G, cs, ev_c, v_c, a_idx)[l2:h2]) ** 2))
        d = image(X, G, cs, a_idx, ev_c, v_c, l2, h2)
        cmn, cmean = principal_angles(
            {"C": cs, "evals": ev_c, "vecs": v_c, "lo": l2, "hi": h2},
            {"C": a_idx, "evals": ev_a, "vecs": v_a, "lo": lo, "hi": hi}, G)
        rows.append({"frames_from_A": k, "overlap": k / T,
                     "cos_mean": cmean, "transfer": crs / own,
                     "power": float(d[valid].mean()),
                     "corr_db_ref": corr(to_db(d, valid), ref_db)})
    return rows


# --------------------------------------------------------------------------
# D/E. lower-edge scan and the independent-group ceiling
# --------------------------------------------------------------------------

def part_de(X, G, idx, valid, ref_db):
    ev, vv = {}, {}
    for k, i in idx.items():
        ev[k], vv[k] = eigs(G, i)
    T = {k: len(i) for k, i in idx.items()}
    rows = []
    for lo in LO_SCAN:
        r = {"lo": lo}
        own = {k: to_db(image(X, G, idx[k], idx[k], ev[k], vv[k], lo, 134), valid) for k in "ABC"}
        crs = {k: to_db(image(X, G, idx["C"], idx[k], ev["C"], vv["C"], lo, 134), valid) for k in "AB"}
        hi_a = min(134, T["A"])
        r["rank_A"] = hi_a - lo
        r["own_power"] = float(image(X, G, idx["A"], idx["A"], ev["A"], vv["A"], lo, 134)[valid].mean())
        r["cross_power"] = float(image(X, G, idx["C"], idx["A"], ev["C"], vv["C"], lo, 134)[valid].mean())
        r["own_corr_ref"] = corr(own["A"], ref_db)
        r["own_dice_ref"] = dice(own["A"], ref_db)
        r["cross_corr_ref"] = corr(crs["A"], ref_db)
        r["cross_dice_ref"] = dice(crs["A"], ref_db)
        # unbiased: C's basis on A vs B's OWN image -- the two images share no frames
        r["ceiling_own_own"] = corr(own["A"], own["B"])
        r["cross_vs_independent_own"] = corr(crs["A"], own["B"])
        r["cross_vs_cross"] = corr(crs["A"], crs["B"])
        rows.append(r)
    return rows


# --------------------------------------------------------------------------
# F. split-half stability inside the calibration group
# --------------------------------------------------------------------------

def part_f(G, c_idx, a_idx, b_idx):
    """Is the retained band even reproducible against a disjoint half of itself?

    This is the only diagnostic here that uses no target group at all, so it is the
    candidate basis of a *principled* (non-tuned) band rule: keep the modes whose
    directions the calibration data can pin down.
    """
    ev_c, v_c = eigs(G, c_idx)
    c1, c2 = c_idx[0::2], c_idx[1::2]
    e1, v1 = eigs(G, c1); e2, v2 = eigs(G, c2)
    ev_a, v_a = eigs(G, a_idx); ev_b, v_b = eigs(G, b_idx)
    rows = []
    for lo in LO_SCAN:
        hh = min(len(c_idx), len(c1), len(c2))
        if hh - lo < 4:
            continue
        def cm(ei, vi, ci, ej, vj, cj):
            return principal_angles(
                {"C": ci, "evals": ei, "vecs": vi, "lo": lo, "hi": hh},
                {"C": cj, "evals": ej, "vecs": vj, "lo": lo, "hi": hh}, G)[1]
        rows.append({"lo": lo, "rank": hh - lo,
                     "cos_half_within_C": cm(e1, v1, c1, e2, v2, c2),
                     "cos_C_A": cm(ev_c, v_c, c_idx, ev_a, v_a, a_idx),
                     "cos_C_B": cm(ev_c, v_c, c_idx, ev_b, v_b, b_idx)})
    return rows


# --------------------------------------------------------------------------
# cross-fold / cross-sequence transfer table (Gram only)
# --------------------------------------------------------------------------

def part_transfer_table(cfg, root, splits_root, rules, sequences):
    out = {}
    for seq in sequences:
        G = load_gram(seq, root)
        seq_rows = {}
        for sp in sorted(Path(splits_root).glob(f"{seq.lower()}_*.json")):
            man = json.loads(Path(sp).read_text(encoding="utf-8"))
            per_rule = {r: [] for r in rules}
            for fold in man["folds"]:
                c = np.asarray(fold["calibration"], dtype=np.int64)
                a = np.asarray(fold["A"], dtype=np.int64)
                ev_c, v_c = eigs(G, c)
                ev_a, v_a = eigs(G, a)
                tr_a = float(np.trace(G[np.ix_(a, a)]).real)
                for rule in rules:
                    lo, hi = rule_edges(rule, cfg, ev_c)
                    hi = min(hi, len(a)); lo = min(lo, hi)
                    if hi - lo < 2:
                        continue
                    own = same_band_energy(G, a, a, ev_a, v_a, (lo, hi))
                    crs = same_band_energy(G, c, a, ev_c, v_c, (lo, hi))
                    la, ha = rule_edges(rule, cfg, ev_a)
                    _, cmean = principal_angles(
                        {"C": c, "evals": ev_c, "vecs": v_c, "lo": lo, "hi": hi},
                        {"C": a, "evals": ev_a, "vecs": v_a, "lo": la, "hi": ha}, G)
                    per_rule[rule].append({
                        "fold_id": fold["fold_id"], "rank": int(hi - lo),
                        "cos_mean": cmean, "own_frac": own / tr_a,
                        "cross_frac": crs / tr_a,
                        "transfer": crs / own if own > 0 else None,
                    })
            seq_rows[man["name"]] = {
                "T_C": man["folds"][0]["T_C"], "T_A": man["folds"][0]["T_A"],
                "rules": {r: {
                    "n_folds": len(per_rule[r]),
                    "rank": float(np.mean([x["rank"] for x in per_rule[r]])),
                    "cos_mean": float(np.mean([x["cos_mean"] for x in per_rule[r]])),
                    "own_frac": float(np.mean([x["own_frac"] for x in per_rule[r]])),
                    "transfer": float(np.mean([x["transfer"] for x in per_rule[r]])),
                    "folds_transfer_gt_half": int(sum(
                        1 for x in per_rule[r] if x["transfer"] and x["transfer"] > 0.5)),
                    "per_fold": per_rule[r],
                } for r in rules},
            }
        out[seq] = seq_rows
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--pairs-root", default="data/processed/crossfit")
    ap.add_argument("--splits-root", default="data/processed/crossfit/splits")
    ap.add_argument("--sequences", nargs="*", default=["SA64", "HSA64"])
    ap.add_argument("--image-sequences", nargs="*", default=["SA64", "HSA64"])
    ap.add_argument("--fold-index", type=int, default=0)
    ap.add_argument("--out", default="data/processed/crossfit/basis_transfer.json")
    ap.add_argument("--figure", default="reports/figures/phase2_basis_transfer.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(args.baseline).read_text(encoding="utf-8"))
    rules = list(cfg["rank_rules"])
    root = Path(args.pairs_root)
    report = {"config": args.config, "rules": rules, "sequences": {}}

    report["transfer_table"] = part_transfer_table(
        cfg, root, args.splits_root, rules, args.sequences)
    print("[B..F] transfer table done")

    figs = {}
    for seq in args.image_sequences:
        spec = base["data"]["files"][seq]
        G = load_gram(seq, root)
        valid = np.load(ROOT / "data/processed/official" / seq / "valid_mask.npy").reshape(-1).astype(bool)
        ref = np.load(ROOT / "data/processed/official" / seq / "pd_linear.npy").astype(np.float64).reshape(-1)
        ref_db = 10 * np.log10(np.maximum(ref[valid], 1e-30) / np.percentile(ref[valid], 99.9))
        man = json.loads(Path(args.splits_root, f"{seq.lower()}_int3.json").read_text(encoding="utf-8"))
        fold = man["folds"][args.fold_index]
        idx = {k: np.asarray(fold[v], dtype=np.int64)
               for k, v in (("C", "calibration"), ("A", "A"), ("B", "B"))}
        print(f"[{seq}] fold {fold['fold_id']}  T_C={len(idx['C'])} T_A={len(idx['A'])} "
              f"T_B={len(idx['B'])}")

        need_image = True
        X = None
        if need_image:
            x = load_iq(str(Path(base["data"]["root"]) / spec["path"]), variable="Data_" + seq)
            X = x.reshape(-1, x.shape[3]).astype(np.complex64).astype(np.complex128)
            del x
            print(f"[{seq}] IQ loaded {X.shape}")

        ev_c, v_c = eigs(G, idx["C"])
        lo, hi = rule_edges("proportional", cfg, ev_c)
        rng = np.random.default_rng(11)
        seq_rep = {
            "fold_id": fold["fold_id"],
            "A": part_a(cfg, X, G, idx, valid, ref_db, rules),
            "B": part_b(G, idx, ev_c, v_c, "C", "A", (lo, hi), 20, rng),
            "C": part_c(cfg, X, G, idx["C"], idx["A"], valid, ref_db, "proportional",
                        (0.0, 0.13, 0.25, 0.37, 0.5, 0.63, 0.75, 0.87, 0.99), rng),
            "DE": part_de(X, G, idx, valid, ref_db),
            "F": part_f(G, idx["C"], idx["A"], idx["B"]),
            "B_image": part_b_image(X, G, idx, ev_c, v_c, valid, ref_db, rng),
        }
        # matched-rank random subspace inside the *deep* portion, the one place the
        # band may be doing real work
        for lo_d in (40, 70):
            seq_rep[f"B_deep_{lo_d}"] = part_b(G, idx, ev_c, v_c, "C", "A",
                                               (lo_d, 134), 20, rng)
        report["sequences"][seq] = seq_rep
        figs[seq] = seq_rep
        del X
        print(f"[{seq}] parts A-F done")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"wrote {out}")

    # ---- figure ----------------------------------------------------------
    n = len(figs)
    fig, axes = plt.subplots(2, n, figsize=(7.2 * n, 9.0), squeeze=False)
    for j, (seq, rep) in enumerate(figs.items()):
        ax = axes[0][j]
        de = rep["DE"]
        ax.plot([r["lo"] for r in de], [r["own_corr_ref"] for r in de], "o-", label="A's own basis -> A")
        ax.plot([r["lo"] for r in de], [r["cross_corr_ref"] for r in de], "s-", label="C's basis -> A")
        ax.plot([r["lo"] for r in de], [r["ceiling_own_own"] for r in de], "--",
                color="grey", label="A's own vs B's own (ceiling)")
        ax.plot([r["lo"] for r in de], [r["cross_vs_independent_own"] for r in de], "^-",
                color="crimson", label="C's basis -> A  vs  B's own (unbiased)")
        ax.axvline(20, color="k", lw=0.8, ls=":")
        ax.text(20, ax.get_ylim()[0], " proportional rule", fontsize=7, va="bottom")
        ax.set_xlabel("band lower edge (mode index of the calibration spectrum)")
        ax.set_ylabel("corr of own-p99.9 dB image with reference")
        ax.set_title(f"{seq}: structure vs band depth")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1][j]
        ov = rep["C"]
        ax.plot([r["overlap"] for r in ov], [r["transfer"] for r in ov], "o-",
                label="energy transfer ratio")
        ax.set_xlabel("fraction of the calibration frames that are the target's own frames")
        ax.set_ylabel("||P_C x_A||^2 / ||P_A x_A||^2")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax2 = ax.twinx()
        ax2.plot([r["overlap"] for r in ov], [r["cos_mean"] for r in ov], "s-",
                 color="crimson", label="cos_mean(C band, A band)")
        ax2.set_ylabel("subspace cosine", color="crimson")
        ax.set_title(f"{seq}: what transfers is the shared data realisation")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fp = Path(args.figure)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=140)
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()

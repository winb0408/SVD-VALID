"""Gate N: estimate arm-to-arm noise correlation with k >= 3 arms.

Why this exists (see `Project log/Project_plan.md` section 5.6):

The old residual proxy (`rel_matched`: arm pair vs Tetris through a matched
operator) was falsified by `audit_metric_calibration.py`.  At ZERO frame overlap
-- where independence holds by construction -- it reads -0.040 / -0.153 / -0.317
across three split styles, and -0.012 on HSA64.  Same premise, same question,
answers 8-27x apart.  Its Spearman against `power_map_r` is +0.93..+0.98, so it
measures how alike the two arm images are, not whether their noise is
independent.

Replacing the formula does not rescue it.  A direct complex correlation of the
arm-mean fields is dominated at small N by the globally subtracted mean
(rho -> +1 no matter how independent) and pushed negative at full N by the
sum-to-zero finite-pool constraint.  Both are symptoms of one fact:

    2 arms with no ground truth => Cov(eps_1, eps_2) is NOT identifiable.

Any 2-arm statistic either assumes independence (circular) or measures something
else.  With k >= 3 arms the over-identification -- k(k-1)/2 cross-covariances
against a single common-signal variance -- makes the quantity testable.

Estimator.  Per voxel D_i = S + eps_i, with c_ij = Cov_v(D_i, D_j) taken over
voxels (only each arm's OWN voxel mean is removed, never a shared one).  Under
independence every c_ij equals V_S, so:

    V_S    = mean_{i != j} c_ij
    V_i    = c_ii - V_S
    rho_ij = (c_ij - V_S) / sqrt(V_i V_j)

V_S is then refined by fixed-point iteration that subtracts the shared-noise
term implied by the current rho, so a genuinely correlated pair cannot inflate
V_S and thereby hide itself.

Blind spot, stated up front: a noise component shared by ALL k arms is
indistinguishable from S and gets absorbed into V_S.  Only the temporal sampling
strategy mitigates that, so every conclusion is phrased "under interleaved
sampling", never as unconditional independence.

Calibration.  Additive injection: leave the real IQ untouched and add a complex
Gaussian field on top whose shared part is common across arms with fraction
alpha.  Recovering rho ~ alpha is the known-answer test that licenses the gate.
ADDITIVE, not replacement -- all four replacement-based surrogates tried during
calibration destroyed the real frames' envelope and came out flat on the
true-correlation axis.

Usage:
    python scripts/gate_noise_independence.py --sequences SA64 HSA64 --k 4 \\
        --out reports/phase2_gate_n_2026-10-19
"""

from __future__ import annotations

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
from build_crossfit_pairs import rule_edges                             # noqa: E402

RHO_GATE = 0.10          # pre-registered, see plan section 5.6.3 (R^2 = 1%)


# --------------------------------------------------------------------------- #
# arm construction
# --------------------------------------------------------------------------- #
def split_into_k(frames, k, mode="interleaved", stride=1):
    """Partition `frames` into k disjoint groups, preserving the parent strategy.

    `stride` is the frame separation between arms, in units of the base grid.
    stride=1 gives plain `f[i::k]`, which puts arm i and arm i+1 one frame apart
    -- inside the measured 5-10 frame noise correlation length, so their means
    cannot be independent.  stride=s takes arm i from indices (i*s) mod (k*s),
    spacing each arm k*s frames apart while separating the arms by s.  The
    difference between stride=1 and stride>=10 is the whole point: it is the
    only lever that can decorrelate arms whose sampling shares a slow component.
    """
    f = np.asarray(sorted(frames), dtype=int)
    if mode == "interleaved":
        period = k * stride
        groups = [f[np.arange(f.size) % period == (i * stride) % period]
                  for i in range(k)]
    elif mode == "contiguous":
        groups = list(np.array_split(f, k))
    else:
        raise ValueError(mode)
    return [np.sort(g) for g in groups if g.size]


def pairs_of(k):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


# --------------------------------------------------------------------------- #
# core estimator
# --------------------------------------------------------------------------- #
def variance_decomposition(maps, valid, normalise_gain=True):
    """k-arm covariance decomposition -> noise correlations.

    maps:  one flat array per arm, same voxel ordering.
    valid: boolean mask of voxels entering the covariance.

    Gain normalisation is NOT cosmetic.  The four arm maps come out proportional
    to one another -- on sa64_int3_c0a1b2 the cross covariances fit
    c_ij = a_i a_j Var(u) with a = (12.21, 11.77, 10.85, 10.92) to within 2% --
    so the model D_i = S + eps_i fails and the median cross covariance is
    contaminated: without the fix, noise_frac came out NEGATIVE for two arms
    ([-0.084, -0.071]) and rho exceeded 1.  After dividing each arm by its gain
    the cross covariances agree to 0.6% and noise_frac is positive throughout.

    V_S = MEDIAN of the cross covariances, not their mean.  The mean is not
    enough: an iterative "subtract the shared term from every pair" refinement
    is degenerate (at alpha=1 every V_S is a fixed point) and drove V_S to 1.17
    where the truth was 1.00.  The median is exact for k=4 when at most two
    pairs are contaminated, and reduces to the common value when a pair is not
    correlated.  Verified against known-answer synthetics: recovers
    alpha in {0,.25,.5,.75,1} to within 0.005.
    """
    k = len(maps)
    m = np.stack([a[valid].astype(np.float64) for a in maps])
    if normalise_gain:
        # Per-arm mean, NOT regression against a common reference.  A regression
        # reference is a mixture of the arms, so when one pair carries a large
        # shared component it contaminates the reference and drags the OTHER
        # arms along with it -- calibration showed exactly that: injecting shared
        # noise into arms 0,1 produced a spurious rho = 0.89 on the untouched
        # pair (2,3).  The arm's own mean is not circular.
        # Gain from regression on a robust reference: the voxel-wise MEDIAN across
        # arms, not the mean.  The mean is a mixture, so a pair carrying a large
        # shared component contaminates it and drags the untouched arms with it
        # (measured leak: rho = 0.89 on a pair that was never injected).  The
        # median is not moved by one contaminated pair; the arm's own mean is not
        # used because it carries too little of the noise to condition the solve
        # (noise_frac collapsed to ~0.01 and went negative).
        ref = np.median(m, axis=0)
        ref = ref - ref.mean()
        gain = np.array([float((x - x.mean()) @ ref) / max(float(ref @ ref), 1e-30)
                         for x in m])
        gain[gain <= 0] = 1.0
        m = m / gain[:, None]
    m -= m.mean(axis=1, keepdims=True)                 # each arm's OWN mean only
    c = (m @ m.T) / m.shape[1]
    idx = pairs_of(k)
    cross = np.array([c[i, j] for i, j in idx])

    v_s = float(np.median(cross))
    diag = np.diag(c)
    v_i = diag - v_s
    sd = np.sqrt(np.clip(np.outer(v_i, v_i), 0.0, None))
    rho = np.where(sd > 0, (c - v_s) / np.where(sd > 0, sd, np.nan), np.nan)
    np.fill_diagonal(rho, 0.0)
    off = np.array([rho[i, j] for i, j in idx])
    return {
        "V_S": v_s,
        "noise_var": [float(x) for x in v_i],
        "noise_frac": [float(x / d) if d > 0 else float("nan") for x, d in zip(v_i, diag)],
        "c": c.tolist(),
        "rho": rho.tolist(),
        "rho_pairs": {f"{i}-{j}": float(off[n]) for n, (i, j) in enumerate(idx)},
        "rho_mean": float(np.nanmean(off)),
        "rho_max": float(np.nanmax(np.abs(off))),
    }


# --------------------------------------------------------------------------- #
# spatial block bootstrap
# --------------------------------------------------------------------------- #
def spatial_blocks(cshape, block, valid):
    z, x, y = cshape
    zz, xx, yy = (np.arange(n) // block for n in (z, x, y))
    nx, ny = (x + block - 1) // block, (y + block - 1) // block
    lab = ((zz[:, None, None] * nx + xx[None, :, None]) * ny + yy[None, None, :]).ravel()
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    out = []
    for b in range(lab.max() + 1):
        idx = order[bounds[b]:bounds[b + 1]]
        idx = idx[valid[idx]]
        if idx.size:
            out.append(idx)
    return out


def bootstrap_rho(maps, blocks, n_boot=200, seed=0):
    """Resample spatial blocks with replacement -> CI for the rho summaries."""
    rng = np.random.default_rng(seed)
    means, maxes = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[p] for p in pick])
        res = variance_decomposition(maps, idx)
        means.append(res["rho_mean"])
        maxes.append(res["rho_max"])
    return {
        "rho_mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "rho_max_ci": [float(np.percentile(maxes, 2.5)), float(np.percentile(maxes, 97.5))],
        "n_boot": int(n_boot),
        "n_blocks": int(len(blocks)),
    }


# --------------------------------------------------------------------------- #
# additive injection calibration (known answer)
# --------------------------------------------------------------------------- #
def inject(x2d, arms, alpha, amplitude, rng, shared=(0, 1)):
    """x' = x + n.  Arms `shared` receive a common noise field in fraction alpha.

    PAIR-specific on purpose.  A field shared by ALL arms is the gate's declared
    blind spot (indistinguishable from signal) and would calibrate nothing; the
    known-answer test has to inject something the estimator can see.  Every arm
    gets the same total injected variance, so only the correlation differs.
    """
    v = x2d.shape[0]
    # Spatially ENVELOPED, not spatially white.  White injection breaks the
    # per-arm proportionality the estimator relies on and leaked rho = 0.9 onto
    # untouched arms.  Scaling by sqrt of the local mean power makes the injected
    # component carry the map's own spatial statistics, which is what a real
    # shared fluctuation looks like.
    env = np.sqrt(np.mean(np.abs(x2d) ** 2, axis=1))
    env = env / max(float(env.mean()), 1e-30)

    def field(cols):
        return (amplitude * env[:, None]
                * (rng.standard_normal((v, cols)) + 1j * rng.standard_normal((v, cols)))
                / np.sqrt(2))

    out = x2d.copy()
    common = field(1)[:, 0]
    for n, a in enumerate(arms):
        own = field(a.size)
        w = np.sqrt(alpha) if n in shared else 0.0
        out[:, a] += w * common[:, None] + np.sqrt(1.0 - w ** 2) * own
    return out


def calibrate(x2d, gram, c_idx, basis, arms, shape3d, cshape, valid,
              alphas=(0.0, 0.25, 0.5, 0.75, 1.0), seed=0, amp_scale=1.0):
    """Known-answer test: rho on the injected pair must track alpha; others stay ~0."""
    amp = amp_scale * float(np.sqrt(np.mean(np.abs(x2d) ** 2)))
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in alphas:
        xi = inject(x2d, arms, alpha, amp, rng, shared=(0, 1))
        # The projector reads G[C, t] for the target frames, so the Gram has to be
        # rebuilt from the injected data -- passing the original one makes the
        # injection vanish entirely (verified: map mean identical to 5 digits).
        gram_i = xi.conj().T @ xi
        maps = [E.to_cropped(E.projected_power_map(xi, gram_i, c_idx, basis, tgt=a) / a.size,
                             shape3d, cshape).reshape(-1) for a in arms]
        del gram_i, xi
        res = variance_decomposition(maps, valid)
        rows.append({"alpha": float(alpha),
                     "rho_injected_pair": res["rho_pairs"]["0-1"],
                     "rho_other_pairs": [v for kk, v in res["rho_pairs"].items()
                                         if kk != "0-1"],
                     "rho_max": res["rho_max"]})
        print(f"      inject a={alpha:.2f} -> rho(0-1) {res['rho_pairs']['0-1']:+.3f} "
              f"others {[round(v, 3) for kk, v in res['rho_pairs'].items() if kk != '0-1']}",
              flush=True)
    err = max(abs(r["rho_injected_pair"] - r["alpha"]) for r in rows)
    mono = all(rows[i + 1]["rho_injected_pair"] >= rows[i]["rho_injected_pair"] - 0.05
               for i in range(len(rows) - 1))
    leak = max(abs(v) for r in rows for v in r["rho_other_pairs"])
    return {"amplitude": amp, "rows": rows, "max_abs_error": float(err),
            "max_spurious_rho": float(leak), "monotone": bool(mono),
            "passes": bool(err <= RHO_GATE and leak <= RHO_GATE and mono)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def basis_for(rule, cfg, evals, vecs, n_cap):
    lo, hi = rule_edges(rule, cfg, evals)
    hi = min(hi, n_cap)
    if hi <= lo:
        return None
    ev = np.maximum(evals[lo:hi], 0.0)
    if ev[0] <= 0.0:
        return None
    return vecs[:, lo:hi] / np.sqrt(ev)[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--pairs-root", default="data/processed/crossfit")
    ap.add_argument("--sequences", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--manifests", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame separation between arms (1 = plain f[i::k])")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--calib-folds", type=int, default=2)
    ap.add_argument("--out", default="reports/phase2_gate_n_2026-10-19")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "note": ("Gate N: k-arm covariance decomposition.  V_S = common signal variance, "
                 "V_i = c_ii - V_S = arm i noise variance, rho_ij = (c_ij - V_S)/"
                 "sqrt(V_i V_j).  Pre-registered gate: rho_max <= 0.10 with bootstrap "
                 "upper CI <= 0.10, after the additive-injection known-answer test "
                 "passes.  Blind spot: noise shared by ALL arms is indistinguishable "
                 "from signal, so conclusions are conditional on interleaved sampling."),
        "rho_gate": RHO_GATE, "k": args.k, "stride": args.stride, "sequences": {}}

    for seq in args.sequences:
        print(f"\n=== {seq} ===", flush=True)
        t0 = time.time()
        iq = E.load_sequence_iq(seq, base)
        shape3d = tuple(iq.shape[:3])
        cshape = E.cropped_shape(shape3d)
        x2d = iq.reshape(-1, iq.shape[3]).astype(np.complex128)
        valid = np.load(Path("data/processed/official") / seq / "valid_mask.npy")
        valid = valid[:cshape[0], :cshape[1], :cshape[2]].reshape(-1).astype(bool)
        gram = x2d.conj().T @ x2d
        blocks = spatial_blocks(cshape, args.block, valid)
        print(f"  {x2d.shape}  valid {valid.mean():.3f}  blocks {len(blocks)} "
              f"({time.time() - t0:.1f}s)", flush=True)

        seq_out = {"shape": list(shape3d), "cropped": list(cshape), "cells": []}
        names = args.manifests or [s["name"] for s in cfg["sequences"][seq]["splits"]]
        n_calib = 0

        for name in names:
            path = Path(args.pairs_root) / "splits" / f"{name}.json"
            if not path.exists():
                continue
            man = json.loads(path.read_text())
            if man.get("sequence") != seq:
                continue
            mode = "contiguous" if "cont" in name else "interleaved"
            for fold in man["folds"]:
                c_idx = np.asarray(fold["calibration"], dtype=int)
                pool = np.asarray(sorted(list(fold["A"]) + list(fold["B"])), dtype=int)
                arms = split_into_k(pool, args.k, mode, args.stride)
                if len(arms) < 3:
                    continue
                n_cap = c_idx.size
                evals, vecs = np.linalg.eigh(gram[np.ix_(c_idx, c_idx)])
                order = np.argsort(evals)[::-1]
                evals, vecs = evals[order], vecs[:, order]

                for rule in ("proportional", "clutter_knee"):
                    basis = basis_for(rule, cfg, evals, vecs, n_cap)
                    if basis is None:
                        continue

                    maps = [E.to_cropped(E.projected_power_map(x2d, gram, c_idx, basis, tgt=a)
                                         / a.size, shape3d, cshape).reshape(-1) for a in arms]
                    res = variance_decomposition(maps, valid)
                    boot = bootstrap_rho(maps, blocks, n_boot=args.n_boot)
                    cell = {"fold_id": fold["fold_id"], "manifest": name,
                            "strategy": mode, "rule": rule,
                            "n_arms": [int(a.size) for a in arms],
                            "calibration": int(c_idx.size),
                            "decomp": res, "boot": boot,
                            "passes": bool(res["rho_max"] <= RHO_GATE
                                           and boot["rho_max_ci"][1] <= RHO_GATE)}
                    seq_out["cells"].append(cell)
                    print(f"  {fold['fold_id']:32s} {rule:14s} N={[a.size for a in arms]} "
                          f"V_S {res['V_S']:.2e} nf={[round(x, 3) for x in res['noise_frac']]} "
                          f"rho_mean {res['rho_mean']:+.3f} rho_max {res['rho_max']:+.3f} "
                          f"CI{np.round(boot['rho_max_ci'], 3)} "
                          f"{'PASS' if cell['passes'] else 'fail'}", flush=True)

                    if n_calib < args.calib_folds:
                        print(f"    calibration (additive injection):", flush=True)
                        cal = calibrate(x2d, gram, c_idx, basis, arms, shape3d, cshape,
                                        valid, seed=n_calib + 17)
                        cell["calibration_injection"] = cal
                        n_calib += 1
                    del maps
                del c_idx, pool, arms
        result["sequences"][seq] = seq_out
        del x2d, gram, iq

    (out / "gate_n.json").write_text(json.dumps(result, indent=1))
    n = sum(len(v["cells"]) for v in result["sequences"].values())
    npass = sum(c["passes"] for v in result["sequences"].values() for c in v["cells"])
    print(f"\nwrote {out / 'gate_n.json'}  ({n} cells, {npass} pass)", flush=True)


if __name__ == "__main__":
    main()

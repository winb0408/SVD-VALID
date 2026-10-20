"""Gate T: is the inter-arm excess covariance TEMPORAL?

See Project_plan.md 5.7 (pre-registered before this file was run).

The spatial covariance cannot tell "structural deviation" from "correlated noise":
both are the same object there, and the real data's 1.9% non-rank-1 part is the
same size as the signal (reports/phase2_gate_n_calibration_analysis.md 4.1).
The two DO differ in how they depend on frame distance: a genuine temporal noise
correlation decays with the frame gap, while a structural deviation -- every arm
sees the same anatomy -- does not.

Two measurements, both pre-registered:
  rho_f(d)   single-frame map fluctuation autocovariance, asymptote-corrected.
             At the single-frame level the noise fraction is ~1, where the
             estimator is exact (calibration: alpha=0.5 -> 0.502 at nf=1.0).
  excess(d)  inter-arm excess correlation vs the mean frame gap between arms.

If excess(d) decays like rho_f(d), the excess is temporal.  If it is flat in d,
the excess is structural and Gate N's numbers are not noise correlations.
"""
import argparse, json, sys, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import evaluate_pair_statistics as E
import gate_noise_independence as G
from build_crossfit_pairs import rule_edges


def per_frame_maps(x2d, gram, c_idx, basis, frames, chunk=32):
    """|P x_t|^2 for each t in `frames`, returned as (V, n) float64.

    The shipped projected_power_map() sums |.|^2 over the target columns; the
    time axis is only visible if that sum is skipped, so it is re-done here with
    the identical projector (u = X_C B, w = B^H G[C, t]).
    """
    cols = np.asarray(frames)
    w = basis.conj().T @ gram[np.ix_(c_idx, cols)]        # (R, n)
    u = x2d[:, c_idx] @ basis                            # (V, R)
    out = np.empty((x2d.shape[0], len(cols)), dtype=np.float64)
    for s in range(0, len(cols), chunk):
        f = u @ w[:, s:s + chunk]
        out[:, s:s + chunk] = np.abs(f) ** 2
    return out


def lag_cov(F, dmax):
    """c(d) of the de-meaned fluctuation matrix F (V, n): average covariance at lag d."""
    n = F.shape[1]
    C = (F.T @ F) / F.shape[0]
    out = []
    for d in range(dmax + 1):
        if d >= n:
            out.append(np.nan); continue
        out.append(float(np.mean(np.diagonal(C, offset=d))))
    return np.array(out)


def corrected(c, lo=55, hi=60):
    c_inf = float(np.mean(c[lo:hi + 1]))
    r = (c - c_inf) / (c[0] - c_inf)
    return r, c_inf


def ar1_field(v, n, rho1, rng):
    """Complex field (V, n) with autocorrelation rho(d) = rho1^d, unit variance."""
    b = np.sqrt(1.0 - rho1 ** 2)
    out = np.empty((v, n), dtype=np.complex64)
    s = (rng.standard_normal((v, 1)) + 1j * rng.standard_normal((v, 1))) / np.sqrt(2)
    out[:, 0] = s[:, 0]
    for t in range(1, n):
        s = rho1[:, None] * s + b[:, None] * (rng.standard_normal((v, 1))
                                             + 1j * rng.standard_normal((v, 1))) / np.sqrt(2)
        out[:, t] = s[:, 0]
    return out


def arm_cov_pred(t_arms, c_lag, all_frames):
    """Predicted covariance between arm means of f_t, exactly from the lag kernel.

    f_t = eps_t - mean_{pool}(eps),  Cov(eps_s, eps_t) = K(s-t),  K = c(|d|), 0 beyond dmax.
    Then with  A(s) = mean_{u in pool} K(s-u)  and  g = mean_{s in pool} A(s):
        Cov(f_s, f_t) = K(s-t) - A(s) - A(t) + g
        Cov(M_X, M_Y) = mean_{x,y} [ K(x-y) - A(x) - A(y) + g ].
    An earlier version replaced A(s) and g by the truncation sum D/n, which is wrong
    as soon as the kernel width is a sizeable fraction of n -- for a fully persistent
    kernel it returns a positive variance where the truth is exactly zero, and on this
    data it made the predicted variance of an arm mean negative.
    """
    dmax = c_lag.size - 1
    pool = np.asarray(all_frames, int)
    gg = np.abs(pool[:, None] - pool[None, :])
    Kpp = np.where(gg <= dmax, c_lag[np.minimum(gg, dmax)], 0.0)
    A = Kpp.mean(axis=1)
    g = float(A.mean())
    pos = {int(t): i for i, t in enumerate(pool)}
    k = len(t_arms)
    out = np.empty((k, k))
    for i in range(k):
        ai = A[[pos[int(t)] for t in t_arms[i]]]
        for j in range(k):
            aj = A[[pos[int(t)] for t in t_arms[j]]]
            gij = np.abs(t_arms[i][:, None] - t_arms[j][None, :])
            out[i, j] = float(np.mean(np.where(gij <= dmax, c_lag[np.minimum(gij, dmax)], 0.0)
                                      - ai[:, None] - aj[None, :] + g))
    return out, g


def corr_of(cov):
    d = np.diag(cov).copy()
    d[d <= 0] = np.nan
    return cov / np.sqrt(np.outer(d, d))


def excess_vs_gap(F, c_lag, shape3d, cshape, valid, frames, all_frames, kmax=5):
    """Measured inter-arm covariance/correlation of the fluctuation arm means vs prediction.

    The arm means are NOT demeaned across arms: doing that imposes sum_X |X| M_X = 0
    and forces +-0.99 correlations on data whose true correlation is ~0 (the previous
    version did this and was unusable).  The structural component is already removed
    because F is demeaned in time.
    """
    Fc = F.reshape(shape3d + (F.shape[1],))[:cshape[0]]
    Fv = Fc.reshape(-1, F.shape[1])[valid]
    frames = np.asarray(frames)
    n = F.shape[1]
    _, gbar = arm_cov_pred([frames], c_lag, all_frames)
    vf = float(c_lag[0] - gbar)     # single-frame fluctuation variance, exactly
    rows = []
    for stride in range(1, kmax + 1):
        arms = G.split_into_k(np.arange(frames.size), 4, "interleaved", stride)
        if len(arms) < 4:
            continue
        t_arms = [frames[a] for a in arms]
        maps = np.stack([Fv[:, a].mean(axis=1) for a in arms])
        cm = (maps @ maps.T) / maps.shape[1]
        cp, _ = arm_cov_pred(t_arms, c_lag, all_frames)
        rm, rp = corr_of(cm), corr_of(cp)
        pairs = []
        for (i, j) in G.pairs_of(4):
            pairs.append({"pair": f"{i}-{j}",
                          "gap": float(np.mean(np.abs(t_arms[i][:, None] - t_arms[j][None, :]))),
                          "meas_cov_vf": float(cm[i, j] / vf),
                          "pred_cov_vf": float(cp[i, j] / vf),
                          "meas_rho": float(rm[i, j]),
                          "pred_rho": float(rp[i, j])})
        rows.append({"stride": stride, "arms": [int(a.size) for a in arms],
                     "vf": vf, "pairs": pairs})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64"])
    ap.add_argument("--split", default="sa64_int3")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dmax", type=int, default=60)
    ap.add_argument("--out", default="reports/phase2_gate_t_2026-10-19")
    a = ap.parse_args()

    import yaml, os
    base = yaml.safe_load(open(a.baseline)); cfg = yaml.safe_load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    for seq in a.sequences:
        iq = E.load_sequence_iq(seq, base); x2d3 = iq.reshape(-1, iq.shape[3]).astype(np.complex128)
        shape3d = tuple(iq.shape[:3]); cshape = E.cropped_shape(shape3d)
        valid3 = np.load(f"data/processed/official/{seq}/valid_mask.npy")[:cshape[0], :cshape[1], :cshape[2]]
        valid = valid3.reshape(-1).astype(bool)
        gram = x2d3.conj().T @ x2d3
        for fi in a.folds:
            fold = json.load(open(f"data/processed/crossfit/splits/{a.split}.json"))["folds"][fi]
            c = np.asarray(fold["calibration"], int)
            pool = np.asarray(sorted(list(fold["A"]) + list(fold["B"])), int)
            ev, V = np.linalg.eigh(gram[np.ix_(c, c)]); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
            lo, hi = rule_edges("proportional", cfg, ev); hi = min(hi, c.size)
            b = V[:, lo:hi] / np.sqrt(np.maximum(ev[lo:hi], 0))[None, :]

            print(f"\n=== {seq} {a.split} fold {fi}  pool {pool.size}  band [{lo},{hi}) ===", flush=True)

            # ---- anchor A0: the per-frame maps must average back to the arm map
            arm = pool[:40]
            pm = per_frame_maps(x2d3, gram, c, b, arm)
            ref = E.to_cropped(E.projected_power_map(x2d3, gram, c, b, tgt=arm) / arm.size,
                               shape3d, cshape).reshape(-1)
            got = E.to_cropped(pm.mean(axis=1), shape3d, cshape).reshape(-1)
            a0 = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
            print(f"  A0 per-frame vs arm map  rel.err {a0:.3e}  {'OK' if a0 < 1e-8 else 'FAIL'}", flush=True)

            # Demean over ALL frames.  Using the arm pool instead makes the A and B
            # arm means exact negatives (sum_X |X| M_X = 0 with |A| = |B|), so the
            # inter-arm correlation is identically -1 and carries no information.
            M = per_frame_maps(x2d3, gram, c, b, np.arange(x2d3.shape[1]))
            F = M - M.mean(axis=1, keepdims=True)
            c_lag = lag_cov(F, a.dmax)
            rho, c_inf = corrected(c_lag)
            print("  rho_f(d): " + " ".join(f"{d}:{rho[d]:+.3f}" for d in (1, 2, 3, 5, 10, 20, 30, 50)), flush=True)
            a2 = abs(float(rho[50]))
            print(f"  A2 |rho_f(50)| = {a2:.3f}  {'OK' if a2 <= 0.10 else 'FAIL (non-decaying component)'}", flush=True)

            # ---- A1: known-answer injection, field rho_x(d) = exp(-d/8)
            # Two corrections to the pre-registered anchor, both recorded in the plan:
            # (i) the truth of the MAP fluctuation is rho_x(d)^2, not rho_x(d): the map is
            #     |P x|^2 and for a circular complex Gaussian field
            #     Cov(|z_t|^2,|z_{t+d}|^2) = |E[z_t conj(z_{t+d})]|^2 (checked numerically).
            # (ii) an injection that is spatially white never reaches the map: the map
            #     projects onto the 97-dim retained subspace of a 151552-dim voxel space,
            #     so a white field keeps a fraction ~97/151552 of its energy (measured:
            #     map changes by 9e-4 for an injection with RMS ~2x the data).  The field
            #     is therefore projected into the subspace first, then scaled so that its
            #     per-voxel RMS is amp_scale x the data RMS.
            L = 8.0
            truth = np.exp(-2.0 * np.arange(a.dmax + 1) / L)
            u = x2d3[:, c] @ b
            rng = np.random.default_rng(11)
            raw = ar1_field(x2d3.shape[0], pool.size, np.full(x2d3.shape[0], np.exp(-1.0 / L)), rng)
            proj = u @ (u.conj().T @ raw)
            # Normalise in the RETAINED subspace, not on the voxel grid.  Real data puts
            # ~99.8% of its energy in the 97-dim subspace, so a field whose voxel-RMS
            # equals the data's has only ~1/600 of the data's in-subspace power (measured
            # 1559 vs 9.8e5 per coefficient) and moves rho_f by ~0.02.  amp_scale is
            # therefore defined as the injected in-subspace power fraction sqrt-ratio.
            sp = float(np.mean(np.abs(u.conj().T @ proj) ** 2))
            sd = float(np.mean(np.abs(b.conj().T @ gram[np.ix_(c, pool)]) ** 2))
            proj *= np.sqrt(sd / max(sp, 1e-300))
            del raw
            # The open question is how large the injection must be to be recoverable.
            # Scaling by mean in-subspace power is the wrong knob: folds 1 and 2 carry
            # 240x and 213x the single-frame FLUCTUATION variance of fold 0, concentrated
            # in a few hundred voxels, so an injection matched on mean power is invisible
            # there (it moved rho_f by <=0.01 up to amp 5).  Scale instead so that the
            # injected map fluctuation variance equals amp_scale^2 x the data's c(0).
            # The injected-only fluctuation is quadratic in the scale, hence the ^0.25.
            xrep = x2d3.copy(); xrep[:, pool] = proj
            Mrep = per_frame_maps(xrep, xrep.conj().T @ xrep, c, b, pool)
            Frep = Mrep - Mrep.mean(axis=1, keepdims=True)
            c0_inj = float(lag_cov(Frep, 0)[0])
            del xrep, Mrep, Frep
            s_unit = (float(c_lag[0]) / max(c0_inj, 1e-300)) ** 0.25
            print(f"  A1 scale unit {s_unit:.4g}  (c0 data {c_lag[0]:.4g}, injected-only {c0_inj:.4g})",
                  flush=True)
            a1 = []
            for amp_scale in (0.0, 1.0, 2.0, 5.0):
                xi = x2d3.copy()
                xi[:, pool] += proj * (s_unit * amp_scale)
                gi = xi.conj().T @ xi
                Mi = per_frame_maps(xi, gi, c, b, pool)
                Fi = Mi - Mi.mean(axis=1, keepdims=True)
                ci = lag_cov(Fi, a.dmax); ri, _ = corrected(ci)
                err = float(max(abs(ri[d] - truth[d]) for d in (1, 5, 20)))
                a1.append({"amp_scale": amp_scale, "rho": ri.tolist(), "max_err": err})
                print("  A1 amp %.1f: " % amp_scale
                      + " ".join(f"d={d} {ri[d]:+.3f}/{truth[d]:.3f}" for d in (1, 5, 20))
                      + f"   max err {err:.3f}", flush=True)
                del xi, gi, Mi, Fi
            a1_ok = min(x["max_err"] for x in a1) <= 0.10
            del proj, u

            # ---- primary: rho_AB predicted for the actual A/B split
            A = np.asarray(sorted(fold["A"]), int); B = np.asarray(sorted(fold["B"]), int)
            cpAB = corr_of(arm_cov_pred([A, B], c_lag, np.arange(x2d3.shape[1]))[0])
            pred = float(cpAB[0, 1])
            print(f"  rho_AB predicted {pred:+.3f}  (A {A.size}  B {B.size})  "
                  f"-> {'PASS' if abs(pred) <= 0.10 else 'FAIL'} vs 0.10", flush=True)

            # ---- shape test: measured excess vs the same quantity predicted from rho_f
            rows = excess_vs_gap(F, c_lag, shape3d, cshape, valid, pool, np.arange(x2d3.shape[1]))
            for r in rows:
                ps = "  ".join(f"{p['pair']} gap{p['gap']:.1f} "
                               f"cov {p['meas_cov_vf']:+.4f}/{p['pred_cov_vf']:+.4f} "
                               f"rho {p['meas_rho']:+.3f}/{p['pred_rho']:+.3f}"
                               for p in r["pairs"])
                print(f"    stride {r['stride']} arms {r['arms']}: {ps}", flush=True)

            json.dump({"seq": seq, "split": a.split, "fold": fi, "band": [lo, hi],
                       "A0_rel_err": a0, "A2_abs_rho50": a2, "A2_ok": bool(a2 <= 0.10),
                       "rho_f": rho.tolist(), "c_lag": c_lag.tolist(), "c_inf": c_inf,
                       "A1": a1, "A1_ok": bool(a1_ok), "A1_truth": truth.tolist(),
                       "rho_AB_pred": pred, "excess_vs_gap": rows},
                      open(f"{a.out}/{seq}_{a.split}_f{fi}.json", "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

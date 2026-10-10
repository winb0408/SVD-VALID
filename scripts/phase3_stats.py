#!/usr/bin/env python3
"""Phase 3: statistical validation of the SVD-guided sampling hypothesis.

Core claim (Project_plan, Innovation 1; GPT-review point on proxy evidence):
    Guided pattern selection picks Tetris patterns whose paired samples have
    LOWER noise-band correlation while PRESERVING signal-band correlation.

Measured directly on sample-level band projections (not RMS-map proxies):

    For pattern k, its four half-resolution subvolumes are built from the
    SIGNAL-band projection (SVD ranks 60-350) and from the NOISE-band
    projection (ranks 350-400) of the raw SA64/HSA64 IQ data.
    rho_signal(k) / rho_noise(k) = mean Pearson correlation over the six
    pairs of the four subvolumes.

Two domains are reported:
    raw      - linear band power, exactly what the network's samples are;
    detrended- local-mean (9^3) removed first, isolating the local noise
               correlation that a Noise2Noise-style loss has to fight (the
               raw correlation is dominated by shared low-frequency tissue
               structure which the network can predict anyway).

Baselines:
    uniform  - mean over the 8 patterns = what per-block random VALID
               sampling (B group) achieves in expectation;
    oracle   - min_k rho_noise(k) per patch = best achievable.

Selection is taken with the SAME selector the training pipeline used
(select_guided_pattern on the prepared signal/noise maps), so the numbers
reflect the actual mechanism, not a re-derivation.

Outputs (runs/phase3/):
  band_stats.json              all numbers below
  figure_phase3_stats.png      per-pattern bars (both domains) + selection
  figure_phase3_selection.png  2D selection maps over the signal MIP
  pattern_selection.npy        guided pattern id per sliding window
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.sampling import TETRIS_PATTERNS, all_phases, select_guided_pattern  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_view import load_sa64  # noqa: E402


def band_power_volume(C, V, lo, hi, shape3d):
    """Linear band power per voxel (sum over band ranks, like official PD)."""
    Q = V[:, lo:hi].astype(np.complex64)
    P = C @ Q                                   # (voxels, band ranks)
    pw = np.sum(np.abs(P) ** 2, axis=1)
    return pw.reshape(shape3d).astype(np.float32)


def pattern_pairwise_corr(power_vol, pattern_ids):
    """Mean Pearson r over the 6 pairs of the pattern's 4 half-res subvolumes."""
    phases = all_phases(torch.as_tensor(power_vol, dtype=torch.float64))
    v = []
    for i in pattern_ids:
        x = phases[i].flatten()
        x = x - x.mean()
        v.append(x / (x.norm() + 1e-12))
    rs = [float((v[i] * v[j]).sum()) for i in range(4) for j in range(i + 1, 4)]
    return float(np.mean(rs)), rs


def patch_stats(sig_pow, noi_pow):
    """rho_signal / rho_noise for all 8 patterns on one patch."""
    rs, rn = [], []
    for pattern in TETRIS_PATTERNS.tolist():
        s, _ = pattern_pairwise_corr(sig_pow, pattern)
        n, _ = pattern_pairwise_corr(noi_pow, pattern)
        rs.append(s)
        rn.append(n)
    return np.array(rs), np.array(rn)


def summarize(name, rows):
    """rows: list of (guided_rn, uniform_rn, oracle_rn, guided_rs, uniform_rs)."""
    from scipy.stats import wilcoxon
    a = np.array(rows)
    out = dict(
        n=len(rows),
        rho_noise_guided=float(a[:, 0].mean()), rho_noise_uniform=float(a[:, 1].mean()),
        rho_noise_oracle=float(a[:, 2].mean()),
        rho_noise_guided_std=float(a[:, 0].std()), rho_noise_uniform_std=float(a[:, 1].std()),
        rho_signal_guided=float(a[:, 3].mean()), rho_signal_uniform=float(a[:, 4].mean()),
    )
    # paired tests: does the guided choice beat the uniform expectation?
    for label, diff in [("noise", a[:, 1] - a[:, 0]),   # uniform - guided, >0 means guided better
                        ("signal", a[:, 3] - a[:, 4])]:
        if np.allclose(diff, diff[0]):
            out[f"wilcoxon_{label}_p"] = 1.0
        else:
            _, p = wilcoxon(diff)
            out[f"wilcoxon_{label}_p"] = float(p)
        out[f"paired_diff_{label}_mean"] = float(diff.mean())
        out[f"paired_diff_{label}_sem"] = float(diff.std() / np.sqrt(len(diff)))
    print(f"[{name}] n={out['n']}")
    print(f"  rho_noise : guided {out['rho_noise_guided']:+.4f}  "
          f"uniform {out['rho_noise_uniform']:+.4f}  oracle {out['rho_noise_oracle']:+.4f}  "
          f"(Wilcoxon p={out['wilcoxon_noise_p']:.2e})")
    print(f"  rho_signal: guided {out['rho_signal_guided']:+.4f}  "
          f"uniform {out['rho_signal_uniform']:+.4f}  "
          f"(Wilcoxon p={out['wilcoxon_signal_p']:.2e})")
    out["noise_reduction_vs_uniform"] = out["rho_noise_uniform"] - out["rho_noise_guided"]
    out["oracle_fraction_captured"] = (
        (out["rho_noise_uniform"] - out["rho_noise_guided"]) /
        max(out["rho_noise_uniform"] - out["rho_noise_oracle"], 1e-9))
    print(f"[{name}] n={out['n']}")
    print(f"  rho_noise : guided {out['rho_noise_guided']:+.4f}  "
          f"uniform {out['rho_noise_uniform']:+.4f}  oracle {out['rho_noise_oracle']:+.4f}")
    print(f"  rho_signal: guided {out['rho_signal_guided']:+.4f}  "
          f"uniform {out['rho_signal_uniform']:+.4f}")
    print(f"  noise-corr reduction vs uniform: {out['noise_reduction_vs_uniform']:+.4f} "
          f"({out['oracle_fraction_captured']*100:.1f}% of oracle)")
    return out


def window_slices(z, x, y, win, stride):
    zs = list(range(0, z - win[0] + 1, stride[0]))
    xs = list(range(0, x - win[1] + 1, stride[1]))
    ys = list(range(0, y - win[2] + 1, stride[2]))
    return zs, xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat")
    ap.add_argument("--signal-map", default="data/processed/sa64_blood_remap/signal_map.npy")
    ap.add_argument("--noise-map", default="data/processed/sa64_blood_remap/noise_map.npy")
    ap.add_argument("--rank-low", type=int, default=60)
    ap.add_argument("--rank-high", type=int, default=350)
    ap.add_argument("--noise-low", type=int, default=350)
    ap.add_argument("--noise-high", type=int, default=400)
    ap.add_argument("--train-patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--map-window", nargs=3, type=int, default=[16, 16, 16])
    ap.add_argument("--map-stride", nargs=3, type=int, default=[4, 4, 4])
    ap.add_argument("--detrend-size", type=int, default=9)
    ap.add_argument("--out", default="runs/phase3")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading + SVD ...")
    raw = load_sa64(args.input)
    z, x, y, t = raw.shape
    C = raw.reshape(-1, t).astype(np.complex64)
    G = (C.conj().T @ C).astype(np.complex64)
    evals, V = np.linalg.eigh(G)
    V = V[:, np.argsort(evals)[::-1]]

    sig_pow = band_power_volume(C, V, args.rank_low, args.rank_high, (z, x, y))
    noi_pow = band_power_volume(C, V, args.noise_low, args.noise_high, (z, x, y))
    del C, G, raw

    print("detrending ...")
    sig_det = (sig_pow - uniform_filter(sig_pow, size=args.detrend_size)).astype(np.float32)
    noi_det = (noi_pow - uniform_filter(noi_pow, size=args.detrend_size)).astype(np.float32)

    sig_map = np.load(args.signal_map).astype(np.float32)
    noi_map = np.load(args.noise_map).astype(np.float32)
    assert sig_map.shape == noi_map.shape == sig_pow.shape, \
        (sig_map.shape, noi_map.shape, sig_pow.shape)

    ze = z - (z % 2)  # even crop for all_phases

    # ---------- selection (once, with the training-time selector) ----------
    tp = args.train_patch
    train_slices = [(slice(z0, z0 + tp[0]), slice(0, x), slice(0, y))
                    for z0 in range(0, z - tp[0] + 1)]
    choices_train = [select_guided_pattern(sig_map[sl], noi_map[sl])[0]
                     for sl in train_slices]

    wz = args.map_window
    zs, xs, ys = window_slices(z, x, y, wz, args.map_stride)
    map_slices = [(slice(z0, z0 + wz[0]), slice(x0, x0 + wz[1]), slice(y0, y0 + wz[2]))
                  for z0 in zs for x0 in xs for y0 in ys]
    sel = np.full((len(zs), len(xs), len(ys)), -1, dtype=np.int8)
    counts = np.zeros(8, dtype=int)
    for idx, sl in enumerate(map_slices):
        pid = select_guided_pattern(sig_map[sl], noi_map[sl])[0]
        sel[idx // (len(xs) * len(ys)), (idx // len(ys)) % len(xs), idx % len(ys)] = pid
        counts[pid] += 1
    np.save(out / "pattern_selection.npy", sel)
    print(f"training-patch choices (z0=0..{len(train_slices)-1}): {choices_train}")
    print(f"sliding-window selection counts: {counts.tolist()}")

    # ---------- statistics in both domains ----------
    results = {}
    for domain, (sig_v, noi_v) in [("raw", (sig_pow, noi_pow)),
                                   ("detrended", (sig_det, noi_det))]:
        print(f"\n===== domain: {domain} =====")

        print("-- whole-volume per-pattern statistics --")
        global_stats = []
        for k, pattern in enumerate(TETRIS_PATTERNS.tolist()):
            s, _ = pattern_pairwise_corr(sig_v[:ze], pattern)
            n, _ = pattern_pairwise_corr(noi_v[:ze], pattern)
            global_stats.append(dict(pattern=k, positions=pattern,
                                     rho_signal=s, rho_noise=n))
            print(f"  pattern {k} {pattern}: rho_signal={s:+.4f}  rho_noise={n:+.4f}")

        print("-- training-size patches --")
        rows_train = []
        for sl, pid in zip(train_slices, choices_train):
            rs, rn = patch_stats(sig_v[sl], noi_v[sl])
            rows_train.append((rn[pid], rn.mean(), rn.min(), rs[pid], rs.mean()))
        train_summary = summarize(f"{domain}/train-patch", rows_train)

        print("-- sliding windows --")
        rows_map = []
        for sl, pid in zip(map_slices, sel.ravel()):
            rs, rn = patch_stats(sig_v[sl], noi_v[sl])
            rows_map.append((rn[int(pid)], rn.mean(), rn.min(), rs[int(pid)], rs.mean()))
        map_summary = summarize(f"{domain}/sliding-window", rows_map)

        results[domain] = dict(global_stats=global_stats,
                               train_patch=dict(size=tp, guided_choices=choices_train,
                                                summary=train_summary),
                               sliding_window=dict(window=args.map_window,
                                                   stride=args.map_stride,
                                                   starts=[zs, xs, ys],
                                                   summary=map_summary,
                                                   selection_counts=counts.tolist()))

    with open(out / "band_stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(
            bands=dict(signal=[args.rank_low, args.rank_high],
                       noise=[args.noise_low, args.noise_high]),
            measure="mean Pearson r over the 6 pairs of a pattern's 4 "
                    "half-res subvolumes, on linear band power",
            detrend_size=args.detrend_size,
            selection_counts_sliding=counts.tolist(),
            **results,
        ), f, indent=2)

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ks = np.arange(8)
    w = 0.38
    for ax, domain in zip(axes, ["raw", "detrended"]):
        gs = results[domain]["global_stats"]
        ax.bar(ks - w / 2, [g["rho_signal"] for g in gs], w,
               label="rho_signal (60-350)", color="tab:red")
        ax.bar(ks + w / 2, [g["rho_noise"] for g in gs], w,
               label="rho_noise (350-400)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(ks, counts, "k^-", label="guided selection count")
        ax2.set_ylabel("selection count")
        ax.set_xlabel("Tetris pattern id")
        ax.set_ylabel("whole-volume mean pairwise r")
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8, ncol=3)
        ax.set_title(f"{domain} domain")
    fig.suptitle("Phase 3: per-pattern signal/noise correlation + guided selection frequency (SA64)")
    fig.tight_layout()
    fig.savefig(out / "figure_phase3_stats.png", dpi=150)
    plt.close(fig)

    # selection maps over the signal MIP, one row per z band
    sig_db = 10.0 * np.log10(np.maximum(sig_pow, 1e-12) / sig_pow.max())
    nz = len(zs)
    fig, axes = plt.subplots(nz, 2, figsize=(9, 2.6 * nz))
    if nz == 1:
        axes = axes[None, :]
    for i, z0 in enumerate(zs):
        band = sig_db[z0:z0 + wz[0]]
        im0 = axes[i, 0].imshow(band.max(axis=0).T, origin="lower", cmap="hot",
                                vmin=-25, vmax=0)
        axes[i, 0].set_title(f"signal band PD, z[{z0}:{z0 + wz[0]}] (dB, MIP)")
        fig.colorbar(im0, ax=axes[i, 0], shrink=0.8)
        im1 = axes[i, 1].imshow(sel[i].T, origin="lower", cmap="tab10",
                                vmin=-0.5, vmax=9.5, interpolation="nearest")
        axes[i, 1].set_title(f"guided pattern id (window {wz[0]}x{wz[1]}x{wz[2]}, stride {args.map_stride[0]})")
        fig.colorbar(im1, ax=axes[i, 1], shrink=0.8, ticks=range(8))
    fig.suptitle("Phase 3: 2D sampling-selection map (SA64)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "figure_phase3_selection.png", dpi=150)
    plt.close(fig)

    print(f"\nsaved figures to {out}/")


if __name__ == "__main__":
    main()

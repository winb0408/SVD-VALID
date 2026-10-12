"""Summary figure for the eight-pattern fixed-sampling scan.

Four panels, all from runs/eval_structure_scan.json + runs/pattern_scan/pattern_scan.json:

  (a) denoising strength vs vessel-core retention, one point per pattern,
  (b) denoising strength per pattern (the axis that says whether the model did
      any work at all -- a model that does nothing scores 1.000 on every keep
      metric, so (a) is only readable together with this panel),
  (c) the selector criterion itself, rho_noise per pattern, on zoomed axes and
      under BOTH definitions of the descriptor (raw whole-volume half-res, and
      detrended), because the two give different levels and different spreads:
      raw 0.902-0.907 (spread 0.005), detrended 0.341-0.363 (spread 0.023).
      Either way the spread is a small fraction of the level, so the selector
      has almost no freedom to choose on,
  (d) role permutations of pattern 4 (sampled voxel set held fixed).

Output: runs/official_mip/figure_scan_overview.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DENOISE_FLOOR = 0.40


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-eval", default="runs/eval_structure_scan.json")
    ap.add_argument("--scan", default="runs/pattern_scan/pattern_scan.json")
    ap.add_argument("--bands", default="runs/phase3/band_stats.json")
    ap.add_argument("--roles-eval", default="runs/eval_structure_sa64_roles.json")
    ap.add_argument("--out", default="runs/official_mip/figure_scan_overview.png")
    args = ap.parse_args()

    scan_eval = {r["name"]: r for r in load(args.scan_eval)["rows"]}
    scan = load(args.scan)
    roles_eval = {r["name"]: r for r in load(args.roles_eval)["rows"]}

    # pattern id -> F{n} row, plus the two reference arms
    pats = sorted(r["pattern"] for r in scan["rows"])
    den = {p: scan_eval[f"F{p}"]["hf_noise_reduction"] for p in pats}
    core = {p: scan_eval[f"F{p}"]["vessel_core_keep"] for p in pats}
    rho = {r["pattern"]: r["rho_noise"] for r in scan["rows"]}
    rho_sig = {r["pattern"]: r["rho_signal"] for r in scan["rows"]}
    assert scan["domain"] == "detrended", (
        f"panel (c) labels the scan curve as 'detrended', but the file says "
        f"domain={scan['domain']!r} -- fix the labels before using this figure")

    # same criterion, raw (non-detrended) whole-volume half-res definition
    bands = load(args.bands)
    raw = {r["pattern"]: r["rho_noise"]
           for r in bands["raw"]["global_stats"]}
    raw_sig = {r["pattern"]: r["rho_signal"]
               for r in bands["raw"]["global_stats"]}

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) strength vs retention -------------------------------------------------
    a = ax[0, 0]
    a.axvline(DENOISE_FLOOR, color="0.6", ls="--", lw=1)
    a.text(DENOISE_FLOOR, 1.02, " denoise floor", color="0.45", fontsize=8, va="top")
    a.axhline(1.0, color="0.75", ls=":", lw=1)
    a.text(0.62, 1.0, "do nothing scores 1.000 here", color="0.45",
           fontsize=8, va="bottom", ha="right")
    for p in pats:
        c = "tab:red" if den[p] < 0 else ("tab:green" if den[p] >= DENOISE_FLOOR
                                          else "tab:orange")
        a.scatter(den[p], core[p], s=70, color=c, zorder=3)
        a.annotate(f"P{p}", (den[p], core[p]), textcoords="offset points",
                   xytext=(7, 4), fontsize=10, fontweight="bold")
    for nm, col in [("B3", "0.35"), ("C3", "tab:blue")]:
        r = scan_eval[nm]
        a.scatter(r["hf_noise_reduction"], r["vessel_core_keep"], s=110, marker="s",
                  facecolor="none", edgecolor=col, linewidths=2, zorder=4)
        a.annotate(nm, (r["hf_noise_reduction"], r["vessel_core_keep"]),
                   textcoords="offset points", xytext=(7, -12), fontsize=10, color=col)
    a.set_xlabel("denoising strength  (hf_noise_reduction)")
    a.set_ylabel("vessel-core retention")
    a.set_title("(a) Only P4 denoises AND retains structure\n"
                "(open squares = original VALID baselines)", fontsize=10)
    a.grid(alpha=0.25)

    # (b) denoising strength per pattern ---------------------------------------
    b = ax[0, 1]
    cols = ["tab:red" if den[p] < 0 else ("tab:green" if den[p] >= DENOISE_FLOOR
                                          else "tab:orange") for p in pats]
    b.bar([f"P{p}" for p in pats], [den[p] for p in pats], color=cols)
    b.axhline(0, color="k", lw=1)
    b.axhline(DENOISE_FLOOR, color="0.6", ls="--", lw=1)
    for i, p in enumerate(pats):
        b.text(i, den[p] + (0.03 if den[p] >= 0 else -0.06), f"{den[p]:+.3f}",
               ha="center", fontsize=8)
    b.set_ylabel("hf_noise_reduction")
    b.set_title("(b) Negative = the model AMPLIFIES high-frequency noise", fontsize=10)
    b.grid(alpha=0.25, axis="y")

    # (c) the criterion itself, under both definitions of the descriptor --------
    c = ax[1, 0]
    c.axis("off")
    c.set_title("(c) The criterion is essentially CONSTANT across patterns\n"
                "-> the selector had nothing to choose on (axes are zoomed)", fontsize=10)
    c1 = c.inset_axes([0.09, 0.58, 0.87, 0.33])
    c2 = c.inset_axes([0.09, 0.12, 0.87, 0.33])
    xs = np.arange(len(pats))

    for ca, cur, sig, tag, col in (
            (c1, raw, raw_sig, "raw (whole-volume half-res)", "tab:purple"),
            (c2, rho, rho_sig, "detrended", "tab:brown")):
        ca.bar(xs, [cur[p] for p in pats], color=col)
        ca.set_xticks(xs)
        ca.set_xticklabels([f"P{p}" for p in pats])
        lo, hi = min(cur.values()), max(cur.values())
        pad = 0.15 * (hi - lo)
        ca.set_ylim(lo - pad, hi + pad)
        ca.axhline(lo, color="0.6", ls=":", lw=1)
        ca.axhline(hi, color="0.6", ls=":", lw=1)
        ca.text(0.02, 0.95,
                f"{tag}:  spread {hi - lo:.4f}  "
                f"({100 * (hi - lo) / ((hi + lo) / 2):.1f}% of level)\n"
                f"rho_signal spread {max(sig.values()) - min(sig.values()):.4f}",
                transform=ca.transAxes, fontsize=8, va="top",
                bbox=dict(fc="lightyellow", ec="0.7"))
        ca.set_ylabel("rho_noise", fontsize=8)
        ca.tick_params(labelsize=8)
        ca.grid(alpha=0.25, axis="y")

    # (d) role permutations ----------------------------------------------------
    d = ax[1, 1]
    names = ["F4", "F4r1", "F4r2", "F4r3", "F4r4"]
    labels = ["F4\n(0,1,2,3)", "r1", "r2", "r3", "r4\n(breaks disp.)"]
    xs = np.arange(len(names))
    dd = [roles_eval[n]["hf_noise_reduction"] for n in names]
    cc = [roles_eval[n]["vessel_core_keep"] for n in names]
    d.bar(xs, dd, color=["tab:green"] * 4 + ["tab:red"], alpha=0.85)
    for i in xs:
        d.text(i, dd[i] + 0.02, f"{dd[i]:+.3f}", ha="center", fontsize=8)
    d.set_xticks(xs)
    d.set_xticklabels(labels, fontsize=8)
    d.set_ylabel("hf_noise_reduction", color="tab:green")
    d.set_ylim(min(dd) - 0.25, max(dd) + 0.15)
    d2 = d.twinx()
    d2.plot(xs, cc, "o-", color="tab:blue", lw=2, ms=7, label="vessel-core retention")
    for i in xs:
        d2.text(i, cc[i] + 0.006, f"{cc[i]:.3f}", ha="center", fontsize=8,
                color="tab:blue")
    d2.set_ylabel("vessel-core retention", color="tab:blue")
    d2.set_ylim(min(cc) - 0.05, max(cc) + 0.03)
    d.set_title("(d) Permuting input->target roles on P4's fixed voxel set:\n"
                "r1-r3 keep the input-pair displacement -> no change", fontsize=10)

    fig.suptitle("Eight-pattern fixed-sampling scan (SA64, single seed)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)

    print("\npattern  denoise   core    rho_noise  rho_signal   (rho = detrended)")
    for p in sorted(pats, key=lambda q: -den[q]):
        print(f"  P{p}   {den[p]:+.4f}  {core[p]:.4f}   {rho[p]:.4f}    {rho_sig[p]:.4f}"
              f"      raw {raw[p]:.4f}")


if __name__ == "__main__":
    main()

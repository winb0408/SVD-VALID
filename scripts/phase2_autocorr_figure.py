#!/usr/bin/env python
"""Figure for Phase 2 item 1: how long the per-frame fluctuation stays correlated.

Left panel answers the design question directly.  The contiguous split is the
only one where every physical lag is available inside one group, so it traces
the full decay curve of the band-power correlation length.  The interleaved
splits can only ever sample that curve at multiples of their group period, and
their measured points are marked on it -- so the panel shows, rather than
asserts, that widening the interleave moves the split down the curve.

Right panel is the mode-resolved lag-1 autocorrelation of the coefficients.  It
is included because its SHAPE is the other Phase 2 finding: the correlation
falls from ~1 at mode 1, crosses zero near mode 100, and goes negative at high
mode index.  A negative lag-1 autocorrelation is the signature of a high-pass
residual, not of white noise, so the high-rank band cannot be called a noise
band on the strength of its spectrum alone.

The project rule applies: an image looking right is never the acceptance
criterion.  This figure is for interpretation only; the numbers live in
data/processed/crossfit/phase2_autocorr.json and the acceptance report.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path("data/processed/crossfit/phase2_autocorr.json")
OUT = Path("reports/figures/phase2_autocorr.png")
RULES = ("proportional", "cum_energy", "clutter_knee", "spectral")
COLORS = {"proportional": "C0", "cum_energy": "C1",
          "clutter_knee": "C2", "spectral": "C3"}


def curve(res, seq, split, rule, field="autocorr_A"):
    """Mean over folds of r(lag) -> (lags, r, null)."""
    acc = {}
    nf = 0
    for f in res[seq]["folds"]:
        if f["split"] != split:
            continue
        nf += 1
        b = f["rules"].get(rule, {})
        if b.get("status") != "OK":
            continue
        for lag, v in b["power"][field]["lags"].items():
            acc.setdefault(int(lag), []).append((v["r"], v["null"]))
    if not acc:
        return np.array([]), np.array([]), np.array([]), nf
    lags = np.array(sorted(acc))
    r = np.array([np.mean([a for a, _ in acc[l]]) for l in lags])
    nul = np.array([np.mean([b for _, b in acc[l]]) for l in lags])
    return lags, r, nul, nf


def main():
    res = json.loads(RES.read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # ---- left: decay curve + where each interleave actually lands ---------
    # The black curve is WITHIN one contiguous 133-frame window: it is the best
    # available estimate of the correlation length of the band power.  The
    # markers are the real interleaved A groups, whose frames span the whole
    # movie; they sit BELOW the window curve at the same lag, and that gap is
    # the point -- spreading frames across the acquisition mixes different
    # drift states and therefore shares less.  Labelling them as points "on"
    # the curve would hide the very effect that makes a wide interleave useful.
    lags, r, nul, _ = curve(res, "SA64", "sa64_cont3", "proportional")
    ax1.plot(lags, r, "o-", color="k", lw=1.8, ms=4,
             label="within one contiguous 133-frame window (proportional)")
    ax1.fill_between([0.9, 55], -nul.mean(), nul.mean(), color="0.85",
                     zorder=0, label=r"null $\pm1/\sqrt{n}$")
    for period, measured in ((3, 0.221), (5, 0.071)):
        ax1.plot([period], [measured], "D", ms=11, mfc="none", mew=2.2,
                 color="C3",
                 label=f"actual interleaved A group, period p={period}: {measured:+.3f}")
        ax1.annotate(f"p={period}", (period, measured),
                     textcoords="offset points", xytext=(9, -13), fontsize=9)
    # the spectral (narrow) band, for contrast: far more correlated
    lags_s, r_s, _, _ = curve(res, "SA64", "sa64_cont3", "spectral")
    ax1.plot(lags_s, r_s, "s--", color=COLORS["spectral"], lw=1.4, ms=3.5,
             label="same window, spectral (narrow) band")
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set_xlabel("physical frame lag $\\ell$")
    ax1.set_ylabel("autocorrelation of per-frame band power")
    ax1.set_title("How long the band power stays correlated\n"
                  "(SA64; curve = within-window, markers = actual interleaved groups)")
    ax1.set_xlim(0.9, 55)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(alpha=0.3)

    # ---- right: mode-resolved lag-1 autocorrelation (the high-pass shape) --
    for seq, ls in (("SA64", "-"), ("HSA64", "--")):
        bins = res[seq]["full_movie"]["mode_bins"]
        centres, vals, fracs = [], [], []
        for name, row in bins.items():
            lo, hi = row["mode_1based"]
            centres.append(np.sqrt(lo * hi))            # geometric centre
            vals.append(row["1"]["mean_re"])
            fracs.append(row["power_fraction"])
        ax2.plot(centres, vals, ls, marker="o", ms=4, color="C0" if seq == "SA64" else "C1",
                 label=f"{seq} (lag 1)")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.axvline(60, color="0.4", ls=":", lw=1.2)
    ax2.axvline(350, color="0.4", ls=":", lw=1.2)
    ax2.text(60, 0.9, " official band edges", fontsize=8, color="0.3")
    ax2.set_xscale("log")
    ax2.set_xlabel("mode index (geometric centre of bin, 1-based)")
    ax2.set_ylabel("lag-1 autocorrelation of coefficients")
    ax2.set_title("Lag-1 autocorrelation vs mode\n"
                  "(negative at high mode = high-pass residual, not white noise)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

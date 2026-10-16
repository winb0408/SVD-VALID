#!/usr/bin/env python
"""Phase 2, items 2--5: the statistical gate on the candidate observation pair.

The question is narrow and it is not about training: is (D_A, D_B) entitled to
be used as a Noise2Noise training pair?  Four things are measured, all on the
volumes Phase 1 already built:

  2. structure consistency -- do both arms show the same vascular network?
  3. intensity consistency -- is there a systematic level/contrast bias between
     the arms, and can a change applied to BOTH arms remove it?
  4. residual correlation  -- is the ARM-SPECIFIC fluctuation shared between the
     arms?  (corr(D_A, D_B) alone cannot answer this: the true vessel signal is
     supposed to be correlated.  Only the fluctuation around the shared
     structure is evidence about the noise.)
  5. cross-fold stability  -- does the direction survive regrouping the frames?

PRE-REGISTERED PROTOCOL (frozen in this docstring; nothing below is tuned after
looking at a result -- see Project_plan.md section 5.4, which allows exactly one
calibration of these numbers, before any model is trained, and requires the
change and its reason to be logged)
--------------------------------------------------------------------------
VESSEL CANDIDATE -- CALIBRATED ONCE, 2026-10-16, BEFORE ANY TRAINING.
The first draft of this file pre-registered the vessel-candidate rule as "power
at or above -20 dB relative to this map's own valid-voxel 99.9th percentile" --
the official display convention (clim = [-20, 0] dB in
main_supplementary_figure_2.m) promoted to a segmentation rule.  First contact
with the data killed it: on SA64 that rule selects EVERY valid voxel,
124328 / 124328 (see runs/pair_stats_smoke.log, the [ref] line, and section 5.4's
one permitted calibration clause).  The reason is a measured property of the
official map, not a coding error -- SA64's full-sequence Power Doppler has only
about 8 dB of usable dynamic range (p50 = -8.23 dB re max; per-z-slice medians
run smoothly from -10.85 dB deep to -6.92 dB shallow), so a -20 dB floor sits
below the entire distribution and segments nothing.  It is a depth-gradient
floor with faint vessel modulation, consistent with L26/L28.  The arms are a
different animal: P1_proportional_A spans -19.97 dB from p50 to max and
P0_A spans -23.42 dB, so no single absolute dB threshold can mean the same thing
to the reference and to an arm, and Otsu-on-reference transfers not at all
(it selects 53.8% of the reference but 0.19% of P1_proportional_A).

The rule actually used is EQUAL-AREA: a voxel is a vessel candidate when its
power is in the top VESSEL_Q percent of that volume's own valid voxels.  This is
a per-volume relative threshold (it cannot be written as a dB offset), it is
applied identically to the reference, to every arm and to the Tetris control,
and at q = 5 it gives the reference 6216 voxels -- a workable mask, checked by
eye against reports/figures/phase1_pair_SA64_sa64_int3_c0a1b2_proportional.png.
It is the same rule Phase 1's pairing figure used, so Phase 1 and Phase 2 score
the same objects.

WHAT THIS COSTS, stated up front.  An equal-area rule cannot detect "this arm
dropped the weak vessels", because it guarantees every map the same candidate
area: exactly the sensitivity the -20 dB draft was reaching for.  That failure
mode is therefore NOT delegated to the structure metrics -- it is caught by the
shared-scale skeleton intensity profile (skeleton_profile), which reads the arm's
level along the REFERENCE skeleton in dB on a scale fixed by the pooled pair, and
so does see a vessel go missing rather than renormalising it away.  The two
numbers must be read together; neither alone is the structure gate.

INTERPRETABILITY OF THE COVERAGE NUMBER.  At 5% area, dilating an arm mask by one
voxel covers a large part of the volume, so centreline coverage can be high for
reasons that have nothing to do with the vessels.  A null distribution is
therefore reported alongside it: the same coverage computed for random equal-area
masks drawn from the valid voxels (N_NULL draws).  A coverage that does not stand
out from its own null is not evidence of structure retention, whatever its
absolute value.

GRID.  The structure and intensity statistics use the full voxel grid.  The
residual-correlation statistics (item 4) use the z-cropped grid 0:zc with
zc = 2*(nz//2), because a 2x2x2 phase partition -- which the Tetris control arm
requires (datasets/sampling.py) -- needs an even axis, while the data is
(nz, 64, 64, 400) with nz = 37.  The comparison is run for ALL pair kinds on the
same cropped grid so the four numbers are commensurable; the cost is the last z
slice, 1/37 = 2.7% of the volume, dropped from item 4 only.

ROI PROTOCOL.  Completely disjoint classes, decided from the FULL-SEQUENCE
reference only (never from A or B, never from the pair being scored):
    background  p < 50     of the reference power over valid voxels
    mid         50 <= p < 99
    vessel      p >= 99
The reference is the pseudoreference of section 5.1 -- it comes from the same
acquisition, so it is not clean ground truth; it is used here only to place
ROIs, and every number computed on an ROI is reported per ROI so the reader can
see whether a conclusion depends on the ROI choice.

DETRENDING KERNEL.  9x9x9 normalised box filter (normalised so that dead voxels
do not pull the local mean down).  9 matches the kernel already used in
scripts/phase3_stats.py for the same purpose.

CENTRELINE TOLERANCE.  1 voxel (0.15 mm).  The arm's candidate mask is dilated
with the 26-neighbourhood before the coverage test, so a skeleton voxel counts
as covered if the arm has a candidate anywhere in its 3x3x3 neighbourhood.

BOOTSTRAP.  20000 draws.  Units are independent folds when a statistic is
per-fold (folds are disjoint frame groupings, so they are independent samples of
the grouping, not of the voxels), and non-overlapping 2x2x2 spatial blocks when
a statistic is voxel-level.  Overlapping sliding windows are never used as
units.

GATE.  section 5.4: centreline coverage >= 0.85 and width median change <= 15% against
the full-sequence reference; no obvious global intensity bias; and a residual
correlation at least 10% LOWER than the Tetris view with a 95% CI excluding 0.
Pair-P0 is NOT a control for the noise gate (section 5.4): it has no official SVD
calibration, its residual is dominated by tissue clutter, and the resulting gap
is not attributable to cross-fitting.  It is reported as a sanity check only.

THE TETRIS VIEW
---------------
Sections 5.4 requires the noise comparison to be "same grid, same official filtering,
only the sampling geometry differs".  VALID's own pair is two complementary
2x2x2 voxel subsets of the SAME frame.  So the control pair is built by
partitioning every 2x2x2 block into its 8 phases and splitting them into the two
complementary Tetris patterns ([0,1,2,3] vs [4,5,6,7] in the space-to-depth
order); each arm's block value is the mean of its four voxels' power, repeated
back over the block so both arms live on the same full grid as the CrossFit
arms.  The ONE operation -- block mean over the arm's voxels -- is applied
identically to all four pair kinds; the only difference is which voxels and
which frames feed each arm.  Two choices are offered for the frames: `pair` (the
fold's own A u B, so only the geometry differs -- the gate's setting) and `all`
(the whole movie, kept as a sensitivity check), and two for the band: `official` (the fold-
independent full-movie band, the literal reading of section 5.4) and `rule` (the
fold's own band, which makes "only the geometry differs" exact).  Both are
reported; the gate uses `rule`.

Residual asymmetry, disclosed: even with `pair` frames, a Tetris arm is the mean
over its block of the power summed across |A u B| = 2|A| frames, while a CrossFit
arm is the mean over |A| frames.  The control therefore has roughly twice the
frames per arm and so slightly LESS arm-specific fluctuation, which raises its
residual correlation and flatters CrossFit.  A control built to be exactly
matched would leave CrossFit looking worse, not better.

DEAD VOXELS.  Every statistic is restricted to the valid mask.  17.96% of voxels
are identically zero in the raw IQ, so an unrestricted comparison would award
free agreement for agreeing on zero.

WHAT THIS FILE DOES NOT DO.  It does not train anything, it does not call the
reference a clean ground truth, and it does not decide the verdict: it writes the
numbers that section 5.4's thresholds are applied to.
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
from datasets.iq_io import load_iq                                      # noqa: E402
from datasets.sampling import TETRIS_PATTERNS, one_phase                # noqa: E402
from scripts.build_crossfit_pairs import eigh_desc, rule_edges          # noqa: E402
from scripts.make_crossfit_splits import assert_fold                    # noqa: E402
from scripts.phase3_stats import block_bootstrap_ci, sign_flip_permutation_test  # noqa: E402

# ---------------------------------------------------------------- protocol ---
VESSEL_Q = 5.0                 # vessel candidate = top q% of OWN valid voxels
ABS_DB_LEGACY = -20.0          # the draft rule; degenerate on SA64, reported
                               # per volume so the failure stays visible
ROI_PCTS = (50.0, 99.0)        # background / mid / vessel cut points
DETREND_SIZE = 9               # normalised box filter, voxels
CENTRELINE_TOL = 1             # voxels; dilation radius before coverage
N_BOOT = 20000
N_NULL = 200                   # random equal-area masks for the coverage null
BLOCK = 2                      # 2x2x2 phase partition for the Tetris control
BOOT_BLOCK = 4                 # spatial unit for the item-4 statistics, voxels
PATTERN_A = tuple(TETRIS_PATTERNS[0].tolist())   # [0,1,2,4]
PATTERN_B = tuple(TETRIS_PATTERNS[7].tolist())   # [3,5,6,7]
ROUGH_DB = -20.0               # MIP display floor


# ------------------------------------------------------------------ loading ---
def load_sequence_iq(seq, base):
    """(V, T) complex64 and the (Z, X, Y, T) view of the same data."""
    f = base["data"]["files"][seq]
    path = str(Path(base["data"]["root"]) / f["path"])
    t0 = time.time()
    x = load_iq(path, variable="Data_" + seq)
    assert x.shape[:3] == tuple(f["shape_zyxt"][:3]), x.shape
    x = x.astype(np.complex64)
    print(f"  [iq] {seq} {x.shape} ({time.time() - t0:.1f}s)")
    return x


def to_db(power, valid, pct=99.9):
    """dB relative to the volume's own valid-voxel upper percentile.

    Scale-free on purpose: this is a SHAPE statement, and renormalising an arm
    before comparing its shape to a reference is legitimate as long as the
    renormalisation is disclosed.  Level statements use `shared_db` instead.
    """
    ref = np.percentile(power[valid], pct)
    return 10.0 * np.log10(np.maximum(power, 1e-30) / max(ref, 1e-30))


def shared_db(power, scale):
    """dB on a scale fixed by SOMEONE ELSE (the pooled pair scale).  Never own."""
    return 10.0 * np.log10(np.maximum(power, 1e-30) / max(scale, 1e-30))


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if d <= 0:
        return float("nan")
    return float(np.sum(a * b) / d)


def relative_independent_amplitude(a, b, mask):
    """How much of one arm the other does not reproduce at all.

    If a and b differ only by independent fluctuation of relative size s, then
    rms(a/b - 1) ~= sqrt(2) s.  This is the most directly readable form of what
    the noise gate is about: the fraction of the image a Noise2Noise loss would
    actually be supervised on.  A pair whose arms agree to 1% has 1% of signal
    to learn from, whatever its residual correlation says.
    """
    aa, bb = np.asarray(a)[mask], np.asarray(b)[mask]
    keep = bb > 0
    if int(keep.sum()) < 64:
        return {"n": int(keep.sum()), "rms_rel_diff": float("nan")}
    d = aa[keep] / bb[keep] - 1.0
    return {"n": int(keep.sum()),
            "rms_rel_diff": float(np.sqrt(np.mean(d ** 2))),
            "p50_abs_rel_diff": float(np.median(np.abs(d)))}


def normalised_box(vol, valid, size=DETREND_SIZE):
    """Local mean that ignores dead voxels instead of averaging zeros in."""
    from scipy.ndimage import uniform_filter
    m = valid.astype(np.float64)
    num = uniform_filter(vol * m, size=size, mode="nearest")
    den = uniform_filter(m, size=size, mode="nearest")
    return num / np.maximum(den, 1e-12)


# ------------------------------------------------------- tetris control arm ---
def projected_power_map(x2d, gram, idx_src, basis, tgt=None, chunk=64):
    """Per-voxel power of P x(t), summed over frames t.

    P = U U^H with U = X_src @ basis, so for the (V, T) data matrix X

        P X = U (U^H X) = (X_src B) (B^H G[src, :])

    which is why the whole thing is a pair of matmuls against a Gram slice: the
    projector is never materialised as a V x V operator, and this is the same
    construction Phase 1 used for D_A / D_B, so the Tetris control is filtered
    by the identical operator rather than by a lookalike.

    Column-chunked: a full (V, 400) complex64 intermediate is 485 MB, which is
    survivable but pointless to hold when only the column-sum of |.|^2 is kept.
    """
    cols = np.arange(x2d.shape[1]) if tgt is None else np.asarray(tgt)
    src = np.arange(x2d.shape[1]) if idx_src is None else np.asarray(idx_src)
    w = basis.conj().T @ gram[np.ix_(src, cols)]                 # (R, n_tgt)
    u = x2d[:, src] @ basis                                      # (V, R)
    out = np.zeros(x2d.shape[0], dtype=np.float64)
    for s in range(0, len(cols), chunk):
        f = u @ w[:, s:s + chunk]
        out += np.sum(np.abs(f) ** 2, axis=1, dtype=np.float64)
    return out


def tetris_pair(power_map, shape3d, valid3d):
    """Split a per-voxel power map into VALID's two complementary half-views.

    Returns (arm_a, arm_b) as full-grid maps: each 2x2x2 block carries the mean
    over the four voxels that belong to that arm, repeated across the block.
    Blocks that contain no valid voxel are left at zero and masked out.
    """
    pd3 = power_map.reshape(shape3d)
    v3 = valid3d
    phases = [one_phase(pd3, i) for i in range(8)]
    vph = [one_phase(v3.astype(np.float32), i) for i in range(8)]

    out = []
    for pat in (PATTERN_A, PATTERN_B):
        num = np.zeros_like(phases[0], dtype=np.float64)
        den = np.zeros_like(phases[0], dtype=np.float64)
        for i in pat:
            num += phases[i].astype(np.float64) * vph[i]
            den += vph[i]
        block = num / np.maximum(den, 1e-12)
        block = block * (den > 0)
        out.append(np.repeat(np.repeat(np.repeat(block, BLOCK, 0), BLOCK, 1), BLOCK, 2))
    return out[0].ravel(), out[1].ravel()


# --------------------------------------------------------- structure metrics ---
def vessel_mask(power, valid, q=VESSEL_Q):
    """Equal-area vessel candidate: top `q` percent of the volume's own voxels.

    `power` and `valid` are flat.  The threshold is taken over valid voxels only,
    so the 17.96% of dead voxels cannot dilute it.  Ties at the cut are kept
    (`>=`), so the selected area can exceed q percent slightly; the realised
    fraction is reported next to every use rather than assumed.
    """
    m = np.asarray(valid).astype(bool)
    if not m.any():
        return np.zeros_like(m)
    thr = np.percentile(power[m], 100.0 - q)
    return m & (power >= thr)


def legacy_candidate_fraction(power, valid):
    """How much of the volume the abandoned -20 dB rule would have taken.

    Kept in the output on purpose: it is the evidence that the draft rule was
    degenerate, and a reader who disagrees with the calibration can see exactly
    what it would have cost.
    """
    return float(np.mean(to_db(power, valid)[valid] >= ABS_DB_LEGACY))


def centreline_coverage_null(valid, ref_skel3d, shape3d, n_cand,
                             n_draws=N_NULL, seed=11):
    """Coverage of the reference skeleton by RANDOM equal-area masks.

    Coverage alone is not interpretable at 5% area and 1-voxel dilation, so the
    same statistic is computed for masks of the same size drawn uniformly from
    the valid voxels.  An arm's coverage says something about its structure only
    insofar as it beats this null.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, 3)
    n = int(ref_skel3d.sum())
    vi = np.flatnonzero(np.asarray(valid).astype(bool))
    n_cand = int(min(n_cand, vi.size))
    if n == 0 or n_cand <= 0:
        return {"n_draws": 0, "n_cand": n_cand}
    rng = np.random.default_rng(seed)
    skel_flat = ref_skel3d.ravel()
    vals = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        m = np.zeros(np.asarray(valid).size, dtype=bool)
        m[rng.choice(vi, size=n_cand, replace=False)] = True
        grown = binary_dilation(m.reshape(shape3d), structure=st,
                                iterations=CENTRELINE_TOL)
        vals[i] = np.logical_and(skel_flat, grown.ravel()).sum() / n
    return {"n_draws": int(n_draws), "n_cand": n_cand,
            "mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max())}


def skeleton(mask3d):
    from skimage.morphology import skeletonize
    return skeletonize(mask3d)


def dice(a, b):
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / s)


def jaccard(a, b):
    u = np.logical_or(a, b).sum()
    if u == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / u)


def centreline_coverage(arm_mask3d, ref_skel3d):
    """Fraction of reference skeleton voxels within CENTRELINE_TOL of the arm."""
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, 3)
    grown = binary_dilation(arm_mask3d, structure=st, iterations=CENTRELINE_TOL)
    n = int(ref_skel3d.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.logical_and(ref_skel3d, grown).sum() / n), n


def width_stats(mask3d, skel3d):
    """Local vessel diameter sampled ON the skeleton: 2x the distance transform.

    Reading the width only where the skeleton runs means the number describes
    vessel calibre rather than how fat the blob is, which is what the 15% width
    criterion is about.
    """
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(mask3d)
    vals = 2.0 * dt[skel3d & mask3d]
    if vals.size == 0:
        return {"n": 0, "median_vox": float("nan"), "mean_vox": float("nan")}
    return {"n": int(vals.size), "median_vox": float(np.median(vals)),
            "mean_vox": float(vals.mean()),
            "p25_vox": float(np.percentile(vals, 25)),
            "p75_vox": float(np.percentile(vals, 75))}


def component_count(mask3d, min_size=8):
    from scipy.ndimage import label, generate_binary_structure
    lab, n = label(mask3d, structure=generate_binary_structure(3, 3))
    if n == 0:
        return 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), int((sizes >= min_size).sum())


def skeleton_profile(arm_power3d, ref_skel3d, valid3d):
    """Vessel contrast of ONE map: its level on the reference skeleton vs off it.

    Both terms come from the same map, so the number carries no cross-pipeline
    normalisation.  That matters here: the official reference sums |.|^2 over the
    whole 400-frame movie while an arm averages over its own ~133 frames and
    occupies a narrower band, so the two are ~26 dB apart by construction (10
    log10 400) and any level statement that spans them is meaningless.  What is
    meaningful, and what the structure gate is really asking, is whether the map
    still puts its own high-power voxels where the reference's skeleton runs.  A
    map that does shows a positive contrast; one that has lost the vessels shows
    zero.

    The reference's own contrast is computed the same way and reported alongside
    -- it is an upper bound by construction (the skeleton IS part of the
    reference's top 5%), so it calibrates the scale of the number rather than
    competing with the arms.
    """
    on = arm_power3d[ref_skel3d & valid3d]
    off = arm_power3d[valid3d & ~ref_skel3d]
    if on.size == 0 or off.size == 0:
        return {}
    med_on, med_off = float(np.median(on)), float(np.median(off))
    return {"n_on": int(on.size), "n_off": int(off.size),
            "median_on": med_on, "median_off": med_off,
            "contrast_db": float(10.0 * np.log10(max(med_on, 1e-30)
                                                 / max(med_off, 1e-30)))}


def intensity_summary(power, valid, rois, scale):
    """Quantiles on a SHARED scale.  Never an arm-by-arm renormalisation."""
    db = shared_db(power, scale)
    out = {"scale_used": float(scale)}
    for name, m in rois.items():
        v = db[m.ravel() & valid]
        if v.size == 0:
            continue
        out[name] = {
            "n": int(v.size),
            "median_db": float(np.median(v)),
            "p10_db": float(np.percentile(v, 10)),
            "p25_db": float(np.percentile(v, 25)),
            "p75_db": float(np.percentile(v, 75)),
            "p90_db": float(np.percentile(v, 90)),
            "mean_db": float(v.mean()),
        }
    return out


# --------------------------------------------------- residual correlation ----
def residual_correlation(a3d, b3d, valid3d, rois3d, detrend=True):
    """Shared-fluctuation correlation, per ROI.

    The local mean absorbs the shared low-frequency structure (the vessel tree
    itself), so what is left is the local fluctuation around it.  If the two
    arms' fluctuation fields are independent this is ~0; a positive value is a
    shared fluctuation and is the thing an N2N loss cannot learn away.  The
    background ROI is the cleanest case: there the shared structure is absent to
    begin with, so its (undetrended) correlation is the most direct noise-
    correlation estimate available, and it is reported whether or not detrending
    is on.
    """
    out = {}
    for name, m in rois3d.items():
        mm = m & valid3d
        n = int(mm.sum())
        if n < 64:
            out[name] = {"n": n, "r": float("nan")}
            continue
        if detrend:
            a = a3d - normalised_box(a3d, valid3d)
            b = b3d - normalised_box(b3d, valid3d)
        else:
            a, b = a3d, b3d
        out[name] = {"n": n, "r": pearson(a[mm], b[mm])}
    return out


def cropped_shape(shape3d):
    """The z-cropped grid the 2x2x2 phase partition forces: even first axis."""
    return (2 * (shape3d[0] // 2),) + tuple(shape3d[1:])


def to_cropped(a_map, shape3d, cshape):
    """Put any pair-kind map on the cropped grid.

    CrossFit arms and the Tetris control are built by different code paths -- the
    arms live on the full grid, the Tetris control on the parity-cropped one --
    so they are normalised here rather than at each call site.
    """
    a = np.asarray(a_map)
    if a.size == int(np.prod(cshape)):
        return a.reshape(cshape)
    return a.reshape(shape3d)[:cshape[0]]


def block_residual_diffs(a3d, b3d, valid3d, size=BOOT_BLOCK):
    """Per-non-overlapping-block residual correlation, as independent units.

    A voxel-level correlation has no honest error bar -- neighbouring voxels of
    the same vessel are not independent samples.  So the same statistic is
    computed inside each non-overlapping `size`-cubed block and the blocks are
    treated as the units.

    `size` must be > BLOCK: a 2x2x2 block holds 8 voxels, which is below any
    sensible minimum for a Pearson r, and an earlier draft of this file silently
    dropped every unit that way.  The grid is derived from the array shape here
    rather than passed in, so an inconsistent axis count cannot recur.
    """
    a = (a3d - normalised_box(a3d, valid3d)).ravel()
    b = (b3d - normalised_box(b3d, valid3d)).ravel()
    m = valid3d.ravel()
    zz, yy, xx = np.indices(a3d.shape)
    n_y = -(-a3d.shape[1] // size)
    n_x = -(-a3d.shape[2] // size)
    bid = (((zz // size) * n_y + (yy // size)) * n_x + (xx // size)).ravel()
    order = np.argsort(bid[m], kind="stable")
    bid_m, a_m, b_m = bid[m][order], a[m][order], b[m][order]
    edges = np.searchsorted(bid_m, np.unique(bid_m))
    edges = np.append(edges, bid_m.size)
    min_vox = max(8, (size ** 3) // 2)
    vals, sizes = [], []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        if e - s < min_vox:
            continue
        x, y = a_m[s:e] - a_m[s:e].mean(), b_m[s:e] - b_m[s:e].mean()
        d = np.sqrt(np.sum(x * x) * np.sum(y * y))
        if d <= 0:
            continue
        vals.append(float(np.sum(x * y) / d))
        sizes.append(int(e - s))
    return np.asarray(vals), np.asarray(sizes)


# ------------------------------------------------------------------ figures ---
def mid_slices(v3):
    z, x, y = v3.shape
    return [v3[z // 2, :, :], v3[:, x // 2, :], v3[:, :, y // 2]]


def make_structure_figure(path, fold_id, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(panels.keys())
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 3.0 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, name in enumerate(rows):
        v3 = panels[name]
        slices = mid_slices(v3)
        mips = [v3.max(0), v3.max(1), v3.max(2)]
        cols = [("z mid", slices[0]), ("x mid", slices[1]), ("y mid", slices[2]),
                ("MIP xy", mips[0]), ("zoom", _zoom(v3))]
        for j, (t, img) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(img, cmap="inferno", vmin=ROUGH_DB, vmax=0.0,
                      aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(t + (" (not a structure claim)" if t == "MIP xy" else ""),
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle(f"{fold_id}: A / B / full-sequence pseudoreference / difference "
                 f"(shared -20..0 dB scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _zoom(v3, frac=0.22):
    z, x, y = v3.shape
    cz, cx, cy = int(z * 0.45), x // 2, y // 2
    hz, hx, hy = max(2, int(z * frac / 2)), max(4, int(x * frac / 2)), max(4, int(y * frac / 2))
    return v3[cz - hz:cz + hz, cx - hx:cx + hx, cy - hy:cy + hy].max(0)


# --------------------------------------------------------------------- main ---
def analyse_fold(seq, fold, rule_names, cfg, base, iq4d, valid, ref_power,
                 ref_db, ref_mask3d, ref_skel3d, ref_width, ref_comp,
                 rois3d, arms, tetris_by_rule):
    """Everything for one fold.  Pure function of the cached artifacts.

    `tetris_by_rule` maps a rule name to the (arm_a, arm_b) control pair built
    with that rule's band, so each rule is compared against a Tetris view of the
    same filtering strength and the comparison isolates the sampling geometry.
    """
    shape3d = tuple(iq4d.shape[:3])
    cshape = cropped_shape(shape3d)
    v3 = valid.reshape(shape3d)
    out = {"fold_id": fold["fold_id"], "split": fold["split"], "sequence": seq,
           "T_C": int(fold["T_C"]), "T_A": int(fold["T_A"]), "rules": {}}

    # full-sequence pseudoreference, scored the same way as any arm
    ref_stats = {
        "dice": 1.0, "jaccard": 1.0, "centreline_coverage": 1.0,
        "components": ref_comp[0], "components_ge_8": ref_comp[1],
        "width": ref_width,
        "candidate_fraction": float(ref_mask3d.sum() / max(v3.sum(), 1)),
    }

    for rule in rule_names:
        key = f"P1_{rule}"
        if key not in arms:
            continue
        r = {}
        pa, pb = arms[key]["A"], arms[key]["B"]
        p0a, p0b = arms["P0"]["A"], arms["P0"]["B"]
        p2a, p2b = arms.get(f"P2_{rule}", {}).get("A"), arms.get(f"P2_{rule}", {}).get("B")

        # --- 3. intensity, on the pooled A u B scale (Phase 1's frozen rule)
        scale = float(np.percentile(np.concatenate([pa[valid], pb[valid]]), 99.9))
        ia = intensity_summary(pa, valid, rois3d, scale)
        ib = intensity_summary(pb, valid, rois3d, scale)
        bias = {name: ia[name]["median_db"] - ib[name]["median_db"]
                for name in ia if name != "scale_used" and name in ib}
        r["intensity"] = {
            "pooled_scale": scale,
            "A": ia, "B": ib,
            "median_bias_db_A_minus_B": bias,
            # NO reference-vs-arm level number here.  The reference and the arms
            # differ by construction in both frame count (sum over 400 vs mean
            # over ~133 -> 26 dB) and band width, so such a number would report
            # the convention difference and nothing else.  The normalisation-free
            # substitute is skeleton_profile, which is a within-map contrast.
        }

        # --- 2. structure
        st = {}
        for label, p in (("A", pa), ("B", pb)):
            cand = vessel_mask(p, valid).reshape(shape3d)
            cov, nsk = centreline_coverage(cand, ref_skel3d)
            sk = skeleton(cand)
            w = width_stats(cand, sk)
            comp = component_count(cand)
            prof = skeleton_profile(p.reshape(shape3d), ref_skel3d, v3)
            st[label] = {
                "dice_vs_ref": dice(cand, ref_mask3d),
                "jaccard_vs_ref": jaccard(cand, ref_mask3d),
                "centreline_coverage": cov,
                "ref_skeleton_voxels": nsk,
                "components": comp[0], "components_ge_8": comp[1],
                "width": w,
                "width_median_change_vs_ref": (
                    (w["median_vox"] - ref_width["median_vox"])
                    / max(ref_width["median_vox"], 1e-12)),
                "candidate_fraction": float(cand.sum() / max(v3.sum(), 1)),
                "skeleton_contrast": prof,
            }
        st["reference"] = ref_stats
        st["A_vs_B_dice"] = dice(vessel_mask(pa, valid).reshape(shape3d),
                                 vessel_mask(pb, valid).reshape(shape3d))
        st["A_vs_B_width_median_change"] = (
            (st["A"]["width"]["median_vox"] - st["B"]["width"]["median_vox"])
            / max(st["B"]["width"]["median_vox"], 1e-12))
        r["structure"] = st

        # --- 4. residual correlation, four pair kinds, one estimator
        # All four kinds are put on the SAME cropped grid, so the comparison is
        # between pair kinds and not between grids (see GRID in the docstring).
        v3c = v3[:cshape[0]]
        rois3dc = {k: m[:cshape[0]] for k, m in rois3d.items()}

        def rc(a_map, b_map):
            a3 = to_cropped(a_map, shape3d, cshape)
            b3 = to_cropped(b_map, shape3d, cshape)
            return {"detrended": residual_correlation(a3, b3, v3c, rois3dc, True),
                    "raw": residual_correlation(a3, b3, v3c, rois3dc, False)}

        pairs = {"P1": (pa, pb), "P0": (p0a, p0b)}
        if p2a is not None:
            pairs["P2"] = (p2a, p2b)
        tpair = tetris_by_rule.get(rule)
        if tpair is not None:
            pairs["TETRIS"] = tpair
        r["residual"] = {k: rc(*v) for k, v in pairs.items()}

        # 4b. how large is the arm-specific part at all?  Reported next to the
        # residual correlation because a pair can have a very high residual r
        # and still be a perfectly usable N2N pair if its arms differ by a lot --
        # and conversely a low r means nothing if the arms are near-identical.
        v3cf = v3c.ravel()
        r["pair_reproducibility"] = {}
        for k, (a_map, b_map) in pairs.items():
            a3 = to_cropped(a_map, shape3d, cshape).ravel()
            b3 = to_cropped(b_map, shape3d, cshape).ravel()
            r["pair_reproducibility"][k] = {
                "all_valid": relative_independent_amplitude(a3, b3, v3cf),
                "mid": relative_independent_amplitude(
                    a3, b3, rois3dc["mid"].ravel() & v3cf),
            }

        # spatial-block units for the primary (detrended, all-valid) statistic
        units, unit_vals = {}, {}
        for k, (a_map, b_map) in pairs.items():
            vals, _ = block_residual_diffs(to_cropped(a_map, shape3d, cshape),
                                           to_cropped(b_map, shape3d, cshape),
                                           v3c)
            unit_vals[k] = vals
            units[k] = {"n_units": int(vals.size),
                        "mean_r": float(vals.mean()) if vals.size else float("nan"),
                        "median_r": float(np.median(vals)) if vals.size else float("nan")}
            if vals.size >= 8:
                lo, hi = block_bootstrap_ci(vals, n_boot=N_BOOT, seed=7)
                units[k]["ci95"] = [lo, hi]
        r["residual_units"] = units

        # the gate comparison: P1 against the Tetris view, paired over blocks
        if tpair is not None:
            va, vb = unit_vals["P1"], unit_vals["TETRIS"]
            n = min(va.size, vb.size)
            diff = va[:n] - vb[:n]
            mean, p = sign_flip_permutation_test(diff, n_perm=N_BOOT, seed=7)
            lo, hi = block_bootstrap_ci(diff, n_boot=N_BOOT, seed=7)
            rel = float(np.mean(va[:n]) - np.mean(vb[:n])) / max(abs(np.mean(vb[:n])), 1e-12)
            r["gate_noise_vs_tetris"] = {
                "unit": "non-overlapping 2x2x2 spatial blocks",
                "n_units": int(n),
                "p1_mean_r": float(np.mean(va[:n])),
                "tetris_mean_r": float(np.mean(vb[:n])),
                "paired_diff_mean": mean, "paired_diff_ci95": [lo, hi],
                "sign_flip_p": p,
                "relative_reduction": rel,
                "passes_10pct_reduction": bool(rel <= -0.10 and hi < 0),
            }
        out["rules"][rule] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--baseline", default="configs/phase0_baseline.yaml")
    ap.add_argument("--sequences", nargs="+", default=["SA64"])
    ap.add_argument("--rules", nargs="+",
                    default=["proportional", "spectral", "cum_energy", "clutter_knee"])
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--fold-limit", type=int, default=None)
    ap.add_argument("--tetris-band", choices=["official", "rule"], default="rule")
    ap.add_argument("--tetris-frames", choices=["pair", "all"], default="pair",
                    help="which frames feed the Tetris control.  'pair' uses "
                         "A u B, the literal reading of section 5.4 ('only the "
                         "sampling geometry differs'); 'all' uses the whole movie "
                         "and is kept as a sensitivity check.  Either way the "
                         "control keeps a frame-count advantage worth noting: its "
                         "arms see |A u B| = 2|A| frames each, so its arm-specific "
                         "fluctuation is if anything smaller than CrossFit's, which "
                         "flatters CrossFit, not the gate.")
    ap.add_argument("--out", default="data/processed/crossfit/pair_statistics.json")
    ap.add_argument("--figures", default="reports/figures")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = yaml.safe_load(open(args.baseline))
    pairs_root = Path(cfg["output"]["pairs_root"])
    splits_root = pairs_root / "splits"
    figdir = Path(args.figures)
    figdir.mkdir(parents=True, exist_ok=True)

    result = {"protocol": {
        "vessel_rule": "equal-area: top q%% of the volume's own valid voxels",
        "vessel_q": VESSEL_Q,
        "abs_db_legacy": ABS_DB_LEGACY,
        "calibration_note": (
            "the draft -20 dB re own p99.9 rule selected 100% of valid voxels on "
            "SA64 (the official PD map has ~8 dB of range) and was replaced under "
            "plan section 5.4's one permitted calibration, before any training; "
            "legacy_candidate_fraction per sequence is the evidence"),
        "roi_pcts": ROI_PCTS, "detrend_size": DETREND_SIZE,
        "centreline_tol": CENTRELINE_TOL, "n_boot": N_BOOT, "n_null": N_NULL,
        "block": BLOCK,
        "tetris_patterns": {"A": PATTERN_A, "B": PATTERN_B},
        "tetris_band": args.tetris_band,
        "tetris_frames": args.tetris_frames,
        "residual_grid": "z-cropped 0:2*(nz//2); the last z slice (1/37 = 2.7%) "
                         "is excluded because the 2x2x2 Tetris partition needs an "
                         "even axis.  Structure and intensity keep the full grid.",
        "gate": {"centreline_coverage_min": 0.85, "width_median_change_max": 0.15,
                 "noise_relative_reduction_max": -0.10, "ci_must_exclude_zero": True},
        "note": "P0 is a sanity check, NOT a control for the noise gate (plan 5.4)",
    }, "sequences": {}}

    for seq in args.sequences:
        print(f"=== {seq} ===")
        iq4d = load_sequence_iq(seq, base)
        shape3d = tuple(iq4d.shape[:3])
        x2d = iq4d.reshape(-1, iq4d.shape[3])
        valid = np.load(Path("data/processed/official") / seq / "valid_mask.npy").ravel()
        valid3d = valid.reshape(shape3d)
        ref_power = np.load(Path("data/processed/official") / seq / "pd_linear.npy").ravel()
        ref_db_map = to_db(ref_power, valid)

        # roi protocol, from the reference only
        pv = ref_power[valid]
        t_bg, t_ves = np.percentile(pv, ROI_PCTS)
        rois3d = {
            "background": ((ref_power < t_bg) & valid).reshape(shape3d),
            "mid": ((ref_power >= t_bg) & (ref_power < t_ves) & valid).reshape(shape3d),
            "vessel": ((ref_power >= t_ves) & valid).reshape(shape3d),
            "all_valid": valid3d.copy(),
        }
        for k, m in rois3d.items():
            print(f"  [roi] {k:10s} {int(m.sum()):7d} voxels")

        ref_mask3d = vessel_mask(ref_power, valid).reshape(shape3d) & valid3d
        ref_skel3d = skeleton(ref_mask3d)
        ref_width = width_stats(ref_mask3d, ref_skel3d)
        ref_comp = component_count(ref_mask3d)
        ref_null = centreline_coverage_null(valid, ref_skel3d, shape3d,
                                            int(ref_mask3d.sum()))
        ref_contrast = skeleton_profile(ref_power.reshape(shape3d), ref_skel3d,
                                        valid3d)
        legacy_frac = legacy_candidate_fraction(ref_power, valid)
        print(f"  [ref] q{VESSEL_Q:g} candidate {int(ref_mask3d.sum())} "
              f"({ref_mask3d.sum() / max(valid.sum(), 1):.3f}) "
              f"skel {int(ref_skel3d.sum())} width_med "
              f"{ref_width['median_vox']:.2f} comps {ref_comp[0]}")
        print(f"  [ref] coverage null over {ref_null['n_draws']} random "
              f"equal-area masks: mean {ref_null.get('mean', float('nan')):.3f} "
              f"p95 {ref_null.get('p95', float('nan')):.3f}")
        print(f"  [ref] legacy {ABS_DB_LEGACY:g} dB rule would select "
              f"{legacy_frac:.4f} of valid voxels")

        # the fold-independent band for the Tetris control
        gram_all = np.load(pairs_root / seq / "gram_all.npy")
        ev_full, vec_full = eigh_desc(gram_all)

        sequences_out = {"shape3d": list(shape3d),
                         "cropped_shape3d": list(cropped_shape(shape3d)), "ref": {
            "vessel_q": VESSEL_Q,
            "width": ref_width, "components": ref_comp[0],
            "candidate_fraction": float(ref_mask3d.sum() / max(valid.sum(), 1)),
            "centreline_coverage_null": ref_null,
            "skeleton_contrast": ref_contrast,
            "legacy_abs_db": ABS_DB_LEGACY,
            "legacy_candidate_fraction": legacy_frac,
            "skel_voxels": int(ref_skel3d.sum()),
            "roi_thresholds": {"background_lt": float(t_bg), "vessel_ge": float(t_ves)},
        }, "folds": []}

        split_names = args.splits or [s["name"] for s in cfg["sequences"][seq]["splits"]]
        folds = []
        for name in split_names:
            man = json.load(open(splits_root / f"{name}.json"))
            for fold in man["folds"]:
                fold = dict(fold)
                fold["split"] = name
                assert_fold(fold)
                folds.append(fold)
        if args.fold_limit:
            folds = folds[:args.fold_limit]

        for fi, fold in enumerate(folds):
            t0 = time.time()
            fold_dir = pairs_root / seq / fold["fold_id"]
            arms = load_arms(fold_dir, args.rules)

            # the Tetris control at matched filtering strength and matched frames
            tetris_by_rule = {}
            if args.tetris_frames == "pair":
                tframes = np.concatenate([np.asarray(fold["A"], dtype=int),
                                          np.asarray(fold["B"], dtype=int)])
            else:
                tframes = None
            c_idx = np.asarray(fold["calibration"], dtype=int)
            if args.tetris_band == "official":
                lo_o, hi_o = 60 - 1, 350                             # 1-based inclusive
                basis = (vec_full[:, lo_o:hi_o]
                         / np.sqrt(ev_full[lo_o:hi_o])[None, :]).astype(np.complex64)
                bands = {r: basis for r in args.rules}
                src = None
            else:
                ev_c, vec_c = eigh_desc(gram_all[np.ix_(c_idx, c_idx)])
                bands, src = {}, c_idx
                for rule in args.rules:
                    lo, hi = rule_edges(rule, cfg, ev_c)
                    hi = min(hi, len(c_idx))
                    if hi <= lo:
                        continue
                    bands[rule] = (vec_c[:, lo:hi]
                                   / np.sqrt(ev_c[lo:hi])[None, :]).astype(np.complex64)
            for rule, basis in bands.items():
                pd_band = projected_power_map(x2d, gram_all, src, basis, tgt=tframes)
                tetris_by_rule[rule] = tuple(
                    a.ravel() for a in tetris_pair(pd_band, shape3d, valid3d))

            out_fold = analyse_fold(seq, fold, args.rules, cfg, base, iq4d, valid,
                                    ref_power, ref_db_map, ref_mask3d, ref_skel3d,
                                    ref_width, ref_comp, rois3d, arms, tetris_by_rule)
            sequences_out["folds"].append(out_fold)
            print(f"  [{fi + 1}/{len(folds)}] {fold['fold_id']} ({time.time() - t0:.1f}s)")

        result["sequences"][seq] = sequences_out

        # representative structure figure: first fold, proportional
        first = sequences_out["folds"][0]
        fid = first["fold_id"]
        panels = {}
        fd = pairs_root / seq / fid
        for nm, lab in ((f"P1_proportional_A.npy", "A (P1)"),
                        (f"P1_proportional_B.npy", "B (P1)"),
                        ("", "full-seq pseudoref")):
            if nm:
                p = np.load(fd / nm).ravel()
            else:
                p = ref_power
            scale = float(np.percentile(p[valid], 99.9))
            panels[lab] = shared_db(p, scale).reshape(shape3d)
        diff = np.abs(panels["A (P1)"] - panels["B (P1)"])
        panels["|A - B| (dB)"] = -diff
        make_structure_figure(figdir / f"phase2_pair_structure_{fid}.png", fid, panels)

        # summary figure: residual correlation by pair kind and rule
        make_summary_figure(figdir / f"phase2_pair_residual_{seq}.png",
                            seq, sequences_out)

        # cross-fold aggregate with CIs over independent folds
        sequences_out["aggregate"] = aggregate_folds(sequences_out["folds"],
                                                     ref_null=ref_null)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


def load_arms(fold_dir, rules):
    """Load one fold's Phase 1 volumes, keyed by pair kind and arm.

    Filenames are P<kind>_<rule>_<arm>.npy, except P0 which has no rule (no SVD
    at all), so the rule field is absent there.  The key for a ruled pair is
    "P1_proportional" etc.; P0 is plain "P0".  A missing ruled pair is skipped
    rather than filled in, so a rule the manifest never built cannot silently
    appear as zeros.
    """
    arms = {"P0": {}}
    for fn in sorted(fold_dir.glob("P*_*.npy")):
        parts = fn.stem.split("_")
        kind, arm, rule = parts[0], parts[-1], "_".join(parts[1:-1])
        key = kind if kind == "P0" else f"{kind}_{rule}"
        if kind != "P0" and rule not in rules:
            continue
        arms.setdefault(key, {})[arm] = np.load(fn).ravel().astype(np.float64)
    if len(arms["P0"]) != 2:
        raise FileNotFoundError(f"{fold_dir}: P0_A/P0_B not both present")
    return arms


def aggregate_folds(folds, ref_null=None):
    """Per-rule means and 95% bootstrap CIs over INDEPENDENT folds."""
    agg = {}
    rules = set()
    for f in folds:
        rules |= set(f["rules"])
    for rule in sorted(rules):
        rows = [f["rules"][rule] for f in folds if rule in f["rules"]]
        if not rows:
            continue
        def col(fn):
            return np.asarray([fn(r) for r in rows], dtype=float)
        cov_a = col(lambda r: r["structure"]["A"]["centreline_coverage"])
        cov_b = col(lambda r: r["structure"]["B"]["centreline_coverage"])
        wch_a = col(lambda r: r["structure"]["A"]["width_median_change_vs_ref"])
        wch_b = col(lambda r: r["structure"]["B"]["width_median_change_vs_ref"])
        bias = col(lambda r: r["intensity"]["median_bias_db_A_minus_B"]["vessel"])
        entry = {
            "n_folds": len(rows),
            "centreline_coverage": {
                "A_mean": float(np.nanmean(cov_a)), "A_ci95": list(block_bootstrap_ci(cov_a, seed=3)),
                "B_mean": float(np.nanmean(cov_b)), "B_ci95": list(block_bootstrap_ci(cov_b, seed=3)),
                "min_over_folds": float(np.nanmin([cov_a.min(), cov_b.min()])),
                "null_p95": (float(ref_null["p95"]) if ref_null
                             and "p95" in ref_null else None),
            },
            "width_median_change_vs_ref": {
                "A_mean": float(np.nanmean(wch_a)), "B_mean": float(np.nanmean(wch_b)),
                "max_abs": float(np.nanmax(np.abs(np.concatenate([wch_a, wch_b])))),
            },
            "vessel_median_bias_db_A_minus_B": {
                "mean": float(np.nanmean(bias)), "ci95": list(block_bootstrap_ci(bias, seed=3)),
            },
            "skeleton_contrast_db": {
                "A_mean": float(np.nanmean(col(
                    lambda r: r["structure"]["A"]["skeleton_contrast"]["contrast_db"]))),
                "B_mean": float(np.nanmean(col(
                    lambda r: r["structure"]["B"]["skeleton_contrast"]["contrast_db"]))),
            },
            "pair_reproducibility_rms_rel_diff": {
                k: float(np.nanmean(col(
                    lambda r, k=k: r["pair_reproducibility"][k]["mid"]["rms_rel_diff"])))
                for k in rows[0]["pair_reproducibility"]
            },
        }
        if "gate_noise_vs_tetris" in rows[0]:
            rel = col(lambda r: r["gate_noise_vs_tetris"]["relative_reduction"])
            hi = np.asarray([r["gate_noise_vs_tetris"]["paired_diff_ci95"][1] for r in rows])
            entry["noise_vs_tetris"] = {
                "relative_reduction_mean": float(np.nanmean(rel)),
                "relative_reduction_ci95": list(block_bootstrap_ci(rel, seed=3)),
                "n_folds_with_ci_below_0": int((hi < 0).sum()),
                "n_folds": len(rows),
            }
        # explicit threshold verdicts
        np95 = entry["centreline_coverage"]["null_p95"]
        entry["verdict"] = {
            "structure_centreline_ge_0.85": bool(
                entry["centreline_coverage"]["min_over_folds"] >= 0.85),
            "structure_centreline_beats_random_equal_area_null": (
                bool(entry["centreline_coverage"]["min_over_folds"] > np95)
                if np95 is not None else None),
            "structure_width_le_15pct": bool(
                entry["width_median_change_vs_ref"]["max_abs"] <= 0.15),
            "noise_10pct_reduction_in_ge_half_folds": bool(
                entry.get("noise_vs_tetris", {}).get("n_folds_with_ci_below_0", 0)
                >= max(1, len(rows) // 2)),
        }
        agg[rule] = entry
    return agg


def make_summary_figure(path, seq, seq_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = seq_out["folds"]
    kinds = [k for k in ("P1", "P2", "TETRIS", "P0")
             if any(k in f["rules"][r]["residual"] for f in folds for r in f["rules"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, roi, title in ((axes[0], "all_valid", "residual corr, all valid voxels"),
                           (axes[1], "background", "residual corr, background ROI")):
        labels, vals = [], []
        for rule in sorted({r for f in folds for r in f["rules"]}):
            for k in kinds:
                v = [f["rules"][rule]["residual"].get(k, {}).get("detrended", {}).get(roi, {}).get("r")
                     for f in folds if rule in f["rules"]]
                v = [x for x in v if x is not None and np.isfinite(x)]
                if not v:
                    continue
                labels.append(f"{rule[:9]}\n{k}" if k != "TETRIS" else f"{rule[:9]}\nTET")
                vals.append((float(np.mean(v)), float(np.std(v) / max(np.sqrt(len(v)), 1))))
        if not vals:
            continue
        x = np.arange(len(vals))
        ax.bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=3,
               color=["C0" if "P1" in l else "C2" if "TET" in l else "C3" for l in labels])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{seq}: {title}", fontsize=9)
        ax.set_ylabel("detrended residual correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 statistics tests: the residual-correlation gate must fail loudly.

Run with:  python -m pytest tests/test_phase2_statistics.py -v

The gate in `scripts/evaluate_pair_statistics.py` decides whether CrossFit's
observation pair is usable for N2N training.  These tests pin the properties
that decision depends on, so that a future edit cannot silently change the
answer:

  1. the estimator is CORRECT          -- identical arms give r ~ 1, independent
                                          noise gives r ~ 0, and it is monotone
                                          in the true noise correlation;
  2. the estimator is SENSITIVE        -- it must respond to noise correlation
                                          over the range real data occupies, not
                                          only in the white-noise limit;
  3. the operator is REPRODUCIBLE      -- the audit's `tetris_operator` with
                                          PATTERN_A/PATTERN_B is bit-identical
                                          to `tetris_pair`;
  4. the operator is NOT FREE          -- applying it changes the statistic, so
                                          P1 and Tetris cannot be compared
                                          without matching it;
  5. degenerate inputs FAIL FAST       -- empty masks, constant volumes and
                                          zero-variance regions raise or return
                                          NaN deliberately rather than silently
                                          producing a plausible-looking number;
  6. block bootstrap pairs LOCATIONS   -- the spatial units line up index for
                                          index, which is what makes the paired
                                          difference meaningful.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_pair_statistics import (                                    # noqa: E402
    BLOCK, PATTERN_A, PATTERN_B, block_residual_diffs, cropped_shape,
    pearson, relative_independent_amplitude, residual_correlation,
    tetris_pair,
)
from audit_pair_statistics import tetris_operator                        # noqa: E402

SHAPE = (24, 48, 48)          # even on every axis; big enough that the
#                               9^3 detrend kernel does not eat the whole field
CROP = cropped_shape(SHAPE)


# ------------------------------------------------------------------ helpers ---
def _all_valid():
    return np.ones(SHAPE, dtype=bool)


def _smooth(rng, sigma=1.0, shape=SHAPE):
    """Gaussian-correlated field with unit variance, via FFT (no scipy needed)."""
    x = rng.normal(size=shape)
    f = [np.fft.fftfreq(n) for n in shape]
    k2 = (f[0][:, None, None] ** 2 + f[1][None, :, None] ** 2
          + f[2][None, None, :] ** 2)
    g = np.exp(-2 * np.pi ** 2 * sigma ** 2 * k2)
    y = np.fft.ifftn(np.fft.fftn(x) * g).real
    return y / y.std()


def _speckle_pair(rng, rho, sigma=1.0, noise_sd=0.3):
    """Two arms sharing a smooth base, with noise correlated at exactly `rho`.

    This is the generative model the gate implicitly assumes: a = base*(1+s*n_a),
    b = base*(1+s*n_b), and the question is whether the statistic recovers rho.
    """
    # base_sig=5 at this shape keeps the smooth structure from leaking into
    # the 9^3-detrended residual, which is what made the small-shape version
    # of this generator produce a flat statistic.
    base = np.abs(_smooth(rng, 5.0)) + 2.0
    n1 = _smooth(rng, sigma)
    n2 = _smooth(rng, sigma)
    n2 = n2 - (n2 * n1).mean() * n1
    n2 /= n2.std()
    nb = rho * n1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * n2
    return base * (1 + noise_sd * n1), base * (1 + noise_sd * nb)


def _r(a, b, valid=None, roi="all_valid", detrend=True):
    v = _all_valid() if valid is None else valid
    a3 = np.asarray(a).reshape(SHAPE)
    b3 = np.asarray(b).reshape(SHAPE)
    return residual_correlation(a3, b3, v, {"all_valid": v}, detrend)[roi]["r"]


# ------------------------------------------------------- 1. estimator is correct ---
def test_identical_input_gives_r_one():
    """Non-degenerate identical arms must give r == 1, not merely 'high'."""
    rng = np.random.default_rng(0)
    a, _ = _speckle_pair(rng, 0.0)
    assert _r(a, a) == pytest.approx(1.0, abs=1e-12)


def test_independent_noise_gives_near_zero():
    rng = np.random.default_rng(1)
    a, b = _speckle_pair(rng, 0.0)
    assert abs(_r(a, b)) < 0.1


def test_statistic_is_monotone_in_true_noise_correlation():
    """The whole gate rests on this: higher true rho => higher measured r.

    Checked over the full range including negative correlation, because the
    gate's claim is about *ordering* two pair kinds, not about an absolute
    threshold.
    """
    rng = np.random.default_rng(2)
    rhos = [-1.0, -0.5, 0.0, 0.5, 1.0]
    measured = []
    for rho in rhos:
        a, b = _speckle_pair(rng, rho)
        measured.append(_r(a, b))
    assert np.all(np.diff(measured) > 0), (
        f"statistic must increase with true noise correlation; got {measured}")
    assert measured[0] < -0.5 and measured[-1] > 0.95


def test_noise_correlation_recovered_within_tolerance():
    """Not just ordered: roughly the right value on the model it assumes."""
    rng = np.random.default_rng(3)
    for rho in (-0.5, 0.0, 0.5, 0.9):
        a, b = _speckle_pair(rng, rho, noise_sd=0.3)
        # r = (s2*rho) / (s2 + s2) with equal noise power in both arms -> rho/2
        # in the low-contrast limit; at s=0.3 the measured value is close to rho.
        assert abs(_r(a, b) - rho) < 0.25, f"rho={rho}: measured {_r(a, b):.3f}"


# ------------------------------------------------ 2. estimator is sensitive ---
def test_statistic_retains_dynamic_range_at_realistic_correlation_length():
    """Regression guard for the audit's central question.

    A white-noise-only check (sigma -> 0) is NOT representative: real residual
    fields have lag-1 autocorrelation ~0.4-0.8 (measured on P1 arms).  At that
    correlation length the statistic must still separate rho = -1 / 0 / +1 by a
    wide margin.  If a future change makes the estimator flat, this fails.
    """
    rng = np.random.default_rng(4)
    vals = [_r(*_speckle_pair(rng, rho, sigma=0.8)) for rho in (-1.0, 0.0, 1.0)]
    spread = vals[2] - vals[0]
    assert spread > 1.2, (
        f"dynamic range collapsed at realistic correlation length: {vals}")
    assert vals[0] < 0.0 < vals[2]


# --------------------------------------------------- 3. operator is reproducible ---
def test_complementary_operator_matches_tetris_pair_bitwise():
    """`tetris_operator` must be the same function as `tetris_pair`.

    The audit's conclusions depend on applying the *identical* operator with
    different phase sets.  If these ever diverge, every audit number is void --
    so this is exact equality, not approx.
    """
    rng = np.random.default_rng(5)
    power = (np.abs(rng.normal(size=SHAPE)) + 0.1).ravel()
    valid = np.ones(SHAPE, dtype=bool)
    ta, tb = tetris_pair(power, SHAPE, valid)
    oa = tetris_operator(power, PATTERN_A, SHAPE, valid)
    ob = tetris_operator(power, PATTERN_B, SHAPE, valid)
    assert np.array_equal(ta, oa), "PATTERN_A must reproduce tetris_pair arm A"
    assert np.array_equal(tb, ob), "PATTERN_B must reproduce tetris_pair arm B"
    assert not np.array_equal(ta, tb), "the two arms must not be the same image"


def test_operator_is_blockwise_constant():
    """Block-average + backfill means every 2x2x2 cell is flat afterwards."""
    rng = np.random.default_rng(6)
    power = (np.abs(rng.normal(size=SHAPE)) + 0.1).ravel()
    out = tetris_operator(power, PATTERN_A, SHAPE, np.ones(SHAPE, bool)).reshape(SHAPE)
    blocks = out.reshape(SHAPE[0] // BLOCK, BLOCK,
                         SHAPE[1] // BLOCK, BLOCK,
                         SHAPE[2] // BLOCK, BLOCK)
    # each 2x2x2 cell holds a single value repeated
    assert np.allclose(blocks.std(axis=(1, 3, 5)), 0.0)


def test_operator_reduces_high_frequency_energy():
    """The operator smooths -- which is exactly why it cannot be left unmatched."""
    rng = np.random.default_rng(7)
    power = (np.abs(rng.normal(size=SHAPE)) + 0.1).ravel()
    out = tetris_operator(power, PATTERN_A, SHAPE, np.ones(SHAPE, bool))
    hf = lambda x: np.abs(np.diff(x.reshape(SHAPE), axis=2)).mean()
    assert hf(out) < hf(power)


# --------------------------------------------------- 4. operator is not free ---
def test_matching_the_operator_changes_the_statistic():
    """P1-as-implemented and P1-through-the-operator must differ materially.

    This is the defect the audit found.  If this test ever passes trivially
    (i.e. the two agree), the confound has been removed by construction and the
    audit's premise no longer holds -- which would itself be worth knowing.

    Checked at rho = 1 (a perfect pair), not rho = 0.  At rho ~ 0 the statistic
    is ~0 either way and the operator's effect is small in absolute terms; the
    audit's finding is about the *level* a good pair is reported at, which only
    exists where the pair is actually good.  The direction is asserted, not just
    the magnitude: the operator can only make a perfect pair look worse, because
    it adds arm-specific (phase-dependent) structure on top of the shared signal.
    """
    rng = np.random.default_rng(8)
    a, b = _speckle_pair(rng, 1.0, sigma=0.8)
    native = _r(a, b)
    matched = _r(tetris_operator(a.ravel(), PATTERN_A, SHAPE, _all_valid()),
                 tetris_operator(b.ravel(), PATTERN_B, SHAPE, _all_valid()))
    assert native > 0.95, f"generator must produce a near-perfect pair, got {native}"
    assert matched < native - 0.1, (
        f"operator had no effect ({native:.3f} vs {matched:.3f}); "
        "the audit's confound would not exist")


def test_matched_operator_preserves_ordering_but_compresses_range():
    """What the matched comparison can and cannot be used for.

    Matching keeps the ORDER of pair quality (so 'is P1 better than Tetris' is
    answerable) but compresses the effect size (so the *magnitude* of the
    reported percentage is not comparable across operators).  Both halves are
    asserted, because an audit that fixes the operator and then quotes the old
    percentage would be overclaiming.
    """
    rng = np.random.default_rng(9)
    rows = []
    for rho in (0.0, 0.5, 1.0):
        a, b = _speckle_pair(rng, rho, sigma=0.8)
        native = _r(a, b)
        matched = _r(tetris_operator(a.ravel(), PATTERN_A, SHAPE, _all_valid()),
                     tetris_operator(b.ravel(), PATTERN_B, SHAPE, _all_valid()))
        rows.append((native, matched))
    natives = [r[0] for r in rows]
    matcheds = [r[1] for r in rows]
    assert np.all(np.diff(natives) > 0)
    assert np.all(np.diff(matcheds) > 0), "matched operator must keep the ordering"
    assert (max(matcheds) - min(matcheds)) < (max(natives) - min(natives)), (
        "matched operator is expected to compress dynamic range")


# ------------------------------------------------------ 5. degenerate inputs ---
def test_empty_mask_returns_nan_not_a_number():
    rng = np.random.default_rng(10)
    a, b = _speckle_pair(rng, 0.0)
    valid = np.zeros(SHAPE, dtype=bool)
    out = residual_correlation(a, b, valid, {"all_valid": valid}, True)
    assert np.isnan(out["all_valid"]["r"]), "empty ROI must be NaN, not 0"


def test_constant_volume_returns_nan_not_zero():
    """A constant arm has zero variance; Pearson is undefined, not 0.

    Returning 0 here would look like 'perfectly independent noise' and would
    silently pass the gate -- the exact failure mode the audit warns about.
    """
    rng = np.random.default_rng(11)
    _, b = _speckle_pair(rng, 0.0)
    const = np.ones(SHAPE)
    out = _r(const, b)
    assert np.isnan(out), f"constant input must be NaN, got {out}"


def test_pearson_of_zero_variance_is_nan():
    assert np.isnan(pearson(np.ones(100), np.arange(100.0)))


def test_block_diffs_skip_blocks_below_min_voxels():
    """Blocks with too few valid voxels must be dropped, not counted as r=0."""
    rng = np.random.default_rng(12)
    a, b = _speckle_pair(rng, 0.0)
    valid = np.zeros(SHAPE, dtype=bool)
    valid[:4] = True                       # only the first z-band is valid
    vals, _ = block_residual_diffs(a, b, valid, size=4)
    n_blocks_with_data = int(valid.sum() // 64)
    assert vals.size <= max(n_blocks_with_data, 1)
    assert np.all(np.isfinite(vals))


def test_dead_voxels_do_not_enter_the_statistic():
    """Changing values in dead voxels must not move the result."""
    rng = np.random.default_rng(13)
    a, b = _speckle_pair(rng, 0.0)
    valid = np.ones(SHAPE, dtype=bool)
    valid[:4, :4, :4] = False
    a2 = a.copy(); b2 = b.copy()
    a2[~valid] = 1e6; b2[~valid] = -1e6
    assert _r(a, b, valid) == pytest.approx(_r(a2, b2, valid), rel=1e-12)


# --------------------------------------------------- 6. bootstrap pairs locations ---
def test_block_units_are_paired_by_location():
    """The paired difference is only meaningful if unit i is the same place.

    Two runs on the same inputs must produce the same unit count and the same
    ordering, and the paired difference must be exactly the elementwise one.
    """
    rng = np.random.default_rng(14)
    a, b = _speckle_pair(rng, 0.0)
    c, d = _speckle_pair(rng, 0.5)
    v1, m1 = block_residual_diffs(a, b, _all_valid(), size=4)
    v2, m2 = block_residual_diffs(c, d, _all_valid(), size=4)
    assert v1.size == v2.size
    assert np.array_equal(m1, m2), "unit masks must be identical across pair kinds"
    v1b, m1b = block_residual_diffs(a, b, _all_valid(), size=4)
    assert np.array_equal(v1, v1b), "block statistic must be deterministic"
    assert np.array_equal(m1, m1b)


def test_block_size_changes_unit_count_as_expected():
    rng = np.random.default_rng(15)
    a, b = _speckle_pair(rng, 0.0)
    n4 = block_residual_diffs(a, b, _all_valid(), size=4)[0].size
    n8 = block_residual_diffs(a, b, _all_valid(), size=8)[0].size
    assert n4 > n8 > 0, "larger blocks must yield fewer units"


# --------------------------------------------- relative_independent_amplitude ---
def test_rms_rel_diff_scales_as_sqrt_2_over_n():
    """The audit's correction to the Phase 2 report, pinned as a test.

    For a difference of two N-frame means the relative-difference RMS scales as
    sigma*sqrt(2/N), NOT sigma/sqrt(N).  The Phase 2 report compared its measured
    1.7% against 1/sqrt(133) = 8.7%; the correct two-arm constant is 12.3%.
    """
    rng = np.random.default_rng(16)
    N, sigma, n_vox = 133, 0.3, 40000
    a = 1 + rng.normal(0, sigma, (N, n_vox)).mean(0)
    b = 1 + rng.normal(0, sigma, (N, n_vox)).mean(0)
    mask = np.ones(n_vox, dtype=bool)
    got = relative_independent_amplitude(a, b, mask)["rms_rel_diff"]
    two_arm = sigma * np.sqrt(2.0 / N)
    one_arm = sigma / np.sqrt(N)
    assert abs(got - two_arm) < 0.15 * two_arm, (
        f"measured {got:.4f} should track sqrt(2/N) = {two_arm:.4f}")
    assert abs(got - one_arm) > 0.25 * one_arm, (
        f"measured {got:.4f} must NOT be explained by 1/sqrt(N) = {one_arm:.4f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

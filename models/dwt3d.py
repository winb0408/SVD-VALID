# Adapted from FDU-donglab/VALID v1.0 (datasets/sampling.py, DWT_3D).
# Upstream: https://github.com/FDU-donglab/VALID/tree/v1.0
# See THIRD_PARTY_LICENSES.md.
#
# This module implements the 3-D Haar discrete wavelet transform used by the
# original VALID Hessian regularizer. The separable slicing formulation below
# is numerically identical to the upstream matrix-multiplication DWT_3D for
# the 'haar' wavelet (rec_lo = [1/sqrt(2), 1/sqrt(2)], rec_hi = [-1/sqrt(2),
# 1/sqrt(2)]), but is autograd-friendly and does not rebuild filter matrices
# on every forward pass.

import torch


def _low(x):
    """Haar low-pass along the last dimension: length halved."""
    return (x[..., ::2] + x[..., 1::2]) / torch.sqrt(torch.tensor(2.0))


def _high(x):
    """Haar high-pass along the last dimension: length halved."""
    return (x[..., 1::2] - x[..., ::2]) / torch.sqrt(torch.tensor(2.0))


def _apply_axis(x, fn, axis):
    """Apply fn along a spatial axis (2, 3 or 4) of a (B,C,D,H,W) tensor."""
    x = x.transpose(axis, -1)
    x = fn(x)
    return x.transpose(axis, -1)


def dwt3d_haar(x):
    """One-level separable 3-D Haar DWT.

    Input:  (B, C, D, H, W) with even spatial dims.
    Output: dict with the 8 subbands; the key names follow the VALID
    convention (first letter: axis D, second: axis H, third: axis W),
    e.g. 'LLH' is low-pass along D and H, high-pass along W.
    """
    if x.ndim != 5 or x.shape[2] % 2 or x.shape[3] % 2 or x.shape[4] % 2:
        raise ValueError("Expected (B,C,D,H,W) with even spatial dims.")
    bands = {}
    for name in ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"):
        y = x
        for axis, letter in zip((2, 3, 4), name):
            y = _apply_axis(y, _low if letter == "L" else _high, axis)
        bands[name] = y
    return bands


VALID_REG_BANDS = ("LLL", "LLH", "LHL", "HLL")

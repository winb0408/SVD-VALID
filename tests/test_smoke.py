#!/usr/bin/env python3
"""Smoke tests for SVD-VALID core components.

Run with:  python tests/test_smoke.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.sampling import (  # noqa: E402
    TETRIS_PATTERNS,
    all_phases,
    valid_random_tetris,
    guided_tetris_subvolumes,
)
from models.dwt3d import dwt3d_haar  # noqa: E402
from models.network import NetworkCNR, HessianConstraintLoss3D  # noqa: E402


def test_phases_partition_volume():
    v = torch.arange(2 * 6 * 6).float().reshape(2, 6, 6)
    ph = all_phases(v)
    stack = torch.stack(ph)
    # Voxels are unique here (arange), so sorted phases must equal the volume.
    assert torch.equal(stack.flatten().sort().values, v.flatten().sort().values), (
        "phases must cover every voxel exactly once"
    )
    print("ok: 8 phases partition the volume")


def test_valid_random_tetris_covers_and_disjoint():
    torch.manual_seed(0)
    v = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    subs = valid_random_tetris(v)
    assert [tuple(s.shape) for s in subs] == [(2, 4, 4)] * 4
    joint = torch.stack(subs).flatten()
    # Tetris keeps 4 of the 8 positions per block: exactly half the voxels,
    # each sampled at most once (values here are unique).
    assert joint.numel() == (4 * 8 * 8) // 2
    assert joint.unique().numel() == joint.numel(), "subvolumes must be disjoint"
    assert torch.isin(joint, torch.as_tensor(v).flatten()).all()
    print("ok: faithful random Tetris yields 4 disjoint half-coverage subvolumes")


def test_pattern_semantics_match_upstream():
    # Upstream VALID idx_pair, verified against FDU-donglab/VALID v1.0.
    upstream = torch.tensor(
        [[0, 1, 2, 4], [0, 1, 3, 5], [1, 2, 3, 7], [0, 2, 3, 6],
         [0, 4, 5, 6], [1, 4, 5, 7], [2, 4, 6, 7], [3, 5, 6, 7]],
        dtype=torch.long,
    )
    assert torch.equal(TETRIS_PATTERNS, upstream)
    print("ok: Tetris patterns identical to upstream VALID")


def test_dwt3d_haar_energy():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 8, 16, 16)
    bands = dwt3d_haar(x)
    energy = sum(b.square().sum() for b in bands.values())
    assert torch.allclose(energy, x.square().sum(), rtol=1e-4), "Haar DWT must be energy-preserving"
    for name, b in bands.items():
        assert tuple(b.shape) == (1, 1, 4, 8, 8), f"band {name} shape {b.shape}"
    print("ok: Haar DWT is energy-preserving with correct band shapes")


def test_crn_forward_backward():
    torch.manual_seed(0)
    model = NetworkCNR(features=8, groups=4)
    x = torch.randn(1, 1, 8, 16, 16)
    y = model(x)
    assert tuple(y.shape) == tuple(x.shape)
    y.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f"ok: CRN forward/backward ({sum(p.numel() for p in model.parameters())/1e6:.3f}M params)")


def test_hessian_loss():
    h = HessianConstraintLoss3D()
    smooth = torch.zeros(1, 1, 8, 8, 8)
    rough = torch.randn(1, 1, 8, 8, 8)
    assert float(h(smooth)) < 1e-10 < float(h(rough))
    print("ok: Hessian loss penalises roughness only")


def test_guided_selection_runs():
    torch.manual_seed(0)
    vol = np.random.rand(8, 16, 16).astype(np.float32)
    sig = np.random.rand(8, 16, 16).astype(np.float32)
    noi = np.random.rand(8, 16, 16).astype(np.float32)
    pid, views, q = guided_tetris_subvolumes(vol, sig, noi)
    assert 0 <= pid < 8 and len(views) == 4 and len(q) == 8
    print(f"ok: guided selection runs (picked pattern {pid})")


def test_train_loss_computation():
    # One full optimisation step end-to-end on CPU with tiny dims.
    # CRN needs every spatial dim >= 8 (AvgPool3d(8) in the first branch),
    # and the Tetris subvolumes halve each dim, so use a 16x32x32 volume.
    from train import tensor, wavelet_hessian_loss

    vol = np.random.rand(16, 32, 32).astype(np.float32)
    model = NetworkCNR(features=4, groups=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    subs = valid_random_tetris(vol)
    xa, xb, ya, yb = [tensor(v, torch.device("cpu")) for v in subs]
    ha, hb = model(xa), model(xb)
    mse = torch.nn.MSELoss()
    loss = (0.5 * mse(ha, ya) + 0.5 * mse(hb, yb) + mse(ha, hb)
            + 1e-4 * wavelet_hessian_loss(ha, hb, HessianConstraintLoss3D()))
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    print(f"ok: full training step (loss={float(loss):.4f})")


if __name__ == "__main__":
    test_pattern_semantics_match_upstream()
    test_phases_partition_volume()
    test_valid_random_tetris_covers_and_disjoint()
    test_dwt3d_haar_energy()
    test_crn_forward_backward()
    test_hessian_loss()
    test_guided_selection_runs()
    test_train_loss_computation()
    print("\nAll smoke tests passed.")

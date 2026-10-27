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
    check_cross_slots,
    guided_tetris_subvolumes,
    one_phase,
    strided_phase,
    strided_views,
    valid_random_tetris,
    valid_random_tetris_cross,
    valid_random_tetris_strided,
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


def test_strided_phase_regresses_to_one_phase():
    """stride=1 时新采样器必须与老的 one_phase 逐体素相同。"""
    v = torch.arange(4 * 8 * 8).float().reshape(4, 8, 8)
    for i in range(8):
        assert torch.equal(strided_phase(v, i, 1), one_phase(v, i)), f"phase {i}"
    # stride>1：形状每轴 n/(2m)；每轴必须能被 2m 整除，否则拒绝（8 个相位不等长）
    for m, shape in ((2, (4, 8, 8)), (3, (6, 12, 24))):
        vol = torch.arange(int(np.prod(shape))).float().reshape(shape)
        for i in range(8):
            s = strided_phase(vol, i, m)
            assert tuple(s.shape) == tuple(n // (2 * m) for n in shape)
        assert strided_phase(vol, 0, m).numel() == vol.numel() // (2 * m) ** 3
    try:
        strided_phase(torch.zeros(4, 8, 8), 0, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("不能被 2m 整除的形状必须被拒绝")
    print("ok: strided_phase regresses to one_phase at m=1")


def test_valid_random_tetris_strided_regresses_to_original():
    """同样的 RNG 消耗顺序下，stride=1 的随机 Tetris 必须与原实现逐体素相同。"""
    v = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    torch.manual_seed(7)
    ref = valid_random_tetris(v)
    torch.manual_seed(7)
    new = valid_random_tetris_strided(v, 1)
    for a, b in zip(ref, new):
        assert torch.equal(a, b)
    print("ok: strided random Tetris regresses to the original at m=1")


def test_cross_pairing_is_disjoint_with_targets_in_the_other_tet():
    """cross 配对：四个视图互不重叠、总数与 within 相同；但目标的 2 个角点在输入 tet 之外。"""
    v = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    torch.manual_seed(3)
    subs = valid_random_tetris_cross(v, 1)
    joint = torch.stack(subs).flatten()
    # 两种配对都是每块 4/8：覆盖一半，且不重复
    assert joint.numel() == v.size // 2
    assert joint.unique().numel() == joint.numel(), "views must be disjoint"
    # 逐块检查：输入的 2 个角必须同属一个 tet，目标必须在其互补 tet 内
    blk = valid_random_tetris_cross(torch.arange(8).float().reshape(2, 2, 2), 1)
    corners = [int(s.reshape(-1)[0]) for s in blk]
    # 注意：只给 2 个输入角点时 tet 并不唯一（如 {0,1} 同时属于 p0 和 p1），
    # 所以判据是"存在某个 tet 使目标恰好是它的互补 tet"。
    # 目标只抽了互补 tet 的 2 个角，所以是"包含于"，不是"等于"
    ok = any(set(corners[2:]) <= set(range(8)) - set(pat)
             for pat in TETRIS_PATTERNS.tolist() if set(corners[:2]) <= set(pat))
    assert ok, f"目标 {corners[2:]} 不在输入 {corners[:2]} 的互补 tet 内"
    print("ok: cross pairing is disjoint and its targets live in the complement tet")


def test_body_diagonal_pair_needs_the_complement_tet():
    """自检：任何 tet 内部都没有体对角对，所以预注册的首选只能在 cross 配对下实现。"""
    def off(i):
        return (i // 4, (i // 2) % 2, i % 2)

    for pat in TETRIS_PATTERNS.tolist():
        for a in pat:
            for b in pat:
                if a >= b:
                    continue
                xor = tuple(u ^ v for u, v in zip(off(a), off(b)))
                assert xor != (1, 1, 1), f"pattern {pat} 竟含体对角对 {(a, b)}"
    # 体对角：1(001)->6(110)、4(100)->3(011)，XOR 全为 111
    assert check_cross_slots([1, 4, 6, 3]) == 0
    # y 轴向：1->3、4->6，只差中间那一位
    assert check_cross_slots([1, 4, 3, 6]) == 0
    for bad in ([0, 1, 2, 4], [0, 1, 2, 7], [0, 1, 7, 7]):
        try:
            check_cross_slots(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} 应被判为非法")
    # 关键隔离：两条臂用**同一批 4 个相位**，输入也相同，只把两个目标对调
    assert set([1, 4, 6, 3]) == set([1, 4, 3, 6]), "两臂必须采到同一批体素"
    assert (7 ^ 1, 7 ^ 4) == (6, 3), "体对角目标的取值"
    assert ((1 ^ 2, 4 ^ 2)) == (3, 6), "y 轴向目标的取值" 
    print("ok: body diagonal requires the complement tet; both arms share the same voxels")


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
    test_strided_phase_regresses_to_one_phase()
    test_valid_random_tetris_strided_regresses_to_original()
    test_cross_pairing_is_disjoint_with_targets_in_the_other_tet()
    test_body_diagonal_pair_needs_the_complement_tet()
    test_phases_partition_volume()
    test_valid_random_tetris_covers_and_disjoint()
    test_dwt3d_haar_energy()
    test_crn_forward_backward()
    test_hessian_loss()
    test_guided_selection_runs()
    test_train_loss_computation()
    print("\nAll smoke tests passed.")


def test_train_accepts_multi_volume_without_train_volume():
    """回归：`--train-volume` 曾是 required=True，导致所有多卷 launcher（A 阶段 2 / B / C）
    在 argparse 阶段就退出、一格训练都跑不出来（2026-10-27 发现并修）。

    这里只验"不再在 argparse 阶段被拒"：给一个不存在的卷，错误必须发生在**读文件**，
    而不是 "the following arguments are required: --train-volume"。
    """
    import subprocess
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "train.py", "--train-volumes", "/nonexistent/nope.npy",
         "--out", "/tmp/_never_written", "--steps-per-epoch", "1", "--epochs", "1"],
        cwd=root, capture_output=True, text=True)
    assert r.returncode != 0, "输入不存在时必须失败（否则说明它悄悄跑了别的卷）"
    err = (r.stderr or "") + (r.stdout or "")
    assert "required: --train-volume" not in err, err[-800:]
    assert "nope.npy" in err, f"应报读文件失败，实际：{err[-800:]}"
    print("ok: train.py 允许只给 --train-volumes")

#!/usr/bin/env python3
"""R3-P06 反例：`--mode dense` 下 identity 项与它自己的梯度**恒为 0**，是空操作。

外部审查（`Chatgpt/V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P06）指出的问题：
`dense_views(arr)` 返回四个**同一个**数组的引用，于是 `xa is xb`，
同一个确定性 CRN 对同一输入给出逐位相同的 `ha = hb`，
`mse(ha, hb)` 恒等于 0、对参数的梯度也恒等于 0。
后果：拿"dense + `--weight-identity 0`"当"关掉 identity 项"的消融，是**假消融**
（两个臂的训练过程逐位相同），任何基于它的归因都无效。

本文件钉住四条：

1. `dense_views` 四个视图是同一个对象（所以"相同"不是数值巧合）；
2. 因此 identity loss == 0 且其梯度范数 == 0（而 neighbor 项梯度非 0 —— 排除"计算图断了"）；
3. 换成两条**直和**视图（同一 patch 加两次独立噪声）后 identity 项立刻非 0 且梯度非 0；
4. `weight_identity` 在 dense 几何下对总损失梯度**没有任何影响**（这就是"空操作"的定义）。

Run with:  python -m pytest tests/test_v3_round3_dense_identity.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network import NetworkCNR                    # noqa: E402
from train import dense_views, grad_norm, loss_terms, tensor  # noqa: E402

SHAPE = (8, 8, 8)


def _model(seed=0):
    torch.manual_seed(seed)
    m = NetworkCNR(in_channels=1, out_channels=1, features=4, groups=2)
    m.eval()
    return m


def _patch(seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal(SHAPE).astype(np.float32)


def test_dense_views_are_the_same_object():
    """dense 的"四条视图"是同一个数组的四个引用 —— 不是四次相同抽样。"""
    arr = _patch()
    views = dense_views(arr)
    assert len(views) == 4
    assert all(v is arr for v in views)


def test_dense_identity_loss_and_grad_are_exactly_zero():
    """同一输入 ⇒ ha == hb 逐位相同 ⇒ identity 项与其梯度都恒为 0。"""
    from models.network import HessianConstraintLoss3D

    model = _model()
    arr = _patch()
    xa, xb, ya, yb = [tensor(v, "cpu") for v in dense_views(arr)]

    ha, hb = model(xa), model(xb)
    assert torch.equal(ha, hb), "确定性 CRN 对同一输入必须逐位相同"

    terms = loss_terms(ha, hb, ya, yb, HessianConstraintLoss3D(), 0.02, 0.05)
    assert float(terms["identity"]) == 0.0
    assert grad_norm(terms["identity"], model) == 0.0
    # 控制：neighbor 项在同一张图上梯度非 0，说明上面的 0 不是因为断了计算图
    assert grad_norm(terms["neighbor"], model) > 0.0


def test_dense_with_two_direct_views_has_nonzero_identity():
    """把 xb 换成"同一 patch 的第二次独立加噪"，identity 项与梯度立刻非 0。

    这是修复方案（直和学习目标）能被**测得**的条件：两路输入必须不同。
    """
    from models.network import HessianConstraintLoss3D

    model = _model()
    rng = np.random.default_rng(7)
    clean = _patch()
    obs1 = clean + 0.3 * rng.standard_normal(SHAPE).astype(np.float32)
    obs2 = clean + 0.3 * rng.standard_normal(SHAPE).astype(np.float32)
    assert not np.array_equal(obs1, obs2)

    xa, xb, ya, yb = [tensor(v, "cpu") for v in (obs1, obs2, clean, clean)]
    ha, hb = model(xa), model(xb)
    terms = loss_terms(ha, hb, ya, yb, HessianConstraintLoss3D(), 0.02, 0.05)
    assert float(terms["identity"]) > 0.0
    assert grad_norm(terms["identity"], model) > 0.0


@pytest.mark.parametrize("w", [0.0, 0.05, 0.5])
def test_weight_identity_is_a_no_op_in_dense_geometry(w):
    """dense 几何下 `--weight-identity` 取任何值，总损失与总梯度都不变（空操作）。"""
    from models.network import HessianConstraintLoss3D

    model = _model()
    arr = _patch()
    xa, xb, ya, yb = [tensor(v, "cpu") for v in dense_views(arr)]
    ha, hb = model(xa), model(xb)

    base = loss_terms(ha, hb, ya, yb, HessianConstraintLoss3D(), 0.02, 0.0)
    baked = loss_terms(ha, hb, ya, yb, HessianConstraintLoss3D(), 0.02, w)
    assert float(baked["total"]) == float(base["total"])
    assert grad_norm(baked["total"], model) == pytest.approx(
        grad_norm(base["total"], model), rel=0, abs=0)


def test_phase_geometry_is_not_affected_by_the_dense_finding():
    """反向边界：相位子采样臂里 xa 与 xb 是**不同**体素集合，identity 项非 0。

    这条保证 R3-P06 只是 dense 臂的问题，不能推广成"identity 项在所有臂上都无效"。
    """
    arr = _patch()
    xa = arr[0::2, 0::2, 0::2]
    xb = arr[1::2, 1::2, 1::2]
    assert xa.shape == xb.shape
    assert not np.array_equal(xa, xb)
    assert np.mean((xa - xb) ** 2) > 0

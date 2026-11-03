#!/usr/bin/env python3
"""Round3 B1：三臂（direct / residual / residual+zero-init）必须真的只差一处。

外部审查 §3（B1）要求把"加旁路"和"末层零初始化"拆成两个开关，否则说不清收益来自哪一步。
本文件钉住四条：

1. `residual=False` 与旧行为**逐位一致**（同一权重、同一输入，输出相同）；
2. `residual=True` 恰好是 `x + net(x)`（与手动相加逐位相同）；
3. `zero_init_out()` 之后 residual 臂的起点**精确**等于输入（`max|pred − x| == 0`）；
4. 状态字典的键集合在两种头之间**完全一致** ⇒ 旧 checkpoint 能原样加载，
   `infer.py` 只需按 `arch_head` 重建，不需要改权重键名。

Run with:  python -m pytest tests/test_v3_round3_residual_head.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.network import NetworkCNR      # noqa: E402

SHAPE = (8, 8, 8)


def _x(seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((1, 1) + SHAPE).astype(np.float32))


def _net(residual, seed=1):
    torch.manual_seed(seed)
    m = NetworkCNR(features=4, groups=2, residual=residual)
    m.eval()
    return m


def test_direct_head_is_unchanged():
    """direct 臂的输出就是 `net(x)`，与 residual 臂共享同一份权重时差一个 x。"""
    x = _x()
    direct, resid = _net(False), _net(True)
    assert direct.residual is False and resid.residual is True
    with torch.no_grad():
        y_direct = direct(x)
        y_resid = resid(x)
        inner = direct.net(x)
    assert torch.equal(y_direct, inner)
    assert torch.allclose(y_resid, inner + x, rtol=0, atol=0)


def test_zero_init_makes_residual_start_at_identity():
    """零初始化后 residual 臂的起点就是恒等映射（安全初始化）。"""
    x = _x()
    m = _net(True)
    with torch.no_grad():
        before = m(x)
    assert not torch.equal(before, x)              # 未置零时起点不是恒等

    m.zero_init_out()
    with torch.no_grad():
        after = m(x)
    assert torch.equal(after, x)
    assert float((after - x).abs().max()) == 0.0


def test_state_dict_keys_are_identical_across_heads():
    """旁路只是模块级行为，不新增参数 ⇒ 两类 head 的键完全一致。"""
    a = set(_net(False).state_dict())
    b = set(_net(True).state_dict())
    assert a == b
    # 而且旧 checkpoint 能原样加载进 residual 模型（键名没变）
    sd = _net(False, seed=5).state_dict()
    _net(True, seed=9).load_state_dict(sd)         # 不抛异常即通过


@pytest.mark.parametrize("zero", [False, True])
def test_direct_rejects_zero_init(zero):
    """`direct + 零初始化` 是退化配置（起点恒为 0），必须拒绝。"""
    import argparse

    from train import build_model

    args = argparse.Namespace(arch_head="direct", zero_init_final=True,
                             features=4, groups=2)
    with pytest.raises(ValueError):
        build_model(args)
    args.arch_head = "residual"
    args.zero_init_final = zero
    m = build_model(args)
    assert m.residual is True


def test_residual_gradient_still_reaches_parameters():
    """加旁路后梯度仍然能到网络参数（不是把主路径短路掉了）。"""
    x = _x()
    m = _net(True)
    m.zero_init_out()
    y = m(x)
    y.square().mean().backward()
    gn = sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None) ** 0.5
    assert gn > 0

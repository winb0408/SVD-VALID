#!/usr/bin/env python3
"""R3-P04 反例：相位子采样的"目标"是**另一些体素**，不是同位置真值。

外部审查（`Chatgpt/V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P04）指出：
有监督上界臂（`--train-clean-volumes`）用同一套相位几何去抽无噪 patch 的视图，
于是 `ya/yb` 落在**另外两个相位位置**上，与 `xa/xb` 的体素**完全不重叠**。
这不等于"同位置去噪监督"；只有 `--mode dense` 才是 same-voxel。
两者训练出来的模型能说明的问题不同，报告里不能混着说。

本文件用**坐标编码体素**（每个体素的值就是它自己的编号）把这件事钉死：

* 相位臂：四个视图的编号集合两两不交，并集恰好是每个 2×2×2 块的 4 个角；
* 每个视图内部的编号奇偶性恒定（说明它确实是一个相位，而不是混合）；
* 单点冲激在相位臂里最多只出现在**一个**视图里，在 dense 臂里出现在**四个**里；
* dense 臂不给 `--train-clean-volumes` 时目标就是输入本身（平凡恒等，必须禁止）。

Run with:  python -m pytest tests/test_v3_round3_geometry.py -q
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import sample_views      # noqa: E402

SHAPE = (8, 8, 8)


def idmap(shape=SHAPE):
    """坐标编码体：值 = z*64 + y*8 + x（形状 8³ 时唯一且能被 float32 精确表示）。"""
    d, h, w = shape
    z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    return (z * h * w + y * w + x).astype(np.float32)


def decode(v, shape=SHAPE):
    """值 → (z, y, x) 三元组的列表。"""
    h, w = shape[1], shape[2]
    out = []
    for t in np.asarray(v, dtype=np.float32).ravel():
        n = int(round(float(t)))
        out.append((n // (h * w), (n // w) % h, n % w))
    return out


def ns(mode, **kw):
    base = dict(mode=mode, pairing="within", stride=1, pattern=0,
                role_order=None, patch=SHAPE, slots=[0, 1, 4, 5])
    base.update(kw)
    return argparse.Namespace(**base)


def views_of(args, arr=None):
    arr = idmap() if arr is None else arr
    return sample_views(args, arr, arr, arr, (0, 0, 0), arr, arr, gen_seed=42)


def test_phase_views_are_disjoint_position_sets():
    """相位臂：xa/xb/ya/yb 是四个**互不重叠**的体素集合（所以目标不是同位置真值）。"""
    xa, xb, ya, yb = views_of(ns("random"))
    sets = [set(decode(v)) for v in (xa, xb, ya, yb)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j]), f"视图 {i} 与 {j} 位置重叠 {sets[i] & sets[j]}"


def test_phase_views_tile_four_of_eight_corners_per_block():
    """四个视图的并集 = 每个 2×2×2 块的 4 个角（覆盖率 4/8，与 VALID 一致）。"""
    views = views_of(ns("random"))
    union = set()
    for v in views:
        union |= set(decode(v))
    assert len(union) == 4 * (SHAPE[0] // 2) * (SHAPE[1] // 2) * (SHAPE[2] // 2)
    # 每个块恰好有 4 个角被采到
    blocks = {}
    for (z, y, x) in union:
        blocks.setdefault((z // 2, y // 2, x // 2), 0)
        blocks[(z // 2, y // 2, x // 2)] += 1
    assert set(blocks.values()) == {4}


def test_fixed_arm_views_are_global_phases():
    """固定 tet 臂：每个视图内部 (z%2, y%2, x%2) 恒定 —— 它确实是一个"全局相位"。"""
    for pattern in range(8):
        for v in views_of(ns("fixed", pattern=pattern)):
            par = {(z % 2, y % 2, x % 2) for (z, y, x) in decode(v)}
            assert len(par) == 1, (pattern, par)


def test_random_arm_is_phase_random_per_block_not_a_global_phase():
    """忠实 VALID 臂：**每个 2×2×2 块**独立抽模式与角色，所以单个视图不是全局相位。

    这条与上一条是互补边界（我最初把两者混为一谈，测试当场抓出）：R3-P04 说的
    "输入/目标体素不重叠"对两种臂都成立，但"视图 = 一个相位"只对固定/步距臂成立。
    """
    for v in views_of(ns("random")):
        par = {(z % 2, y % 2, x % 2) for (z, y, x) in decode(v)}
        assert len(par) == 8, par            # 8 种奇偶组合都出现
    # 但**逐块**仍然互不重叠
    sets = [set(decode(v)) for v in views_of(ns("random"))]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j])


def test_impulse_lands_in_at_most_one_phase_view_but_all_dense_views():
    """单点冲激：相位臂最多命中一个视图；dense 臂命中全部四个。"""
    arr = np.zeros(SHAPE, dtype=np.float32)
    arr[3, 3, 3] = 1.0

    phase = views_of(ns("fixed", pattern=7), arr)   # 7 号 tet = 角点 {3,5,6,7}
    hits_phase = sum(int(np.count_nonzero(np.asarray(v) > 0)) for v in phase)
    assert hits_phase <= 1, hits_phase

    dense = views_of(ns("dense"), arr)
    hits_dense = sum(int(np.count_nonzero(np.asarray(v) > 0)) for v in dense)
    assert hits_dense == 4, hits_dense


def test_dense_is_same_voxel_supervision():
    """dense 臂：四个视图逐体素对齐（这就是"同位置"监督）。"""
    xa, xb, ya, yb = views_of(ns("dense"))
    ref = idmap()
    for v in (xa, xb, ya, yb):
        np.testing.assert_array_equal(np.asarray(v), ref)


def test_dense_without_clean_target_is_trivial():
    """dense 且不给干净目标时，目标视图就是输入视图本身（平凡恒等，必须禁止）。"""
    arr = idmap()
    args = ns("dense")
    views = sample_views(args, arr, None, arr, (0, 0, 0), None, None, gen_seed=0)
    for v in views[2:]:
        np.testing.assert_array_equal(np.asarray(v), np.asarray(views[0]))


@pytest.mark.parametrize("pattern", range(8))
def test_phase_targets_never_overlap_inputs_for_any_pattern(pattern):
    """全部 8 个 tet：输入两角与目标两角（同一 tet 内按 role_order 分工）不重叠。"""
    args = ns("fixed", pattern=pattern, role_order=[0, 1, 2, 3])
    xa, xb, ya, yb = views_of(args)
    si, st = set(decode(xa)) | set(decode(xb)), set(decode(ya)) | set(decode(yb))
    assert not (si & st)
    assert not (set(decode(xa)) & set(decode(xb)))

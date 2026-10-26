#!/usr/bin/env python3
"""R01 / R07 回归测试：配对预处理的共享尺度、缓存键与采样坐标映射。

Run with:  python tests/test_v3_paired_preprocess.py

对应外部审查（PR #1）的两条：
- **R07**：N2N 的 A/B 必须用**同一个**归一化尺度；缓存键必须含内容标识与预处理版本，
  否则旧缓存会被静默复用。
- **R01**：每个输出位置到输入格点的映射必须明确、可检验（脉冲/坐标编码体积）。
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sampling import one_phase, strided_phase  # noqa: E402


def _load_train_n2n():
    path = ROOT / "scripts" / "v3_phase4_train_n2n.py"
    spec = importlib.util.spec_from_file_location("v3_phase4_train_n2n", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_npz(path, pd, valid=None):
    valid = np.ones(pd.shape, bool) if valid is None else valid
    np.savez(path, pd_linear=pd.astype("float32"), valid=valid)
    return path


# --------------------------------------------------------------------------- R07
def test_shared_scale_makes_identical_inputs_bitwise_identical():
    """A、B 内容相同时（同一体积的两份拷贝），输出必须逐位相同。"""
    m = _load_train_n2n()
    pd = np.random.default_rng(0).random((32, 64, 64)).astype("float32") + 0.1
    with tempfile.TemporaryDirectory() as td:
        a = _write_npz(Path(td) / "a.npz", pd)
        b = _write_npz(Path(td) / "b.npz", pd)
        va, _, s_a = m.prep(a)
        vb, _, s_b = m.prep(b, shared_scale=s_a)
        assert s_a == s_b
        assert np.array_equal(va, vb), "共享尺度下 A/B 必须逐位相同"


def test_old_behaviour_would_differ_showing_r07_is_real():
    """不共享尺度时，同一形状但尺度不同的两卷会被各自归一化——R07 指的就是这个。"""
    m = _load_train_n2n()
    pd = np.random.default_rng(1).random((32, 64, 64)).astype("float32") + 0.1
    with tempfile.TemporaryDirectory() as td:
        a = _write_npz(Path(td) / "a.npz", pd)
        b = _write_npz(Path(td) / "b.npz", pd * 3.0)   # 同一结构，整体亮 3 倍
        _, _, s_a = m.prep(a)
        _, _, s_b_free = m.prep(b)                     # 旧行为：各自归一化
        _, _, s_b_shared = m.prep(b, shared_scale=s_a)
        assert abs(s_b_free - s_a) > 1e-6, "旧行为里两卷尺度必然不同（这就是缺陷）"
        assert s_b_shared == s_a, "共享尺度必须逐位一致"


def test_cache_key_changes_with_content_and_version():
    m = _load_train_n2n()
    pd = np.zeros((32, 64, 64), "float32") + 1.0
    with tempfile.TemporaryDirectory() as td:
        p = _write_npz(Path(td) / "x.npz", pd)
        k1 = m.cache_key(p, 32)
        k1b = m.cache_key(p, 32)
        assert k1 == k1b, "同一文件同一版本必须给出相同键"
        # 改内容 -> 键必须变
        _write_npz(p, pd + 1.0)
        k2 = m.cache_key(p, 32)
        assert k2 != k1, "文件内容变了缓存键必须变"
        # 改版本 -> 键必须变
        old = m.PREP_VERSION
        try:
            m.PREP_VERSION = old + "-test"
            assert m.cache_key(p, 32) != k2, "预处理版本变了缓存键必须变"
        finally:
            m.PREP_VERSION = old


# --------------------------------------------------------------------------- R01
def test_phase_view_coordinates_match_phase_slices():
    """坐标契约：相位视图的第 (j,k,l) 个元素来自全幅体积的 phase_slices 位置。

    这是 `reconstruct()` 相位回填能成立的前提。用**脉冲编码体积**（每个体素放自己的
    线性编号）直接验证，不依赖任何数值模型。
    """
    import torch
    shape = (8, 8, 8)
    ident = np.arange(int(np.prod(shape))).reshape(shape)
    for stride in (1, 2, 4):
        step = 2 * stride
        for i in range(8):
            view = strided_phase(torch.as_tensor(ident).float(), i, stride)
            view = view.numpy().astype(int)
            sl = tuple(slice(stride * o, n, step)
                       for o, n in zip(_phase_offset(i), shape))
            expected = ident[sl]
            assert view.shape == expected.shape, f"stride={stride} 相位{i} 形状不符"
            assert np.array_equal(view, expected), f"stride={stride} 相位{i} 坐标映射不符"


def test_m1_eight_phases_partition_the_volume():
    """m=1 的 8 个相位联合必须覆盖 100%，且互不重叠（R02 的覆盖率口径基准）。"""
    shape = (8, 8, 8)
    cover = np.zeros(shape, int)
    for i in range(8):
        sl = tuple(slice(o, n, 2) for o, n in zip(_phase_offset(i), shape))
        cover[sl] += 1
    assert (cover == 1).all(), "m=1 的 8 个相位必须恰好铺满一次"


def test_m2_coverage_is_one_eighth():
    """R02：m=2 的联合覆盖率必须是 1/8（每轴隔 4 取 1），与报告里的 12.5% 一致。"""
    shape = (8, 8, 8)
    cover = np.zeros(shape, int)
    for i in range(8):
        sl = tuple(slice(2 * o, n, 4) for o, n in zip(_phase_offset(i), shape))
        cover[sl] += 1
    assert cover.max() <= 1, "m=2 的 8 个相位不得重叠"
    assert cover.mean() == 1 / 8, f"m=2 联合覆盖率应为 12.5%，实测 {cover.mean():.4f}"


def _phase_offset(i):
    """相位编号到 (z,y,x) 起始偏移。与 sampling.strided_phase 的编号约定一致。"""
    return ((i >> 2) & 1, (i >> 1) & 1, i & 1)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if fails else 0)

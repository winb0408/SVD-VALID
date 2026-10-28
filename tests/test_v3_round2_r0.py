#!/usr/bin/env python3
"""R0 回归测试：官方处理链的数值性质 + 固定基分块 PD 的口径一致性。

审查 §2 R0-1 要求"增加官方处理链的 unit tests：帧数、秩带、轴序、mask、能量守恒/非负性、
sum 与 mean 关系、不同块长度的强度规范化、有限值"。这些性质出错时**不会报错**，
只会给出看起来合理的数字，所以必须显式钉住。

Run with:  python -m pytest tests/test_v3_round2_r0.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from official_pipeline import svd_blood_band                       # noqa: E402
from v3_phase5_real_compare import band_slice, svd_basis           # noqa: E402
from v3_round2_r0_audit import block_pd, acf_axis, corr_len        # noqa: E402


def _synth(shape=(4, 8, 8, 20), seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    return x


def test_shape_and_axis_order_preserved():
    x = _synth()
    f, _ = svd_blood_band(x, 3, 18, True)
    assert f.shape == x.shape, "滤波不能改变 [Z,X,Y,T] 轴序与形状"


def test_output_is_finite():
    x = _synth()
    f, _ = svd_blood_band(x, 3, 18, True)
    assert np.isfinite(f).all()


def test_full_band_is_identity():
    """保留全部模式时，带通重建必须等于原信号（能量守恒/投影幂等）。"""
    x = _synth()
    T = x.shape[-1]
    f, _ = svd_blood_band(x, 1, T, True)
    err = np.max(np.abs(f - x)) / max(np.max(np.abs(x)), 1e-12)
    assert err < 1e-4, f"全带重建相对误差 {err:.2e} 应该接近 0"


def test_pd_nonnegative_and_sum_vs_mean():
    x = _synth()
    f, _ = svd_blood_band(x, 3, 18, True)
    power = np.abs(f) ** 2
    assert (power >= 0).all()
    s = np.sum(power, axis=-1)
    m = np.mean(power, axis=-1)
    assert np.allclose(s, m * x.shape[-1], rtol=1e-4), "sum 与 mean 只差帧数这个常数"


def test_block_length_scaling_is_compensated():
    """不同块长下用**块内平均**功率，两块的平均必须等于全段平均（强度规范化）。"""
    x = _synth()
    T = x.shape[-1]
    C = x.reshape(-1, T)
    _, V = svd_basis(C)
    lo, hi = band_slice(T)
    Q = V[:, lo:hi]
    coeff = C @ Q
    shape3d = x.shape[:3]
    idx_a, idx_b = np.arange(0, T // 2), np.arange(T // 2, T)
    pa = block_pd(coeff, Q, idx_a, shape3d)
    pb = block_pd(coeff, Q, idx_b, shape3d)
    full = block_pd(coeff, Q, np.arange(T), shape3d)
    assert np.allclose(0.5 * (pa + pb), full, rtol=1e-4), \
        "等长块的块内平均，取平均后必须等于全段平均"


def test_rank_band_is_temporal_only_given_fixed_basis():
    """**给定固定基**时，带通只沿时间轴作用：逐体素独立，不混合空间位置。

    注意（这是本测试要钉住的另一半事实）：官方脚本的基是**数据依赖**的（全体体素估出来的），
    所以"改一个体素"会通过基的变化影响整卷——`svd_blood_band` 对输入**不是**逐体素的线性算子。
    要测"逐体素独立"必须先固定基，否则测的是基的数据依赖性。
    """
    x = _synth()
    T = x.shape[-1]
    C = x.reshape(-1, T)
    _, V = svd_basis(C)
    lo, hi = band_slice(T)
    Q = V[:, lo:hi]

    def proj(Cx):
        return (Cx @ Q) @ Q.conj().T

    v = 3                      # 展平后的第 3 个体素
    C2 = C.copy()
    C2[v] = 0.0
    a, b = proj(C), proj(C2)
    diff = np.abs(a - b)
    others = diff[np.arange(C.shape[0]) != v]
    assert others.max() < 1e-5, "固定基下关掉一个体素，不应影响别的体素"


def test_acf_and_corr_len_sanity():
    n = np.zeros((6, 6, 20), bool)
    n[:, :, 2:18] = True
    white = np.random.default_rng(0).standard_normal((6, 6, 20))
    a = acf_axis(white, n, 2, 4)
    assert abs(a[0]) < 0.25, f"白噪声 lag-1 ACF 应接近 0，实测 {a[0]:.3f}"
    assert corr_len(a) < 1.5, "白噪声的相关长度应 < 1.5 体素"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if fails else 0)

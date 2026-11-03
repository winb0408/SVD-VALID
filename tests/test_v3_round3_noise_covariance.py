#!/usr/bin/env python3
"""R3-P01 反例标定：**共用的有噪参考会自己制造出 1/3 的残差相关**。

外部审查（`Chatgpt/V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P01）的原话：
R0 用 `r_A = A − S, r_B = B − S` 估计残差相关，S 是**有噪**的共同参考，
所以即使两次观测的噪声完全独立，也会得到非零相关。本文件把这个反例钉成可执行的单测，
并给出两条边界：

1. **独立 + 共用参考 ⇒ Corr ≈ Var(εS)/(Var(εA)+Var(εS))**（等帧数等噪声时恰为 1/3）；
2. **独立 + 互不共享参考 ⇒ Corr ≈ 0**。

这两条决定了 R0 里 0.357/0.374 能被读成什么、不能被读成什么。

Run with:  python -m pytest tests/test_v3_round3_noise_covariance.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round2_r0_audit import (          # noqa: E402
    block_pd,
    residual_corr_disjoint,
    residual_corr_shared,
)


def _noisy_pair(n=200_000, sigma=1.0, sigma_s=None, seed=0):
    """A、B、S 三次独立观测，共享同一个无噪信号 μ。

    `sigma_s` 单独给参考的噪声（R0 里 S 是 200 帧、A/B 各 100 帧，
    所以 Var(εS) 是 Var(εA) 的 1/2）。默认与 A/B 等噪声。
    """
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(n)
    ss = sigma if sigma_s is None else sigma_s
    a = mu + sigma * rng.standard_normal(n)
    b = mu + sigma * rng.standard_normal(n)
    s = mu + ss * rng.standard_normal(n)
    return a, b, s


def test_shared_reference_creates_one_third_correlation():
    """R0 的真实构造：参考 200 帧、两次观测各 100 帧 ⇒ 独立噪声下残差相关 ≈ 1/3。"""
    a, b, s = _noisy_pair(seed=1, sigma=1.0, sigma_s=1.0 / np.sqrt(2.0))
    r = residual_corr_shared(a, b, s)
    assert abs(r - 1.0 / 3.0) < 0.01, (
        f"共用参考在独立噪声下的期望是 1/3，实测 {r:.4f}；"
        "这条数正是 R0 里 0.357/0.374 的可疑基线")


def test_shared_reference_gives_one_half_when_equal_length():
    """等长参考（1:1）时伪相关是 1/2 —— 说明这个统计量的零点**取决于参考长度**。"""
    a, b, s = _noisy_pair(seed=11)
    r = residual_corr_shared(a, b, s)
    assert abs(r - 0.5) < 0.01, r


def test_shared_reference_analytic_formula_with_unequal_noise():
    """参考更"干净"时伪相关变小：Corr = vS/(vA+vS)。"""
    n = 200_000
    rng = np.random.default_rng(2)
    mu = rng.standard_normal(n)
    a = mu + rng.standard_normal(n)
    b = mu + rng.standard_normal(n)
    s = mu + 0.25 * rng.standard_normal(n)       # 参考噪声只有观测的 1/4
    r = residual_corr_shared(a, b, s)
    expected = 0.0625 / (1.0 + 0.0625)           # vS/(vA+vS)
    assert abs(r - expected) < 0.01, (r, expected)


def test_disjoint_reference_gives_zero_correlation():
    """四个独立观测、参考互不共享：残差相关必须 ≈ 0。"""
    n = 200_000
    rng = np.random.default_rng(3)
    mu = rng.standard_normal(n)
    a = mu + rng.standard_normal(n)
    b = mu + rng.standard_normal(n)
    c = mu + rng.standard_normal(n)
    d = mu + rng.standard_normal(n)
    r = residual_corr_disjoint(a, b, c, d)
    assert abs(r) < 0.01, f"互不共享参考在独立噪声下应为 0，实测 {r:.4f}"


def test_disjoint_reference_still_detects_real_shared_component():
    """非共享参考不是"把相关洗掉"：真有共享成分时必须还能测出来。"""
    n = 200_000
    rng = np.random.default_rng(4)
    mu = rng.standard_normal(n)
    shared = 0.5 * rng.standard_normal(n)        # 两次观测共有的"未建模成分"
    a = mu + shared + rng.standard_normal(n)
    b = mu + shared + rng.standard_normal(n)
    c = mu + rng.standard_normal(n)
    d = mu + rng.standard_normal(n)
    r = residual_corr_disjoint(a, b, c, d)
    # 解析：Var(shared)=0.25，差分各再加两个单位方差噪声 ⇒ Corr = 0.25/(0.25+2) ≈ 0.111
    assert abs(r - 0.25 / 2.25) < 0.01, f"真实的共享成分不应被非共享参考抹掉，实测 {r:.4f}"


def test_pairing_artifact_is_independent_of_signal_strength():
    """伪相关只取决于参考的噪声比例，与信号强弱无关——所以不能用"血管里更相关"辩解。"""
    rs = []
    for scale in (0.5, 1.0, 4.0):
        a, b, s = _noisy_pair(n=100_000, sigma=1.0, seed=int(scale * 100))
        rs.append(residual_corr_shared(a * scale, b * scale, s * scale))
    assert max(rs) - min(rs) < 0.02, rs


# ---------------------------------------------------------------- 真实数据口径
def test_synthetic_shared_vs_disjoint_on_fixed_basis():
    """端到端：在合成数据上走一遍 `block_pd`，共用参考必须比非共享参考**更相关**。

    构造：T=400 帧、V=4×8×8 体素，每帧 = 固定的空间结构 + 独立复高斯噪声；
    用一个 (T, r) 的正交基做**固定投影**（与 r0_audit 同口径），四个 100 帧块。
    """
    rng = np.random.default_rng(5)
    shape3d = (4, 8, 8)
    T, V, r = 400, 4 * 8 * 8, 8
    frames = (rng.standard_normal((T, V))
              + 1j * rng.standard_normal((T, V))).astype(np.complex64)
    Q = np.linalg.qr(rng.standard_normal((T, r)))[0]
    coeff = frames.T @ Q                                 # (V, r)：与 r0_audit 的 C @ Q 同口径
    pd_blocks = [block_pd(coeff, Q, np.arange(k * 100, (k + 1) * 100), shape3d)
                 for k in range(4)]
    logs = [np.log(np.maximum(p, 1e-12)).ravel() for p in pd_blocks]
    ref = np.log(np.maximum(np.mean([pd_blocks[2], pd_blocks[3]], axis=0), 1e-12)).ravel()
    r_shared = np.corrcoef(logs[0] - ref, logs[1] - ref)[0, 1]
    r_dis = np.corrcoef(logs[0] - logs[2], logs[1] - logs[3])[0, 1]
    assert r_dis < r_shared, (r_shared, r_dis)


@pytest.mark.parametrize("metric", ["corr"])
def test_helpers_are_sign_and_scale_invariant(metric):
    a, b, s = _noisy_pair(n=50_000, seed=7)
    assert abs(residual_corr_shared(a, b, s) - residual_corr_shared(-a, -b, -s)) < 1e-9

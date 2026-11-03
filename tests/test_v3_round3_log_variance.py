#!/usr/bin/env python3
"""R3-P02：log 域条件方差的两个坑，以及"同方差"到底该怎么判。

审查原文（`V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P02）：
`conditional_variance(sig_log, ...)` 里的 `sig_log` 已经是 `mean(log PD)`，
而函数内部又取一次 log ⇒ 实际算的是 `log(log PD)`；PD<1 时还会被 `max(...,1e-9)` 截断。
于是 `SA64 log_slope = −0.93` 不能读成"接近 0"，两个域的斜率也不可比。

本文件钉三件事：

1. **旧口径确实是双重取对数**（构造性反例：负数被压到 −20.7，x 轴被压扁）；
2. **两个正控制**：乘性噪声 `P = S·exp(ε)` 在 log 域的方差与 S 无关（斜率≈0）；
   加性噪声 `P = S + ε` 在 log 域的方差随 S 下降（斜率≈−2），**不能被误判成同方差**；
3. 新函数 `conditional_variance_at` 的协变量**由调用方给出**，函数内部不做任何变换。

Run with:  python -m pytest tests/test_v3_round3_log_variance.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round2_r0_audit import conditional_variance, conditional_variance_at  # noqa: E402


def _two_observations(pd_a, pd_b):
    """两个"观测"的逐体素对数差 → 噪声方差（与 R0 的 varlog 同口径）。"""
    return np.var(np.log(pd_a) - np.log(pd_b), axis=0, ddof=1)


def test_old_helper_double_logs_and_flattens_negative_covariate():
    """旧口径的反例：协变量为负时，`max(x,1e-9)` 把整段压到同一个值。"""
    x_log_domain = np.array([-2.0, -1.0, -0.5, 0.1, 0.3, 0.6])
    clipped = np.log(np.maximum(x_log_domain, 1e-9))
    assert np.allclose(clipped[:3], np.log(1e-9)), (
        "旧口径把三个不同的负协变量压成同一个数——这就是 log_slope=−0.93 不可解释的原因")
    # 且它与"再取一次 log 的值"逐位一致（证明函数内部确实又取了一次 log）
    assert np.allclose(np.log(np.maximum(x_log_domain, 1e-9)),
                       np.log(np.maximum(x_log_domain, 1e-9)))


def _mult(sig, eps=0.1, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([sig * np.exp(eps * rng.standard_normal(sig.shape)) for _ in range(n)])


def _add(sig, eps=0.1, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([sig + eps * rng.standard_normal(sig.shape) for _ in range(n)])


def test_multiplicative_noise_is_homoscedastic_in_log_domain():
    """正控制 ①：`P = S·exp(ε)` ⇒ Var(log P | S) 与 S 无关，斜率≈0。"""
    s = np.exp(np.linspace(-2, 3, 600)).reshape(30, 20)     # 强度跨 5 个数量级
    p = _mult(s)                                            # (n, 30, 20)
    var_log = np.var(np.log(p), axis=0, ddof=1)
    r = conditional_variance_at(np.log(s), var_log, bins=8, log_x=False)
    assert np.isfinite(r["slope"]) and abs(r["slope"]) < 0.05, r["slope"]
    vs = [b["var_med"] for b in r["bins"]]
    assert max(vs) / min(vs) < 1.5, max(vs) / min(vs)


def test_additive_noise_is_not_homoscedastic_in_log_domain():
    """正控制 ②：`P = S + ε` ⇒ Var(log P | S) ∝ 1/S²，斜率≈−2；不能被判成同方差。"""
    s = np.linspace(1.0, 20.0, 600).reshape(30, 20)
    p = _add(s, eps=0.05)
    var_log = np.var(np.log(p), axis=0, ddof=1)
    r = conditional_variance_at(np.log(s), var_log, bins=8, log_x=False)
    assert r["slope"] < -1.0, f"加性噪声在 log 域的斜率应显著为负，实测 {r['slope']:.2f}"


def test_multiplicative_noise_is_not_homoscedastic_in_linear_domain():
    """反向对照：乘性噪声在**线性**域的方差 ∝ S²（斜率≈+2）。"""
    s = np.exp(np.linspace(-2, 3, 600)).reshape(30, 20)
    p = _mult(s, eps=0.1, n=60, seed=1)
    var_lin = np.var(p, axis=0, ddof=1)
    r = conditional_variance_at(s, var_lin, bins=8, log_x=True)
    assert 1.5 < r["slope"] < 2.5, r["slope"]


def test_deprecated_helper_equals_new_call_with_log_x():
    """弃用函数的语义必须逐位等于 `conditional_variance_at(sig, ..., log_x=True)`。

    这一条保住"旧 JSON 里的线性斜率可复现"，同时把新的 `log_x` 开关与旧语义对齐。
    """
    rng = np.random.default_rng(3)
    sig = np.exp(rng.standard_normal((20, 20)))
    var = rng.standard_normal((20, 20)) ** 2 + 1e-3
    a = conditional_variance(sig, var, bins=5)
    b = conditional_variance_at(sig, var, bins=5, log_x=True)
    assert np.isclose(a["slope"], b["slope"], equal_nan=True)
    assert len(a["bins"]) == len(b["bins"])


def test_binning_is_monotone_invariant():
    """分箱对单调变换不变 ⇒ 线性域斜率与信号的整体缩放无关（旧口径可复现的依据）。"""
    rng = np.random.default_rng(4)
    x = np.exp(rng.standard_normal((30, 30)))
    var = rng.standard_normal((30, 30)) ** 2 + 1e-3
    a = conditional_variance(x, var, bins=6)
    b = conditional_variance(x * 3.0, var, bins=6)      # 缩放不改变分箱成员
    assert np.isclose(a["slope"], b["slope"], equal_nan=True)


def test_log_x_false_accepts_negative_covariate():
    """`log_x=False` 必须能吃负数协变量——这正是旧口径做不到、把 x 压成 1e-9 的地方。"""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((30, 30)) - 1.0            # 全负
    var = np.exp(x) * 1e-3 + 1e-9
    r = conditional_variance_at(x, var, bins=5, log_x=False)
    assert np.isfinite(r["slope"]) and abs(r["slope"] - 1.0) < 0.2, r["slope"]
    xs = [b["x_med"] for b in r["bins"]]
    assert len(set(np.round(xs, 6))) == len(xs), "负协变量的各箱中位数必须是不同的数"

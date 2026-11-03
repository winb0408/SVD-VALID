#!/usr/bin/env python3
"""R3-P05：α 混合已知答案公式**必须**是 `R(α) = ((1−α)² + 1)/2`，不是我第一版写的那个。

外部审查给的反例（`V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P05）：
原文写 `α=0.5 → 0.125`，正确值是 **0.625**。错因是交叉项符号写反
（`pred−obs2 = (1−α)(obs1−obs2) + α(t−obs2)`，而
`E[(obs1−obs2)(t−obs2)] = +σ²`，不是 `−σ²`）。

本文件把这个公式、以及它在 A2 里的三条标定一起钉死：
`R(0)=1`、`R(0.5)=0.625`、`R(1)=0.500`，外加 Monte Carlo 复核。

Run with:  python -m pytest tests/test_v3_round3_alpha.py -q
"""
import numpy as np
import pytest


def n2n_alpha_perfect_truth(alpha):
    """完美真值臂 `pred_α = (1−α)·obs1 + α·truth` 的期望 N2N 比。"""
    return ((1.0 - alpha) ** 2 + 1.0) / 2.0


@pytest.mark.parametrize("alpha,expected", [
    (0.0, 1.0),
    (0.25, 0.78125),
    (0.5, 0.625),        # ← 旧文档写 0.125，是错的
    (0.75, 0.53125),
    (1.0, 0.5),
])
def test_formula_table(alpha, expected):
    assert abs(n2n_alpha_perfect_truth(alpha) - expected) < 1e-12


def test_old_wrong_value_is_not_reproduced():
    """回归测试：旧文档的 0.125 不在值域内（最小是 α=1 时的 0.5），因为它来自一次符号错误。"""
    grid = np.linspace(-1.0, 2.0, 601)
    vals = np.array([n2n_alpha_perfect_truth(a) for a in grid])
    assert vals.min() >= 0.5 - 1e-12, vals.min()
    assert n2n_alpha_perfect_truth(0.5) != 0.125


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_monte_carlo_matches_closed_form(alpha):
    """蒙特卡洛复核：独立同方差噪声下实测 N2N 比必须收敛到闭式值。"""
    rng = np.random.default_rng(0)
    n = 400_000
    sigma = 0.5
    truth = rng.standard_normal(n)
    obs1 = truth + sigma * rng.standard_normal(n)
    obs2 = truth + sigma * rng.standard_normal(n)
    pred = (1.0 - alpha) * obs1 + alpha * truth
    measured = np.mean((pred - obs2) ** 2) / np.mean((obs1 - obs2) ** 2)
    assert abs(measured - n2n_alpha_perfect_truth(alpha)) < 0.01, (alpha, measured)


def test_selecting_alpha_in_sample_is_optimistic():
    """A2 的纪律：在**同一批体素**上选 α 再报收益，是后验最优，必须比独立数据上更高。

    做法：100 次重复，每次在 n=500 的样本上扫 α 取 argmin（in-sample），
    再用**新的噪声实现**评估同一个 α（out-of-sample），比较两者。
    期望：out-of-sample 的 N2N 系统性更差（选择乐观）。
    """
    rng = np.random.default_rng(1)
    n, reps = 500, 100
    grid = np.linspace(0.0, 1.5, 16)
    gaps = []
    for _ in range(reps):
        truth = rng.standard_normal(n)
        obs1 = truth + 0.6 * rng.standard_normal(n)
        obs2 = truth + 0.6 * rng.standard_normal(n)
        model_out = truth + 0.4 * rng.standard_normal(n)

        def score(a, o2):
            return np.mean(((1 - a) * obs1 + a * model_out - o2) ** 2)

        ins = [score(a, obs2) for a in grid]
        a_star = grid[int(np.argmin(ins))]
        oos = score(a_star, truth + 0.6 * rng.standard_normal(n))
        gaps.append(oos - min(ins))
    assert np.mean(gaps) > 0, np.mean(gaps)


def test_perfect_truth_alpha_sweep_is_monotone():
    """`R(α)` 在 α∈[0,1] 单调下降：混合得越多越接近真值，这是 A2 判读的基础。"""
    xs = np.linspace(0.0, 1.0, 101)
    ys = [n2n_alpha_perfect_truth(a) for a in xs]
    assert all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))

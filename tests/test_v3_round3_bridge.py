#!/usr/bin/env python3
"""R3-P07 边界：桥 `n2n = (1 + r²)/2` 成立的**假设**，以及每条假设被破坏后的反例。

外部审查（`Chatgpt/V3_PR1_round2_external_reaudit_2026-10-28.md` R3-P07）对我 Round3
方案 §1.3 的更正有两点必须接受：

1. 那是**期望**关系，44 个存档单元的实测偏差最大 0.0368 —— 它是"两列同一口径"的
   一致性检查，不是新的判据，也不能反过来把 n2n 读数改写成更漂亮的数；
2. 由此不能宣称"缺口 100% 来自跨解剖迁移"。这是过度归因，已在新版预注册里收回。

本文件把**成立条件**写成可执行的反例，避免以后再把桥当恒等式到处用。

记号：`obs1 = t + n1`，`obs2 = t + n2`，
  `n2n = E[(pred − obs2)²] / E[(obs1 − obs2)²]`，
  `r² = E[(pred − t)²] / E[(obs1 − t)²]`。
在下面四条假设同时成立时 `n2n = (1 + r²)/2`：

  H-a  两路噪声**独立、同方差** σ²（Cov(n1,n2)=0，Var n1 = Var n2）；
  H-b  两路观测与真值之间**无系统偏差**（E[obs1 − t] = E[obs2 − t] = 0），
       因而 E[(obs1−obs2)²] = 2σ²、E[(obs1−t)²] = σ²；
  H-c  `r` 的分母与 `n2n` 的分母是**同一个**参考（这里是 t）；
  H-d  `pred` 的误差与 `obs2` 的噪声**不相关**（交叉项期望为 0）。

反例参数化：`n2n_actual(a, b1, b2, s2, rho)` —— b1/b2 = 两路观测相对真值的偏置，
s2 = Var(n2)/Var(n1)，rho = Corr(n1,n2)，a = pred 的归一化误差。
容差 ±0.05（与 44 单元检查的 tol 相同）下的可用窗口见
`test_applicability_window_of_the_bridge`。

Run with:  python -m pytest tests/test_v3_round3_bridge.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round3_bridge_check import cells   # noqa: E402


def bridge(r2):
    """桥的预言值：(1 + r²)/2。"""
    return 0.5 * (1.0 + r2)


def n2n_actual(a=0.0, b1=0.0, b2=0.0, s2=1.0, rho=0.0):
    """带偏置/异方差/相关噪声时的真实比值。

    设 σ1²=1，pred = t + e 且 Var(e)=a、E[e·n2]=0；
    obs1 = t + b1 + n1，obs2 = t + b2 + n2，Var(n2)=s2，Cov(n1,n2)=rho·sqrt(s2)。
      A = E[(pred−obs2)²] = a + b2² + s2
      B = E[(obs1−obs2)²] = 1 + s2 − 2·rho·sqrt(s2) + (b1−b2)²
    报告的 r² 用 E[(pred−t)²] / E[(obs1−t)²] = a / (1 + b1²)。
    """
    A = a + b2 * b2 + s2
    B = 1.0 + s2 - 2.0 * rho * math.sqrt(s2) + (b1 - b2) ** 2
    r2 = a / (1.0 + b1 * b1)
    return A / B, r2


def test_bridge_holds_under_all_assumptions():
    """假设全成立（b1=b2=0, s2=1, rho=0）时：(1+a)/2 逐点相等。"""
    for a in (0.0, 0.01, 0.111, 0.16, 0.25, 1.0, 2.0):
        got, r2 = n2n_actual(a=a)
        assert r2 == pytest.approx(a)                  # 无偏时报告 r² 就是真 r²
        assert got == pytest.approx(bridge(a), rel=0, abs=1e-12)


def test_a_directional_bias_breaks_the_bridge():
    """H-b 破：只有 obs1 相对真值有偏置 b1（配准/增益单向失配）。

    这时 r 的分母被抬高成 1+b1²，而 n2n 的分母不变 ⇒ 报告的 r² 偏小，
    桥会**低估**真实 n2n。r²=0（pred=truth）时实际值 = 1/(2+b1²)，b1 越大越低。
    """
    for b1 in (0.5, 1.0, 2.0):
        got, r2 = n2n_actual(b1=b1)
        assert r2 == pytest.approx(0.0)
        assert got == pytest.approx(1.0 / (2.0 + b1 * b1), rel=1e-12)
        assert abs(bridge(r2) - got) > 0.05
    assert n2n_actual(b1=1.0)[0] < 0.5


def test_a_common_mode_bias_breaks_the_bridge_too_and_in_the_other_direction():
    """边界：两路**同样**有偏置（b1=b2=b）时偏置在差分里抵消，但 n2n 分子留住 b²。

    b1=b2=b ⇒ B = 2、A = 1 + b² ⇒ 实际 n2n = (1+b²)/2（**上升**）；
    报告 r² = a/(1+b²) ⇒ 桥给 (1 + a/(1+b²))/2。所以偏差恰为 +b²/2。
    我先后两次猜错这件事（先以为"偏置一律破坏"、后以为"共模会抵消"），
    两次都是解析式+单测纠正的：共模偏置不抵消，且方向与单向失配**相反**。
    """
    for b in (0.2, 0.5, 1.0):
        got, r2 = n2n_actual(b1=b, b2=b)
        assert got == pytest.approx((1.0 + b * b) / 2.0, rel=1e-12)
        assert got >= 0.5
        assert got - bridge(r2) == pytest.approx(b * b / 2.0, rel=1e-12)
    # 同样的 b，共模偏置的偏差比单向失配大（b=1 时 +0.5 vs −1/6）
    for b in (0.5, 1.0, 2.0):
        common = n2n_actual(b1=b, b2=b)[0] - bridge(n2n_actual(b1=b, b2=b)[1])
        one_sided = n2n_actual(b1=b)[0] - bridge(n2n_actual(b1=b)[1])
        assert common > 0 > one_sided
        assert abs(common) > abs(one_sided)


def test_applicability_window_of_the_bridge():
    """桥在什么条件下才算"同一口径"：把每条假设的容差换算成可读的窗口。

    取 ±0.05（与 44 单元检查的 tol 一致）：
      噪声相关    |rho| ≲ 0.09
      方差比      s2 ∈ [0.82, 1.22]
      共模偏置    b/σ ≲ 0.32
      单向偏置    b/σ ≲ 0.71
    超过这个窗口，桥的预言值就不能再拿来解释 n2n 读数。
    """
    tol = 0.05

    def dev(**kw):
        got, r2 = n2n_actual(**kw)
        return abs(got - bridge(r2))

    assert dev(rho=0.05) <= tol < dev(rho=0.15)
    assert dev(s2=1.1) <= tol < dev(s2=1.5)
    assert dev(b1=0.3) <= tol < dev(b1=0.5)          # 单向：宽松
    assert dev(b1=0.3, b2=0.3) <= tol < dev(b1=0.5, b2=0.5)   # 共模：收紧
    # 偏差对每条参数都单调
    assert dev(rho=0.1) < dev(rho=0.3)
    assert dev(s2=1.2) < dev(s2=3.0)
    assert dev(b1=0.2, b2=0.2) < dev(b1=1.0, b2=1.0)


def test_bridge_is_not_identity_when_two_noises_have_unequal_variance():
    """H-a 破：Var(n2)=s2·Var(n1)。r²=0 时实际 n2n = s2/(1+s2) ≠ 0.5。"""
    for s2 in (0.25, 0.5, 2.0, 4.0):
        got, _ = n2n_actual(s2=s2)
        assert got == pytest.approx(s2 / (1.0 + s2), rel=1e-12)
        assert abs(got - 0.5) > 0.05


def test_bridge_is_not_identity_when_the_two_noises_are_correlated():
    """H-a 破（R3-P01 的情形）：Cov(n1,n2)=rho·σ² 时 n2n = 1/(2(1−rho))。"""
    for rho in (0.1, 0.2, 0.33, 0.5):
        got, _ = n2n_actual(rho=rho)
        assert got == pytest.approx(1.0 / (2.0 * (1.0 - rho)), rel=1e-12)
        assert got > 0.5


def test_bridge_is_not_identity_when_r_uses_a_different_reference():
    """H-c 破：r 的分母若来自另一个参考（imaging ≠ clean truth），两边不同口径。

    最干净的形式：真 r² = a = 0，但 `rmse_input` 量的是对**另一个**参考 X 的误差，
    于是报告的 r² 多了一个参考自身的失配项 δ。
    """
    a, delta = 0.0, 0.5
    got, _ = n2n_actual(a=a)                # 实际 n2n = 0.5
    assert abs(bridge(a + delta) - got) > 0.05


def test_monte_carlo_agrees_with_the_analytic_bridge():
    """用仿真复核解析式（说明上面几条不是推导错误）。"""
    rng = np.random.default_rng(0)
    n = 400_000
    t = rng.standard_normal(n)
    cases = [dict(a=0.0), dict(a=0.25), dict(a=1.0),
             dict(b1=0.4), dict(b1=0.3, b2=0.3),
             dict(s2=2.0), dict(rho=0.3), dict(s2=0.5, rho=0.2, a=0.2)]
    for c in cases:
        a, s2, rho = c.get("a", 0.0), c.get("s2", 1.0), c.get("rho", 0.0)
        b1, b2 = c.get("b1", 0.0), c.get("b2", 0.0)
        n1 = rng.standard_normal(n)
        n2z = rng.standard_normal(n)
        n2 = math.sqrt(s2) * (rho * n1 + math.sqrt(max(1 - rho * rho, 0)) * n2z)
        obs1, obs2 = t + b1 + n1, t + b2 + n2
        pred = t + math.sqrt(a) * rng.standard_normal(n)
        got = np.mean((pred - obs2) ** 2) / np.mean((obs1 - obs2) ** 2)
        r2_rep = np.mean((pred - t) ** 2) / np.mean((obs1 - t) ** 2)
        want, r2 = n2n_actual(a=a, b1=b1, b2=b2, s2=s2, rho=rho)
        assert got == pytest.approx(want, abs=0.03)
        assert r2_rep == pytest.approx(r2, abs=0.03)


def test_bridge_predicts_a_lower_bound_floor():
    """用法边界：桥给出的下界是 0.5，所以 n2n 读数 <0.5 一定来自噪声相关或口径不一致。

    这正是 vlow 臂 n2n≈0.80 与 oracle≈0.50 之间那段空间的含义 —— 也说明
    "把 n2n 再压低"必须靠 r，而不能靠分母（分母是数据定的）。
    """
    assert bridge(0.0) == 0.5
    assert bridge(0.1) < 0.6
    assert bridge(1.0) == 1.0                     # r=1（和输入一样差）⇔ 打平 identity
    for r2 in (0.0, 0.04, 0.16, 0.36, 1.0):
        assert bridge(r2) == pytest.approx(0.5 * (1.0 + r2))


def test_cells_flags_a_unit_whose_two_columns_are_not_comparable(tmp_path):
    """检查脚本的实现边界：`budget_checkpoints` 之类的非体积键被跳过，而口径不一致的单元
    会被偏差量**抓出来**（不是静默通过）。"""
    import json

    payload = {
        "controls": {"input": {"v1": {"rmse_vs_imaging": 2.0}, "budget_checkpoints": {}}},
        "arms": {"arm1": {"per_volume": {
            "v1": {"n2n_ratio": 1.05, "rmse_vs_imaging": 2.0},
            "budget_checkpoints": {"n2n_ratio": 9.9},
        }}},
    }
    p = tmp_path / "m.json"
    p.write_text(json.dumps(payload))
    rows = list(cells(p))
    assert len(rows) == 1, rows
    arm, vol, n2n, r = rows[0]
    assert (arm, vol) == ("arm1", "v1")
    assert r == pytest.approx(1.0)                  # r=1 ⇒ 桥预言 1.0
    assert bridge(r * r) == pytest.approx(1.0)
    assert abs(bridge(r * r) - n2n) > 0.05          # 该单元会被记为偏差（口径可疑）

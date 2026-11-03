#!/usr/bin/env python3
"""实验 B 四条门槛的回归测试（合成数字的已知答案校准）。

重点覆盖三种最容易判错的边界：
* 差值小于种子极差 → **不能**说"可分辨"；
* 只赢一头（n2n 最低但 r_roi 不是最高）→ **无胜者**；
* 三个表示都 ≥ 1.0 → T-B3 必须报"表示不是瓶颈"（不许被用来救实验 A）。

Run with:  python tests/test_v3_expB_gate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_expB_gate as B  # noqa: E402


def make(reps, seeds=("3407", "3408"), vols=("s30", "s31")):
    """reps = {rep: {(vol, seed_index): (n2n, r_roi, mass, spearman, low3share)}}"""
    rec = {"arms": {}}
    for rep, cells in reps.items():
        for si, sd in enumerate(seeds):
            per = {}
            for v in vols:
                n2n, rr, mass, spr, low3 = cells[(v, si)]
                per[v] = {
                    "n2n_ratio_lin": n2n, "r_roi": rr,
                    "res_mass_frac_roi": mass, "spearman_abs_res_vs_input": spr,
                    "residual_by_decile": {"share": [low3 / 3] * 3 + [0.0] * 7},
                }
            rec["arms"][f"B_{rep}_s{sd}"] = {"per_volume": per}
    return rec


def rep_cells(n2n, rr, mass=0.20, spr=0.40, low3=0.18, jitter=(0.0, 0.0)):
    d = {}
    for si in (0, 1):
        for v in ("s30", "s31"):
            d[(v, si)] = (n2n + jitter[si], rr, mass, spr, low3)
    return d


def three(log1p_best=True):
    if log1p_best:
        return {"linear": rep_cells(1.20, 0.70), "log1p": rep_cells(0.80, 0.85),
                "logeps": rep_cells(1.00, 0.75)}
    return {"linear": rep_cells(0.80, 0.85), "log1p": rep_cells(1.20, 0.70),
            "logeps": rep_cells(1.00, 0.75)}


# ------------------------------------------------------------------ T-B1
def test_tb1_separable_when_delta_exceeds_seed_spread():
    arms = B.load_arms(make(three()))
    r = B.tb1(arms)
    assert r["status"] == "已判定"
    p = r["pairs"]["linear_vs_log1p"]
    assert p[B.PRIMARY]["separable"] is True
    assert p[B.PRIMARY]["same_sign"] is True


def test_tb1_not_separable_when_delta_smaller_than_spread():
    reps = three()
    # 两臂均值只差 0.03，但 log1p 的种子极差是 0.10 → 分不出高下
    reps["linear"] = rep_cells(0.83, 0.80)
    reps["log1p"] = rep_cells(0.78, 0.85, jitter=(0.0, 0.10))
    arms = B.load_arms(make(reps))
    p = B.tb1(arms)["pairs"]["linear_vs_log1p"][B.PRIMARY]
    assert p["seed_spread"] > 0.05, p
    assert abs(p["deltas"]["s30"]) < p["seed_spread"], p
    assert p["separable"] is False, p


def test_tb1_not_separable_when_signs_disagree_between_volumes():
    reps = three()
    # 让 log1p 在 s31 上反而更差 → 跨体反号
    reps["log1p"][("s31", 0)] = (1.5, 0.60, 0.2, 0.4, 0.18)
    reps["log1p"][("s31", 1)] = (1.5, 0.60, 0.2, 0.4, 0.18)
    p = B.tb1(B.load_arms(make(reps)))["pairs"]["linear_vs_log1p"][B.PRIMARY]
    assert p["same_sign"] is False and p["separable"] is False


# ------------------------------------------------------------------ T-B2
def test_tb2_winner_when_one_rep_wins_both_endpoints():
    r = B.tb2(B.load_arms(make(three())))
    assert r["status"] == "已判定"
    assert r["winner"] == "log1p", r


def test_tb2_no_winner_when_endpoints_disagree():
    reps = three()
    # linear: n2n 最低但 r_roi 不是最高
    reps["linear"] = rep_cells(0.60, 0.70)
    reps["log1p"] = rep_cells(0.90, 0.90)
    reps["logeps"] = rep_cells(1.10, 0.80)
    r = B.tb2(B.load_arms(make(reps)))
    assert r["winner"] is None
    assert "取舍" in r["verdict"]


# ------------------------------------------------------------------ T-B3
def test_tb3_all_above_one_means_representation_is_not_the_bottleneck():
    reps = {"linear": rep_cells(1.4, 0.6), "log1p": rep_cells(1.2, 0.7),
            "logeps": rep_cells(1.1, 0.65)}
    r = B.tb3(B.load_arms(make(reps)))
    assert r["all_ge_1"] is True
    assert "表示不是瓶颈" in r["verdict"]


def test_tb3_flags_when_some_rep_goes_below_one():
    r = B.tb3(B.load_arms(make(three())))
    assert r["all_ge_1"] is False


# ------------------------------------------------------------------ T-B4
def test_tb4_hypotheses_are_reported_with_falsification():
    reps = {
        "linear": rep_cells(1.2, 0.70, mass=0.30, spr=0.60, low3=0.10),
        "log1p": rep_cells(0.8, 0.85, mass=0.20, spr=0.30, low3=0.30),
        "logeps": rep_cells(1.0, 0.75, mass=0.22, spr=0.35, low3=0.25),
    }
    r = B.tb4(B.load_arms(make(reps)))
    h = r["hypotheses"]
    assert h["H1_linear_highest_res_mass_roi"]["holds"] is True
    assert h["H2_linear_lowest_darkest3_share"]["holds"] is True
    assert h["H3_linear_largest_spearman"]["holds"] is True
    assert r["all_hold"] is True


def test_tb4_records_failed_hypothesis_instead_of_spinning_it():
    reps = {
        "linear": rep_cells(1.2, 0.70, mass=0.10, spr=0.20, low3=0.40),
        "log1p": rep_cells(0.8, 0.85, mass=0.20, spr=0.30, low3=0.30),
        "logeps": rep_cells(1.0, 0.75, mass=0.22, spr=0.35, low3=0.25),
    }
    r = B.tb4(B.load_arms(make(reps)))
    assert r["hypotheses"]["H1_linear_highest_res_mass_roi"]["holds"] is False
    assert r["all_hold"] is False


# ------------------------------------------------------------------ 未运行
def test_missing_metrics_file_is_not_run(tmp_path):
    rec = {}
    assert not (tmp_path / "x.json").exists()
    assert B.load_arms({}) == {}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""实验 C 判据的回归测试：T-C0（覆盖率/坐标前提）与 T-C1/C2/C3 的比较逻辑。

用**合成数字**做已知答案校准：判据必须在"先验确实更好"时通过、在"只差半点"时不通过、
在只有一个种子时明确报"部分可判"，不许把不可判读成通过。

Run with:  python tests/test_v3_expC_gate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_expC_gate as C  # noqa: E402


def metrics(arms, seeds=("3407", "3408"), vols=("v2_eval_s30", "v2_eval_s31")):
    """arms = {arm: {vol: [(n2n, r_roi), ...]}} → 与 v3_expA_v2_eval 同形的 JSON。"""
    rec = {"arms": {}}
    for arm, per_vol in arms.items():
        for si, sd in enumerate(seeds):
            per = {}
            for v in vols:
                seq = per_vol[v]
                n2n, r = seq[min(si, len(seq) - 1)]
                per[v] = {"n2n_ratio": n2n, "r_roi": r}
            rec["arms"][f"{arm}_s{sd}"] = {"per_volume": per}
    return rec


# --------------------------------------------------------------------- T-C0
def test_tc0_coverage_is_identical_across_fixed_arms():
    f = C.coverage_facts()
    assert f["T-C0"] == "通过", f["T-C0_text"]
    assert f["same_union_frac"] is True
    fracs = {a: v["union_frac"] for a, v in f["per_arm"].items()}
    assert set(fracs.values()) == {0.5}, fracs


def test_tc0_role_swap_uses_same_coordinates_but_wrong_iso_does_not():
    f = C.coverage_facts()
    sc = f["same_coordinate_set"]
    assert sc["C_correct_vs_C_role_swap"] is True, "role swap 必须用完全相同的 4 个体素"
    assert sc["C_correct_vs_C_wrong_iso"] is False, "wrong_iso 是另一组相位，坐标集合不同"


def test_tc0_records_the_prereg_number_discrepancy():
    """prereg 写 12.5%，代码实测"单视图 12.5% / 四视图并集 50%"，这个更正必须写进 JSON。"""
    f = C.coverage_facts()
    assert abs(f["per_arm"]["C_correct"]["per_view_frac"] - 0.125) < 1e-12
    assert "12.5%" in f["prereg_number_note"] and "50%" in f["prereg_number_note"]


# --------------------------------------------------------------------- T-C1
def _better(better_by=0.05, r_better=0.03):
    return {
        "C_correct": {"v2_eval_s30": [(0.60, 0.90), (0.62, 0.89)],
                      "v2_eval_s31": [(0.65, 0.88), (0.66, 0.87)]},
        "C_role_swap": {"v2_eval_s30": [(0.60 + better_by, 0.90 - r_better),
                                        (0.62 + better_by, 0.89 - r_better)],
                        "v2_eval_s31": [(0.65 + better_by, 0.88 - r_better),
                                        (0.66 + better_by, 0.87 - r_better)]},
    }


def test_tc1_passes_when_both_endpoints_and_both_volumes_agree():
    pa = C._per_arm(metrics(_better()), [])
    r = C.judge("T-C1", ("C_correct", "C_role_swap"), pa)
    assert r["verdict"] == "通过", r


def test_tc1_fails_when_effect_is_smaller_than_seed_spread():
    arms = _better(better_by=0.005, r_better=0.002)
    # 把种子差异拉大：极差 0.05 > 差值 0.005
    arms["C_role_swap"]["v2_eval_s30"] = [(0.50, 0.95), (0.75, 0.70)]
    arms["C_role_swap"]["v2_eval_s31"] = [(0.50, 0.95), (0.75, 0.70)]
    pa = C._per_arm(metrics(arms), [])
    r = C.judge("T-C1", ("C_correct", "C_role_swap"), pa)
    assert r["verdict"] == "不成立", r
    assert "没有可测增益" in r["verdict_text"]


def test_tc1_fails_when_sign_flips_between_volumes():
    arms = _better()
    # 第二个体上反转：C_correct 反而更差
    arms["C_correct"]["v2_eval_s31"] = [(0.80, 0.80), (0.81, 0.79)]
    arms["C_role_swap"]["v2_eval_s31"] = [(0.60, 0.90), (0.61, 0.89)]
    pa = C._per_arm(metrics(arms), [])
    r = C.judge("T-C1", ("C_correct", "C_role_swap"), pa)
    assert r["verdict"] == "不成立", r


def test_tc1_partial_when_only_one_seed():
    pa = C._per_arm(metrics(_better(), seeds=("3407",)), [])
    r = C.judge("T-C1", ("C_correct", "C_role_swap"), pa)
    assert r["verdict"] == "部分可判", r
    assert "种子极差" in r["verdict_text"]


def test_tc2_and_tc3_report_not_run_when_arm_missing():
    arms = _better()
    pa = C._per_arm(metrics(arms), [])
    assert C.judge("T-C2", ("C_correct", "C_random"), pa)["status"] == "未运行"
    assert C.judge("T-C3", ("C_correct", "C_wrong_iso"), pa)["status"] == "未运行"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))

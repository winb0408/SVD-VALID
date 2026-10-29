#!/usr/bin/env python3
"""实验 A 两道门的回归测试：用**合成数字**验证判定方向正确（已知答案校准）。

为什么必须测：门是可宣布"通过/未通过"的东西，一旦判反，后面所有实验的解冻决定都错。
本测试覆盖：T1 三种结局、T2 三种结局、zero 归因规则、以及"文件不存在 ⇒ 未运行（不猜）"。

Run with:  python tests/test_v3_expA_v2_gate.py   或   pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v3_expA_v2_gate as G  # noqa: E402


# --------------------------------------------------------------------------- 合成输入
def stage1(arms):
    """arms = {'A_rand': [(r_ves, n2n), ...]}（每个 seed 一格）"""
    return {
        "baseline": {"input": {"mean": {"r_ves": 0.9456, "n2n_ratio": 1.0}}},
        "arms": {a: {str(3407 + i): {"r_ves": rv, "n2n_ratio": n2n}
                     for i, (rv, n2n) in enumerate(v)}
                 for a, v in arms.items()},
    }


def stage2(arms, zero=0.5, input_r=0.80):
    """arms = {'A_rand': [(n2n, r_roi), ...]}（每格一个 seed×体）"""
    vols = ["v2_eval_s30", "v2_eval_s31"]
    rec = {"controls": {"input": {v: {"n2n_ratio": 1.0, "r_roi": input_r} for v in vols},
                        "zero": {v: {"n2n_ratio": zero, "r_roi": 0.10} for v in vols}},
           "arms": {}}
    for a, cells in arms.items():
        per = {}
        for i, (n2n, r) in enumerate(cells):
            per[vols[i % len(vols)] + (f"_r{i}" if i >= len(vols) else "")] = {
                "n2n_ratio": n2n, "r_roi": r}
        rec["arms"][f"{a}_s3407"] = {"per_volume": per}
    return rec


# --------------------------------------------------------------------------- T1
def test_t1_pass_requires_both_arms_beat_input():
    j = stage1({"A_rand": [(0.96, 0.90)], "A_fixed4": [(0.97, 0.85)]})
    r = G.check_t1(j)
    assert r["verdict"] == "T1-通过", r
    assert "降级" in r["verdict_text"]


def test_t1_fail_when_both_arms_lose_and_noise_rises():
    j = stage1({"A_rand": [(0.90, 1.30)], "A_fixed4": [(0.88, 1.10)]})
    r = G.check_t1(j)
    assert r["verdict"] == "T1-未通过", r


def test_t1_undetermined_when_mixed_and_it_does_not_pick_the_good_cell():
    # 一条臂过、一条臂过门槛但噪声变差 → 必须"未定"，不许挑好的那条
    j = stage1({"A_rand": [(0.99, 1.20)], "A_fixed4": [(0.90, 1.50)]})
    r = G.check_t1(j)
    assert r["verdict"] == "T1-未定", r
    assert "不挑选" in r["verdict_text"]


def test_t1_flags_literal_threshold_mismatch():
    j = stage1({"A_rand": [(0.96, 0.9)], "A_fixed4": [(0.96, 0.9)]})
    j["baseline"]["input"]["mean"]["r_ves"] = 0.9300       # 与预注册 0.9456 不一致
    r = G.check_t1(j)
    assert r["threshold_literal_check"]["input_r_ves_matches_prereg"] is False


# --------------------------------------------------------------------------- T2
def test_t2_pass_when_both_arms_beat_input_and_zero_loses():
    j = stage2({"A_rand": [(0.85, 0.90), (0.88, 0.86)],
                "A_fixed4": [(0.90, 0.87), (0.92, 0.85)]}, zero=2.5)
    r = G.check_t2(j)
    assert r["verdict"] == "T2-通过", r
    assert r["zero_control"]["zero_still_degenerate"] is False


def test_t2_fail_allowed_only_when_zero_control_is_not_degenerate():
    j = stage2({"A_rand": [(1.20, 0.70), (1.10, 0.75)],
                "A_fixed4": [(1.30, 0.60), (1.15, 0.72)]}, zero=2.5)
    r = G.check_t2(j)
    assert r["verdict"] == "T2-未通过"
    assert "可以归因于方法" in r["attribution"]


def test_t2_fail_is_unattributable_if_zero_still_wins():
    # zero 的 n2n=0.5 优于输入的 1.0 → 判据仍退化 → 失败不得归因于方法
    j = stage2({"A_rand": [(1.20, 0.70), (1.10, 0.75)],
                "A_fixed4": [(1.30, 0.60), (1.15, 0.72)]}, zero=0.5)
    r = G.check_t2(j)
    assert r["verdict"] == "T2-未通过"
    assert r["zero_control"]["zero_still_degenerate"] is True
    assert "不可归因于方法" in r["attribution"]


def test_t2_reports_arm_mean_when_strict_fails():
    # 一格过、一格不过 → strict 不过；同时要报出"按体求均值"这种更宽松口径下的结果，供口径敏感性核对
    j = stage2({"A_rand": [(0.90, 0.90), (1.30, 0.60)],
                "A_fixed4": [(1.20, 0.65), (1.20, 0.65)]}, zero=2.5)
    r = G.check_t2(j)
    a = r["per_arm"]["A_rand"]
    assert a["strict_pass"] is False
    assert "arm_mean_pass" in a


# --------------------------------------------------------------------------- 未运行
def test_missing_files_are_reported_as_not_run(tmp_path):
    rec = {}
    assert G._load(str(tmp_path / "nope.json"), "k", rec) is None
    assert rec["k"]["status"] == "未运行"
    assert "不猜测" in rec["k"]["why"]


# --------------------------------------------------------------------------- 单调性
def test_monotonicity_reports_decreasing_r_ves():
    a = stage1({"A_rand": [(0.93, 1.1)], "A_fixed4": [(0.92, 1.2)]})
    b = stage1({"A_rand": [(0.91, 1.2)], "A_fixed4": [(0.90, 1.3)]})
    pts = [{"label": "4k", "path": "a", "json": a, "loss": {"epoch": 10, "loss": 0.02}},
           {"label": "200k", "path": "b", "json": b, "loss": {"epoch": 200, "loss": 0.015}}]
    rec = {}
    m = G.monotonicity(pts, rec)
    assert m["trend"]["A_rand"]["decreasing"] is True
    assert m["trend"]["A_rand"]["sequence"][-1][1] == pytest.approx(0.91)


def test_parse_loss_reads_last_epoch(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("epoch=001 loss=0.050\nepoch=099 loss=0.016\nepochs: 50%|xx\n")
    assert G.parse_loss(p) == {"epoch": 99, "loss": 0.016}
    assert G.parse_loss(tmp_path / "missing.log") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_zero_nan_r_roi_is_recorded_not_silently_ignored():
    """全零输出的 r_roi 是 0/0。nan 不能被静默当作"没赢"，必须写进 JSON 供核对。"""
    j = stage2({"A_rand": [(0.90, 0.85), (0.95, 0.85)],
                "A_fixed4": [(0.92, 0.84), (0.93, 0.83)]}, zero=300.0)
    for v in j["controls"]["zero"]:
        j["controls"]["zero"][v]["r_roi"] = float("nan")
    r = G.check_t2(j)
    assert r["verdict"] == "T2-通过", r
    for vol, row in r["zero_control"]["rows"].items():
        assert "r_roi" in row["nan_keys"], row
        assert row["nan_treated_as"] and "not_beating" in row["nan_treated_as"]

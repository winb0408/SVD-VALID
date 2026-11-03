#!/usr/bin/env python3
"""R03 回归测试：采样器配置的冻结与校验必须真的能拦住漂移。

这里测的不是"配置里有几个字段"，而是**校验函数本身有没有牙齿**：
把配置里的任何一类冻结值改掉、或把磁盘上的 ρ 曲线/角色序/有效区掩膜换掉，
`check()` 都必���报出来。如果哪天有人把校验写成恒返回空列表，这组测试会失败。

Run with:  python tests/test_v3_sampler_config.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import v3_sampler_config as sc  # noqa: E402


def _cfg():
    return json.loads(sc.CONFIG.read_text())


def test_config_exists_and_passes_check():
    assert sc.CONFIG.exists(), "先跑 python scripts/v3_sampler_config.py --build"
    assert sc.check() == [], f"配置与磁盘产物不一致：{sc.check()}"


def test_band_and_psf_drift_are_caught():
    cfg = _cfg()
    cfg["band"]["modes"] = 290  # 旧口径（少一个模式）
    assert sc.check(cfg), "秩带被改成旧的 290 模式却没报错"
    cfg = _cfg()
    cfg["imaging_constants"]["psf_sigma_zyx"] = [1.0, 1.0, 1.0]
    assert sc.check(cfg), "PSF σ 改变了却没报错（oracle 对照会偷步）"


def test_pattern_drift_is_caught():
    cfg = _cfg()
    cfg["geometry"]["tetris_patterns"][0] = [0, 1, 2, 3]
    assert sc.check(cfg), "tet 角点表被改了却没报错"
    cfg = _cfg()
    cfg["geometry"]["complement_pattern"] = list(range(8))
    assert sc.check(cfg), "互补表被改了却没报错"


def test_version_drift_is_caught():
    cfg = _cfg()
    cfg["sampler_config_version"] = "R03-v0"
    assert sc.check(cfg), "版本号不一致却没报错"


def test_prior_file_swap_is_caught(monkeypatch, tmp_path):
    """换掉 ρ 曲线文件（内容不同、路径随便）必须被 sha1 抓住。"""
    fake = tmp_path / "corr.json"
    j = json.loads(sc.CORR.read_text())
    j["SA64"]["P1"]["acf"]["z"] = 0.0  # 任何一处改动都算换先验
    fake.write_text(json.dumps(j))
    monkeypatch.setattr(sc, "CORR", fake)
    bad = sc.check()
    assert any("v3_correlation_length" in b for b in bad), bad


def test_valid_mask_swap_is_caught(monkeypatch, tmp_path):
    fake = tmp_path / "valid_mask.npy"
    np.save(fake, np.ones((37, 64, 64), bool))  # 假设"全有效"，与官方扇形成像足迹不符
    monkeypatch.setattr(sc, "VALID_SA64", fake)
    bad = sc.check()
    assert any("valid_mask" in b for b in bad), bad


def test_sim_valid_mask_is_not_official():
    """仿真的有效区**不是**官方有效区，这一点必须留在配置里，不能被悄悄抹平。"""
    vm = _cfg()["valid_mask"]
    cov_sim = vm["sim"]["coverage"]
    cov_real = vm["official_SA64"]["coverage"]
    assert abs(cov_sim - cov_real) > 0.05, "两者覆盖应当明显不同（0.919 vs 0.820）"
    assert vm["sim_vs_official_SA64"]["iou"] < 0.95
    assert "不等于" in vm["sim_vs_official_SA64"]["disclaimer"]


def test_negative_controls_are_declared_not_run():
    """R03 要的错误先验负控制属实验 C：配置里必须写明「只有规格、未运行」。"""
    nc = _cfg()["negative_controls"]
    assert nc["status"] == "spec_frozen_not_run"
    names = {a["name"] for a in nc["arms"]}
    assert {"correct_prior", "wrong_role_swap", "wrong_prior_isotropic",
            "wrong_prior_axis_permuted", "random_cross"} <= names
    swap = [a for a in nc["arms"] if a["name"] == "wrong_role_swap"][0]
    assert "角色" in swap["isolates"]
    # 轴错位这条臂已事前证伪为空操作，必须保留这个标注而不是悄悄删掉
    perm = [a for a in nc["arms"] if a["name"] == "wrong_prior_axis_permuted"][0]
    assert perm["status"] == "void_undetectable" and "映射到自身" in perm["why_void"]


if __name__ == "__main__":
    import inspect
    import traceback

    class _MP:
        """极简 monkeypatch（本仓库测试直接 python tests/xxx.py 运行，没有 pytest fixture）。"""

        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)
            self._undo.clear()

    import tempfile

    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    ok = 0
    for f in fns:
        mp, tmp = _MP(), Path(tempfile.mkdtemp())
        try:
            kwargs = {}
            for p in inspect.signature(f).parameters:
                kwargs[p] = mp if p == "monkeypatch" else tmp
            f(**kwargs)
            print(f"PASS {f.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL {f.__name__}")
            traceback.print_exc()
        finally:
            mp.undo()
    print(f"\n{ok}/{len(fns)} passed")
    raise SystemExit(0 if ok == len(fns) else 1)

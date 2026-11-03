#!/usr/bin/env python3
"""S0 回归测试：钉住本轮**踩过的坑**，而不是钉住好看的数。

三条都是"出错时不会报错、只会给出看起来合理的数"的地方：

1. `--mode dense` 不配 `--train-clean-volumes` 时必须**报错退出**——否则输入=目标=同一张图，
   训练只会学恒等，而 n2n/`r_roi` 会给出漂亮但**平凡**的分数。
   （2026-10-27 加入 dense 诊断臂时同步加的牙齿。）
2. 干净目标必须在 `valid==False` 处置 0：第一版漏了这一步，`std(clean - log1p(gt_imaging))`
   是 0.2098 而不是 0.0204，训练已启动才发现，只能杀掉重跑。
3. 噪声档必须满足 low < mid < high 且与 manifest 里的 `sigma_eps` 一致——
   档位标签写错会让"噪声阶梯"这个结论整个失效。

Run with:  python -m pytest tests/test_v3_round2_s0.py -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATA = ROOT / "data/processed/v3_round2_s0"


def test_dense_mode_without_clean_targets_is_rejected():
    """dense 模式缺干净目标 ⇒ 必须失败，不能悄悄学恒等。"""
    log1p = sorted((DATA / "low/train").glob("*_log1p.npy"))
    if not log1p:
        pytest.skip("S0 数据未生成")
    r = subprocess.run(
        [PY, "train.py", "--train-volumes", str(log1p[0]), "--mode", "dense",
         "--epochs", "1", "--steps-per-epoch", "1", "--patch", "4", "8", "8",
         "--out", "/tmp/_s0_test_should_not_exist"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode != 0, "dense 模式没有干净目标却成功启动了"
    assert "只能学到恒等" in (r.stderr + r.stdout), r.stderr[-2000:]


def test_clean_targets_are_masked_outside_valid():
    """干净目标在 valid==False 处必须为 0（与 log1p 输入的口径一致）。"""
    npz = sorted((DATA / "low/train").glob("*.npz"))
    npy = sorted((DATA / "low/train").glob("*_clean.npy"))
    if not npz or not npy:
        pytest.skip("S0 数据未生成")
    d = np.load(npz[0])
    valid = d["valid"].astype(bool)
    s = float(d["s"])
    clean = np.load(npy[0]).astype(np.float32)
    expect = np.log1p(np.maximum(d["gt_imaging_pd"].astype(np.float32), 0.0) / max(s, 1e-12))
    assert np.all(clean[~valid] == 0.0), "valid==False 处必须恰为 0"
    assert np.allclose(clean[valid], expect[valid], atol=1e-6), "valid 处必须等于 log1p(gt_imaging/s)"


def test_noise_levels_are_ordered_and_match_manifest():
    sig = {}
    for level in ("low", "mid", "high"):
        p = DATA / level / "manifest.json"
        if not p.exists():
            pytest.skip(f"{level} 档未生成")
        sig[level] = float(json.loads(p.read_text())["sigma_eps"])
    assert sig["low"] < sig["mid"] < sig["high"], sig
    # 档位是按 2 倍递增造的，允许 1% 的生成误差
    assert 1.9 < sig["mid"] / sig["low"] < 2.1, sig
    assert 1.9 < sig["high"] / sig["mid"] < 2.1, sig

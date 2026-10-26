#!/usr/bin/env python3
"""R13 回归测试：官方血带口径是 1-based 闭区间 60:350 = 0-based [59,350) = 291 个模式。

Run with:  python tests/test_v3_band_convention.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from v3_phase5_real_compare import band_slice  # noqa: E402


def test_official_band_is_291_modes_at_T400():
    lo, hi = band_slice(400, official=True)
    assert (lo, hi) == (59, 350), f"官方口径应为 (59,350)，实测 {(lo, hi)}"
    assert hi - lo == 291, "官方 MATLAB 60:350 含 291 个模式"


def test_legacy_band_was_one_mode_short():
    lo, hi = band_slice(400, official=False)
    assert (lo, hi) == (60, 350)
    assert hi - lo == 290, "2026-10-26 之前的旧口径少一个模式（R13 指出的正是这个）"


def test_half_volume_band_scales_proportionally():
    lo, hi = band_slice(200, official=True)
    assert (lo, hi) == (29, 175), f"半卷官方口径应为 (29,175)，实测 {(lo, hi)}"
    assert hi - lo == 146


def test_default_is_official():
    assert band_slice(400) == (59, 350), "默认必须走官方口径"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1; print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if fails else 0)

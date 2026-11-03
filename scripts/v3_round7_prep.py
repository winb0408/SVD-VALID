#!/usr/bin/env python3
"""Round7 数据准备：四列终稿对照需要的四个体（**不训练**）。

为什么这样做（改动原因）
------------------------
终稿只要四列：原始带噪 / 普通 SVD / SVD-SN2N / 本方法。四列的定义冻结在
`reviews/V3_PR1_round7_prereg_final_comparison.md` §1：

    C1 原始带噪 : PD_raw = Σ_t |IQ_t|²  （全部 400 帧，**不做 SVD 杂波抑制**）
    C2 普通 SVD : 官方血流带（1-based 60:350 = 0-based [59,350)）SVD 带通 → PD → dB
    C3 / C4 的输入都是 C2 的对数表示。

C1 必须单独算：`data/processed/official/*/pd_linear.npy` 已经是**带内**的 PD，
拿它当"原始带噪"会把杂波抑制这一步藏起来，正是上一轮四列图说不清的地方。
C2 直接复用 `official/` 下已核过官方索引的产物（不再重算，避免出现第二套口径）。

产物
----
    data/processed/v3_round7/{seq}/pd_raw_linear.npy   C1 线性 PD
    data/processed/v3_round7/{seq}/pd_raw_db.npy       C1 dB（10log10(PD/max)）
    data/processed/v3_round7/{seq}/pd_svd_linear.npy   C2 线性 PD（= official/pd_linear.npy）
    data/processed/v3_round7/{seq}/pd_svd_db.npy       C2 dB
    data/processed/v3_round7/{seq}/valid_mask.npy
    data/processed/v3_round7/metadata.json

用法
----
    python -u scripts/v3_round7_prep.py --seqs SA64 HSA64
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.iq_io import load_iq                                    # noqa: E402
from official_pipeline import SEQUENCE_SETTINGS                       # noqa: E402

OUT_ROOT = ROOT / "data/processed/v3_round7"
OFFICIAL = ROOT / "data/processed/official"


def raw_power_doppler(seq: str) -> np.ndarray:
    """未做杂波抑制的 PD：把全部帧的复数 IQ 直接按时积分。"""
    st = SEQUENCE_SETTINGS[seq]
    x = load_iq(st["path"] if "path" in st else str(Path(SOURCES[seq])), st["variable"])
    if st["sum_arrays"] and x.ndim == 5:
        x = np.squeeze(x.sum(axis=-1))
    if x.ndim != 4:
        raise ValueError(f"{seq}: expected [Z,X,Y,T], got {x.shape}")
    return np.sum(np.abs(x) ** 2, axis=3).astype(np.float32)


SOURCES = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}


DB_FLOOR = -60.0   # 扇区外体素 PD 恒为 0 ⇒ dB 是 -inf；统一压到这个地板值


def to_db(pd: np.ndarray) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(pd, 1e-12) / pd.max())
    return np.maximum(np.nan_to_num(db, nan=DB_FLOOR, neginf=DB_FLOOR), DB_FLOOR).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    meta = {"definition": "reviews/V3_PR1_round7_prereg_final_comparison.md §1", "seqs": {}}

    for seq in args.seqs:
        out = out_root / seq
        out.mkdir(parents=True, exist_ok=True)

        pd_raw = raw_power_doppler(seq)
        np.save(out / "pd_raw_linear.npy", pd_raw)
        np.save(out / "pd_raw_db.npy", to_db(pd_raw))

        src = OFFICIAL / seq
        shutil.copyfile(src / "pd_linear.npy", out / "pd_svd_linear.npy")
        # official/pd_db.npy 含 -inf（扇区外 PD=0），直接喂网络会出 NaN；这里统一压地板
        np.save(out / "pd_svd_db.npy", to_db(np.load(src / "pd_linear.npy")))
        shutil.copyfile(src / "valid_mask.npy", out / "valid_mask.npy")

        pd_svd = np.load(out / "pd_svd_linear.npy")
        valid = np.load(out / "valid_mask.npy")
        meta["seqs"][seq] = {
            "shape_zyx": list(pd_raw.shape),
            "raw_max": float(pd_raw.max()),
            "svd_max": float(pd_svd.max()),
            "valid_voxels": int(valid.sum()),
            "raw_to_svd_energy_ratio": float(pd_raw.sum() / max(pd_svd.sum(), 1e-12)),
        }
        print(f"[{seq}] raw PD {pd_raw.shape} max={pd_raw.max():.4g} | "
              f"SVD PD max={pd_svd.max():.4g} | valid={int(valid.sum())}")

    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_root/'metadata.json'}")


if __name__ == "__main__":
    main()

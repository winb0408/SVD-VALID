#!/usr/bin/env python3
"""Round7 C4（本方法）的数据准备：全帧 PD 的 logeps 域张量（**不训练**）。

与 Round6 的唯一区别是**用全部 400 帧**（`pd_odd + pd_even` 的那一整卷），
因为终稿的用法就是"输入一笔带噪数据、输出一笔完整去噪数据"，不是拆半评估。
尺度锚点 `s` 与标准化常数沿用 Round6/B1 冻结值（`data/processed/v3_round6/<seq>/metadata.json`
的 `s`、`reports/v3_round4/b1_norm.json`），保证与已有结果同一尺度。

产物
----
    data/processed/v3_round7_b1/logeps/{seq}_full_x.npy   （训练体）
    data/processed/v3_round7_b1/logeps/{seq}_full_y.npy   （cross 配对下 train.py 只用 _x）
    reports/v3_round7/provenance.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import fwd                       # noqa: E402

D7 = ROOT / "data/processed/v3_round7"
B1 = ROOT / "data/processed/v3_round7_b1"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())


def main():
    prov = {}
    for seq in ("SA64", "HSA64"):
        pd = np.load(D7 / seq / "pd_svd_linear.npy")
        valid = np.load(D7 / seq / "valid_mask.npy")
        s = float(json.loads((ROOT / f"data/processed/v3_round6/{seq}/metadata.json")
                             .read_text())["s"])
        m, sd = float(NORM["logeps"]["mean"]), float(NORM["logeps"]["std"])
        z = ((fwd("logeps", pd, s, valid) - m) / sd).astype(np.float32)

        out = B1 / "logeps"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{seq}_full_x.npy", z)
        np.save(out / f"{seq}_full_y.npy", z)

        prov[seq] = {"shape": list(z.shape), "s": s, "n_frames": "all (400)",
                     "domain": "logeps = log(PD/s + 1e-3), 用 B1 冻结 mean/std 标准化",
                     "valid_voxels": int(valid.sum()),
                     "x_mean": float(z[valid].mean()), "x_std": float(z[valid].std())}
        print(f"[{seq}] {z.shape} mean={z[valid].mean():.4f} std={z[valid].std():.4f}")

    (ROOT / "reports/v3_round7").mkdir(parents=True, exist_ok=True)
    p = ROOT / "reports/v3_round7/provenance_ours.json"
    p.write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round10 部署臂的数据准备：**全 400 帧** PD 的三域张量（**不训练**）。

为什么需要
----------
量化判定（预注册 §4）必须在留出半上做，所以训练用的是 `v3_round6_b1` 的奇/偶半卷。
但用户要看的是"这笔数据最终去噪成什么样"，那是 400 帧全卷的产物。两者不能混用一套权重
（半卷模型没见过另一半的体素），所以部署图单独训 2 个作业，数据由本脚本产出。

全卷 PD 就是两半之和（`pd_odd + pd_even`，Round6 已验证与官方 `pd_linear.npy` 相对误差 ≤1e-7），
不重新做 SVD 带通 —— 重做会让秩带的物理含义与已有结果漂移。

产物
----
    data/processed/v3_round10_b1/{domain}/{seq}_full_x.npy    训练体（同形状的 _y 只占位）
    data/processed/v3_round10/{seq}/pd_full.npy               全帧线性 PD（画图用）
    reports/v3_round10/provenance_full.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import ARMS, fwd                     # noqa: E402

HALF = ROOT / "data/processed/v3_round6"
OUT = ROOT / "data/processed/v3_round10"
B1 = ROOT / "data/processed/v3_round10_b1"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())


def main() -> int:
    prov = {}
    for seq in ("SA64", "HSA64"):
        d = HALF / seq
        pd_odd = np.load(d / "odd/pd_linear.npy").astype(np.float64)
        pd_even = np.load(d / "even/pd_linear.npy").astype(np.float64)
        pd_full = (pd_odd + pd_even).astype(np.float32)
        valid = np.load(d / "valid_mask.npy").astype(bool)
        s = float(json.loads((d / "metadata.json").read_text())["s"])

        (OUT / seq).mkdir(parents=True, exist_ok=True)
        np.save(OUT / seq / "pd_full.npy", pd_full)

        for arm in ARMS:
            m, sd = float(NORM[arm]["mean"]), float(NORM[arm]["std"])
            z = ((fwd(arm, pd_full, s, valid) - m) / sd).astype(np.float32)
            od = B1 / arm
            od.mkdir(parents=True, exist_ok=True)
            np.save(od / f"{seq}_full_x.npy", z)
            np.save(od / f"{seq}_full_y.npy", z)   # within 配对下 train.py 只用 _x

        prov[seq] = {"s": s, "n_valid": int(valid.sum()),
                     "pd_full_max": float(pd_full.max()),
                     "check_sum_exact": float(np.abs(pd_full.astype(np.float64)
                                                    - (pd_odd + pd_even)).max())}
        print(f"[{seq}] s={s:.6g} valid={int(valid.sum())} "
              f"PD_max={pd_full.max():.4g} 两半求和的数值误差={prov[seq]['check_sum_exact']:.2e}")

    (ROOT / "reports/v3_round10").mkdir(parents=True, exist_ok=True)
    p = ROOT / "reports/v3_round10/provenance_full.json"
    p.write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

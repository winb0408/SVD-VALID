"""S0 归因诊断④配套：生成训练体的**第二次观测** `*_log1p2.npy`（只读不写原产物）。

为什么需要它
------------
`--mode dense` 把训练几何换成满分辨率之后，还剩下一个问题：**满分辨率下的配对从哪来？**
相位抽稀那套"互补视图"是半尺寸的，正是几何不一致的来源。满分辨率下唯一严格独立的
第二观测就是**另一次含噪采集**（同一 clean、独立噪声）。S0 的造数脚本本来就按
`seed+20_000` 生成过第二次观测，只是训练集当时为了省盘没有落盘（`with_second=False`）。

本脚本用**同一套确定性生成**（同 seed、同 rng 派生规则）把这第二次观测算出来，
只写 `*_log1p2.npy`，**不改任何已有文件**；写完核对 `log1p`（第一次观测）与原文件
逐位一致，作为"生成过程没有漂移"的证据。

用途仅限归因诊断臂（`--mode dense --train-clean-volumes <log1p2>`）：
此臂**不在预注册门槛内，不用于任何判定**，只用来回答"满分辨率 + 独立第二观测能否过门"。

用法
----
    python -u scripts/v3_round2_s0_second_view.py --level low
    # 输出 data/processed/v3_round2_s0/low/train/s0_low_s200_log1p2.npy ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round2_s0_dataset import build_one                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    args = ap.parse_args()

    base = ROOT / f"data/processed/v3_round2_s0/{args.level}"
    man = json.loads((base / "manifest.json").read_text())
    dom, corr = man["noise_domain"], man["corr_sigma"]
    if corr is not None:
        corr = tuple(float(v) for v in corr)

    n_ok = 0
    for t in man["train"]:
        d = build_one(int(t["seed"]), args.level, dom, corr, with_second=True)
        s = float(t["s"])
        lg1 = np.log1p(np.maximum(d["pd_linear"], 0.0) / s).astype(np.float32)
        lg2 = np.log1p(np.maximum(d["pd2_linear"], 0.0) / s).astype(np.float32)
        valid = d["valid"].astype(bool)
        ref = np.load(ROOT / t["log1p"])
        # 原文件在 valid==False 处被置 0（与输入表示的口径一致），所以逐位比对要**分区**做：
        # valid 内必须完全相同，valid 外必须恰为 0。
        same = (np.array_equal(ref[valid], lg1[valid]) and np.all(ref[~valid] == 0.0))
        print(f"  s{t['seed']}: 第一次观测与原文件逐位一致={same}")
        if not same:
            raise SystemExit("生成过程漂移：log1p 与原文件不一致，停止（不写任何文件）")
        lg2[~valid] = 0.0
        out = ROOT / t["log1p"].replace("_log1p.npy", "_log1p2.npy")
        np.save(out, lg2)
        # 自检：两次观测之差应当只剩噪声，与 clean 的偏差量级应约等于 sqrt(2)·sigma_eff
        print(f"    std(obs2−obs1)={np.std((lg2 - lg1)[valid]):.4f} "
              f"std(obs1−clean)={np.std((lg1 - np.log1p(np.maximum(d['gt_imaging_pd'], 0) / s))[valid]):.4f}"
              f" → {out.name}")
        n_ok += 1
    print(f"[{args.level}] 写出 {n_ok} 个 log1p2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

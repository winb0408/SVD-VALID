"""Round4-B1 数据准备：把 low 档训练/校准/留出卷转成 three-arm 表示域（**不训练**）。

冻结定义见 `reviews/V3_PR1_round4_prereg_b1_amendment.md` §3：

    u        = PD / s                       （s 取自每卷 npz，三臂共用）
    linear   : x = u                 inverse  PD = s·u
    log1p    : x = log1p(u)          inverse  PD = s·expm1(x)
    logeps   : x = log(u + 1e-3)     inverse  PD = s·(exp(x) - 1e-3)

无效体素填该域的**零信号值**（linear/log1p → 0，logeps → log(1e-3)），
因为"填 0"在 logeps 域反变换后是 `s·(1-1e-3) ≈ s`，会人为造出一块亮区。

标准化：每域用**训练集**所有 valid 体素的一阶/二阶统计量做仿射标准化，
常数写进 `reports/v3_round4/b1_norm.json`，**不在留出集重算**。

产物（每档一套，训练 12 卷 + 校准/留出 6 卷）
    data/processed/v3_round4_b1/<arm>/{train/*.npy, eval/*.npy}
    reports/v3_round4/b1_norm.json

用法
----
    python -u scripts/v3_round4_b1_prep.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-3
LEVEL = "low"
ARMS = ("linear", "log1p", "logeps")


def fwd(arm: str, pd: np.ndarray, s: float, valid: np.ndarray) -> np.ndarray:
    u = np.maximum(pd.astype(np.float64), 0.0) / max(s, 1e-12)
    if arm == "linear":
        x = u
    elif arm == "log1p":
        x = np.log1p(u)
    elif arm == "logeps":
        x = np.log(u + EPS)
    else:
        raise ValueError(arm)
    zero = 0.0 if arm != "logeps" else float(np.log(EPS))
    out = np.where(valid, x, zero).astype(np.float32)
    return out


def inv(arm: str, x: np.ndarray, s: float) -> np.ndarray:
    x = x.astype(np.float64)
    if arm == "linear":
        pd = x
    elif arm == "log1p":
        pd = np.expm1(x)
    else:
        pd = np.exp(x) - EPS
    return np.maximum(pd, 0.0) * max(s, 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default=LEVEL)
    ap.add_argument("--out-root", default="data/processed/v3_round4_b1")
    args = ap.parse_args()

    src = ROOT / f"data/processed/v3_round2_s0/{args.level}"
    train_npz = sorted((src / "train").glob("*.npz"))
    # 校准/留出：e900-e905 在 <level>6/eval/
    evals = sorted((ROOT / f"data/processed/v3_round2_s0/{args.level}6/eval").glob("*.npz"))
    evals = [p for p in evals if any(f"_e{s}.npz" in p.name for s in range(900, 906))]
    print(f"train {len(train_npz)} 卷, eval {len(evals)} 卷")

    outroot = ROOT / args.out_root
    stats = {a: {"n": 0, "sum": 0.0, "sumsq": 0.0} for a in ARMS}
    for p in train_npz:
        d = np.load(p)
        s = float(d["s"])
        valid = d["valid"].astype(bool)
        for arm in ARMS:
            x = fwd(arm, d["pd_linear"], s, valid)
            xv = x[valid].astype(np.float64)
            stats[arm]["n"] += int(xv.size)
            stats[arm]["sum"] += float(xv.sum())
            stats[arm]["sumsq"] += float((xv ** 2).sum())
    norm = {}
    for arm in ARMS:
        n = max(stats[arm]["n"], 1)
        mean = stats[arm]["sum"] / n
        var = max(stats[arm]["sumsq"] / n - mean ** 2, 1e-12)
        norm[arm] = {"mean": float(mean), "std": float(np.sqrt(var)), "n_vox": n,
                     "eps": EPS, "level": args.level}
    # 留证：logeps 与 log1p 不是仿射等价（本次修订的直接理由）
    p0 = train_npz[0]
    d0 = np.load(p0)
    v0 = d0["valid"].astype(bool)
    a = fwd("log1p", d0["pd_linear"], float(d0["s"]), v0)[v0]
    b = fwd("logeps", d0["pd_linear"], float(d0["s"]), v0)[v0]
    norm["degeneracy_check"] = {
        "volume": p0.name,
        "corr_log1p_logeps": float(np.corrcoef(a, b)[0, 1]),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "note": "原预注册的 log(PD·ε/s+ε) 与 log1p(PD/s) 只差常数 log ε ⇒ 标准化后逐位相同；"
                "本修订把 ε 移到分母侧，两臂才真正不同",
    }

    for arm in ARMS:
        m, sd = norm[arm]["mean"], norm[arm]["std"]
        for split, paths, key in (("train", train_npz, "pd_linear"),
                                  ("eval", evals, "pd_linear")):
            od = outroot / arm / split
            od.mkdir(parents=True, exist_ok=True)
            for p in paths:
                d = np.load(p)
                s = float(d["s"])
                valid = d["valid"].astype(bool)
                x = fwd(arm, d[key], s, valid)
                # 目标：同一变换作用于 PSF 匹配的成像真值
                y = fwd(arm, d["gt_imaging_pd"], s, valid)
                z = (x - m) / sd
                zy = (y - m) / sd
                np.save(od / f"{p.stem}_x.npy", z.astype(np.float32))
                np.save(od / f"{p.stem}_y.npy", zy.astype(np.float32))
    p = ROOT / "reports/v3_round4/b1_norm.json"
    p.write_text(json.dumps(norm, indent=1, ensure_ascii=False))
    print(json.dumps(norm, indent=1, ensure_ascii=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

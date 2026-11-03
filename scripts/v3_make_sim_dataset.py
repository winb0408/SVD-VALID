"""生成 R06/v2 的多体仿真训练集与评估集。

为什么需要这个脚本（对应用户给定的整改方向 (ii) "多体仿真训练集"）
--------------------------------------------------------------
实验 A 只用了**一个**仿真体（`data/processed/v3_sim/sim_seed0.npz`）训练，
于是"网络是不是把这一例的血管背下来了"无法与"任务定义本身有问题"分开。
本脚本用同一套 v2 前向模型生成多个**血管几何不同**的体：

* 训练体 `--n-train` 个（默认 16），seed 从 `--train-seed0` 起连号；
* 评估体 `--n-eval` 个（默认 2），seed 从 `--eval-seed0` 起连号，
  与训练体**不重叠**，且每个评估体额外生成一次**独立观测**
  （`noise_seed = seed + 700000`、`static_seed = seed`：固定散斑与解剖相同、
  动态/血流/热噪声独立），用于 n2n 类判据。

每个体都存：
    pd_linear     带通后的观测 PD（SVD 带官方 1-based M(:,60:350)）
    gt_flow_pd    只对血流项做同样带通得到的真值 PD
    vessels / rho 布尔掩膜 / 连续密度（v2 的 rho 才是真正的前向输入）
    valid         近场死区外的有效区
评估体再多存 `pd2_linear`（第二次观测）。

同时为每个体写出训练用的 `*_log1p.npy`：`log1p(pd / s)`，`s = median(pd[valid])`。
**注意尺度口径**：`s` 是逐体各自的中位数。训练时 `train.py --train-volumes`
不做跨体归一（每个体自己已经归一过），这与单体的 `train_log1p.npy` 口径一致；
评估时输入与真值用**同一个** `s`（存在 npz 的 `s` 字段里），保证两者可比（R07）。

用法
----
    python scripts/v3_make_sim_dataset.py --n-train 16 --n-eval 2

产物
----
    data/processed/v3_sim_v2/train/v2_train_s{seed}.npz  + _log1p.npy
    data/processed/v3_sim_v2/eval/v2_eval_s{seed}.npz
    data/processed/v3_sim_v2/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_simulator import SimCfg, band_and_pd, background_mask, simulate  # noqa: E402


def build_one(seed: int, forward: str, contrast_db: float,
              with_second: bool) -> dict:
    """跑一个种子；with_second=True 时再多跑一次独立噪声的观测。"""
    cfg = SimCfg(seed=seed, forward=forward, flow_contrast_db=contrast_db)
    t0 = time.time()
    d = simulate(cfg)
    pd, _ = band_and_pd(d["X"], d["valid"])
    gt, _ = band_and_pd(d["flow"], d["valid"])
    del d["X"]
    rec = {
        "pd_linear": pd.astype(np.float32),
        "gt_flow_pd": gt.astype(np.float32),
        "vessels": d["vessels"],
        "rho": d["rho"].astype(np.float32),
        "valid": d["valid"],
        "s": float(np.median(pd[d["valid"]])),
        "cfg": json.dumps(asdict(cfg)),
        "seed": seed,
    }
    if with_second:
        cfg2 = SimCfg(seed=seed, forward=forward, flow_contrast_db=contrast_db,
                      noise_seed=seed + 700_000, static_seed=seed)
        d2 = simulate(cfg2)
        pd2, _ = band_and_pd(d2["X"], d2["valid"])
        del d2["X"]
        rec["pd2_linear"] = pd2.astype(np.float32)
        # 逐位自检：同解剖、固定散斑同一条流 ⇒ 噪声独立但结构共享
        r2 = np.asarray(d2["rho"], np.float32)
        rec["rho_same"] = bool(np.array_equal(rec["rho"], r2))
        rec["rho_maxdiff"] = float(np.max(np.abs(rec["rho"] - r2)))
        del d2
    # 记录带通后真正达到的血管/背景对比度（与 v3_simulator.py CLI 同一定义）
    core = rec["rho"] >= 0.5
    bgm = background_mask(pd, rec["valid"], 0.20)
    if core.any() and bgm.any():
        rec["contrast_db"] = float(10.0 * np.log10(
            max(float(np.median(gt[core])), 1e-30)
            / max(float(np.median(pd[bgm])), 1e-30)))
    rec["gt_zero_frac"] = float(np.mean(gt[rec["valid"]] <= 0.0))
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/v3_sim_v2")
    ap.add_argument("--forward", default="v2", choices=["v1", "v2"])
    ap.add_argument("--contrast-db", type=float, default=6.0)
    ap.add_argument("--n-train", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--train-seed0", type=int, default=100)
    ap.add_argument("--eval-seed0", type=int, default=30)
    args = ap.parse_args()

    out = ROOT / args.out
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)

    man: dict = {"forward": args.forward, "contrast_db": args.contrast_db,
                 "train": [], "eval": []}

    train_seeds = [args.train_seed0 + i for i in range(args.n_train)]
    eval_seeds = [args.eval_seed0 + i for i in range(args.n_eval)]

    for k, seed in enumerate(train_seeds):
        npz = out / "train" / f"v2_train_s{seed}.npz"
        npy = out / "train" / f"v2_train_s{seed}_log1p.npy"
        if npz.exists() and npy.exists():
            print(f"[train {k+1}/{len(train_seeds)}] seed={seed} cached", flush=True)
        else:
            print(f"[train {k+1}/{len(train_seeds)}] seed={seed} ...", flush=True)
            rec = build_one(seed, args.forward, args.contrast_db, with_second=False)
            x = np.log1p(rec["pd_linear"] / max(rec["s"], 1e-12)).astype(np.float32)
            x[~rec["valid"]] = 0.0
            np.save(npy, x)
            np.savez_compressed(npz, **rec)
            print(f"    {rec['seconds']}s  s={rec['s']:.4g}  "
                  f"contrast={rec.get('contrast_db', float('nan')):.2f} dB  "
                  f"gt0={rec['gt_zero_frac']:.4f}", flush=True)
        man["train"].append({"seed": seed, "npz": str(npz.relative_to(ROOT)),
                             "log1p": str(npy.relative_to(ROOT))})

    for k, seed in enumerate(eval_seeds):
        npz = out / "eval" / f"v2_eval_s{seed}.npz"
        if npz.exists():
            print(f"[eval {k+1}/{len(eval_seeds)}] seed={seed} cached", flush=True)
        else:
            print(f"[eval {k+1}/{len(eval_seeds)}] seed={seed} (+ independent obs) ...",
                  flush=True)
            rec = build_one(seed, args.forward, args.contrast_db, with_second=True)
            np.savez_compressed(npz, **rec)
            print(f"    {rec['seconds']}s  s={rec['s']:.4g}  "
                  f"rho_same={rec['rho_same']}  "
                  f"contrast={rec.get('contrast_db', float('nan')):.2f} dB  "
                  f"gt0={rec['gt_zero_frac']:.4f}", flush=True)
        man["eval"].append({"seed": seed, "npz": str(npz.relative_to(ROOT))})

    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\nwrote {out/'manifest.json'}  "
          f"({len(man['train'])} train / {len(man['eval'])} eval)")


if __name__ == "__main__":
    main()

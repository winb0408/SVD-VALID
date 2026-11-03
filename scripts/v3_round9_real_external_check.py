#!/usr/bin/env python3
"""Round9：真实 SA64 / HSA64 的**只读外部检查**（计划要求的交付项，无训练）。

补记说明：这一项是预注册 §7 要求的交付，但在 Round9 定稿的 `DECISION.md` 清单里**漏列并漏跑**
（用户复核时问"这两笔真实数据训了吗"才发现）。本脚本于定稿后补写、补跑，0 个新训练作业，
结果补记在 `reports/v3_round9/DECISION.md`；它**不改变** `STOP_BASELINE`。

说明为什么需要这一步：Round9 的正对照是仿真的 S0，噪声是加在 log 域的、σ=0.0385
（信噪比约 11:1）。真实数据是完全不同的工况。这一步只回答一个问题：
**把忠实训练出来的两个 checkpoint 直接用到真实数据上，输出会不会崩**
（整体偏移、动态范围塌缩、结构被抹平、背景变吵）。
**不**回答"去噪有没有效"——真实数据没有干净真值，这里不做任何 GT-free 的成功宣称。

输入域必须与训练域一致：Round9 训练用**原始** `log1p(PD/s)`；
真实数据体存的是**另一个**域的标准化值 `(logeps − mean)/std`，其中
`logeps = log(PD/s + 1e-3)`。两个域之间不是仿射关系，所以不能只反标准化：
必须 `logeps_raw = x·std + mean` → `u = exp(logeps_raw) − 1e-3` → `log1p_raw = log1p(u)`。
（这一步写错会让网络吃到错域的输入，检查结论就作废——所以在这里显式写出链条并断言往返一致。）
推理结束后再按 `pd = s·expm1(y)` 回到线性 PD。

显示口径与 Round8 一致：`db = 10log10(pd / refmax)`，`refmax` = **输入** PD 在 mask 内的最大值，
**所有方法共用一个参考**；色标固定 `[-25, 0]` dB，绝不逐图拉伸。

用法：
    python -u scripts/v3_round9_real_external_check.py --seqs SA64 HSA64 --gpu 2
输出：
    reports/v3_round9/real_external_check.json
    reports/v3_round9/figs/{seq}_real_external_check.png
    data/processed/v3_round9_real/{seq}/{method}_log1p.npy
（对应的 `real_external_check.md` 是据该 JSON 手写的报告，不由本脚本生成。）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener as scipy_wiener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PY = "/data2/wjb/anaconda3/envs/threePM/bin/python"
PATCH = [32, 64, 64]
OVERLAP = 0.1
GAUSS_SIGMA = 0.60          # 与 Round9 校准一致（calibration.json）
VMIN, VMAX = -25.0, 0.0
OUT = ROOT / "reports/v3_round9"


def sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def roughness(y, m):
    """背景粗糙度：标准化前域的 mean|∇y|²，只取 mask 内部（不含边界伪差）。"""
    g = np.gradient(np.asarray(y, np.float64))
    s = np.zeros_like(g[0])
    for gi in g:
        s += gi ** 2
    return float(s[m].mean())


def run_infer(ckpt_dir: Path, npy: Path, gpu: int, tag: str) -> np.ndarray:
    dst = ROOT / "runs/_round9_real" / tag
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "infer.py", "--volume", str(npy),
           "--checkpoint", str(ckpt_dir / "latest.pt"),
           "--patch", *[str(p) for p in PATCH], "--overlap", str(OVERLAP),
           "--device", "cuda", "--output", str(dst)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"infer.py 失败 {ckpt_dir}\n{r.stdout[-1200:]}\n{r.stderr[-1200:]}")
    return np.load(dst / "denoised.npy").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--arms", nargs="+", default=["A", "B"])
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()

    norm = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())["logeps"]
    res = {
        "purpose": "真实数据只读外部检查：只问‘会不会崩’，不问‘有没有效’（无干净真值）",
        "input_domain": ("log1p(PD/s) = log1p(exp(standardized*"
                         f"{norm['std']:.6f}+{norm['mean']:.6f}) - 1e-3)  "
                         "（真实体存的是标准化 logeps，非仿射换算，已按定义换域）"),
        "output_domain": "pd = s*expm1(y)，y 为 log1p 域网络输出",
        "display": "db = 10log10(pd/refmax)，refmax 由**输入** PD 在 mask 内最大值定，所有方法共用；色标 [-25,0] dB",
        "patch": PATCH, "overlap": OVERLAP, "gauss_sigma": GAUSS_SIGMA,
        "new_training_jobs": 0,
        "checkpoints": {a: {"path": f"runs/v3_round9/{a}/latest.pt",
                            "sha256": sha256(ROOT / f"runs/v3_round9/{a}/latest.pt")}
                        for a in args.arms},
        "sequences": {},
    }

    tmp = ROOT / "runs/_round9_real_npy"
    tmp.mkdir(parents=True, exist_ok=True)

    for seq in args.seqs:
        src = ROOT / f"data/processed/v3_round7_b1/logeps/{seq}_full_x.npy"
        mask_p = ROOT / f"data/processed/official/{seq}/valid_mask.npy"
        meta_p = ROOT / f"data/processed/v3_round6/{seq}/metadata.json"
        for p in (src, mask_p, meta_p):
            if not p.exists():
                raise FileNotFoundError(f"缺少真实数据：{p}")
        x_std = np.load(src).astype(np.float32)
        dims = tuple(2 * (v // 2) for v in x_std.shape)   # 与 Round8 同样的偶数支撑
        sl = tuple(slice(0, k) for k in dims)
        x_std = x_std[sl]
        mask = np.load(mask_p).astype(bool)[sl]
        s = float(json.loads(meta_p.read_text())["s"])

        # 标准化 logeps → 原始 logeps → 训练域 log1p
        eps = 1e-3
        logeps = x_std.astype(np.float64) * norm["std"] + norm["mean"]
        u = np.maximum(np.exp(logeps) - eps, 0.0)          # u = PD/s
        x_log = np.log1p(u)
        # 往返断言：按定义反推回 logeps，必须与反标准化结果逐位一致
        back = np.log(u + eps)
        assert np.allclose(back, logeps, atol=1e-6), "logeps→log1p 往返不自洽"
        # 与 S0 训练数据一致：mask 外为 0
        x_log = np.where(mask, x_log, 0.0).astype(np.float32)
        npy = tmp / f"{seq}_log1p_raw.npy"
        np.save(npy, x_log)
        print(f"[{seq}] 训练域输入 log1p：mean={x_log[mask].mean():.4f} "
              f"std={x_log[mask].std():.4f}  （S0 训练集量级 mean≈0.76 std≈0.43）")

        pd_in = np.maximum(s * np.expm1(x_log.astype(np.float64)), 0.0)
        refmax = max(float(pd_in[mask].max()), 1e-12)
        q_bright = float(np.quantile(pd_in[mask], 0.9))
        q_bg = float(np.quantile(pd_in[mask], 0.3))
        bright, background = mask & (pd_in >= q_bright), mask & (pd_in <= q_bg)

        preds = {
            "svd_input": x_log.astype(np.float32),
            f"fixed_gauss_s{GAUSS_SIGMA:.2f}": gaussian_filter(
                x_log.astype(np.float64), GAUSS_SIGMA, mode="nearest").astype(np.float32),
            "fixed_wiener3d": scipy_wiener(x_log.astype(np.float64), mysize=3).astype(np.float32),
        }
        for a in args.arms:
            preds[a] = run_infer(ROOT / f"runs/v3_round9/{a}", npy, args.gpu, f"{a}_{seq}")

        methods = {}
        for name, y in preds.items():
            y = np.asarray(y, np.float64)
            pd = np.maximum(s * np.expm1(y), 0.0)
            db = 10 * np.log10(np.maximum(pd, 1e-12) / refmax)
            methods[name] = {
                # 原始浮点统计（计划要求：输入/输出动态范围不同必须一起报）
                "log1p_mean": float(y[mask].mean()), "log1p_std": float(y[mask].std()),
                "log1p_min": float(y[mask].min()), "log1p_max": float(y[mask].max()),
                "pd_mean": float(pd[mask].mean()), "pd_std": float(pd[mask].std()),
                "pd_max": float(pd[mask].max()),
                # 共同参考 dB 统计
                "db_p50": float(np.median(db[mask])),
                # 结构/噪声代理（无真值，只能自比）
                "core_mean_ratio": float(pd[bright].mean() / max(pd_in[bright].mean(), 1e-12)),
                "bg_std_ratio": float(pd[background].std() / max(pd_in[background].std(), 1e-12)),
                "roughness_bg_log1p": roughness(y, background),
            }
        res["sequences"][seq] = {"s": s, "shape": list(dims), "mask_voxels": int(mask.sum()),
                                 "refmax": refmax, "methods": methods}
        save = ROOT / f"data/processed/v3_round9_real/{seq}"
        save.mkdir(parents=True, exist_ok=True)
        for name, y in preds.items():
            np.save(save / f"{name}_log1p.npy", np.asarray(y, np.float32))
        print(f"[{seq}] refmax={refmax:.4f}")
        for name, m in methods.items():
            print(f"   {name:22s} core={m['core_mean_ratio']:.3f} "
                  f"bg_std={m['bg_std_ratio']:.3f} rough={m['roughness_bg_log1p']:.4f} "
                  f"pd_mean={m['pd_mean']:.3f}")

        # 图：MIP + 中间层，共用参考与色标
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(preds)
        fig, ax = plt.subplots(2, len(keys), figsize=(2.6 * len(keys), 5.6))
        for i, k in enumerate(keys):
            pd = np.maximum(s * np.expm1(np.asarray(preds[k], np.float64)), 0.0)
            db = 10 * np.log10(np.maximum(pd, 1e-12) / refmax)
            for r, img in enumerate((db.max(axis=0), db[db.shape[0] // 2])):
                ax[r, i].imshow(img, cmap="hot", vmin=VMIN, vmax=VMAX)
                ax[r, i].set_xticks([]), ax[r, i].set_yticks([])
                if r == 0:
                    ax[r, i].set_title(k, fontsize=8)
            ax[0, i].set_ylabel("MIP" if i == 0 else "", fontsize=8)
        ax[1, 0].set_ylabel("mid z", fontsize=8)
        fig.suptitle(f"{seq}: Round9 checkpoints on real data (read-only) · "
                     f"shared dB ref = input PD max = {refmax:.3f} · [-25,0] dB", fontsize=8)
        fig.tight_layout()
        fp = OUT / "figs" / f"{seq}_real_external_check.png"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, dpi=160)
        plt.close(fig)
        print("   figure:", fp.relative_to(ROOT))

    (OUT / "real_external_check.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print("\nwrote", OUT / "real_external_check.json")


if __name__ == "__main__":
    main()

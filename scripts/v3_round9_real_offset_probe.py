"""Round9 真实数据「电平偏移」诊断探针（只读，不训练）。

存在理由（为什么单列一个脚本）：
    reports/v3_round9/real_external_check.md 观察到 A/B 在两个真实序列上把 log1p 域
    斜率压到 0.82/0.76（SA64）与 0.69/0.67（HSA64）。这个现象有两种可能来源：
      (i) checkpoint 本身学到的就是「压暗」映射 —— 那模拟阳性对照里也该看到；
      (ii) 真实数据的输入电平整体高于 S0 训练分布（局部均值 1.08/1.26 vs 0.75），
           网络在超出训练电平后收缩 —— 那是输入分布偏移。
    外检报告不能只写「观察到压暗」而不区分这两者，否则 GPT 复核无法判断
    STOP_BASELINE 的诊断是该改损失还是该改数据域。本探针只做一件事：
    把整卷减去一个常数 c（用 S0 训练数据 nonzero 均值 0.7314 对齐），
    重跑同一个 checkpoint，看收缩是否消失。c 只作用于输入，不改任何指标定义。

    这是**干预探针**，不是结果：偏移后的输入不是真实数据，不能拿来报性能，
    也不改变 STOP_BASELINE（该 gate 建立在模拟阳性对照上）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_round9_real_external_check import OUT, ROOT, run_infer  # noqa: E402

S0_NONZERO_MEAN = 0.7314        # data/processed/v3_round2_s0/low/train/*.npy, nonzero voxels


def load_log1p(seq: str):
    x = np.load(ROOT / f"data/processed/v3_round9_real/{seq}/svd_input_log1p.npy").astype(np.float32)
    mask = np.load(ROOT / f"data/processed/official/{seq}/valid_mask.npy").astype(bool)[: x.shape[0]]
    return x, mask


def affine(x, y, m):
    """mask 内最小二乘斜率/截距。

    注意除以 N：np.dot(xx-x̄, yy-ȳ) 是**求和**不是均值，漏掉 /N 会把斜率放大 N 倍
    （本卷 N=120232，第一版就因此报出 1.1e5 的假斜率）。
    """
    xx, yy = x[m].astype(np.float64), y[m].astype(np.float64)
    v = float(xx.var())
    if v <= 0:
        return float("nan"), float("nan")
    a = float(np.dot(xx - xx.mean(), yy - yy.mean()) / (xx.size * v))
    return a, float(yy.mean() - a * xx.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--arms", nargs="+", default=["A"])
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    res = {
        "purpose": "电平偏移干预探针：区分『checkpoint 学到压暗』与『输入电平超训练分布导致收缩』；"
                   "偏移后输入非真实数据，不得据此报性能",
        "s0_nonzero_mean": S0_NONZERO_MEAN,
        "new_training_jobs": 0,
        "seqs": {},
    }
    tmp = ROOT / "runs/_round9_real/offset_probe"
    tmp.mkdir(parents=True, exist_ok=True)

    for seq in args.seqs:
        x, mask = load_log1p(seq)
        c = float(x[mask].mean() - S0_NONZERO_MEAN)     # 对齐到 S0 训练电平
        # 只在 mask 内平移：训练约定是「mask 外恒为 0」，整卷平移会把外部也变成
        # -c，给网络一个它从没见过的负平台，探针结论就不可信。
        xp = np.where(mask, x - c, 0.0).astype(np.float32)
        print(f"\n=== {seq}: 原始 mask 内均值 {x[mask].mean():.4f} → 减 c={c:.4f} "
              f"→ {xp[mask].mean():.4f}（S0 训练 {S0_NONZERO_MEAN}）；"
              f"偏移后 min={xp[mask].min():.3f}")
        npy = tmp / f"{seq}_shifted.npy"
        np.save(npy, xp)

        entry = {"shift_c": c, "arms": {}}
        for arm in args.arms:
            tag = f"{arm}_{seq}_shifted"
            raw_f = ROOT / "runs/_round9_real" / tag / "denoised.npy"
            if raw_f.exists():
                y = np.load(raw_f).astype(np.float32)
                print(f"  [{arm}] 复用已有输出 {raw_f}")
            else:
                y = run_infer(ROOT / f"runs/v3_round9/{arm}", npy, args.gpu, tag)
            # 注意：比较斜率时用的是「偏移后的输入 vs 输出」，两者都未加回 c，
            # 因此斜率只反映映射的形状，不受常数影响。
            a_orig, _ = affine(x, y, mask)
            a_shift, b_shift = affine(xp, y, mask)
            core_in = mask & (x >= np.quantile(x[mask], 0.9))
            entry["arms"][arm] = {
                "slope_vs_shifted_input": a_shift,
                "intercept_vs_shifted_input": b_shift,
                "slope_vs_original_input": a_orig,
                "out_mean_log1p": float(y[mask].mean()),
                "core_mean_ratio_log1p": float(y[core_in].mean() / max(xp[core_in].mean(), 1e-9)),
            }
            print(f"  [{arm}] 斜率：对偏移后输入 {a_shift:.4f}（对原始输入 {a_orig:.4f}），"
                  f"输出均值 {y[mask].mean():.4f}，"
                  f"最亮10%core比 {y[core_in].mean() / max(xp[core_in].mean(), 1e-9):.3f}")
        res["seqs"][seq] = entry

    dst = OUT / "real_offset_probe.json"
    dst.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n写出 {dst}")


if __name__ == "__main__":
    main()

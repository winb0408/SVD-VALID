"""R06 续：仿真与真实数据的"可去除噪声占比"同口径标定。

为什么必须先做这一步
--------------------
R06 审计发现仿真真值退化的同时，暴露了一个更根本的问题：在 v1 仿真里，
**两次独立观测之间"不一样"的部分只占 log-PD 方差的 6.6% 左右**（`n2n_den = 2.1e-3`
对 `bg_std² = 0.032`）。Noise2Noise 能去掉的**只可能**是这部分；剩下 93% 是两次观测共享的
静态散斑，任何 N2N 类方法都无权删除。于是"把血管从散斑里分离出来"这个任务在 N2N 框架下
**在数学上就不可达**，而实验 A 的所有方法（包括原版 VALID）都输在这上面。

这一步把同一个量在**真实数据**上量出来，用来回答两个问题：
  ① 仿真的"可去除噪声占比"是否与真实数据同量级？（不同则仿真的难度是假的）
  ② 真实数据上，N2N 类方法最多能去掉多少？（决定判据能做到多好才叫成功）

定义（严格同口径，仿真与真实完全一致）
--------------------------------------
把 400 帧按时间奇偶拆成两半，**用同一次带通滤波的结果**分别重建 PD（不重做 SVD）：
    pd1 = Σ_{t 偶} |filt_t|²,  pd2 = Σ_{t 奇} |filt_t|²,  pd = pd1 + pd2（精确）
在 log1p(pd/median) 域里取 l1、l2：
    `noise_frac = mean((l1 − l2)²) / (2 · var(0.5·(l1 + l2)))`
分母是"整幅图的总方差"，分子是"两半之差"，取 2 的因子是因为两半各自独立。
  · 因为两半各只有 200 帧，其估计噪声是 400 帧图像的 2 倍（PD 估计方差 ∝ 1/T），
    所以 (l1−l2) 的方差 ≈ 4·Var(l_full)；再除以 2 得 `noise_frac_full ≈ 2·Var(l_full)/Var(l)`。
  · 也就是说 **`noise_frac` 读作"如果把帧数翻倍能消掉的那部分方差占比"**，
    这是任何时间平均类方法（含 N2N）在该数据上能触及的上限的一阶估计。
真实与仿真都用这个式子，所以数值可直接对比。

用法
----
    python scripts/v3_testretest_calibration.py --sim-seeds 0 1
产物
----
    reports/v3_r06/testretest_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.iq_io import load_iq                       # noqa: E402
from official_pipeline import SEQUENCE_SETTINGS, svd_blood_band  # noqa: E402


def half_pds(filt):
    """同一次带通结果的帧奇偶拆分：pd1 + pd2 == Σ_t|filt_t|² 精确成立。"""
    pw = np.abs(filt) ** 2
    return pw[..., 0::2].sum(axis=3), pw[..., 1::2].sum(axis=3)


def noise_frac(pd1, pd2, valid):
    """log1p 域里"两半之差 / 总方差"，见模块文档。"""
    pd = pd1 + pd2
    s = float(np.median(pd[valid]))
    l1 = np.log1p(pd1[valid] / s)
    l2 = np.log1p(pd2[valid] / s)
    l = 0.5 * (l1 + l2)
    num = float(np.mean((l1 - l2) ** 2))
    den = float(2.0 * np.var(l))
    return {"noise_frac": num / den if den else float("nan"),
            "num_mean_sq_diff": num, "den_2var": den,
            "var_full_log": float(np.var(np.log1p(pd[valid] / s))),
            "n": int(valid.sum()), "scale_s": s}


def real_one(name):
    # 与 official_pipeline 完全相同的定位方式，避免另建一套路径约定
    path = Path("/ssd1/wjb/Data/Dyrad/data") / name / f"Data_{name}.mat"
    if not path.exists():
        return {"error": f"{name}: 原始 IQ 未找到（{path}）"}
    st = SEQUENCE_SETTINGS[name]
    x = load_iq(path, st["variable"])
    if st["sum_arrays"] and x.ndim == 5:
        x = np.squeeze(x.sum(axis=-1))
    filt, _ = svd_blood_band(x, st["cut1"], st["cut2"], True)
    pd1, pd2 = half_pds(filt)
    valid = (pd1 + pd2) > 0
    return noise_frac(pd1, pd2, valid)


def sim_one(seed, fwd):
    from v3_simulator import SimCfg, simulate, band_and_pd
    d = simulate(SimCfg(seed=seed, forward=fwd))
    _, filt = band_and_pd(d["X"], d["valid"])
    pd1, pd2 = half_pds(filt)
    return noise_frac(pd1, pd2, d["valid"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--sim-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--forwards", nargs="+", default=["v1", "v2"])
    ap.add_argument("--out", default="reports/v3_r06/testretest_calibration.json")
    args = ap.parse_args()

    res = {}
    for name in args.seqs:
        r = real_one(name)
        res[name] = r
        print(f"{name:8s} " + (f"noise_frac={r['noise_frac']:.4f}  (n={r['n']})"
                               if "error" not in r else r["error"]))
    for fwd in args.forwards:
        for sd in args.sim_seeds:
            r = sim_one(sd, fwd)
            key = f"sim_{fwd}_seed{sd}"
            res[key] = r
            print(f"{key:18s} noise_frac={r['noise_frac']:.4f}  (n={r['n']})")

    real = [res[n]["noise_frac"] for n in args.seqs if "error" not in res[n]]
    simv = [v["noise_frac"] for k, v in res.items() if k.startswith("sim_")]
    out = {
        "definition": "mean((l1-l2)^2) / (2*var(0.5*(l1+l2))) in log1p(pd/median) domain; "
                      "halves = odd/even frames of the SAME band-pass filtered IQ",
        "reads_as": "占整幅图 log-PD 方差的比例；上界≈'帧数翻倍/时间平均类方法能消掉的方差'",
        "caveat": "两半各 200 帧，估计噪声是 400 帧的 2 倍；本式已按此换算，"
                  "但它是**时间平均**口径，不等价于 N2N 用两次独立采集能做的全部",
        "results": res,
        "summary": {"real_mean": float(np.mean(real)) if real else None,
                    "sim_mean": float(np.mean(simv)) if simv else None},
    }
    dst = ROOT / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n真实均值 {out['summary']['real_mean']:.4f} vs 仿真均值 "
          f"{out['summary']['sim_mean']:.4f}  → {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

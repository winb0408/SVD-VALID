"""V3 Phase 3 / 最小端到端探针（4 臂 × 多个种子）。

这一步只回答两个问题，别的不答：

  Q1 **12.5% 的覆盖率还能不能训起来？**（m=2 每块只采 8/64 个体素）
  Q2 **角色配对真的能分出高低吗？**（体对角 vs y 轴向，机制双向可证伪）

两问都不能用真实数据回答，因为真实数据没有真值。所以探针跑在仿真器上
（`scripts/v3_simulator.py`，血流项已修成逐帧独立散斑，见 2026-10-20 修正），
训练用一个种子、评估用**另一个种子**的仿真体（血管几何不同），
避免"把这一例的血管背下来"。

四条臂（同一网络、同一优化器、同一预算、同一种子集合）：

| 臂 | stride | 配对 | 槽位 (xa,xb,ya,yb) | 用途 |
|---|---|---|---|---|
| A | 1 | cross | 每块随机 | 基线：m=1 的互补正交采样 |
| B | 2 | cross | 每块随机 | 只改步距，看 m=2 能不能训 |
| C | 2 | cross | 1 4 6 3 | 体对角（泄漏 0.017，信号最远） |
| D | 2 | cross | 1 4 3 6 | y 轴向（泄漏 0.082，信号较近） |

**C 与 D 的关键隔离**：两者采到的体素完全相同（{1,3,4,6} 四个相位）、输入也相同
（相位 1 和 4），连两个目标体的集合都相同（{3,6}）——唯一差别是**哪个目标配哪个输入**。
所以 C 与 D 的任何差别都不可能是"采到了不同的数据"造成的。

`--pairing cross` 是必需的：8 个 tet 每个只从 4 组对顶角里各取一个，
**任何 tet 内部都没有体对角对**，预注册的首选只能在互补 tet 里实现
（自检见 tests/test_smoke.py::test_body_diagonal_pair_needs_the_complement_tet）。

用法：
    python -u scripts/v3_phase3_probe.py --seeds 3407 3408 3409 --gpus 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

ARMS = {
    # name: (stride, extra train.py args)
    "A_m1_rand": (1, ["--pairing", "cross", "--mode", "random"]),
    "B_m2_rand": (2, ["--pairing", "cross", "--mode", "random"]),
    "C_m2_bodydiag": (2, ["--pairing", "cross", "--mode", "fixed",
                          "--slots", "1", "4", "6", "3"]),
    "D_m2_yaxis": (2, ["--pairing", "cross", "--mode", "fixed",
                       "--slots", "1", "4", "3", "6"]),
    # E 是**诊断臂**，不是候选方案：把 C 的固定角色对原样搬到 m=1。
    # 目的只有一个——判断"学不到跨相位映射"是 m=2 特有的，还是这套设定本来就这样。
    "E_m1_bodydiag": (1, ["--pairing", "cross", "--mode", "fixed",
                          "--slots", "1", "4", "6", "3"]),
}


def prep_volume(src: Path, dst: Path):
    """把仿真观测 PD 变成训练用的体数据：log1p(PD/中位数)。

    为什么取 log：仿真 PD 的血管内均值是 5.8e7、背景是 0，动态范围跨 7 个数量级，
    直接喂给 Hessian/DWT 正则会让 loss 被少数血管体素主导。取 log1p 之后
    血管与背景的差距被压到 O(1)，与真实数据上 Phase 1 用的变换一致
    （`log1p(PD/s)`，s = 该体数据的 PD 中位数）。
    这一步对四条臂完全相同，不构成臂间差别。
    """
    d = np.load(src)
    pd = d["pd_linear"].astype(np.float32)
    valid = d["valid"].astype(bool)
    s = float(np.median(pd[valid]))
    x = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    x[~valid] = 0.0
    np.save(dst, x)
    return dst


def run_arm(arm: str, seed: int, gpu: int, train_vol: Path, out_root: Path,
            epochs: int, steps: int, patch: list[int], features: int):
    stride, extra = ARMS[arm]
    out = out_root / f"{arm}_s{seed}"
    if (out / "latest.pt").exists():
        return out, "cached"
    cmd = [
        PY, "-u", "train.py",
        "--train-volume", str(train_vol),
        "--stride", str(stride),
        "--patch", *[str(p) for p in patch],
        "--epochs", str(epochs), "--steps-per-epoch", str(steps),
        "--features", str(features), "--seed", str(seed),
        "--out", str(out), *extra,
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = out.with_suffix(".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{arm} seed {seed} failed, see {log}")
    return out, f"{time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sim", default="data/processed/v3_sim/sim_seed0.npz")
    ap.add_argument("--eval-sim", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--gpus", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--prep", default="data/processed/v3_sim/train_log1p.npy")
    ap.add_argument("--out", default="runs/v3_probe")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    prep = prep_volume(Path(args.train_sim), Path(args.prep))
    print(f"训练体数据 {prep}  评估体 {args.eval_sim}")

    jobs = [(a, s) for a in args.arms for s in args.seeds]
    gpu_of = {j: args.gpus[i % len(args.gpus)] for i, j in enumerate(jobs)}
    print(f"{len(jobs)} 个训练任务，GPU 分配 {sorted(set(gpu_of.values()))}")

    # 按 GPU 分桶串行，桶间并行
    buckets: dict[int, list] = {}
    for j in jobs:
        buckets.setdefault(gpu_of[j], []).append(j)

    procs = []
    meta = {}
    for gpu, items in buckets.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        lines = []
        for arm, seed in items:
            stride, extra = ARMS[arm]
            out = out_root / f"{arm}_s{seed}"
            if (out / "latest.pt").exists():
                continue
            cmd = [PY, "-u", "train.py", "--train-volume", str(prep),
                   "--stride", str(stride),
                   "--patch", *[str(p) for p in args.patch],
                   "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps),
                   "--features", str(args.features), "--seed", str(seed),
                   "--out", str(out), *extra]
            lines.append("cd %s && %s >> %s.log 2>&1" % (ROOT, " ".join(cmd), out))
        if not lines:
            continue
        script = out_root / f"runner_gpu{gpu}.sh"
        script.write_text("#!/bin/bash\nset -e\n" + "\n".join(lines) + "\n")
        script.chmod(0o755)
        log = out_root / f"runner_gpu{gpu}.log"
        procs.append((gpu, subprocess.Popen(["bash", str(script)], cwd=ROOT, env=env,
                                            stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        meta[gpu] = [j for j in items]

    for gpu, p in procs:
        p.wait()
        print(f"GPU {gpu} 完成，返回码 {p.returncode}")
        if p.returncode:
            raise SystemExit(f"GPU {gpu} 上有任务失败，见 {out_root}/runner_gpu{gpu}.log")

    done = [str(out_root / f"{a}_s{s}") for a, s in jobs
            if (out_root / f"{a}_s{s}" / "latest.pt").exists()]
    print(f"完成 {len(done)}/{len(jobs)} 个运行")
    (out_root / "arms.json").write_text(json.dumps(
        {k: {"stride": v[0], "args": v[1]} for k, v in ARMS.items()}, indent=1))
    print("下一步：python -u scripts/v3_phase3_eval.py")


if __name__ == "__main__":
    main()

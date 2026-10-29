"""V3 Phase 4 单旋钮实验：`--weight-identity` 扫描。

## 这个实验要判什么

L45 的审计发现：`train.py` 里 `weight_identity` 默认 **1.0**，而邻居数据项合计只有 0.5+0.5。
一致性项作用在 tet 两个**互不相交**的角点格点上（`h_a`、`h_b` 逐元素比较两套不同位置），
最省力的解就是把两边都压成同一个低振幅平滑场。实测输出对输入的回归斜率：
A 0.198 / E 0.169 / B 0.016，输出 std 只有输入的 0.30 / 0.37 / 0.10——
**这正是"向均值收缩"的指纹**，也解释了目视的三条观察（血管轮廓在、背景没压、对比度掉）。

所以只需要一个旋钮就能分清两件事：

  - 如果某个权重下 `slope` 回到 ≈0.8–1.0 **且** n2n 判据通过 ⇒ 之前的失败是**设置问题**；
  - 如果扫完所有权重 `slope` 都上不去、或上去了但判据不过 ⇒ 是**方法/前提问题**。

V1/V2 记过 `idt=0` 的代价（暗区转负、色阶跨度虚高、越界翻倍），所以 0 只作上界探测，
不作推荐值；本实验不预先看好任何一档。

除 `--weight-identity` 外，其余全部与臂 A（m=1 / cross / random）逐位相同。

用法：
    python -u scripts/v3_phase4_identity_sweep.py --seeds 3407 3408 3409 --gpus 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# 名字: 传给 train.py 的额外参数
GRID = {
    "idt100": ["--weight-identity", "1.0"],   # 现状（VALID v1.0 默认）
    "idt030": ["--weight-identity", "0.3"],
    "idt010": ["--weight-identity", "0.1"],
    "idt000": ["--weight-identity", "0.0"],   # 上界探测，V1/V2 记过它的代价
}
BASE = ["--pairing", "cross", "--mode", "random"]   # 与臂 A 相同


def prep_volume(src: Path, dst: Path) -> Path:
    """复刻 v3_phase3_probe.prep_volume：log1p(PD/中位数)。"""
    import numpy as np
    if dst.exists():
        return dst
    d = np.load(src)
    pd = d["pd_linear"].astype(np.float32)
    valid = d["valid"].astype(bool)
    s = float(np.median(pd[valid]))
    x = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    x[~valid] = 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, x)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sim", default="data/processed/v3_sim/sim_seed0.npz")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--gpus", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--grid", nargs="+", default=list(GRID))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--prep", default="data/processed/v3_sim/train_log1p.npy")
    ap.add_argument("--out", default="runs/v3_idt_sweep")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    prep = prep_volume(Path(args.train_sim), Path(args.prep))

    jobs = [(g, s) for g in args.grid for s in args.seeds]
    buckets: dict[int, list] = {}
    for i, j in enumerate(jobs):
        buckets.setdefault(args.gpus[i % len(args.gpus)], []).append(j)
    print(f"{len(jobs)} 个训练任务，GPU 分桶 { {k: len(v) for k, v in buckets.items()} }")

    procs = []
    for gpu, items in buckets.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        lines = []
        for name, seed in items:
            out = out_root / f"{name}_s{seed}"
            if (out / "latest.pt").exists():
                continue
            cmd = [PY, "-u", "train.py", "--train-volume", str(prep),
                   "--stride", "1", "--patch", *[str(p) for p in args.patch],
                   "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps),
                   "--features", str(args.features), "--seed", str(seed),
                   "--out", str(out), *BASE, *GRID[name]]
            lines.append("cd %s && %s >> %s.log 2>&1" % (ROOT, " ".join(cmd), out))
        if not lines:
            continue
        script = out_root / f"runner_gpu{gpu}.sh"
        script.write_text("#!/bin/bash\nset -e\n" + "\n".join(lines) + "\n")
        script.chmod(0o755)
        log = out_root / f"runner_gpu{gpu}.log"
        procs.append((gpu, subprocess.Popen(["bash", str(script)], cwd=ROOT, env=env,
                                            stdout=open(log, "w"), stderr=subprocess.STDOUT)))
    for gpu, p in procs:
        p.wait()
        print(f"GPU {gpu} 完成，返回码 {p.returncode}")
        if p.returncode:
            raise SystemExit(f"GPU {gpu} 上有任务失败，见 {out_root}/runner_gpu{gpu}.log")
    (out_root / "grid.json").write_text(json.dumps(GRID, indent=1))
    print("完成。下一步：python -u scripts/v3_phase4_scoreboard.py --runs", out_root)


if __name__ == "__main__":
    main()

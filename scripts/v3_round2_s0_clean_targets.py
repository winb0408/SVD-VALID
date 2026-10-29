"""S0 归因用：为有监督上界臂生成与噪声体逐卷对应的**无噪** log1p 体。

为什么需要它（第二轮审查要求"失败必须给机制，不能只给结论"）：
S0 的自监督正控制失败时，有两种可能都解释得通——(a) 自监督配对（I1）不行，
(b) 训练预算/实现不行。要分开，只需把**目标**这一个变量换掉：
输入视图、网络、优化器、步数、种子全部不动，只把目标从"互补稀疏视图"换成真值。
这就是 `runs/v3_round2_s0/launch_supervised.sh` 里 `--train-clean-volumes` 用的文件。

口径与训练体一致：`log1p(gt_imaging_pd / s)`，同一卷的同一个 `s`（存在 npz 里），
实部 float32。**不是** `gt_flow_pd`——S0 的主终点是"完整成像期望"，不是血流分量。

用法
----
    python -u scripts/v3_round2_s0_clean_targets.py --level low
输出写回同一目录：`<tag>_s<seed>_clean.npy`，并打印清单。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--data-dir", default=None,
                    help="自定义数据目录（诊断臂用，如 data/processed/v3_round2_s0/low60）；"
                         "不给则用 data/processed/v3_round2_s0/<level>")
    a = ap.parse_args()

    base = Path(a.data_dir) if a.data_dir else ROOT / f"data/processed/v3_round2_s0/{a.level}"
    man_p = base / "manifest.json"
    man = json.loads(man_p.read_text())
    made = []
    for split in ("train", "eval"):
        for e in man[split]:
            npz = ROOT / e["npz"]
            d = np.load(npz)
            s = float(d["s"])
            clean = np.log1p(np.maximum(d["gt_imaging_pd"].astype(np.float32), 0.0) / s)
            # 与噪声体逐位同口径：valid 之外置 0（噪声体写盘时也是这么做的）
            clean[~d["valid"].astype(bool)] = 0.0
            out = npz.with_name(npz.stem + "_clean.npy")
            np.save(out, clean.astype(np.float32))
            made.append({"npz": e["npz"], "clean": str(out.relative_to(ROOT)),
                         "s": s, "min": float(clean.min()), "max": float(clean.max())})
            print(f"{out.name:34s} s={s:.4f} range=[{clean.min():.4f}, {clean.max():.4f}]")
    (ROOT / f"data/processed/v3_round2_s0/{a.level}/clean_targets.json").write_text(
        json.dumps({"level": a.level, "files": made,
                    "definition": "log1p(gt_imaging_pd / s)，与训练体同口径"}, ensure_ascii=False, indent=1))
    print(f"\n{len(made)} 个 clean 体已写出（{a.level}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

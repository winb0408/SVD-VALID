"""Phase 4 方案②：**中心可见**的 N2N 训练（Plain2Plain 与 Residual 两种输出形式）。

## 为什么要这个（L47/L48）

L47 判定：现有失败不是参数问题也不是采样法问题，而是"漏防式重建不许看被预测体素"
这一部署构造，其闭式上界 9.353（比什么都不做差 9 倍），训练结果 9.83 已贴着它。
而一旦允许估计器用 x_v 本身，同一任务能到 0.909（全局单系数）/ 0.584（分区收缩）。

但"允许看 x_v"在**单次采集的自监督**里会退化：目标就是 x_v，令输出=x_v 即得零损失。
所以②必须换监督信号——用**同一场景的两次独立观测**（真实数据里就是
`sa64_odd_remap` / `sa64_even_remap`，奇偶帧各 200 帧；模拟器里是 `--noise-seed`）。

    输入 = 观测 A 的完整体积（**含中心体素**）
    目标 = 观测 B 的同位置值（独立噪声实现）

这样"看 x_v"不再泄漏：目标来自另一次实现，不是输入自己。

## 两种输出形式

- `--residual 0`（Plain2Plain）：out = f(x)
- `--residual 1`（Residual）  ：out = x + f(x)，即网络只学"要减掉什么"

残差形式在低噪声区更容易保住结构，但要验它是否真的更好，不能假设。

## 与既有流程的区别（必须在报告里写清）

- 不再用 `strided_phase` 相位对，因此 `infer` 标记为 `full`：推理是一次整卷前向，
  不是一个体素一个体素地由别的相位散回来。
- 因此 checkpoints 里 `args["infer"]="full"`，`v3_phase4_scoreboard.py` 会走整卷前向。

用法（训练用 A→B，**评分必须用第三张实现 C**）：
    python scripts/v3_phase4_train_n2n.py \
        --input-npz data/processed/v3_sim/sim_seed1.npz \
        --target-npz data/processed/v3_sim/sim_seed1_n101.npz \
        --out runs/v3_n2n --gpu 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = "/data2/wjb/anaconda3/envs/threePM/bin/python"


PREP_VERSION = "prep-v2-shared-scale"   # R07: 缓存键与预处理口径的唯一版本号


def prep(npz: Path, shape_div: int = 32, shared_scale: float | None = None):
    """log1p(pd/s) 预处理。

    R07 更正（2026-10-26，外部审查）：**A、B 必须用同一个 s**。
    原实现里 A 与 B 各自调用本函数，于是各自用自己体积的 `median(pd[valid])` 归一化；
    对 N2N 而言这等于给两卷加了**不同的确定性增益**，破坏了"两卷只差噪声"的前提。
    现在由调用方先算出输入卷的 s，再用 `shared_scale=` 传给目标卷。

    `shared_scale=None` 时仍按自己算（单卷场景用），但 N2N 训练路径不允许走这条。
    返回 (volume, valid, s)，调用方负责断言 A/B 的 s 逐位相同。
    """
    import numpy as np
    d = np.load(npz)
    pd = d["pd_linear"].astype("float32")
    valid = d["valid"].astype(bool)
    shape = tuple((n // shape_div) * shape_div for n in pd.shape)
    crop = tuple(slice(0, n) for n in shape)
    pd = pd[crop]
    valid = valid[crop]
    s = float(np.median(pd[valid])) if shared_scale is None else float(shared_scale)
    return np.log1p(pd / max(s, 1e-12)).astype("float32"), valid.astype("float32"), s


def cache_key(npz: Path, shape_div: int) -> str:
    """R07: 缓存键 = 文件内容哈希 + 预处理版本 + 形状参数。

    用**内容**哈希而不是 mtime：mtime 在同一秒内可能不变，会让"文件改了但缓存没失效"
    这件事悄悄发生（本测试已复现）。文件大小在 MB 级，一次 sha1 的开销远小于重新跑预处理。
    """
    import hashlib
    h = hashlib.sha1()
    with open(npz, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"|{PREP_VERSION}|{shape_div}".encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", nargs="+", default=["data/processed/v3_sim/sim_seed1.npz"])
    ap.add_argument("--target-npz", nargs="+", default=["data/processed/v3_sim/sim_seed1_n101.npz"])
    ap.add_argument("--val-input-npz", default=None,
                    help="留出场景的观测 A；给了就每 --val-every 个 epoch 算一次留出比值并保存 best.pt")
    ap.add_argument("--val-target-npz", default=None)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--out", default="runs/v3_n2n")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--residual", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--gpus", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    src = out_root / "_prepared"
    src.mkdir(exist_ok=True)
    assert len(args.input_npz) == len(args.target_npz), "输入与目标必须一一对应"
    def prepare_pair(tag_a, tag_b, a_npz, b_npz, i):
        """R07：先算输入卷的 s，再用同一个 s 处理目标卷；缓存键含内容与版本。

        返回 (输入npy, 目标npy, valid_npy)。三者都用内容键缓存，任何一项缺失即重算。
        """
        import numpy as np
        a_npz, b_npz = Path(a_npz), Path(b_npz)
        ka, kb = cache_key(a_npz, 32), cache_key(b_npz, 32)
        da = src / f"{tag_a}_{ka}.npy"
        db = src / f"{tag_b}_{kb}.npy"
        dv = src / f"V{i}_{ka}.npy"
        meta_a = da.with_suffix(".json")
        s_a = json.loads(meta_a.read_text())["s"] if meta_a.exists() else None
        if s_a is None or not da.exists() or not dv.exists():
            vol, valid, s_a = prep(a_npz)
            np.save(da, vol)
            np.save(dv, valid)
            meta_a.write_text(json.dumps({"s": s_a, "src": str(a_npz),
                                          "prep_version": PREP_VERSION}))
            print(f"prepared {da.name}  (valid 占比 {valid.mean():.3f}, s={s_a:.6g})")
        if not db.exists():
            vol, _, s_b = prep(b_npz, shared_scale=s_a)
            assert s_b == s_a, "目标卷必须复用输入卷的 s"
            np.save(db, vol)
            db.with_suffix(".json").write_text(json.dumps(
                {"s": s_b, "src": str(b_npz), "shared_from": str(a_npz),
                 "prep_version": PREP_VERSION}))
            print(f"prepared {db.name}  (shared s={s_b:.6g})")
        return da, db, dv

    pair_paths = []
    for i, (a_npz, b_npz) in enumerate(zip(args.input_npz, args.target_npz)):
        da, db, dv = prepare_pair(f"A{i}", f"B{i}", a_npz, b_npz, i)
        pair_paths.append((da, db, dv))
    val = (None, None, None)
    if args.val_input_npz:
        val = prepare_pair("AV", "BV", args.val_input_npz, args.val_target_npz, "V")

    def launch(gpu, items):
        lines = []
        for res, seed in items:
            name = f"{'res' if res else 'plain'}_s{seed}"
            out = out_root / name
            if (out / "latest.pt").exists():
                continue
            cmd = [PY, "-u", "scripts/train_n2n.py",
                   "--input", *[str(a) for a, _, _ in pair_paths],
                   "--target", *[str(b) for _, b, _ in pair_paths],
                   "--valid", *[str(v) for _, _, v in pair_paths],
                   "--patch", *[str(p) for p in args.patch],
                   "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps_per_epoch),
                   "--features", str(args.features), "--groups", str(args.groups),
                   "--lr", str(args.lr), "--seed", str(seed),
                   "--residual", str(res), "--out", str(out)]
            if val[0]:
                cmd += ["--val-input", str(val[0]), "--val-target", str(val[1]),
                        "--val-valid", str(val[2]), "--val-every", str(args.val_every)]
            lines.append("cd %s && %s >> %s.log 2>&1" % (ROOT, " ".join(cmd), out))
        if not lines:
            return
        sh = out_root / f"runner_gpu{gpu}.sh"
        sh.write_text("#!/bin/bash\nset -e\n" + "\n".join(lines) + "\n")
        sh.chmod(0o755)
        print(f"== GPU {gpu} ==")
        for ln in lines:
            print("   " + ln.split("&& ")[1][:150])
        if not args.dry_run:
            subprocess.Popen(["bash", str(sh)], cwd=ROOT,
                             env={"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                                  "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                             stdout=open(out_root / f"runner_gpu{gpu}.log", "a"),
                             stderr=subprocess.STDOUT)

    items = [(res, s) for res in args.residual for s in args.seeds]
    for i, gpu in enumerate(args.gpus):
        launch(gpu, items[i::len(args.gpus)])
    (out_root / "grid.json").write_text(json.dumps(
        {"residual": args.residual, "seeds": args.seeds,
         "input": args.input_npz, "target": args.target_npz,
         "val_input": args.val_input_npz, "val_target": args.val_target_npz}, indent=1))
    print(f"\nwrote {out_root}/grid.json")


if __name__ == "__main__":
    main()

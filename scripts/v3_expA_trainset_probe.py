"""实验 A 附带诊断：在**网络自己训练过的那个体**上做同样的评价。

为什么需要这一步
----------------
主实验（4000 step）显示 faithfully 配对的 VALID 在留出仿真体上 `r_ves` 0.69–0.77（输入 0.946）、
`bg_std_ratio` 1.6–1.8（噪声被放大）。两种解释无法从留出体上区分：

  (a) **训练失败**：网络连自己见过的体都没学会降散斑（预算/损失/结构问题）；
  (b) **训练成功但不过关**：在训练体上有效，只是没泛化到留出体（仿真体间统计不匹配 / 记忆）。

本脚本给出决定性的一刀：把训练用的那个体（`sim_seed0`）当成"留出体"再评一次。
- 若训练体上同样差 → 属 (a)，长预算对照才有意义，且问题在任务/损失而不在泛化。
- 若训练体上明显好、留出体上差 → 属 (b)，这本身就是一个负结果（仿真协议不足以训练可迁移的盲点网络）。

同时校验 `train_log1p.npy` 与"从 `sim_seed0.npz` 现场重算的输入"逐位一致，
以确保被评价的体与训练时喂进去的体是同一个（缓存/预处理没有偷偷漂移）。

用法
----
    python scripts/v3_expA_trainset_probe.py --runs runs/v3_expA --gpu 0
产物
----
    reports/v3_expA/trainset_probe.json
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

from v3_expA_faithful import (infer, metrics, vessel_mask_dilated,  # noqa: E402
                             WIENER_W)
from v3_phase5_real_compare import wiener                            # noqa: E402


def load_sim(src: Path):
    d = np.load(src)
    # 与 v3_expA_faithful.py 一致：不裁体积（infer.py 自己补偶数再裁回）
    shape = tuple(int(n) for n in d["pd_linear"].shape)
    crop = tuple(slice(0, n) for n in shape)
    pd = d["pd_linear"][crop].astype(np.float32)
    gt = d["gt_flow_pd"][crop].astype(np.float32)
    vessels = d["vessels"][crop].astype(bool)
    valid = d["valid"][crop].astype(bool)
    s = float(np.median(pd[valid]))                  # 输入与真值共用同一个 s
    vol = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    vol[~valid] = 0.0
    truth = np.log1p(gt / max(s, 1e-12)).astype(np.float32)
    ves = vessel_mask_dilated(vessels, 1) & valid
    return vol, truth, valid, ves, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="data/processed/v3_sim/sim_seed0.npz",
                    help="训练所用的那个仿真体")
    ap.add_argument("--train-npy", default="data/processed/v3_sim/train_log1p.npy")
    ap.add_argument("--runs", default="runs/v3_expA")
    ap.add_argument("--arms", nargs="+", default=["A_rand", "A_fixed4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/v3_expA/trainset_probe.json")
    args = ap.parse_args()

    vol, truth, valid, ves, s = load_sim(ROOT / args.sim)
    ref_bg_std = float(np.std(vol[valid & ~ves]))

    # 一致性校验：训练喂进去的体 == 现场从 npz 重算的体？
    train_vol = np.load(ROOT / args.train_npy).astype(np.float32)
    same = bool(train_vol.shape == vol.shape and np.allclose(train_vol, vol, atol=1e-6))
    print(f"训练体 vs 重算体逐位一致: {same} (max|Δ|="
          f"{float(np.abs(train_vol - vol).max()) if train_vol.shape == vol.shape else 'shape mismatch'})")

    out = {"sim": args.sim, "train_npy": args.train_npy, "train_vol_matches": same,
           "scale_s": s, "valid_voxels": int(valid.sum()),
           "vessel_voxels_dilated": int((valid & ves).sum()),
           "note": "同一套 metrics()，只是把评价体换成**训练过的那个体**；这不是泛化读数",
           "baseline": {}, "arms": {}}

    out["baseline"]["input"] = metrics(vol, truth, valid, ves, ref_bg_std)
    w_out, _, _ = wiener(vol, WIENER_W, valid)
    out["baseline"][f"wiener_w{WIENER_W}"] = metrics(w_out, truth, valid, ves, ref_bg_std)
    out["baseline"]["zero"] = metrics(np.zeros_like(vol), truth, valid, ves, ref_bg_std)

    vol_npy = Path(args.out).parent / "trainset_input_log1p.npy"
    vol_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(vol_npy, vol)

    runs = ROOT / args.runs
    for arm in args.arms:
        for sd in args.seeds:
            ck = runs / f"{arm}_s{sd}"
            if not (ck / "latest.pt").exists():
                print(f"  [缺] {ck}/latest.pt — 跳过")
                continue
            pred = infer(ck, vol_npy, args.gpu)
            m = metrics(pred, truth, valid, ves, ref_bg_std)
            out["arms"][f"{arm}_s{sd}"] = m
            print(f"{arm}_s{sd:<5} r_ves={m['r_ves']:.4f} bg_std_ratio={m['bg_std_ratio']:.3f} "
                  f"rmse_ves={m['rmse_ves']:.4f} sep={m['sep_ratio']:.3f}")

    print("\n基线（同一个体）：")
    for k, v in out["baseline"].items():
        print(f"  {k:<12} r_ves={v['r_ves']:.4f} bg_std_ratio={v['bg_std_ratio']:.3f} "
              f"rmse_ves={v['rmse_ves']:.4f} sep={v['sep_ratio']:.3f}")

    dst = ROOT / args.out
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n→ {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

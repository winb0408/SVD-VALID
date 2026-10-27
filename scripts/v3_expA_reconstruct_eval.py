"""实验 A 的**推理口径更正**：同一批 checkpoint，改用相位重建推理再评一次。

发现的问题（2026-10-26 深夜）
------------------------------
`train.py` 喂给网络的是 `valid_random_tetris()` 返回的**子采样体**，形状 (D/2, H/2, W/2)——
每 2×2×2 块只取一个位置（随机 Tetris 图案），因此**训练时的输入密度是原图的 1/8**。
而 `infer.py` 把**满分辨率整卷**喂进同一个网络（滑窗 patch 32×64×64）。

→ 训练/推理的输入密度差 8 倍，空间频率、对比度、感受野所覆盖的结构尺度全部错位。
这足以单独解释实验 A 观察到的"输出 ≈ 输入的轻度低通 + 向均值收缩"：
网络从来没见过满密度的输入。

`scripts/v3_phase3_eval.py:reconstruct()` 的做法才是与训练一致的：把体积按 8 个相位各抽一次
（每个相位 = (D/2,H/2,W/2) 的子采样体），分别前向，再回填到各自格点。
上一轮我把 `reconstruct()` 标注为"非上游的八相位散回"并排除，**这个判断是错的**：
它是唯一与训练输入分布匹配的推理方式。本脚本用同一批 checkpoint 重算，给出更正后的数字。

用法
----
    python scripts/v3_expA_reconstruct_eval.py --runs runs/v3_expA --gpu 0
    python scripts/v3_expA_reconstruct_eval.py --runs runs/v3_expA_long --gpu 1
产物
----
    <out>/reconstruct_metrics.json（与 expA_metrics.json 同结构，另存，不覆盖旧记录）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.network import NetworkCNR                       # noqa: E402
from v3_phase3_eval import phase_slices                     # noqa: E402
from v3_expA_faithful import (load_vol_and_truth, metrics,   # noqa: E402
                              n2n_metrics, summarise, vessel_mask_dilated)
from datasets.sampling import strided_phase                  # noqa: E402


def reconstruct_model(model, vol, stride, device):
    """8 相位各前向一次并回填（与训练输入密度一致）。"""
    out = np.zeros(vol.shape, np.float32)
    with torch.no_grad():
        for i in range(8):
            view = strided_phase(torch.from_numpy(vol), i, stride)
            y = model(view[None, None].to(device))[0, 0].detach().cpu().numpy()
            out[phase_slices(vol.shape, stride, i)] = y
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sim", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--pair-sim", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--runs", default="runs/v3_expA")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["A_rand", "A_fixed4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(args.runs).parent / "reports" / Path(args.runs).name
    out_dir = ROOT / out_dir if not out_dir.is_absolute() else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    vol, truth, valid, ves, s, vol2 = load_vol_and_truth(
        ROOT / args.eval_sim, ROOT / args.pair_sim)
    # 相位抽取要求每个轴能被 2*stride 整除 → 37 层必须裁成 36。
    # 这与 infer.py 口径**不同**（那边补偶数再裁回，用满 37 层），差异只有 1/37 的体积，
    # 但两条口径的绝对数字因此不能直接混用，只能各自与同口径的输入基线比。
    shape = tuple((n // 2) * 2 for n in vol.shape)
    cr = tuple(slice(0, n) for n in shape)
    vol, truth, valid, ves = vol[cr], truth[cr], valid[cr], ves[cr]
    vol2 = vol2[cr]
    ref_bg_std = float(np.std(vol[valid & ~ves]))
    print(f"评估体 {args.eval_sim} shape={vol.shape}（相位抽取要求偶数，37→36） "
          f"valid={int(valid.sum())} vessel ROI={int((valid & ves).sum())} s={s:.4g}")

    rec = {"eval_sim": args.eval_sim, "pair_sim": args.pair_sim,
           "shape": list(vol.shape), "valid_voxels": int(valid.sum()),
           "vessel_voxels_dilated": int((valid & ves).sum()), "scale_s_eval": s,
           "inference_api": "phase reconstruction (8 phases, stride=1) — 与训练输入密度一致",
           "note": "同一批 checkpoint、同一真值/ROI，只换推理口径；"
                   "与 expA_metrics.json 的 infer.py 口径并存，不覆盖",
           "arms": {}, "baseline": {}}

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for arm in args.arms:
        for sd in args.seeds:
            ck = ROOT / args.runs / f"{arm}_s{sd}" / "latest.pt"
            if not ck.exists():
                print(f"  [缺] {ck}")
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            a = blob["args"]
            model = NetworkCNR(features=a.get("features", 16), groups=a.get("groups", 4)).to(device)
            model.load_state_dict(blob["model"])
            model.eval()
            pred = reconstruct_model(model, vol, int(a.get("stride", 1)), device)
            m = metrics(pred, truth, valid, ves, ref_bg_std)
            m.update(n2n_metrics(pred, vol, vol2, valid, ves))
            m["ckpt_budget"] = {"epochs": a.get("epochs"), "steps_per_epoch": a.get("steps_per_epoch")}
            rec["arms"].setdefault(arm, {})[str(sd)] = m
            print(f"{arm:9s} s{sd}  r_ves={m['r_ves']:.3f}  bg_std_ratio={m['bg_std_ratio']:.3f}  "
                  f"n2n={m['n2n_ratio']:.3f}  rmse_ves={m['rmse_ves']:.4f}")

    rec["summary"] = summarise(rec["arms"])
    base = metrics(vol, truth, valid, ves, ref_bg_std)
    base.update(n2n_metrics(vol, vol, vol2, valid, ves))
    rec["baseline"]["input"] = {"mean": base, "min": base, "max": base, "n_seeds": 1}
    print(f"\n输入基线  r_ves={base['r_ves']:.4f}  bg_std_ratio=1.000  n2n=1.000")
    print("\n=== 与输入的直接对照（相位重建口径）===")
    for a in sorted(rec["summary"]):
        sm = rec["summary"][a]
        print(f"{a:9s} Δr_ves={sm['r_ves']['mean'] - base['r_ves']:+.3f}"
              f"(r_ves={sm['r_ves']['mean']:.3f})  n2n={sm['n2n_ratio']['mean']:.3f}  "
              f"bg_std_ratio={sm['bg_std_ratio']['mean']:.3f}")

    dst = out_dir / "reconstruct_metrics.json"
    dst.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\n→ {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""S0 归因诊断②：网络在**它自己训练的那个几何里**到底学到了没有？

为什么要做这个诊断
------------------
S0 的部署口径（`v3_round2_s0_eval.py`，与 `infer.py` 同义：满分辨率 patch 过网络）
全部 n2n > 1。这留下两种完全不同的解释，必须分开：

  (i)  网络**没学到**任何去噪算子（训练本身失败）；
  (ii) 网络学到了，但那个算子只在**训练几何**（半尺寸相位视图、输入相位→配对相位）
       里有效，部署时换了几何所以用不上。

判别方法：把训练用的那套采样几何**原样**搬到评价端，用同一个 checkpoint，
问"给定输入视图 `x_a`，网络预测配对视图 `y_a` 能不能比 `x_a` 本身更接近 `y_a`"。
这与 `n2n_ratio` 是同一个构造（分母就是输入本身），只是换了配对与几何：

    ratio = [MSE(pred_a, y_a) + MSE(pred_b, y_b)] / [MSE(x_a, y_a) + MSE(x_b, y_b)]

* ratio < 1 ⇒ 网络在该几何里**确实**比"什么都不做"更接近目标；
* ratio ≥ 1 ⇒ 连训练几何里也没学到，问题在优化/容量。

同时在**训练体**与**留出体**上各算一遍：两者接近 ⇒ 不是过拟合；
留出体上仍 < 1 ⇒ 学到的是可泛化的算子（只是几何不对）。

用法
----
    python -u scripts/v3_round2_s0_train_geometry.py --level low \
        --runs runs/v3_round2_s0/low --arms A_rand S_rand --seeds 3407 3408
输出 `reports/v3_round2/s0_train_geometry.json`。
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

from datasets.sampling import valid_random_tetris, tetris_from_pattern   # noqa: E402
from v3_round2_s0_eval import load_eval                                   # noqa: E402
from v3_round2_s0_phase_infer import load_model                           # noqa: E402

PATCH = (32, 64, 64)
N_PATCH = 8


def sample_origins(shape, rng):
    return [tuple(int(rng.integers(0, max(s - p, 0) + 1)) for s, p in zip(shape, PATCH))
            for _ in range(N_PATCH)]


def views_of(patch, mode, gen):
    if mode == "fixed":
        return [v.numpy() for v in tetris_from_pattern(patch, 4)]
    return [v.numpy() for v in valid_random_tetris(patch, generator=gen)]


@torch.no_grad()
def ratio_on_volume(model, vol, mode, device, seed):
    """返回 {mse_pred, mse_input, ratio, per_role}，口径见模块 docstring。"""
    rng = np.random.default_rng(seed)
    gen = torch.Generator()
    num = den = 0.0
    per_role = {"a": [0.0, 0.0], "b": [0.0, 0.0]}
    for origin in sample_origins(vol.shape, rng):
        patch = vol[origin[0]:origin[0] + PATCH[0],
                    origin[1]:origin[1] + PATCH[1],
                    origin[2]:origin[2] + PATCH[2]]
        gen.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
        xa, xb, ya, yb = views_of(patch, mode, gen)
        for tag, xi, yi in (("a", xa, ya), ("b", xb, yb)):
            t = torch.from_numpy(np.ascontiguousarray(xi, np.float32))[None, None].to(device)
            p = model(t)[0, 0].cpu().numpy()
            n = float(np.mean((p - yi) ** 2))
            d = float(np.mean((xi - yi) ** 2))
            num += n
            den += d
            per_role[tag][0] += n
            per_role[tag][1] += d
    return {"mse_pred": num, "mse_input": den, "ratio": num / den,
            "per_role": {k: v[0] / v[1] for k, v in per_role.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--arms", nargs="+", default=["A_rand", "S_rand"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/v3_round2/s0_train_geometry.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runs = Path(args.runs or f"runs/v3_round2_s0/{args.level}")
    man = json.loads((ROOT / f"data/processed/v3_round2_s0/{args.level}/manifest.json").read_text())

    train_vols = [np.load(ROOT / t["log1p"]).astype(np.float32) for t in man["train"]]
    eval_es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]
    print(f"[{args.level}] train 体 {len(train_vols)} 卷；留出体 {[Path(E['npz']).stem for E in eval_es]}")

    rec = {"level": args.level, "patch": list(PATCH), "n_patch_per_vol": N_PATCH,
           "note": ("ratio = MSE(pred,配对视图)/MSE(输入视图,配对视图) 在**训练几何**里算；"
                    "<1 表示网络在该几何里确实优于恒等。train=训练体，eval=留出体。"),
           "arms": {}}
    for arm in args.arms:
        mode = "fixed" if "fixed" in arm else "random"
        for seed in args.seeds:
            ck = ROOT / runs / f"{arm}_s{seed}" / "latest.pt"
            if not ck.exists():
                print(f"  [skip] {ck} 不存在")
                continue
            model = load_model(ck, device)
            tr = [ratio_on_volume(model, v, mode, device, 9000 + i) for i, v in enumerate(train_vols)]
            ev = [ratio_on_volume(model, E["vol"], mode, device, 8000 + i)
                  for i, E in enumerate(eval_es)]
            agg = lambda L: sum(r["mse_pred"] for r in L) / sum(r["mse_input"] for r in L)
            tr_r, ev_r = agg(tr), agg(ev)
            rec["arms"][f"{arm}_s{seed}"] = {
                "mode": mode,
                "train_ratio": tr_r, "eval_ratio": ev_r,
                "train_ratio_per_vol": [r["ratio"] for r in tr],
                "eval_ratio_per_vol": [r["ratio"] for r in ev],
                "eval_per_role": [r["per_role"] for r in ev],
            }
            print(f"  [{arm} s{seed}] 训练几何 ratio: train={tr_r:.3f} eval={ev_r:.3f} "
                  f"（<1 = 在该几何里优于恒等）")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

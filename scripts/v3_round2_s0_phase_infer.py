"""S0 归因诊断：**训练/推理的空间尺度是否一致**？

为什么要做这个诊断
------------------
训练时网络看到的输入是 `valid_random_tetris` / `tetris_from_pattern` 产生的
**相位视图**：每 2×2×2 块只取 4 个（或 1 个）角点，尺寸是原 patch 的一半。
推理时 `infer.py` 直接把网络套在**满分辨率** patch 上（`infer.py` 无任何相位逻辑）。
如果网络学到的映射依赖"输入是 2 倍稀的网格"，那么满分辨率推理就会把同一个算子
作用在"频率只有一半"的内容上——表现为系统性偏移/糊，而不是随机噪声。

这不是"猜测"，可以直接测：
* 方案 A（现状）：满分辨率 patch 推理；
* 方案 B（相位一致）：把体按 8 个相位各自抽稀 → 网络 → 按相位拼回原分辨率。
两条都用同一个 checkpoint、同一台机器、同一个评价函数，只差"输入的采样方式"。
若 B 明显优于 A，则训练/推理尺度失配被证实，且 S0 的负结果**不能**归因于方法。

用法
----
    python -u scripts/v3_round2_s0_phase_infer.py --level low \
        --runs runs/v3_round2_s0/low --arms A_rand A_fixed4 S_rand S_fixed4
输出 `reports/v3_round2/s0_phase_infer.json`。
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

from models.network import NetworkCNR                    # noqa: E402
from datasets.sampling import strided_phase               # noqa: E402
from v3_round2_s0_eval import load_eval, metrics_s0       # noqa: E402


def load_model(ckpt: Path, device):
    ck = torch.load(ckpt, map_location=device)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    feats = 16
    for k, v in sd.items():
        if k.endswith("first.weight") or (".first." in k and k.endswith("weight") and v.ndim == 5):
            feats = int(v.shape[0]); break
    m = NetworkCNR(features=feats, groups=4).to(device).eval()
    m.load_state_dict(sd)
    return m


@torch.no_grad()
def infer_full(model, vol: np.ndarray, device) -> np.ndarray:
    """现状口径：满分辨率整卷过网络（与 infer.py 同义，只是不分 patch——体很小）。"""
    x = torch.from_numpy(np.ascontiguousarray(vol, np.float32))[None, None].to(device)
    return model(x)[0, 0].cpu().numpy()


@torch.no_grad()
def infer_phase_consistent(model, vol: np.ndarray, device) -> np.ndarray:
    """相位一致口径：8 个相位各自抽稀 → 网络 → 拼回原分辨率。

    各轴若不是偶数（37 行），先补一行再切，最后裁回原尺寸。
    """
    z, y, x = vol.shape
    pz, py, px = (z + z % 2), (y + y % 2), (x + x % 2)
    v = np.zeros((pz, py, px), np.float32)
    v[:z, :y, :x] = vol
    out = np.zeros((pz, py, px), np.float32)
    for i in range(8):
        ph = strided_phase(v, i, 1)
        t = torch.from_numpy(np.ascontiguousarray(ph, np.float32))[None, None].to(device)
        o = model(t)[0, 0].cpu().numpy()
        dz, dy, dx = i >> 2, (i >> 1) & 1, i & 1
        out[dz::2, dy::2, dx::2] = o
    return out[:z, :y, :x]


@torch.no_grad()
def infer_avg8(model, vol: np.ndarray, device) -> np.ndarray:
    """方案 C：8 个相位各自过网络后**平均再升采样**（不是各自拼回原位）。

    B 失败的一个可能原因是"按相位原位贴回"引入了格点错位；C 用重复升采样把 8 个
    相位的输出平均到每个体素，避开了相位索引，只保留"稀网格 → 满网格"的尺度换算。
    若 C 也不比 A 好，则"训练/推理尺度失配可以被简单的相位重装配救回"这条路被关掉。
    """
    z, y, x = vol.shape
    pz, py, px = (z + z % 2), (y + y % 2), (x + x % 2)
    v = np.zeros((pz, py, px), np.float32)
    v[:z, :y, :x] = vol
    acc = np.zeros((pz, py, px), np.float32)
    for i in range(8):
        ph = strided_phase(v, i, 1)
        t = torch.from_numpy(np.ascontiguousarray(ph, np.float32))[None, None].to(device)
        o = model(t)[0, 0].cpu().numpy()
        acc += np.repeat(np.repeat(np.repeat(o, 2, 0), 2, 1), 2, 2)[:pz, :py, :px]
    return (acc / 8.0)[:z, :y, :x]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--arms", nargs="+", default=["A_rand", "A_fixed4", "S_rand", "S_fixed4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    runs = Path(args.runs or f"runs/v3_round2_s0/{args.level}")
    man = json.loads((ROOT / f"data/processed/v3_round2_s0/{args.level}/manifest.json").read_text())
    Es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]
    rec = {"level": args.level, "ends": {}, "note":
           "A=满分辨率推理（现状，与 infer.py 同义）；B=相位一致推理（8 相位抽稀→网络→按相位原位拼回）；"
           "C=8 相位输出平均后重复升采样（避开相位索引，只保留稀→满的尺度换算）"}
    print(f"[{args.level}] 评估体 {[Path(E['npz']).stem for E in Es]}")

    for arm in args.arms:
        for seed in args.seeds:
            ck = ROOT / runs / f"{arm}_s{seed}" / "latest.pt"
            if not ck.exists():
                continue
            model = load_model(ck, device)
            per = {}
            for E in Es:
                name = Path(E["npz"]).stem
                pa = metrics_s0(infer_full(model, E["vol"], device).astype(np.float32), E)
                pb = metrics_s0(infer_phase_consistent(model, E["vol"], device).astype(np.float32), E)
                pc = metrics_s0(infer_avg8(model, E["vol"], device).astype(np.float32), E)
                keys = ("n2n_ratio", "r_roi", "bg_std_ratio", "rmse_vs_imaging")
                per[name] = {"full_res": {k: pa[k] for k in keys},
                             "phase_consistent": {k: pb[k] for k in keys},
                             "avg8_upsample": {k: pc[k] for k in keys}}
                print(f"  [{arm} s{seed}] {name}: "
                      f"A(full) n2n={pa['n2n_ratio']:.3f} r_roi={pa['r_roi']:.4f} | "
                      f"B(phase) n2n={pb['n2n_ratio']:.3f} r_roi={pb['r_roi']:.4f} | "
                      f"C(avg8) n2n={pc['n2n_ratio']:.3f} r_roi={pc['r_roi']:.4f}")
            rec["ends"][f"{arm}_s{seed}"] = per
    out = ROOT / "reports/v3_round2/s0_phase_infer.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V3 Phase 2 / T2-1（前半）+ T3-1 骨架：步距 m 的 Tetris 采样器与**几何/覆盖合法性**核验。

现有 `datasets/sampling.py::one_phase` 把 2×2×2 块的一个角点抽成一张半分辨率体数据，
8 个相位铺满整幅体积（每个体素恰属于 1 个相位）。把视图间距放大到 m 之后，
一个块里只有 8 个角点被采到，块内其余 (2m)³−8 个体素**不属于任何相位**。
这不是实现细节，它决定了两件事：
  1. 训练时每一步能看到的体素比例；
  2. 推理时要不要用 m³ 个平移格点把整幅体积补全（每个格点一次前向）。

本脚本不训练，只把这两件事算成表格，并对 m=1 做一次**与现有实现的回归一致性检查**
（m=1 时新采样器必须与 `one_phase` 逐体素相同，不同就说明采样器写错了）。

用法：
    python -u scripts/v3_physics_sampler.py
输出 `data/processed/v3_sampling_geometry.json`。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.sampling import TETRIS_PATTERNS, one_phase  # noqa: E402

OUT = Path("data/processed/v3_sampling_geometry.json")
SHAPE = (37, 64, 64)


def usable(n, m):
    """该轴可用长度：能被 2m 整除的最大前缀。与 `one_phase` 在 m=1 时一致（37 -> 36）。"""
    return (n // (2 * m)) * (2 * m)


def phase_index(shape, m, index, shift=(0, 0, 0)):
    """相位 index 覆盖的体素下标（各轴一维数组）。off = m*(o) + shift。"""
    dz, dy, dx = index // 4, (index // 2) % 2, index % 2
    ax = []
    for n, o, s in zip(shape, (dz, dy, dx), shift):
        off = m * o + s
        # 块内角点位置：off, off+2m, off+4m, ...（只在可用前缀内，保证 8 个相位等长）
        ax.append(np.arange(off, usable(n, m), 2 * m))
    return ax


def phase_mask(shape, m, index, shift=(0, 0, 0)):
    mask = np.zeros(shape, bool)
    z, y, x = phase_index(shape, m, index, shift)
    mask[np.ix_(z, y, x)] = True
    return mask


def coarse(volume, m, index, shift=(0, 0, 0)):
    """抽成稠密的下采样体数据（形状 = 每轴块数），即网络实际看到的一张视图。"""
    z, y, x = phase_index(volume.shape, m, index, shift)
    return volume[np.ix_(z, y, x)]


def check_partition(shape, m, shift=(0, 0, 0)):
    """8 个相位必须互斥；与"块内只采 8 个角点"一致。"""
    masks = [phase_mask(shape, m, i, shift) for i in range(8)]
    stack = np.stack(masks).sum(0)
    return {"disjoint": bool((stack <= 1).all()),
            "n_sampled": int(stack.sum()),
            "no_overlap_pairs": int(sum(masks[a].astype(int).sum()
                                        for a in range(8)) == stack.sum())}


def complement_of(p):
    return next(q for q in range(8) if set(TETRIS_PATTERNS[q].tolist())
                == set(range(8)) - set(TETRIS_PATTERNS[p].tolist()))


def main():
    rec = {"shape": SHAPE, "m": {}}
    vol = np.arange(np.prod(SHAPE)).reshape(SHAPE).astype(np.float64)

    # --- 与现有实现的回归一致性（只在 m=1 有意义）
    ok = True
    for i in range(8):
        ref = one_phase(torch.as_tensor(vol), i).numpy()
        new = coarse(vol, 1, i)
        if ref.shape != new.shape or not np.array_equal(ref, new):
            ok = False
            print(f"  ✗ phase {i}: 形状 {ref.shape} vs {new.shape}")
    print(f"m=1 与 datasets/sampling.py::one_phase 逐体素一致: {ok}")
    if not ok:
        raise SystemExit("采样器与现有实现不一致，先修采样器再往下算")

    for m in (1, 2, 3):
        chk = check_partition(SHAPE, m)
        nb = [len(phase_index(SHAPE, m, 0)[k]) for k in range(3)]
        per_block = (2 * m) ** 3
        crop = tuple(usable(n, m) for n in SHAPE)
        cov = chk["n_sampled"] / np.prod(crop)      # 相对裁剪后体积
        # 用 m³ 个平移格点可以铺满整幅体积（每轴 m 个平移量）
        row = {
            "blocks_per_axis": nb,
            "view_shape_per_phase": nb,
            "voxels_per_view": int(np.prod(nb)),
            "sampled_voxels": chk["n_sampled"],
            "coverage": float(cov),
            "crop": list(crop),
            "dropped_frac_of_original": float(1 - np.prod(crop) / np.prod(SHAPE)),
            "coverage_ideal_block": 8.0 / per_block,
            "disjoint": chk["disjoint"],
            "lattices_for_full_coverage": int(m ** 3),
            "relative_forward_cost": float(m ** 3),
            "inference_voxels_ratio": 1.0,
        }
        rec["m"][m] = row
        print(f"\nm={m}: 每轴块数 {nb}  单视图 {nb} = {np.prod(nb)} 体素"
              f"  采到 {chk['n_sampled']} = 体积的 {cov*100:.2f}%"
              f"  (块内理想值 {8/per_block*100:.2f}%)")
        print(f"   8 相位互斥: {chk['disjoint']}   铺满全幅需 {m**3} 个平移格点"
              f"（推理前向次数 ×{m**3}）")

    # --- m=3 的裁边方案：三个轴都要能被 2m 整除
    print("\n裁边方案（要求每轴能被 2m 整除）：")
    for m in (1, 2, 3):
        crop = tuple(usable(n, m) for n in SHAPE)
        nb = tuple(n // (2 * m) for n in crop)
        print(f"  m={m}: {SHAPE} -> {crop}   每轴块数 {nb}"
              f"   丢弃原体积 {100 * (1 - np.prod(crop) / np.prod(SHAPE)):.1f}%")
        rec["m"][m]["crop_blocks"] = list(nb)

    rec["m1_matches_one_phase"] = ok
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

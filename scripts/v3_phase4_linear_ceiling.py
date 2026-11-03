"""盲点法的**闭式线性**上界：给定同一 2x2x2 块里其余 7 个相位，被扣掉那个体素的
最优线性预测是什么样。

## 为什么要做这个（2026-10-21，见 L46/L47）

`strided_phase` 交出的是**按相位下采样**的视图，所以训练任务是：
用同一体数据在偶数子格的 7/8 体素，预测剩下 1/8 体素。这是一个**插值**任务。
如果场的相关长度只有 1–2 体素，那么 E[被扣体素 | 其余 7 个] 必然被**衰减**——
斜率趋近于相关系数 ρ，而不是 1。Phase 1 实测 stride=1 的跨相位 ρ 均值是 0.353。
而 L45 实测训练出的网络斜率是 0.198（arm A），本轮把 `weight_identity` 降到 0 也只到
0.336 —— 与 ρ=0.353 几乎重合。

若"最优线性预测"的斜率与得分与训练出的网络相当，则结论是：
**这个衰减是采样几何决定的，不是训练缺陷、不是损失缺陷**，动任何超参数都到不了
斜率 1。这是"方法本身受限"的判定证据，而不是"再调调就好"。

## 做法

把体积按 2x2x2 分块，每块 8 个角点。对每个角点 c：
  特征 = 其余 7 个角点的值（同一块内，空间相邻）
  目标 = 角点 c 的值
在**整个体数据上**做一次最小二乘（场近似平稳），得到该角点的最优线性预测。
8 个角点各拟合一次，拼回完整体积。

判据与 `v3_phase4_scoreboard.py` 完全一致（同一掩膜、同一 n2n 定义、同一 slope 定义），
以便与训练出的臂逐数对比。

用法：
    python scripts/v3_phase4_linear_ceiling.py \
        --obs1 data/processed/v3_sim/sim_seed1.npz \
        --obs2 data/processed/v3_sim/sim_seed1_n201_s1.npz \
        --out reports/v3_phase4_2026-10-21/linear_ceiling.json
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

from v3_phase3_eval import M_EVAL, usable, support_mask  # noqa: E402
from scipy.ndimage import distance_transform_edt              # noqa: E402

CORNERS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def to_log(lin, s):
    return np.log1p(lin / max(s, 1e-12)).astype(np.float32)


def blocks(x):
    """(Z,Y,X) -> (nz,ny,nx,8) 的 2x2x2 块角点视图。"""
    z, y, xx = x.shape
    b = x.reshape(z // 2, 2, y // 2, 2, xx // 2, 2)
    b = b.transpose(0, 2, 4, 1, 3, 5)            # (nz, ny, nx, 2,2,2)
    return b.reshape(z // 2, y // 2, xx // 2, 8)


def fit_predict(corner_values, valid_flat):
    """对每个角点拟合 目标 ~ 其余 7 个 + 截距，返回拼回原形状的预测。"""
    nz, ny, nx, _ = corner_values.shape
    preds = np.empty((nz, ny, nx, 8), dtype=np.float32)
    coefs = []
    for ci in range(8):
        y = corner_values[..., ci].reshape(-1)
        F = np.delete(corner_values.reshape(-1, 8), ci, axis=1).astype(np.float64)
        m = valid_flat
        A = np.concatenate([F[m], np.ones((int(m.sum()), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(A, y[m].astype(np.float64), rcond=None)
        Af = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
        preds[..., ci] = (Af @ coef).astype(np.float32)
        coefs.append(coef.tolist())
    # 拼回 (Z,Y,X)
    out = preds.reshape(nz, ny, nx, 2, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    return out.reshape(nz * 2, ny * 2, nx * 2)


def score(pred, obs1, obs2, truth, masks):
    common, ves, far = masks["common"], masks["ves"], masks["far"]

    def mse(a, b, m):
        return float(np.mean((a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2))

    x = obs1[common].astype(np.float64)
    y = pred[common].astype(np.float64)
    return {
        "n2n_ratio": mse(pred, obs2, common) / mse(obs1, obs2, common),
        "n2n_ratio_bg": mse(pred, obs2, far) / mse(obs1, obs2, far),
        "r": float(np.corrcoef(pred[common], truth[common])[0, 1]),
        "r_ves": float(np.corrcoef(pred[ves], truth[ves])[0, 1]),
        "slope_vs_input": float(np.polyfit(x, y, 1)[0]),
        "std_ratio": float(y.std() / x.std()),
        "bg_mean": float(np.mean(pred[far])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/linear_ceiling.json")
    args = ap.parse_args()

    d1 = np.load(args.obs1)
    shape = usable(d1["pd_linear"].shape, M_EVAL)
    shape = tuple((n // 32) * 32 for n in shape)
    crop = tuple(slice(0, n) for n in shape)

    pd1 = d1["pd_linear"][crop].astype(np.float32)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))
    obs1 = to_log(pd1, s)
    obs2 = to_log(np.load(args.obs2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)

    common = support_mask(shape, M_EVAL) & valid
    ves = vessels & common
    far = (common & (distance_transform_edt(~vessels) >= 4)
           & (distance_transform_edt(valid) >= 2))
    masks = {"common": common, "ves": ves, "far": far}

    cv = blocks(obs1)
    # 拟合只用块内 8 个角点都有效的块，避免填充带进回归
    ok = blocks(valid.astype(np.float32)).min(axis=-1) > 0.5
    # 块级有效掩膜：块内 8 个角点全有效才参与拟合（每块对应 F 的一行）
    ok_flat = ok.reshape(-1)

    nz, ny, nx = ok.shape
    preds = np.empty_like(cv)
    coefs = []
    for ci in range(8):
        y = cv[..., ci].reshape(-1)
        F = np.delete(cv.reshape(-1, 8), ci, axis=1).astype(np.float64)
        A = np.concatenate([F[ok_flat], np.ones((int(ok_flat.sum()), 1))], axis=1)
        coef, *_ = np.linalg.lstsq(A, y[ok_flat].astype(np.float64), rcond=None)
        Af = np.concatenate([F, np.ones((F.shape[0], 1))], axis=1)
        preds[..., ci] = (Af @ coef).astype(np.float32).reshape(nz, ny, nx)
        coefs.append(coef)
    pred = preds.reshape(nz, ny, nx, 2, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    pred = pred.reshape(nz * 2, ny * 2, nx * 2)

    ctx_sum = np.array([c[:7].sum() for c in coefs])
    res = {
        "shape": list(shape),
        "note": "线性最小二乘盲点预测；特征=同一 2x2x2 块内其余 7 个角点",
        "coef_sum_per_corner": ctx_sum.tolist(),
        "coef_sum_mean": float(ctx_sum.mean()),
        "coef_const_per_corner": [float(c[7]) for c in coefs],
        "linear_ceiling": score(pred, obs1, obs2, truth, masks),
        "input": score(obs1, obs1, obs2, truth, masks),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print(f"masks: common={int(common.sum())} ves={int(ves.sum())} far={int(far.sum())}")
    print(f"7 个上下文系数之和（= 斜率上界的理论量）: 均值 {ctx_sum.mean():.4f}  "
          f"范围 [{ctx_sum.min():.4f}, {ctx_sum.max():.4f}]")
    print(f"\n{'对象':22s} {'n2n':>7s} {'n2n_bg':>7s} {'r':>7s} {'r_ves':>7s} "
          f"{'slope':>7s} {'stdR':>7s}")
    for k in ("input", "linear_ceiling"):
        m = res[k]
        print(f"{k:22s} {m['n2n_ratio']:7.3f} {m['n2n_ratio_bg']:7.3f} {m['r']:7.3f} "
              f"{m['r_ves']:7.3f} {m['slope_vs_input']:7.3f} {m['std_ratio']:7.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

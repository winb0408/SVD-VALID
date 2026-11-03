"""Phase 4 方案③：**结构保持的方差自适应收缩（SPW, structure-preserving Wiener）**。

## 为什么要它（L49）

L48 的 ①（`v3_phase4_wiener.py`）已经通过：n2n 0.754（w=7），但对它做分区拆解后，
n2n 的**分子**几乎全落在两类体素上，且两类的最优处理方向相反：

    区域              体素占比   纯局部均值(b=0) 的 n2n   身份(b=1) 的 n2n
    远端背景 (far)     74%         0.607                  1.000
    血管内  (ves)      1.3%      101.8                    1.000

背景里"共享成分"是缓慢的灰度平台，平滑能实打实去掉噪声；血管里共享成分是 1 体素
尺度的亮峰，平滑把峰削掉，误差发散。所以任何单一窗口的 `out = mu + b(x−mu)`
都被这两端拉扯。① 用局部方差自动在两端之间插值，因此成立，但它有两处系统性浪费：

  1. `b` 被 clip 到 [0,1]，血管处只能"不处理"。可是实测**单点锐化**
     `x + γ(x−mu)`（γ=0.05, w=5）能把 ves 的 n2n 压到 0.517 —— 血管处最优方向不是
     "保持"，而是"轻微锐化"（噪声在血管内相对幅度更小，锐化带来的结构恢复大于噪声放大）。
  2. `mu` 用**均匀窗**，会把邻接血管的亮峰糊进背景体素，使背景的 mu 系统性偏高；
     这就是 far 的 n2n 停在 0.607–0.64 而理论可更低的原因之一。

③ 就是把这两点补上：用**引导滤波（guided filter）**做保边平滑得到 mu（平坦区退化为
局部均值，亮峰处 a→1 自动保留），并允许 `b` 上探到 `bmax ≥ 1`。

    mu  = guided(x; r, eps)                    保边平滑
    lv  = 局部方差（原始 x 上的窗 w）
    b   = clip(bmax · (1 − σ²/lv), 0, bmax)    σ² = median(lv over common)
    out = mu + b (x − mu)

注意 `1 − σ²/lv` 恒 ≤ 1，所以"血管处要不要锐化"这件事**不能**靠给 `b` 加截断上限来实现
（第一版就是这样错的，`bmax` 完全是死参数，实测三种上限给出逐位相同的 n2n）。
必须把 `bmax` 乘进去：平坦区 lv≈σ² ⇒ b≈0（强平滑）；结构区 lv≫σ² ⇒ b→bmax，
`bmax > 1` 时结构区就是"轻微锐化"。这正是分区 oracle 指出的两端解
（背景 b≈0、血管 b≈1.09）。

全部量只用 obs1，不用 obs2、不用掩膜、不用训练。自由参数 (r, eps, w, bmax)。

## 判据（沿用预注册，不改）

n2n ≤ 0.95 ∧ r_ves ≥ 输入 r_ves − 0.01 ∧ gap = (1−n2n)/(1−ceiling) ≥ 0.50，
ceiling 由与 obs2 不相交的实现均值给出。

用法：
    python scripts/v3_phase4_spw.py --out reports/v3_phase4_2026-10-21/spw
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase3_eval import M_EVAL, usable, support_mask          # noqa: E402
from v3_phase4_scoreboard import to_log, score                   # noqa: E402


def guided_smooth(x, r, eps):
    """引导滤波（引导图 = x 自身）：平坦区→局部均值，边缘处→保留 x。"""
    k = 2 * r + 1
    m = uniform_filter(x, k)
    v = np.maximum(uniform_filter(x * x, k) - m * m, 0.0)
    a = v / (v + eps)
    b = m - a * m
    return uniform_filter(a, k) * x + uniform_filter(b, k)


def local_var(x, w):
    mu = uniform_filter(x, w)
    return mu, np.maximum(uniform_filter(x * x, w) - mu * mu, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--ceiling-reals", nargs="+",
                    default=["data/processed/v3_sim/sim_seed1_n202_s1.npz",
                             "data/processed/v3_sim/sim_seed1_n203_s1.npz",
                             "data/processed/v3_sim/sim_seed1_n204_s1.npz"])
    ap.add_argument("--mu-types", nargs="+", default=["uniform", "guided"])
    ap.add_argument("--radii", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--eps-scale", nargs="+", type=float, default=[2.0, 8.0])
    ap.add_argument("--windows", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--bmax", nargs="+", type=float, default=[1.0, 1.05, 1.1, 1.15])
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/spw")
    args = ap.parse_args()

    d1 = np.load(args.obs1)
    shape = tuple((n // 32) * 32 for n in usable(d1["pd_linear"].shape, M_EVAL))
    crop = tuple(slice(0, n) for n in shape)
    pd1 = d1["pd_linear"][crop].astype(np.float32)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))

    x = to_log(pd1, s).astype(np.float64)
    obs2 = to_log(np.load(args.obs2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)
    acc = np.zeros(shape, np.float64)
    for p in args.ceiling_reals:
        acc += to_log(np.load(p)["pd_linear"][crop].astype(np.float32), s)
    acc /= len(args.ceiling_reals)

    common = support_mask(shape, M_EVAL) & valid
    masks = {
        "common": common,
        "ves": vessels & common,
        "far": (common & (distance_transform_edt(~vessels) >= 4)
                & (distance_transform_edt(valid) >= 2)),
    }
    ceiling = score(acc, x.astype(np.float32), obs2, truth, masks)["n2n_ratio"]
    ref = score(x.astype(np.float32), x.astype(np.float32), obs2, truth, masks)
    sig2 = {}
    for w in args.windows:
        _, lv = local_var(x, w)
        sig2[w] = float(np.median(lv[common]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    print(f"天花板（与 obs2 不相交的 {len(args.ceiling_reals)} 实现均值）= {ceiling:.4f}  "
          f"输入 r_ves = {ref['r_ves']:.4f}")
    print(f"{'mu':>8s} {'r':>3s} {'c':>4s} {'w':>3s} {'bmax':>5s} {'n2n':>7s} {'bg':>7s} "
          f"{'ves':>7s} {'r_ves':>7s} {'gap':>6s} {'b(bg/ves)':>11s} 判定")
    grid = []
    for mu_type in args.mu_types:
        cs = [1.0] if mu_type == "uniform" else args.eps_scale
        grid += [(mu_type, r, c, w, bm) for r, c, w, bm in
                 itertools.product(args.radii, cs, args.windows, args.bmax)]
    for mu_type, r, c, w, bmax in grid:
        if mu_type == "uniform":
            mu = uniform_filter(x, 2 * r + 1)
        else:
            mu = guided_smooth(x, r, c * sig2[w])
        _, lv = local_var(x, w)
        b = np.clip(bmax * (1.0 - sig2[w] / np.maximum(lv, 1e-9)), 0.0, bmax)
        pred = (mu + b * (x - mu)).astype(np.float32)
        m = score(pred, x.astype(np.float32), obs2, truth, masks)
        gap = (1.0 - m["n2n_ratio"]) / (1.0 - ceiling)
        passed = (m["n2n_ratio"] <= 0.95 and m["r_ves"] >= ref["r_ves"] - 0.01
                  and gap >= 0.50)
        key = f"{mu_type}_r{r}_c{c:g}_w{w}_b{bmax:g}"
        rows[key] = {"mu": mu_type, "r": r, "eps_scale": c, "w": w, "bmax": bmax,
                     "sigma2": sig2[w], "gap": gap, "pass": bool(passed),
                     "b_bg": float(b[masks["far"]].mean()),
                     "b_ves": float(b[masks["ves"]].mean()), **m}
        print(f"{mu_type:>8s} {r:3d} {c:4g} {w:3d} {bmax:5g} {m['n2n_ratio']:7.4f} "
              f"{m['n2n_ratio_bg']:7.4f} {m['n2n_ratio_ves']:7.4f} {m['r_ves']:7.4f} "
              f"{gap:6.3f} {b[masks['far']].mean():.2f}/{b[masks['ves']].mean():.2f}  "
              f"{'通过' if passed else ''}")

    best = min(rows.items(), key=lambda kv: kv[1]["n2n_ratio"])
    print(f"\n最优 {best[0]}: n2n={best[1]['n2n_ratio']:.4f} "
          f"通过={best[1]['pass']}  通过项数={sum(r['pass'] for r in rows.values())}/{len(rows)}")
    res = {"method": "SPW 结构保持方差自适应收缩（引导滤波 mu + 允许 b>1）",
           "ceiling_n2n": ceiling, "input": ref, "settings": rows, "best": best[0]}
    (out / "spw.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"wrote {out}/spw.json")


if __name__ == "__main__":
    main()

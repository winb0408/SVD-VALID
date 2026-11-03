"""V3 Phase 1 前置检查：8 个 Tetris 模式到底能买到多少"噪声可预测性"差异。

两个问题，都不需要训练、不需要读 IQ：

A. 几何表：8 个模式各自把 4 个角色放在 2x2x2 块的哪 4 个角，以及任意两角色之间
   的物理位移多重集。这是纯算术。

B. 上界：把噪声建模成平稳高斯场（自相关 rho），对每个 (输入相位 a -> 目标相位 y)
   算"目标点噪声中能被输入相位噪声线性预测的比例"
       L(a->y) = c^T C^{-1} c / sigma^2  in [0,1]
   其中 C 是输入相位格点在窗口内的协方差，c 是目标点与它们的协方差。
   这是任何估计器（含非线性 CRN）能泄漏噪声的**上界**，因为可预测部分不可能
   超过高斯条件期望的方差。若 8 个模式之间的 L 差异在本可行区间内可以忽略，
   那么"选模式"这条支线在物理上就买不到东西，不管 PSF 测得多准。

C. 实测：在无血管背景区测 Power Doppler 的空间自相关，把 rho 钉到实际尺度上。

用法：
    python -u scripts/v3_precheck_sampling_geometry.py --mode all
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

PATS = [
    [0, 1, 2, 4], [0, 1, 3, 5], [1, 2, 3, 7], [0, 2, 3, 6],
    [0, 4, 5, 6], [1, 4, 5, 7], [2, 4, 6, 7], [3, 5, 6, 7],
]


def off(p):
    """corner index -> (dz,dy,dx), matching datasets/sampling.index_offset."""
    return np.array([p // 4, (p // 2) % 2, p % 2], dtype=float)


# ---------------------------------------------------------------- A. geometry
def geometry_table():
    rows = []
    for k, pat in enumerate(PATS):
        d = np.array(sorted(
            (np.abs(off(a) - off(b)).sum(), np.linalg.norm(off(a) - off(b)))
            for a, b in itertools.combinations(pat, 2)
        ))[:, 1]
        dist2 = sorted(int(((off(a) - off(b)) ** 2).sum())
                       for a, b in itertools.combinations(pat, 2))
        rows.append({
            "pattern": k,
            "corners": pat,
            "offsets": [list(map(int, off(p))) for p in pat],
            "edge_dist2": dist2,  # squared euclidean in voxel units
            "n_dist1": dist2.count(1),
            "n_dist2": dist2.count(2),
            "n_dist3": dist2.count(3),
        })
    return rows


# ------------------------------------------------------- B. predictable share
CELLS = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=float)


def rho_iso(d, L):
    L = np.broadcast_to(np.asarray(L, float), (3,))
    return np.exp(-0.5 * np.sum((d / L) ** 2, axis=-1))


def leak(pattern, a, y, L, m=1.0):
    """目标相位 y 的噪声中，能被输入相位 a 的噪声线性预测的比例。

    m 是整体放大倍数：m=1 就是原 2x2x2 Tetris；m=2 表示视图间距整体翻倍。
    """
    sa = m * (2.0 * CELLS + off(a))
    sy = m * off(y)
    C = rho_iso(sa[:, None, :] - sa[None, :, :], L)
    C = C + 1e-8 * np.eye(len(sa))
    c = rho_iso(sy[None, :] - sa, L)
    w = np.linalg.solve(C, c)
    return float(np.clip(c @ w, 0.0, 1.0))


def leak_aniso(L, ):
    """各向异性相关长度下的模式 × 角色对泄漏表。L = (l_z, l_y, l_x)。"""
    per_pat, per_pair = {}, {}
    for k, pat in enumerate(PATS):
        vals = {}
        for a in pat:
            for y in pat:
                if a == y:
                    continue
                v = leak(pat, a, y, L)
                vals[f"{a}->{y}"] = v
        per_pat[k] = {"min": min(vals.values()), "mean": float(np.mean(list(vals.values()))),
                      "max": max(vals.values()), "pairs": vals}
        per_pair[k] = vals
    allv = [v for k in per_pat for v in (per_pat[k]["min"], per_pat[k]["max"])]
    user = {k: (per_pat[k]["min"], per_pat[k]["max"]) for k in per_pat}
    return {"L": list(map(float, L)), "per_pattern": user,
            "global_min": min(allv), "global_max": max(allv),
            "spread": max(allv) - min(allv),
            "rel_spread": (max(allv) - min(allv)) / max(max(allv), 1e-12)}


def stride_table(L):
    """固定实测相关长度，扫"视图间距放大倍数 m"，看泄漏怎么掉。"""
    rows = []
    for m in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
        allv = []
        for pat in PATS:
            for a in pat:
                for y in pat:
                    if a != y:
                        allv.append(leak(pat, a, y, L, m=m))
        rows.append({"m": m, "min": min(allv), "mean": float(np.mean(allv)),
                     "max": max(allv)})
    return rows


def leak_table(Ls):
    out = []
    for L in Ls:
        per_pat = {}
        for k, pat in enumerate(PATS):
            vals = [leak(pat, a, y, L) for a in pat for y in pat if a != y]
            per_pat[k] = {"min": min(vals), "mean": float(np.mean(vals)),
                          "max": max(vals)}
        allv = [v for d in per_pat.values() for v in (d["min"], d["max"])]
        out.append({
            "L": L if np.isscalar(L) else list(L),
            "per_pattern": per_pat,
            "global_min": min(allv),
            "global_max": max(allv),
            "spread": max(allv) - min(allv),
            "rel_spread": (max(allv) - min(allv)) / max(max(allv), 1e-12),
        })
    return out


# ------------------------------------------------------------ C. measured ACF
LAGS = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 2, 0), (0, 3, 0),
        (1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]


def measured_acf(pd_path, mask_path, quantile=0.20):
    P = np.load(pd_path).astype(np.float64)
    M = np.load(mask_path).astype(bool)
    assert P.shape == M.shape, (P.shape, M.shape)
    v = P[M]
    thr = np.quantile(v, quantile)
    bg = M & (P <= thr)          # 无血管背景：有效体内低功率分位 20% 以下
    x = np.log(np.maximum(P, 1e-12))
    mu = float(x[bg].mean())
    var = float(((x[bg] - mu) ** 2).mean())
    res = {"shape": list(P.shape), "n_valid": int(M.sum()), "n_bg": int(bg.sum()),
           "bg_quantile": quantile, "bg_threshold": float(thr),
           "bg_frac_of_valid": float(bg.sum() / M.sum()), "acf": {}}
    for d in LAGS:
        # 用切片对齐两幅图
        a0 =tuple(slice(max(0, -di), n - max(0, di)) for n, di in zip(P.shape, d))
        b0 = tuple(slice(max(0, di), n - max(0, -di)) for n, di in zip(P.shape, d))
        A, B = x[a0], x[b0]
        Ma, Mb = bg[a0], bg[b0]
        m = Ma & Mb
        if m.sum() < 500:
            res["acf"][str(d)] = None
            continue
        num = float(((A[m] - mu) * (B[m] - mu)).mean())
        res["acf"][str(d)] = {"rho": num / var, "pairs": int(m.sum())}
    return res


def boxmean(a, k=5):
    """k^3 均值滤波（可分离、边界用边缘复制），用来剥掉很慢的杂波分量。"""
    out = a.astype(np.float64)
    r = k // 2
    for ax in range(3):
        p = np.pad(out, [(r, r) if i == ax else (0, 0) for i in range(3)], mode="edge")
        c = np.cumsum(p, axis=ax)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=ax)), c], axis=ax)
        n = out.shape[ax]
        hi = np.take(c, np.arange(k, k + n), axis=ax)
        lo = np.take(c, np.arange(0, n), axis=ax)
        out = (hi - lo) / k
    return out


def acf_sensitivity(pd_path, mask_path):
    """分位数 × 变换 × 去慢分量 的敏感性：结果不是单点，是一个区间。"""
    P = np.load(pd_path).astype(np.float64)
    M = np.load(mask_path).astype(bool)
    variants = {}
    for q in (0.05, 0.10, 0.20, 0.30):
        bg = M & (P <= np.quantile(P[M], q))
        # 只保留 log：box-mean 高通版本会让 rho>1（去均值引入负长程相关、方差被低估），
        # 归一化不成立，已实测确认，故不进入结论。
        for tag, x in (("log", np.log(np.maximum(P, 1e-12))),):
            mu = float(x[bg].mean())
            var = float(((x[bg] - mu) ** 2).mean())
            key = f"q{int(q*100)}_{tag}"
            variants[key] = {"var": var, "acf": {}}
            for d in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
                a0 = tuple(slice(max(0, -di), n - max(0, di)) for n, di in zip(P.shape, d))
                b0 = tuple(slice(max(0, di), n - max(0, -di)) for n, di in zip(P.shape, d))
                m = bg[a0] & bg[b0]
                if m.sum() < 500:
                    continue
                num = float(((x[a0][m] - mu) * (x[b0][m] - mu)).mean())
                variants[key]["acf"][str(d)] = num / var
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["geometry", "leak", "aniso", "acf", "acf2", "all"])
    ap.add_argument("--out", default="data/processed/v3_precheck_sampling_geometry.json")
    args = ap.parse_args()

    out = {}
    if args.mode in ("geometry", "all"):
        out["geometry"] = geometry_table()
    if args.mode in ("leak", "all"):
        Ls = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0]
        out["leak"] = leak_table(Ls)
    if args.mode in ("acf", "all"):
        out["acf"] = {}
        for name in ("SA64", "HSA64"):
            base = Path("data/processed/official") / name
            out["acf"][name] = measured_acf(base / "pd_linear.npy",
                                            base / "valid_mask.npy")
    if args.mode in ("aniso", "all"):
        # 由实测 ACF 反推的 Gaussian 等效相关长度 (z,y,x)，见报告 C 节
        base_ell = np.array([1.47, 2.35, 1.28])
        out["leak_aniso"] = [leak_aniso(s * base_ell) for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
        out["stride"] = stride_table(base_ell)
    if args.mode in ("acf2", "all"):
        out["acf2"] = {}
        for name in ("SA64", "HSA64"):
            base = Path("data/processed/official") / name
            out["acf2"][name] = acf_sensitivity(base / "pd_linear.npy",
                                                base / "valid_mask.npy")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    # ---- 打印 ----
    if "geometry" in out:
        print("\n== A. 几何：8 个模式的 4 个角与两两位移平方 ==")
        print(f"{'pat':>3} {'corners':<16} {'offsets':<28} {'d^2 多重集':<22} "
              f"{'#1':>3}{'#2':>3}{'#3':>3}")
        for r in out["geometry"]:
            print(f"{r['pattern']:>3} {str(r['corners']):<16} "
                  f"{str(r['offsets']):<28} {str(r['edge_dist2']):<22} "
                  f"{r['n_dist1']:>3}{r['n_dist2']:>3}{r['n_dist3']:>3}")

    if "leak" in out:
        print("\n== B. 噪声可预测比例 L（上界），按假定相关长度 L ==")
        print(f"{'L':>5} {'global_min':>11} {'global_max':>11} {'spread':>9} "
              f"{'rel':>8}   per-pattern mean (0..7)")
        for row in out["leak"]:
            means = " ".join(f"{row['per_pattern'][k]['mean']:.3f}" for k in range(8))
            print(f"{row['L']:>5} {row['global_min']:>11.4f} "
                  f"{row['global_max']:>11.4f} {row['spread']:>9.4f} "
                  f"{row['rel_spread']:>8.3f}   {means}")

    if "acf" in out:
        print("\n== C. 实测：无血管背景区 log PD 的空间自相关 ==")
        for name, r in out["acf"].items():
            print(f"-- {name}  shape={r['shape']}  background={r['n_bg']} "
                  f"({100*r['bg_frac_of_valid']:.1f}% of valid)")
            for d in LAGS:
                e = r["acf"][str(d)]
                if e:
                    print(f"   lag {str(d):<12} rho={e['rho']:+.4f}  pairs={e['pairs']}")
    if "leak_aniso" in out:
        print("\n== B2. 各向异性：(L_z,L_y,L_x) = s x 实测等效值 (1.47,2.35,1.28) ==")
        print(f"{'s':>5} {'L':<24} {'min':>7} {'max':>7} {'spread':>8} {'rel':>7}  "
              f"per-pattern [min,max]")
        for row in out["leak_aniso"]:
            s = row["L"][0] / 1.47
            pp = " ".join(f"[{row['per_pattern'][k][0]:.2f},{row['per_pattern'][k][1]:.2f}]"
                          for k in range(8))
            print(f"{s:>5.2f} {str([round(v,2) for v in row['L']]):<24} "
                  f"{row['global_min']:>7.3f} {row['global_max']:>7.3f} "
                  f"{row['spread']:>8.4f} {row['rel_spread']:>7.3f}  {pp}")

    if "stride" in out:
        print("\n== B3. 固定实测相关长度，扫视图间距放大倍数 m ==")
        print(f"{'m':>5} {'min':>7} {'mean':>7} {'max':>7}")
        for r in out["stride"]:
            print(f"{r['m']:>5.2f} {r['min']:>7.3f} {r['mean']:>7.3f} {r['max']:>7.3f}")

    if "acf2" in out:
        print("\n== C2. 敏感性：分位数 × 变换 × 去慢分量（ACF @ 1 体素） ==")
        print(f"{'seq':>6} {'variant':<14} {'(0,0,1)':>8} {'(0,1,0)':>8} "
              f"{'(1,0,0)':>8} {'(1,1,0)':>8} {'(1,1,1)':>8}")
        for name, variants in out["acf2"].items():
            for key, v in variants.items():
                a = v["acf"]
                print(f"{name:>6} {key:<14} "
                      + " ".join(f"{a.get(str(d), float('nan')):>8.3f}"
                                 for d in [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                           (1, 1, 0), (1, 1, 1)]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

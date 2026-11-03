"""V3 Phase 1 / T1-4b：泄漏上界 L 的**已知答案标定**。

要问的问题：`L = c^T C^{-1} c` 是我们用来决定 m 的尺子。它写成"任何估计器能泄漏的噪声方差比例"的
**上界**，但上界有多松？如果实测泄漏远小于 L，那 m=1 也许够用，建议 m=2 就是过度保守；
如果两者贴合甚至实测超过 L，说明 L 既紧又会被 ρ 模型的失配打破，m=2 是必需的。

做法：在**有已知真值**的仿真器（`v3_simulator.py`，血流关掉 = 纯噪声场）上，把"从输入子格预测目标体素"
这件事**实际做一遍**，与同一个位点集合上算出的 L 对比：

  - 输入集合严格照 `leak()` 的定义：同一角色 a、cell 邻域 ±w 内的全部位点
    （这正是 VALID 的一个相位的空间布局，网络能看到的就这些）；
  - 目标：cell 0 处的角色 y；
  - 理论：`L = c^T C^{-1} c`，ρ 由**同一个残差场自己**的 ACF 拟合（自洽，不跨源）；
  - 实测：留出块的**线性**最小二乘 R²（权重全块共享），加一个 RBF 核岭回归作**非线性**探针。
    留出而不是样本内，因为 343 个特征在样本内必然虚高。

残差场两种定义，对应报告里的两种读法：
  - `raw`  ：log PD − 全局均值（几乎不动，含慢平台）
  - `fast` ：log PD − 宽箱均值（去掉慢平台，对应 Gate 1 采用的快速分量）
两个读法各自算自己的 ρ、自己的 L、自己的 R²，所以比较始终是**同位点集合、同场**。

用法：
    python -u scripts/v3_leak_validate.py
输出 `data/processed/v3_leak_validate.json`。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_simulator import SimCfg, simulate, band_and_pd  # noqa: E402
from v3_correlation_length import acf_all_axes  # noqa: E402
from v3_precheck_sampling_geometry import rho_iso, off, PATS  # noqa: E402

OUT = Path("data/processed/v3_leak_validate.json")


# --------------------------------------------------------------- 工具
def boxmean(a, k):
    """可分离 k^3 居中箱式滤波（边界复制），输出同形状。"""
    out = a.astype(np.float64)
    r = k // 2
    for ax in range(3):
        n = out.shape[ax]
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=ax)
        zero = [slice(None)] * out.ndim
        zero[ax] = slice(0, 1)
        c = np.concatenate([np.zeros_like(c[tuple(zero)]), c], axis=ax)
        sl_hi = [slice(None)] * out.ndim
        sl_lo = [slice(None)] * out.ndim
        sl_hi[ax] = slice(k, k + n)
        sl_lo[ax] = slice(0, n)
        out = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / k
    return out


def residual(pd, k_hp):
    lg = np.log(np.maximum(pd, 1e-12))
    return lg - (lg.mean() if k_hp is None else boxmean(lg, k_hp))


def fit_ell(curves):
    """Gaussian 等效长度，按 lag-1 反推（与 `reports/v3_phase1_correlation_length.md` 一致）。

    早期版本用 lag1..4 的 log-linear 拟合，但去平台后尾部会变负，负值被 clip 到 1e-6 后
    log 变成 -13.8，直接把斜率拽陡、ℓ 被算小 —— 这是本文件的已知坑，不要再改回拟合。
    """
    return tuple(float(np.sqrt(-0.5 / np.log(min(max(curves[a][0][0], 1e-6), 0.999))))
                 for a in ("z", "y", "x"))


def leak_sites(sites_in, site_y, rho):
    """同位点集合上的 L = c^T C^{-1} c（rho 为任意可调用，参数是位移数组）。"""
    A = np.asarray(sites_in, float)
    C = rho(A[:, None, :] - A[None, :, :]) + 1e-8 * np.eye(len(A))
    c = rho(np.asarray(site_y, float)[None, :] - A)
    return float(np.clip(c @ np.linalg.solve(C, c), 0.0, 1.0))


def rho_sep_from(curves):
    """由逐轴 ACF 曲线（[value, n] 形式）构造可分离非参数 ρ。"""
    c = {ax: np.maximum(np.array([v[0] for v in curves[ax]], float), 0.0)
         for ax in ("z", "y", "x")}

    def rho(d):
        d = np.abs(np.asarray(d, float))
        out = np.ones(d.shape[:-1], float)
        for ax, i in (("z", 0), ("y", 1), ("x", 2)):
            cc = np.concatenate([[1.0], c[ax]])
            out = out * np.maximum(np.interp(d[..., i], np.arange(len(cc)), cc,
                                             left=1.0, right=0.0), 0.0)
        return out
    return rho


def cell_lattice(shape, m):
    """每轴 cell 的数目 nc，以及各 cell 的目标格点坐标。"""
    nc = [int(((d - 1) // m + 1) // 2) for d in shape]
    return nc


def gather(F, c0, offv, m, nc, dc):
    """取出 (n_c0, len(dc), 3) 的位点坐标，越界或无效的返回 None 掩码。"""
    zs = m * (2 * (c0[0][:, None, None] + dc[0]) + offv[0])
    ys = m * (2 * (c0[1][None, :, None] + dc[1]) + offv[1])
    xs = m * (2 * (c0[2][None, None, :] + dc[2]) + offv[2])
    return zs, ys, xs


def build_design(F, m, nc, w, role_in, role_tgt, okvox):
    """构造 X（n_blk x n_sites）与 y。返回 (X, y, offsets)。"""
    cells = [[c for c in range(w, n - w)] for n in nc]
    if min(len(c) for c in cells) <= 0:
        return None
    c0 = [np.array(c) for c in cells]
    dc_grid = np.array(list(itertools.product(range(-w, w + 1), repeat=3)), float)

    # 目标格点：role_tgt、cell 0
    tz = (m * (2 * c0[0] + off(role_tgt)[0])).astype(int)
    ty = (m * (2 * c0[1] + off(role_tgt)[1])).astype(int)
    tx = (m * (2 * c0[2] + off(role_tgt)[2])).astype(int)

    nz, ny, nx = F.shape
    Xs, offs, keep = [], [], None
    tgt_valid = okvox[tz[:, None, None], ty[None, :, None], tx[None, None, :]]
    for k, (dz, dy, dx) in enumerate(dc_grid):
        sz = (m * (2 * (c0[0][:, None, None] + dz) + off(role_in)[0])).astype(int)
        sy = (m * (2 * (c0[1][None, :, None] + dy) + off(role_in)[1])).astype(int)
        sx = (m * (2 * (c0[2][None, None, :] + dx) + off(role_in)[2])).astype(int)
        inb = ((sz >= 0) & (sz < nz)) & ((sy >= 0) & (sy < ny)) & ((sx >= 0) & (sx < nx))
        ok = inb & tgt_valid
        ok &= okvox[np.clip(sz, 0, nz - 1), np.clip(sy, 0, ny - 1), np.clip(sx, 0, nx - 1)]
        keep = ok if keep is None else (keep & ok)
        Xs.append(F[np.clip(sz, 0, nz - 1), np.clip(sy, 0, ny - 1), np.clip(sx, 0, nx - 1)])
        offs.append((m * (2 * dz + off(role_in)[0] - off(role_tgt)[0]),
                     m * (2 * dy + off(role_in)[1] - off(role_tgt)[1]),
                     m * (2 * dx + off(role_in)[2] - off(role_tgt)[2])))
    X = np.stack([x[keep] for x in Xs], axis=1)
    y = F[tz[:, None, None], ty[None, :, None], tx[None, None, :]][keep]
    return X, y, np.array(offs, float), np.zeros(3)


def r2_holdout(X, y, frac=0.3, seed=0):
    """按块划分的留出 R²（线性，含截距）。"""
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * (1 - frac))
    tr, te = idx[:k], idx[k:]
    A = np.concatenate([X[tr], np.ones((k, 1))], 1)
    w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    pred = np.concatenate([X[te], np.ones((len(te), 1))], 1) @ w
    res = y[te] - pred
    return float(1.0 - res.var() / y[te].var())


def r2_krr(X, y, n_tr=2000, n_te=800, seed=0):
    """RBF 核岭回归的非线性探针（留出 R²）。"""
    n = len(y)
    if n < n_tr + n_te:
        n_tr, n_te = int(n * 0.6), int(n * 0.3)
    if n_tr < 100:
        return None
    idx = np.random.default_rng(seed).permutation(n)
    tr, te = idx[:n_tr], idx[n_tr:n_tr + n_te]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / A.shape[1]
    K = np.exp(-gamma * d2)
    lam = 1e-1
    alpha = np.linalg.solve(K + lam * np.eye(len(A)), y[tr])
    d2t = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    pred = np.exp(-gamma * d2t) @ alpha
    res = y[te] - pred
    return float(1.0 - res.var() / y[te].var())


# --------------------------------------------------------------- 主流程
def main():
    cfg = SimCfg(seed=0, flow_amp=0.0)          # 纯噪声场
    print("仿真（血流关闭，纯噪声场）…")
    d = simulate(cfg)
    pd, _ = band_and_pd(d["X"], d["valid"])
    okvox = boxmean(d["valid"].astype(np.float64), 5) > 0.999   # 5^3 窗口全在体内
    print(f"  有效内部体素 {okvox.sum()} / {okvox.size}")

    rec = {"cases": []}
    for label, k_hp in (("raw", None), ("fast", 15)):
        F = residual(pd, k_hp)
        acf = acf_all_axes(F, okvox, lags=[1, 2, 3, 4, 5, 6])
        ell = fit_ell(acf)
        print(f"\n=== 残差 [{label}]  Gaussian 等效长度 (z,y,x) = {np.round(ell, 3)}")
        rec[label] = {"ell": list(ell),
                      "acf": {ax: [c[0] for c in acf[ax]] for ax in ("z", "y", "x")}}
        for m in (1, 2, 3):
            nc = cell_lattice(pd.shape, m)
            w = 3
            while w >= 1:
                nb = int(np.prod([max(n - 2 * w, 0) for n in nc]))
                if nb >= 800:
                    break
                w -= 1
            if w < 1:
                print(f"  m={m}: 块数不足，跳过")
                continue
            for role_tgt in (3, 5, 6, 7):
                role_in = 0     # 输入角色固定为 0（PATS[0] 含 0），目标取其补集
                got = build_design(F, m, nc, w, role_in, role_tgt, okvox)
                if got is None:
                    continue
                X, y, offs, ysite = got
                if len(y) < 300:
                    print(f"  m={m} w={w} tgt={role_tgt}: 样本 {len(y)} 太少，跳过")
                    continue
                L_g = leak_sites(offs, ysite, lambda d: rho_iso(d, ell))
                L_s = leak_sites(offs, ysite, rho_sep_from(acf))
                r2 = r2_holdout(X, y)
                rk = r2_krr(X, y)
                print(f"  m={m} w={w} tgt={role_tgt}: n={len(y):>6} "
                      f"L_gauss={L_g:.3f}  L_sep={L_s:.3f}  "
                      f"R²lin={r2:.3f}  R²krr={'-' if rk is None else f'{rk:.3f}'}")
                rec["cases"].append({"mode": label, "m": m, "w": w, "role_in": role_in,
                                     "role_tgt": int(role_tgt), "n": int(len(y)),
                                     "L_gauss": L_g, "L_sep": L_s,
                                     "r2_lin": r2, "r2_krr": rk,
                                     "n_sites": int(len(offs))})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

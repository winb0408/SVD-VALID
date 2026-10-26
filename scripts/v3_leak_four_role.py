"""V3 Phase 1 / T1-4c：多视图输入变体的泄漏上界（**不采用**，只作为越界记录）。

**先说结论式的定位**（2026-10-20 核对架构后确认）：本仓库的 CRN 是**单视图输入**——
`models/model_CRN.py` 的 `in_channels=1`、`out_channels=1`，`train.py` 里
`xa, xb, ya, yb = views` 是把一张模式的 4 个角点拆成**两对独立的 输入→目标**
（`mse(model(xa), ya)` 与 `mse(model(xb), yb)`），两次前向互不耦合。
所以 Gate 1 的"单角色输入"泄漏表**就是**任务本身的泄漏，判据定义没问题。

那这个脚本算什么？算**如果**把网络改成"4 个角点一起当输入通道"（一个诱人的提 SNR 改法）
会泄漏多少。答案（实测 ρ，±3 cell）：m=1 时 0.540–0.871（均值 0.727），m=2 时 0.244–0.712
（均值 0.490），m=3 时 0.069–0.416（均值 0.254）。**也就是说，单视图输入不是省事，是这个方法的前提**：
一旦把 4 个角点喂进同一个网络，m=2 下平均 49% 的目标噪声是可以线性算出来的。

用途有两个，都不是"改判据"：
  (i) 冻结这条边界——将来任何"多加几个视图会不会更好"的想法，先看这张表；
  (ii) 顺手核验同一套 `leak_L` 机器在多视图、重共线输入下没被打坏（条件数 1e1–8e2，未爆炸）。

内容：
  1. 4 角色输入的 L 表：8 个模式 × 4 个目标角色，m=1/2/3，两个序列；
  2. 4 角色输入下窗口半径 w 的饱和性；
  3. `--sim`：在仿真器（血流关闭 = 纯噪声场）上用 4 角色设计矩阵做留出线性 R²，
     与解析 L 对比。**默认不跑**：这一步校验的是一个不采用的量，不占机时。

用法：
    python -u scripts/v3_leak_four_role.py            # 只算表 + 饱和性
    python -u scripts/v3_leak_four_role.py --sim      # 追加仿真器校验
输出 `data/processed/v3_leak_four_role.json`。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402
from v3_leak_nonparametric import fast_curve, rho_sep  # noqa: E402

OUT = Path("data/processed/v3_leak_four_role.json")
CORR = Path("data/processed/v3_correlation_length.json")


# --------------------------------------------------------------- 位点集合
def source_roles(p):
    return [int(x) for x in PATS[p]]


def target_roles(p):
    return sorted(set(range(8)) - set(source_roles(p)))


def sites_of(roles, m, w):
    """4 个角色 × (2w+1)^3 个 cell 的全部绝对坐标（相对目标所在 cell 的原点）。"""
    cells = np.array(list(itertools.product(range(-w, w + 1), repeat=3)), float)
    return np.concatenate([m * (2.0 * cells + off(a)) for a in roles], 0)


# --------------------------------------------------------------- 解析 L
def leak_L(S, sy, rho):
    """L = c^T C^{-1} c，相对岭 1e-6·tr(C)/n。返回 (L, 条件数估计)。"""
    C = rho(S[:, None, :] - S[None, :, :])
    n = len(S)
    ridge = 1e-6 * np.trace(C) / n
    C = C + ridge * np.eye(n)
    c = rho(sy[None, :] - S)
    L = float(np.clip(c @ np.linalg.solve(C, c), 0.0, 1.0))
    return L, float(np.linalg.cond(C))


def table(cur, m, w=3):
    """返回 per-(p,y) 的 L 明细。"""
    rho = lambda d, c=cur: rho_sep(d, c)  # noqa: E731
    rows = []
    for p in range(8):
        src = source_roles(p)
        S4 = sites_of(src, m, w)
        for y in target_roles(p):
            sy = m * off(y)
            L4, cd = leak_L(S4, sy, rho)
            L1, _ = leak_L(sites_of([src[0]], m, w), sy, rho)
            rows.append({"p": p, "y": y, "L4": L4, "L1": L1, "cond": cd})
    v4 = np.array([r["L4"] for r in rows])
    return rows, {"min": float(v4.min()), "max": float(v4.max()),
                  "mean": float(v4.mean()), "spread": float(v4.max() - v4.min())}


# --------------------------------------------------------------- 仿真器校验
def gather_roles(F, m, w4, roles_in, role_tgt, okvox, nc):
    """4 角色设计矩阵：每块一行，列 = 4 角色 × (2w+1)^3 个相对 cell。"""
    cells = [np.arange(w4, n - w4) for n in nc]
    if min(len(c) for c in cells) <= 0:
        return None
    c0 = [np.array(c) for c in cells]
    dc = np.array(list(itertools.product(range(-w4, w4 + 1), repeat=3)), float)

    tz = (m * (2 * c0[0] + off(role_tgt)[0])).astype(int)
    ty = (m * (2 * c0[1] + off(role_tgt)[1])).astype(int)
    tx = (m * (2 * c0[2] + off(role_tgt)[2])).astype(int)
    nz, ny, nx = F.shape
    keep = okvox[tz[:, None, None], ty[None, :, None], tx[None, None, :]]

    cols = []
    for a in roles_in:
        for dz, dy, dx in dc:
            sz = (m * (2 * (c0[0][:, None, None] + dz) + off(a)[0])).astype(int)
            sy = (m * (2 * (c0[1][None, :, None] + dy) + off(a)[1])).astype(int)
            sx = (m * (2 * (c0[2][None, None, :] + dx) + off(a)[2])).astype(int)
            inb = ((sz >= 0) & (sz < nz)) & ((sy >= 0) & (sy < ny)) & ((sx >= 0) & (sx < nx))
            ok = inb & okvox[np.clip(sz, 0, nz - 1), np.clip(sy, 0, ny - 1), np.clip(sx, 0, nx - 1)]
            keep &= ok
            cols.append(F[np.clip(sz, 0, nz - 1), np.clip(sy, 0, ny - 1),
                            np.clip(sx, 0, nx - 1)][keep])
    X = np.stack(cols, axis=1)
    y = F[tz[:, None, None], ty[None, :, None], tx[None, None, :]][keep]
    return X, y


def r2_holdout(X, y, frac=0.3, seed=0, ridge_rel=1e-3):
    """按块留出的线性 R²（含截距，岭回归；p 与 n 同量级时必须用岭）。"""
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * (1 - frac))
    tr, te = idx[:k], idx[k:]
    A = np.concatenate([X[tr], np.ones((k, 1))], 1)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    At = (X[tr] - mu) / sd
    Ae = (X[te] - mu) / sd
    lam = ridge_rel * len(tr)
    G = At.T @ At + lam * np.eye(At.shape[1])
    wv = np.linalg.solve(G, At.T @ (y[tr] - y[tr].mean()))
    pred = Ae @ wv + y[tr].mean()
    res = y[te] - pred
    return float(1.0 - res.var() / y[te].var())


def sim_check(cur, cases):
    """对给定 (m, w4, p, y) 做已知答案校验。"""
    from v3_simulator import SimCfg, simulate, band_and_pd
    from v3_leak_validate import boxmean, residual, cell_lattice

    print("\n仿真（血流关闭，纯噪声场）…", flush=True)
    d = simulate(SimCfg(seed=0, flow_amp=0.0))
    pd, _ = band_and_pd(d["X"], d["valid"])
    okvox = boxmean(d["valid"].astype(np.float64), 5) > 0.999
    F = residual(pd, 15)          # 与 Gate 1 同一读法（去平台）
    print(f"  有效内部体素 {okvox.sum()} / {okvox.size}", flush=True)

    rho = lambda dd, c=cur: rho_sep(dd, c)  # noqa: E731
    out = []
    for m, w4, p, y in cases:
        nc = cell_lattice(F.shape, m)
        got = gather_roles(F, m, w4, source_roles(p), y, okvox, nc)
        if got is None:
            continue
        X, yv = got
        if len(yv) < 200:
            print(f"  m={m} w={w4} p={p} y={y}: 样本 {len(yv)} 太少，跳过", flush=True)
            continue
        S = sites_of(source_roles(p), m, w4)
        L4,  _ = leak_L(S, m * off(y), rho)
        r2 = r2_holdout(X, yv)
        print(f"  m={m} w={w4} pat{p}->tgt{y}: 块 {len(yv):>6} 特征 {X.shape[1]:>4}  "
              f"L4={L4:.3f}  R²lin={r2:.3f}  Δ={r2 - L4:+.3f}", flush=True)
        out.append({"m": m, "w": w4, "p": p, "y": y, "n": int(len(yv)),
                    "n_feat": int(X.shape[1]), "L4": L4, "r2_lin": r2})
    return out


# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()

    J = json.loads(CORR.read_text())
    rec = {"note": "4 角色输入（真实 CRN 定义）的泄漏上界；ρ = 实测可分经验曲线（去平台快速分量）",
           "curves": {}, "table": {}, "saturation": {}, "sim": []}

    for seq in J:
        P1 = J[seq]["summary"]["P1"]
        raw = {a: [v[0] for v in J[seq]["P1"]["acf"][a]] for a in ("z", "y", "x")}
        cur = {a: list(fast_curve(raw[a], P1[a]["plateau"])) for a in ("z", "y", "x")}
        rec["curves"][seq] = {a: [round(float(v), 4) for v in cur[a]] for a in cur}

        print(f"\n=== {seq}   4 角色输入（±3 cell，相对岭 1e-6·tr/n）", flush=True)
        print("  m   单角色 L      4 角色 L（min–max，均值）", flush=True)
        rec["table"][seq] = {}
        for m in (1, 2, 3):
            rows, agg = table(cur, m, w=3)
            rec["table"][seq][f"m{m}"] = {"rows": rows, "agg": agg,
                                          "L1_min": float(min(r["L1"] for r in rows)),
                                          "L1_max": float(max(r["L1"] for r in rows))}
            print(f"  {m}   {min(r['L1'] for r in rows):.3f}–{max(r['L1'] for r in rows):.3f}   "
                  f"{agg['min']:.3f}–{agg['max']:.3f}（{agg['mean']:.3f}）  "
                  f"cond≤{max(r['cond'] for r in rows):.0e}", flush=True)

        # 4 角色下的窗口饱和性
        rec["saturation"][seq] = {}
        for m in (1, 2, 3):
            rows_w = {w: table(cur, m, w=w)[1] for w in (1, 2, 3)}
            span = (abs(rows_w[3]["min"] - rows_w[1]["min"])
                    + abs(rows_w[3]["max"] - rows_w[1]["max"]))
            rec["saturation"][seq][f"m{m}"] = {f"w{w}": rows_w[w] for w in rows_w}
            print(f"    饱和性 m={m}: w=1 [{rows_w[1]['min']:.3f},{rows_w[1]['max']:.3f}] "
                  f"→ w=3 [{rows_w[3]['min']:.3f},{rows_w[3]['max']:.3f}]  跨度变化 {span:.3f}",
                  flush=True)

        if args.sim:
            # 每 m 取最泄漏与最不泄漏的 (p,y) 各一例；w4=1 保证块数 ≥ 特征数
            cases = []
            for m, w4 in ((1, 1), (2, 1)):
                t = rec["table"][seq]["m%d" % m]["rows"]
                lo = min(t, key=lambda r: r["L4"])
                hi = max(t, key=lambda r: r["L4"])
                cases += [(m, w4, lo["p"], lo["y"]), (m, w4, hi["p"], hi["y"])]
            rec["sim"].append({"seq": seq,
                               "cases": sim_check(cur, cases)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

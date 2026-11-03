"""V3 Phase 2 / T2-2：信号侧 vs 噪声侧，按位移类一对一配对（受控仿真）。

背景：`reports/v3_phase2_role_order.md` 只冻结了**噪声侧**排序（泄漏 L）。体对角线位移
3.46 体素，"泄漏最少"也可能是"信号最远"——所以必须同时量信号侧，否则不能选角色。

**为什么不用回归 R² 作信号侧代理（两次失败，写在这里）**：
  1. 初版把 Gate 1 的"快速分量"变换（`log − 15 体素箱均值`）套到信号场上，R²_s 全 ≈ 0/负
     ——那个高通把慢变的血管信号当平台减掉了（仪器用错）。
  2. 换成 `log1p(PD/s)` 后仍然 R²_s ≈ 0：m=2 的目标格点只有 9×16×16 ≈ 2300 个，
     而血管只占体素 ~1%，落在血管上的目标点只有 ~20 个，回归无法估计。
所以信号侧改用**两参数统计量：跨视图相关系数** `r_s`（在**全部内部体素**上算，
血管稀疏不再是问题），噪声侧同时报 `r_n`（同一位移、同一格点集合）与解析泄漏 `L`（窗口版）。

判据（**跑之前写下**，见 `Project_plan.md` T2-2）：
  S_min 用"信号可传递性"的下限表达：要求所选位移类的 `r_s` 不低于
  **所有位移类中最小的 r_s 的 0.9 倍**（即不能挑一个信号明显传不过去的方向）；
  在满足该约束的类里，取解析泄漏 `L` 最小者。

仿真器：`SimCfg(seed=0)`，血流幅度保持出厂值（不调）；`v3_simulator.py` 的血流项
2026-10-20 已修正为逐帧独立散斑（初版是秩 1，会被带通 SVD 整条删掉）。

用法：
    python -u scripts/v3_signal_transfer.py [--m 2]
输出 `data/processed/v3_signal_transfer.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_simulator import SimCfg, simulate, band_and_pd  # noqa: E402
from v3_leak_validate import boxmean  # noqa: E402
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402

OUT = Path("data/processed/v3_signal_transfer.json")
CORR = Path("data/processed/v3_correlation_length.json")


def representative_pairs():
    """按有符号位移分组，每类取第一例（与 v3_role_order.py 同分组）。"""
    seen, pairs = set(), []
    for p in range(8):
        src = [int(x) for x in PATS[p]]
        tgt = sorted(set(range(8)) - set(src))
        for a in src:
            for y in tgt:
                sh = tuple(int(round(v)) for v in (off(a) - off(y)))
                if sh in seen:
                    continue
                seen.add(sh)
                pairs.append({"p": p, "a": a, "y": y, "shift": sh})
    return pairs


def lattice(shape, m, role, okvox):
    """角色 role 在 m=2 下的格点索引，返回 (z,y,x) 三个索引数组 + 有效掩码。"""
    ax = []
    for n, o in zip(shape, off(role)):
        start = int(m * o)
        ax.append(np.arange(start, n, 2 * m))
    z, y, x = ax
    valid = okvox[np.ix_(z, y, x)]
    return z, y, x, valid


def cross_r(F, shape, m, a, y, okvox):
    """两角色格点上 field 的 Pearson 相关（只取两者都有效的体素）。"""
    za, ya, xa, va = lattice(shape, m, a, okvox)
    zy, yy, xy, vy = lattice(shape, m, y, okvox)
    A, B = F[np.ix_(za, ya, xa)], F[np.ix_(zy, yy, xy)]
    n = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1]), min(A.shape[2], B.shape[2])
    A, B, V = A[:n[0], :n[1], :n[2]], B[:n[0], :n[1], :n[2]], va[:n[0], :n[1], :n[2]] & vy[:n[0], :n[1], :n[2]]
    a_v, b_v = A[V], B[V]
    if a_v.size < 100 or a_v.std() < 1e-12 or b_v.std() < 1e-12:
        return float("nan"), int(a_v.size)
    return float(np.corrcoef(a_v, b_v)[0, 1]), int(a_v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--smooth", type=int, default=3,
                    help="PD 先做 k^3 局部平均再变换：血管只占 ~1%%，逐体素相关会被稀疏性压死，"
                         "而卷积网络能用的本来就是局部平均（k=1 时退化回逐体素）")
    args = ap.parse_args()
    m = args.m

    print(f"仿真（血流开，seed=0）… m={m}", flush=True)
    d = simulate(SimCfg(seed=0))
    obs = d["X"].astype(np.complex64)
    flow = d["flow"].astype(np.complex64)
    noise = obs - flow
    pd = {k: band_and_pd(v, d["valid"])[0] for k, v in
          (("obs", obs), ("noise", noise), ("sig", flow))}
    if args.smooth > 1:
        pd = {k: boxmean(v, args.smooth) for k, v in pd.items()}
    s = float(np.median(pd["obs"][d["valid"]]))
    F = {k: np.log1p(np.maximum(v, 0.0) / s) for k, v in pd.items()}
    okvox = boxmean(d["valid"].astype(np.float64), 5) > 0.999
    print(f"  s={s:.4g}  方差 obs {np.var(F['obs'][okvox]):.4f}  "
          f"noise {np.var(F['noise'][okvox]):.4f}  sig {np.var(F['sig'][okvox]):.4f}", flush=True)

    # 真实数据的解析泄漏（设计决策用这一列；仿真只负责信号侧相对排序）
    J = json.loads(CORR.read_text())
    from v3_leak_nonparametric import fast_curve, rho_sep
    from v3_leak_four_role import leak_L, sites_of
    realL = {}
    for seq in J:
        P1 = J[seq]["summary"]["P1"]
        raw = {a: [v[0] for v in J[seq]["P1"]["acf"][a]] for a in ("z", "y", "x")}
        cur = {a: list(fast_curve(raw[a], P1[a]["plateau"])) for a in ("z", "y", "x")}
        rho = lambda dd, c=cur: rho_sep(dd, c)  # noqa: E731
        realL[seq] = {tuple(int(round(v)) for v in (off(p["a"]) - off(p["y"]))):
                      leak_L(sites_of([p["a"]], m, 3), m * off(p["y"]), rho)[0]
                      for p in representative_pairs()}

    rows = []
    print(f"\n  {'位移':>14} {'体素':>12} {'L(SA64)':>8} {'L(HSA)':>7} "
          f"{'r_obs':>7} {'r_sig':>7} {'r_noise':>8}", flush=True)
    for p in representative_pairs():
        sh = tuple(p["shift"])
        ro, _ = cross_r(F["obs"], pd["obs"].shape, m, p["a"], p["y"], okvox)
        rs, ns = cross_r(F["sig"], pd["obs"].shape, m, p["a"], p["y"], okvox)
        rn, _ = cross_r(F["noise"], pd["obs"].shape, m, p["a"], p["y"], okvox)
        rows.append({**p, "vox": [v * m for v in sh], "r_obs": ro, "r_sig": rs, "r_noise": rn,
                     "L_SA64": realL["SA64"][sh], "L_HSA64": realL["HSA64"][sh]})
        print(f"  {str(tuple(v * m for v in sh)):>14} {'':>12} "
              f"{realL['SA64'][sh]:8.3f} {realL['HSA64'][sh]:7.3f} "
              f"{ro:7.3f} {rs:7.3f} {rn:8.3f}", flush=True)

    rs_all = np.array([r["r_sig"] for r in rows])
    S_min = float(rs_all.min() * 0.9) if np.isfinite(rs_all).any() else None
    ok = [r for r in rows if np.isfinite(r["r_sig"]) and r["r_sig"] >= S_min]
    ok.sort(key=lambda r: r["L_SA64"])
    print(f"\n  判据：S_min = 0.9 × min(r_sig) = {S_min:.3f}；通过 {len(ok)} / {len(rows)}", flush=True)
    if ok:
        b = ok[0]
        print(f"  选中：位移 {b['vox']}  L(SA64)={b['L_SA64']:.3f}  r_sig={b['r_sig']:.3f}", flush=True)
        worst_s = min(rows, key=lambda r: (r["L_SA64"]))
        print(f"  对照（只按泄漏最小）：位移 {worst_s['vox']} L={worst_s['L_SA64']:.3f} "
              f"r_sig={worst_s['r_sig']:.3f}", flush=True)

    rec = {"m": m, "smooth": args.smooth, "s": s, "okvox": int(okvox.sum()),
           "var": {k: float(np.var(v[okvox])) for k, v in F.items()},
           "rows": rows, "S_min": S_min, "chosen": ok[0] if ok else None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

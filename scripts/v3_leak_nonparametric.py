"""V3 Phase 1 / T1-4c：用**实测的非参数 ρ**（可分离、逐轴实测曲线）重算泄漏上界表。

为什么要这一步：`Project_plan.md` 的 Gate 1 量化判据用 `L = c^T C^{-1} c`，其中 ρ 写成
**Gaussian** `exp(-½ Σ (d_i/ℓ_i)²)`，ℓ 由实测 ACF 的 lag-1 反推。Gaussian 形状与实测形状差得很远：
例如 SA64 去平台后 z 轴 lag1/2/3 = 0.604/0.466/0.316，Gaussian(ℓ=0.996) 给 0.607/0.135/0.011。
**形状错，L 就错**，而 Gate 1 是拿 L 来否证分支的，所以必须换成实测形状再算一遍。

本脚本不动任何实测数据，只换 ρ 的估计方式，并在**完全相同的位点集合**上对比：
  gauss(ℓ)     —— 方案原用法，对照
  sep-emp(实测) —— 逐轴实测曲线（lag 1..6 线性插值、负数截 0、lag>6 取 0）逐轴相乘得 3D ρ
同时扫位点窗口半径 w（±w 个 cell）以确认结果对窗口饱和，并报告 C 的条件数——
条件数上万时 `c^T C^{-1} c` 已经不是可信的数，这类行不能进结论。

用法：
    python -u scripts/v3_leak_nonparametric.py
输出 `data/processed/v3_leak_nonparametric.json` 与终端表。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_precheck_sampling_geometry import off, rho_iso  # noqa: E402

SRC = Path("data/processed/v3_correlation_length.json")
OUT = Path("data/processed/v3_leak_nonparametric.json")
COND_MAX = 1e4          # 超过这个条件数的行标为不可用


def fast_curve(raw, plateau):
    """报告 §7 采用的去平台变换（与 acf 曲线同一步）。"""
    return (np.asarray(raw, float) - plateau) / max(1.0 - plateau, 1e-12)


def rho_sep(d, curves):
    """可分离非参数 ρ：逐轴在 lag 1..6 上线性插值相乘，负数截 0，lag>6 取 0。"""
    d = np.abs(np.asarray(d, float))
    out = np.ones(d.shape[:-1], float)
    for ax, i in (("z", 0), ("y", 1), ("x", 2)):
        c = np.concatenate([[1.0], np.asarray(curves[ax], float)])
        v = np.interp(d[..., i], np.arange(0, len(c)), c, left=1.0, right=0.0)
        out = out * np.maximum(v, 0.0)
    return out


def ell_from_lag1(curves):
    return tuple(float(np.sqrt(-0.5 / np.log(min(max(c[0], 1e-6), 0.999))))
                 for c in (curves["z"], curves["y"], curves["x"]))


def table(rho, m, w, ridge_rel=1e-6):
    """8 角色 × 7 目标（含同模式与互补模式）的泄漏区间；返回 min/mean/max + 最大条件数。"""
    cells = np.array(list(itertools.product(range(-w, w + 1), repeat=3)), float)
    vals, conds = [], []
    for a in range(8):
        for y in range(8):
            if a == y:
                continue
            S = m * (2.0 * cells + off(a))
            sy = m * off(y)
            C = rho(S[:, None, :] - S[None, :, :])
            lam = np.linalg.eigvalsh(C)[-1]
            C = C + ridge_rel * lam * np.eye(len(S))
            conds.append(float(np.linalg.cond(C)))
            c = rho(sy[None, :] - S)
            vals.append(float(np.clip(c @ np.linalg.solve(C, c), 0.0, 1.0)))
    v = np.array(vals)
    return {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max()),
            "cond": float(np.max(conds)), "ok": bool(np.max(conds) < COND_MAX),
            "n_sites": int(len(cells))}


def main():
    j = json.loads(SRC.read_text())
    rec, lines = {}, []
    for seq in j:
        P1 = j[seq]["summary"]["P1"]
        raw = {ax: [v[0] for v in j[seq]["P1"]["acf"][ax]] for ax in ("z", "y", "x")}
        plat = {ax: float(P1[ax]["plateau"]) for ax in ("z", "y", "x")}
        fast = {ax: list(fast_curve(raw[ax], plat[ax])) for ax in ("z", "y", "x")}
        ell = {"raw": ell_from_lag1(raw), "fast": ell_from_lag1(fast)}
        rec[seq] = {"raw": raw, "fast": fast, "ell": ell, "rows": []}
        lines.append(f"\n=== {seq}   ℓ_raw={np.round(ell['raw'],3)}  ℓ_fast={np.round(ell['fast'],3)}")
        for mode in ("fast", "raw"):
            cur, e = fast if mode == "fast" else raw, ell[mode]
            for m in (1.0, 2.0, 3.0):
                for w in (1, 2, 3):
                    if m * w > 6:                    # 窗口超出 ACF 可测范围就别算了
                        continue
                    emp = table(lambda d, cc=cur: rho_sep(d, cc), m, w)
                    gau = table(lambda d, ee=e: rho_iso(d, ee), m, w)
                    row = {"mode": mode, "m": m, "w": w, "emp": emp, "gauss": gau}
                    rec[seq]["rows"].append(row)
                    tag = "" if emp["ok"] else "  ⚠条件数过大，不可用"
                    lines.append(
                        f"  [{mode}] m={m} w={w}: "
                        f"实测ρ {emp['min']:.3f}/{emp['mean']:.3f}/{emp['max']:.3f} (cond {emp['cond']:.0e})"
                        f"   Gaussian {gau['min']:.3f}/{gau['mean']:.3f}/{gau['max']:.3f}{tag}")
    txt = "\n".join(lines)
    print(txt)
    rec["_trace"] = txt
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

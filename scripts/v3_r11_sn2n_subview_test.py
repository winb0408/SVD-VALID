"""R11 前置检验（不需要训练）：SN2N 的对角子视图前提，在我们的数据上成立吗？

SN2N v0.3.5 把一帧切成 2×2 块，取两条对角像素的均值作为两张"独立观测"：

    A = (ul + dr)/2 ,  B = (ur + dl)/2        # ul=(0,0) dr=(1,1) ur=(0,1) dl=(1,0)

这只有在"相邻像素的噪声彼此独立"时才是两张独立观测。若噪声在 1 像素尺度上就已相关，
两张视图的噪声会被同一条相关结构共享，自监督的约束就变成了"从相关噪声预测同一相关噪声"。

我们用两个互补的办法量它：
1. **实测**：在背景区（低分位体素）直接算 `corr(A, B)`；
2. **解析上界**：可分离近似下
       cov(A,B) = 2ρ(1,0) + 2ρ(0,1)
       var(A)   = 2 + 2ρ(1,1) ≈ 2 + 2ρ(0,1)ρ(1,0)
       corr(A,B) = (ρ_y + ρ_x) / (1 + ρ_y ρ_x)
   ρ 从 `data/processed/v3_correlation_length.json`（真实数据，去平台快速分量）
   与 `reports/v3_r06/v2_endpoint_audit.json`（v2 仿真）读。

阈值写在跑之前：**cross-view 噪声相关 > 0.10 即判定"SN2N 的独立子视图前提在本数据上不成立"**。
0.10 不是调出来的：SN2N 的自约束项要求两张视图噪声无关，噪声相关 0.1 意味着仍有 10% 的方差
被两视图共享，已经足以让"预测另一视图"部分地退化为"复制同一噪声"。

用法
----
    python -u scripts/v3_r11_sn2n_subview_test.py
输出 `reports/v3_R11/sn2n_subview_test.json`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CORR = ROOT / "data/processed/v3_correlation_length.json"
SIM_AUDIT = ROOT / "reports/v3_r06/v2_endpoint_audit.json"
REAL_PD = ROOT / "data/processed/sa64_blood_remap/power_doppler.npy"
REAL_MASK = ROOT / "data/processed/official/SA64/valid_mask.npy"
OUT = ROOT / "reports/v3_R11/sn2n_subview_test.json"

PREMISE_THR = 0.10          # 写死在跑之前
BG_QUANTILE = 0.20


def cross_view_views(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SN2N 的 `block3d` 在给定 2D 平面上的两条对角均值视图（形状各减半）。"""
    ul = plane[0::2, 0::2]
    dr = plane[1::2, 1::2]
    ur = plane[0::2, 1::2]
    dl = plane[1::2, 0::2]
    # 奇数尺寸时 0::2 与 1::2 的元素个数差 1，四种视图必须裁到公共大小才能相加
    n0 = min(ul.shape[0], dr.shape[0], ur.shape[0], dl.shape[0])
    n1 = min(ul.shape[1], dr.shape[1], ur.shape[1], dl.shape[1])
    A = 0.5 * (ul[:n0, :n1] + dr[:n0, :n1])
    B = 0.5 * (ur[:n0, :n1] + dl[:n0, :n1])
    return A, B


def empirical(vol: np.ndarray, bg: np.ndarray, plane_axes: tuple[int, int]) -> dict:
    """在背景体素上直接测 corr(A,B) 与两张视图各自的方差/均值。"""
    a, b = plane_axes
    others = [i for i in range(3) if i not in (a, b)]
    As, Bs, Bg = [], [], []
    for k in range(vol.shape[others[0]]):
        plane = np.take(vol, k, axis=others[0])
        mask = np.take(bg, k, axis=others[0])
        A, B = cross_view_views(plane)
        # 一个 2h×2w 块里 4 个像素全在背景中，才算干净的背景块
        m_ul = mask[0::2, 0::2][: A.shape[0], : A.shape[1]]
        m_dr = mask[1::2, 1::2][: A.shape[0], : A.shape[1]]
        m_ur = mask[0::2, 1::2][: A.shape[0], : A.shape[1]]
        m_dl = mask[1::2, 0::2][: A.shape[0], : A.shape[1]]
        m = m_ul & m_dr & m_ur & m_dl
        if m.sum() == 0:
            continue
        As.append(A[m])
        Bs.append(B[m])
    if not As:
        return {"n": 0}
    A = np.concatenate(As)
    B = np.concatenate(Bs)
    r = float(np.corrcoef(A, B)[0, 1]) if A.size > 3 else float("nan")
    # 每张视图自身也有"块内平均"带来的方差降低：SN2N 的两张视图各是 2 个像素的均值
    return {"n": int(A.size), "corr_A_B": r,
            "std_A": float(A.std()), "std_B": float(B.std())}


def analytic(rho_a: float, rho_b: float, rho_diag: float | None = None) -> dict:
    if rho_diag is None:
        rho_diag = rho_a * rho_b
    r = (rho_a + rho_b) / (1.0 + rho_diag)
    return {"rho_a": rho_a, "rho_b": rho_b, "rho_diag_assumed": rho_diag,
            "corr_A_B": float(r)}


def main() -> int:
    rec = {"test": "R11 前置：SN2N 对角子视图的噪声独立性", "premise_threshold": PREMISE_THR,
           "threshold_written_before_running": True,
           "formula": "corr(A,B) = (rho_a + rho_b) / (1 + rho_a*rho_b)（可分离近似）",
           "sources": {}, "results": {}}

    # ---------- 真实数据（解析，来自实测 ACF） ----------
    J = json.loads(CORR.read_text())
    real = {}
    for seq in J:
        s = J[seq]["summary"]["P1"]
        # SN2N 在 2D 平面上做对角；我们把 (y,x) 当平面（超声图像平面），另给两轴组合作敏感性
        for name, (a, b) in {"plane_yx": ("y", "x"), "plane_zy": ("z", "y"),
                             "plane_zx": ("z", "x")}.items():
            raw = analytic(s[a]["rho_lag1"], s[b]["rho_lag1"])
            fast = analytic(s[a]["rho_fast_lag1"], s[b]["rho_fast_lag1"])
            real[f"{seq}/{name}"] = {
                "raw_plateau_included": raw,
                "fast_deplataeu": fast,
                "premise_fails_raw": raw["corr_A_B"] > PREMISE_THR,
                "premise_fails_fast": fast["corr_A_B"] > PREMISE_THR,
            }
        rec["sources"][seq] = str(CORR.relative_to(ROOT))
    rec["results"]["real_analytic"] = real

    # ---------- v2 仿真（解析） ----------
    A = json.loads(SIM_AUDIT.read_text())
    sim = {}
    for name, v in A.items():
        ax = v.get("q4_acf", {}).get("per_axis")
        if not ax:
            continue
        for plane, (a, b) in {"plane_yx": ("y", "x"), "plane_zy": ("z", "y"),
                              "plane_zx": ("z", "x")}.items():
            an = analytic(ax[a]["sim_lag1"], ax[b]["sim_lag1"])
            sim[f"{name}/{plane}"] = {**an, "premise_fails":
                                      an["corr_A_B"] > PREMISE_THR}
    rec["results"]["sim_analytic"] = sim
    rec["sources"]["sim"] = str(SIM_AUDIT.relative_to(ROOT))

    # ---------- 实测（真实 SA64 背景 + v2 仿真 eval 背景） ----------
    emp = {}
    if REAL_PD.exists() and REAL_MASK.exists():
        pd = np.load(REAL_PD).astype(np.float64)
        vm = np.load(REAL_MASK).astype(bool)
        bg = vm & (pd <= np.quantile(pd[vm], BG_QUANTILE))
        for plane, (a, b) in {"plane_yx": (1, 2), "plane_zy": (0, 1),
                              "plane_zx": (0, 2)}.items():
            e = empirical(pd, bg, (a, b))
            e["premise_fails"] = bool(e.get("corr_A_B", np.nan) > PREMISE_THR)
            emp[f"SA64/{plane}"] = e
        # SN2N 会先做逐帧 percentile(20,99.9) 归一化：对相关结构无影响（单调变换），记录一句
        emp["note"] = ("percentile 归一化是逐帧单调变换，不改变块内的相关结构；"
                       "这里用原始 PD 计算，等价。")
    man = ROOT / "data/processed/v3_sim_v2/manifest.json"
    if man.exists():
        for e in json.loads(man.read_text())["eval"]:
            d = np.load(ROOT / e["npz"])
            pd = d["pd_linear"].astype(np.float64)
            vm = d["valid"].astype(bool)
            bg = vm & (pd <= np.quantile(pd[vm], BG_QUANTILE))
            key = Path(e["npz"]).stem
            for plane, (a, b) in {"plane_yx": (1, 2), "plane_zy": (0, 1),
                                  "plane_zx": (0, 2)}.items():
                r = empirical(pd, bg, (a, b))
                r["premise_fails"] = bool(r.get("corr_A_B", np.nan) > PREMISE_THR)
                emp[f"{key}/{plane}"] = r
    rec["results"]["empirical"] = emp

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    print(f"阈值（写于跑之前）：cross-view 噪声相关 > {PREMISE_THR} 判定前提不成立\n")
    print("== 实测 corr(A,B)（背景区，2×2 对角线均值视图）")
    for k, v in emp.items():
        if isinstance(v, dict) and "corr_A_B" in v:
            print(f"  {k:26s} corr={v['corr_A_B']:.3f}  n={v['n']:6d}  "
                  f"前提{'不成立' if v['premise_fails'] else '成立(勉强)'}")
    print("\n== 解析（实测 ACF 快速分量，平面 (y,x)）")
    for seq in J:
        print(f"  {seq:6s} raw={real[f'{seq}/plane_yx']['raw_plateau_included']['corr_A_B']:.3f}  "
              f"fast={real[f'{seq}/plane_yx']['fast_deplataeu']['corr_A_B']:.3f}")
    for name in A:
        if f"{name}/plane_yx" in sim:
            print(f"  {name:12s} sim={sim[f'{name}/plane_yx']['corr_A_B']:.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

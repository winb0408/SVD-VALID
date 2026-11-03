"""Round6 §7 要求的**分区读数**：把同一个支撑集按亮度分层，
看各臂是在**削背景散斑**还是在**削亮结构**。

## 为什么必须做

预注册 §7 明确要求"每臂的血管区 / 背景区分开读数（判断是去噪还是在削结构）"。
真实采集**没有血管掩膜**（仿真才有 `rho`），分层只能借亮度代理。

## 分层为什么只能用两半的平均，不能用 A 或 B 单独

主指标 `X² = E[(f(A)−B)²]` 之所以无偏，靠的是交叉项为零。
一旦按某个**亮度函数**分带，交叉项只有在"带是 A 的函数"或"带是 B 的函数"时才出现单边偏差：

- 只按 `B` 分带 ⇒ `E[B−sig|带] ≠ 0`；任何系统性压暗的臂都会被额外惩罚到"亮带全面变差"，
  连高斯滤波都变成 `X/X_in > 1`，这是**选择偏差**而不是去噪结论。
- 只按 `A` 分带 ⇒ 反过来，把 A 的噪声尖峰框进亮带，去噪臂白拿好处。

两半平均 `h = (L(A)+L(B))/2` 是唯一对两边噪声对称的亮度代理，取它的分位数分带。
代价是残余污染 `≈ 2·m·μ`（`m` = 带内系统偏差，`μ` = 带内 `B−h` 的均值，量级只有单边的一半），
所以带内 `X/X_in` **只用于臂间比较，不当作绝对误差**；`0.707` 是**全支撑集**的理想值，
单带的理想值因该带 `σ²_B` 不同而不同，不得套用。

## 三个读数

| 名字 | 定义 | 怎么读 |
|---|---|---|
| `ratio` | 带内 `X/X_in` | `<1` 有改善；只作臂间比较 |
| `bias_shift` | `mean(L(f_α(A)) − L(A))` | 负 = 把这一带整体压暗 |
| `slope` | 带内 `L(f_α(A))` 对 `h` 的回归斜率 | `<1` = 压缩对比度（收缩型估计），`≈1` = 保幅 |

`slope` 是关键：它不需要真值。input 臂的斜率按构造 ≈1；某臂斜率明显小于 1，
就说明它的"改善"有一部分来自把亮结构往中间压——这是收缩估计的固有代价，也是
N2N 类方法在真实散斑上最容易犯的错（往条件均值压缩）。

## 口径

- 方向 2（`A=even → B=odd`），与主结论同一方向；`α` 沿用方向 1 存下的值，**不重拟合**。
- 支撑集与主口径完全相同（`pair2` 的 30011 体素 ∩ valid），只是再按亮度切三份。
- **这是主指标的分解，不是新判据**：不参与 SUPPORT/WORSE 判定，只用来读机制。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v3_round6_eval as E                                    # noqa: E402
from v3_round5_eval import usable                             # noqa: E402

OUT = ROOT / "reports/v3_round6/regions.json"
BANDS = (("bg", 0.00, 0.80), ("mid", 0.80, 0.95), ("bright", 0.95, 1.00))
# 方向 2 里每条几何被选中的配置（来自 selections.json / N2N 单配置）
SELECTED = {
    "SA64": {"gauss": None, "D1": ("linear", 6), "BD2": ("log1p", 6), "N2N": ("log1p", 6)},
    "HSA64": {"gauss": None, "D1": ("log1p", 6), "BD2": ("log1p", 6), "N2N": ("log1p", 6)},
}


def bands_of(h: np.ndarray, sup: np.ndarray) -> dict:
    v = h[sup]
    q_lo = float(np.quantile(v, 0.80))
    q_hi = float(np.quantile(v, 0.95))
    out = {}
    for name, lo, hi in BANDS:
        if name == "bg":
            m = v <= q_lo
        elif name == "mid":
            m = (v > q_lo) & (v <= q_hi)
        else:
            m = v > q_hi
        mm = np.zeros_like(sup)
        mm[sup] = m
        out[name] = mm
    out["_cuts"] = {"p80": q_lo, "p95": q_hi}
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """y 对 x 的最小二乘斜率（带内、均值口径）。"""
    vx = float(np.var(x))
    if vx <= 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def read_arm(pd_tr, pd_hat, pd_ev, sup, s, alpha):
    """返回该臂在总支撑集与三个带上的读数。"""
    lt, lh, le = E._lspace(pd_tr, s), E._lspace(pd_hat, s), E._lspace(pd_ev, s)
    if alpha is None:
        lo = lt
    else:
        lo = lt + alpha * (lh - lt)
    h = 0.5 * (lt + le)                      # 两半平均：对两边噪声对称的亮度代理
    bands = bands_of(h, sup)
    cuts = bands.pop("_cuts")
    rec = {}
    for name, m in {"all": sup, **bands}.items():
        n = int(m.sum())
        d_out = (lo - le)[m]
        d_in = (lt - le)[m]
        x_in = float(np.sqrt(np.mean(d_in ** 2)))
        x = float(np.sqrt(np.mean(d_out ** 2)))
        rec[name] = {
            "n": n,
            "x_in": x_in, "x": x,
            "ratio": x / x_in if x_in > 0 else None,
            # 带内"去掉了 A 自身噪声的百分之几"（负 = 反而加了）；只作臂间比较
            "noise_removed": 2.0 * (1.0 - (x / x_in) ** 2) if x_in > 0 else None,
            "bias_in": float(np.mean(d_in)),
            "bias_out": float(np.mean(d_out)),
            "bias_shift": float(np.mean((lo - lt)[m])),   # 该带被整体压暗/加亮多少
            "slope": _slope(h[m], lo[m]),                 # <1 = 压缩对比度
            "mu_b": float(np.mean((le - h)[m])),          # 残余污染的 μ（B 相对两半平均）
            # 该带自身的方差被削掉多少：完美去噪的正确上限就是"该带噪声占比"。
            # 整体（all 带）这一列可直接与 reports/v3_r06/testretest_calibration.json
            # 的 noise_frac 比：超过就说明削掉的不只是噪声，还削了结构。
            "var_red": float(1.0 - np.var(lo[m]) / np.var(lt[m])) if np.var(lt[m]) > 0 else None,
        }
    rec["_cuts"] = cuts
    return rec


def main() -> int:
    from v3_round4_b1_prep import fwd
    from v3_round4_metrics import masked_gauss

    E.load_alpha()
    refs = json.loads((ROOT / "reports/v3_round6/x_scores.json").read_text())["references"]
    out = {}
    for seq, arms in SELECTED.items():
        # 方向 2：A = even，B = odd
        pd_tr, valid, s = E.load_half(seq, "even")
        pd_ev, _, _ = E.load_half(seq, "odd")
        shape = usable(pd_tr.shape, 2)
        sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
        pd_tr, pd_ev, valid = pd_tr[sl], pd_ev[sl], valid[sl]
        sup = E.bd2_support(shape, E.BD2_MODE) & valid
        out[seq] = {}

        # 参照臂 gauss：σ/α 全部沿用方向 1
        sig = str(refs[seq]["odd2even"]["gauss"]["best_sigma"])
        a_g = float(refs[seq]["odd2even"]["gauss"]["grid"][sig]["alpha"])
        lg = fwd("log1p", pd_tr, s, valid)
        pd_g = np.maximum(np.expm1(masked_gauss(lg, valid, float(sig))), 0.0) * s
        out[seq]["gauss"] = {"sigma": float(sig), "alpha": a_g,
                             **read_arm(pd_tr, pd_g, pd_ev, sup, s, a_g)}

        for geom in ("D1", "BD2", "N2N"):
            dom, ep = arms[geom]
            run = ROOT / f"runs/v3_round6/{seq}_{geom}_{dom}_e{ep}_even2odd"
            z = np.load(ROOT / f"data/processed/v3_round6_b1/{dom}/{seq}_even_x.npy")
            y, _ = E.denoise(run / "latest.pt", dom, z, geom, E.BD2_MODE)
            norm = json.loads(E.NORM_PATH.read_text())[dom]
            from v3_round4_b1_prep import inv
            pd_hat = np.maximum(inv(dom, y * norm["std"] + norm["mean"], s), 0.0)
            a = E.ALPHA.get((seq, geom, dom, int(ep)))
            out[seq][geom] = {"domain": dom, "epochs": int(ep), "alpha": a,
                              **read_arm(pd_tr, pd_hat, pd_ev, sup, s, a)}
            print(f"done {seq} {geom} {dom} e{ep} alpha={a:.4f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))

    # 终端摘要
    for seq, arms in out.items():
        cuts = arms["gauss"]["_cuts"]
        print(f"\n=== {seq}  (p80={cuts['p80']:.3f} p95={cuts['p95']:.3f}, "
              f"方向2 A=even→B=odd) ===")
        for name in ("all", "bg", "mid", "bright"):
            print(f"-- {name} (n={arms['gauss'][name]['n']})")
            print(f"   {'arm':6s} {'X/X_in':>8s} {'噪声去除%':>9s} "
                  f"{'bias_shift':>10s} {'slope':>7s} {'var削%':>7s} {'mu_b':>7s}")
            for arm, r in arms.items():
                b = r[name]
                print(f"   {arm:6s} {b['ratio']:8.4f} {100*b['noise_removed']:9.2f} "
                      f"{b['bias_shift']:+10.4f} {b['slope']:7.3f} "
                      f"{100*b['var_red']:7.2f} {b['mu_b']:+7.4f}")
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

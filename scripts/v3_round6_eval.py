#!/usr/bin/env python3
"""Round6 评价：真实采集上的交叉实现误差 `X_RMSE`（口径见预注册 §1/§2b）。

为什么用这个口径
----------------
真实采集没有干净真值。把 400 帧按奇偶拆成两半（同一次 SVD 带通，`pd_odd+pd_even`
精确等于全帧 PD），两半共享解剖、噪声独立。在 A 半训练、去噪 A、与 B 半比：

    E[(f(A) − B)²] = 真信号误差(f) + σ_B²

B 的噪声与 A 独立 ⇒ 交叉项 = 0 ⇒ **排序即真信号误差的排序**，方法间可直接比大小。
`X/X_in = 1` 是"没去噪"，`≈0.5` 是"完美去噪"（两半噪声同量级时）。

主口径（预注册 §2b）：两种几何都只在 BD-2 用训练过的角色对（`1→6`、`4→3`）直接覆盖
的那批体素上算。BD-2 用 16 次前向，D1 全图 8 次前向再取到同一批体素——
**完全同体素、零外推**，不需要 `target = input XOR 7` 那条未训练的外推规则。

一个免费的自由参数（增益校准）
------------------------------
原始输出 `f(A)` 的问题是**幅度**：修正量 `Δ = L(f(A)) − L(A)` 与真噪声负相关
（方向是对的），但总能量常常大于噪声本身，净效果变成加噪。做 1 参数校准

    L(f_α(A)) = L(A) + α·Δ,   α* = −⟨n, Δ⟩ / ‖Δ‖²

α* 有闭式解、不用重训。**α 只在方向 1 拟合，方向 2 沿用方向 1 存下的值**——
否则就等于偷看参照半，无偏性就没了。

用法
----
    python scripts/v3_round6_eval.py --what grid     # 评所有 run → x_scores.json
    python scripts/v3_round6_eval.py --what select   # 每几何选最优 → selections.json
    python scripts/v3_round6_eval.py --what report   # 打印汇总表
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import inv                                  # noqa: E402
from v3_round5_eval import SCATTER, reconstruct_full, usable      # noqa: E402

NORM_PATH = ROOT / "reports/v3_round4/b1_norm.json"
NAME_RE = re.compile(r"^(SA64|HSA64)_(D1|BD2|N2N)_(linear|log1p|logeps)_e(\d+)_(odd|even)2(odd|even)$")
_GEOM_STRIDE = {"D1": 1, "BD2": 2, "N2N": 1}   # N2N = 正对照（满分辨率 + 另一半做目标）


def L(pd: np.ndarray, s: float) -> np.ndarray:
    """冻结的 log1p 指标域 `log1p(pd/s)`。"""
    return np.log1p(np.maximum(pd, 0.0) / s)


def x_rmse(pd_hat: np.ndarray, pd_ref: np.ndarray, support: np.ndarray, s: float) -> float:
    a, b = L(pd_hat, s)[support], L(pd_ref, s)[support]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_half(seq: str, tag: str):
    d = ROOT / "data/processed/v3_round6" / seq
    pd = np.load(d / tag / "pd_linear.npy").astype(np.float64)
    valid = np.load(d / "valid_mask.npy")
    s = float(json.loads((d / "metadata.json").read_text())["s"])
    return pd, valid, s


class _Identity:
    """只为拿到几何覆盖掩膜：输出恒等于输入。"""

    def __call__(self, x):
        return x


_SUPPORT_CACHE: dict = {}
BD2_MODE = "pair2"          # 主口径；"pair4" 为稳健性检验，见 main() 的 --bd2-mode


def bd2_scatter(mode: str) -> dict | None:
    """BD-2 回填映射。`pair2` = 预注册主口径（只 2 对已训练映射）；`pair4` = 4 对全用。"""
    if mode == "pair2":
        return SCATTER["O_log"]
    if mode == "pair4":
        return {1: 6, 4: 3, 6: 1, 3: 4}          # --slots 1 4 6 3 --pairing cross 学到的 4 条
    raise ValueError(f"未知 BD-2 回填模式 {mode}")


def bd2_support(shape, mode: str = "pair2") -> np.ndarray:
    """BD-2 在指定回填映射下直接覆盖的体素集（只由几何决定，与模型无关）。"""
    key = (tuple(shape), mode)
    if key not in _SUPPORT_CACHE:
        _, filled = reconstruct_full(_Identity(), np.zeros(usable(shape, 2), np.float32),
                                     2, "cpu", scatter_to=bd2_scatter(mode))
        _SUPPORT_CACHE[key] = filled
    return _SUPPORT_CACHE[key]


def denoise(ckpt: Path, domain: str, z: np.ndarray, geom: str, bd2_mode: str = "pair2"):
    """返回 (标准化域的输出, 覆盖掩膜)——反变换交给调用方（需要 s）。"""
    import torch
    from models.network import NetworkCNR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device)
    ta = st.get("args", {})
    model = NetworkCNR(features=ta.get("features", 16), groups=ta.get("groups", 4),
                       residual=(ta.get("arch_head", "residual") == "residual")).to(device)
    model.load_state_dict(st["model"])
    model.eval()
    scat = None if geom in ("D1", "N2N") else bd2_scatter(bd2_mode)
    return reconstruct_full(model, z, _GEOM_STRIDE[geom], device, scatter_to=scat)


def eval_one(run: Path) -> dict | None:
    mm = NAME_RE.match(run.name)
    if not mm:
        return None
    seq, geom, domain, epochs, th, eh = mm.groups()

    pd_tr, valid, s = load_half(seq, th)
    pd_ev, _, _ = load_half(seq, eh)
    shape = usable(pd_tr.shape, 2)
    sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
    pd_tr, pd_ev, valid = pd_tr[sl], pd_ev[sl], valid[sl]

    z = np.load(ROOT / f"data/processed/v3_round6_b1/{domain}/{seq}_{th}_x.npy")
    y, filled = denoise(run / "latest.pt", domain, z, geom, BD2_MODE)
    norm = json.loads(NORM_PATH.read_text())[domain]
    pd_hat = np.maximum(inv(domain, y * norm["std"] + norm["mean"], s), 0.0)

    sup = bd2_support(shape, BD2_MODE) & valid   # 支撑集：两种几何完全相同的体素
    x_raw = x_rmse(pd_hat, pd_ev, sup, s)

    # 增益校准：方向 1 现场拟合 α*；方向 2 只允许沿用方向 1 存下的 α
    alpha_fit, x_gain_fit = fit_gain(pd_tr, pd_hat, pd_ev, sup, s)
    if th == "odd":
        alpha_used, alpha_src = alpha_fit, "fitted_on_direction1"
    else:
        alpha_used = ALPHA.get((seq, geom, domain, int(epochs)))
        alpha_src = "from_selections" if alpha_used is not None else "unavailable"
    if alpha_used is None:
        x_gain = None
    else:
        x_gain = x_rmse(apply_gain(pd_tr, pd_hat, s, alpha_used), pd_ev, sup, s)

    return {"run": run.name, "seq": seq, "geom": geom, "domain": domain,
            "epochs": int(epochs), "train_half": th, "eval_half": eh,
            "s": s, "n_support": int(sup.sum()), "coverage": float(filled.mean()),
            "x_pair": x_raw,
            "x_in_pair": x_rmse(pd_tr, pd_ev, sup, s),
            "alpha_fit": alpha_fit, "x_gain_fit": x_gain_fit,
            "alpha_used": alpha_used, "alpha_src": alpha_src, "x_gain": x_gain}


SEL_PATH = ROOT / "reports/v3_round6/selections.json"
ALPHA: dict = {}


def load_alpha() -> None:
    """把方向 1 定下的 α 读进来；方向 2 的评估只认这张表，不现场重拟合。

    两个来源：`selections.json`（选中的几何配置）与 `x_scores.json` 的方向 1 行
    （`alpha_fit`）。后者是 N2N 正对照唯一的来源——它不参与选型，不会出现在 selections 里。
    """
    ALPHA.clear()
    if SEL_PATH.exists():
        sel = json.loads(SEL_PATH.read_text())
        for seq, per_geom in sel.items():
            for geom, v in per_geom.items():
                b = v.get("best", {})
                if "alpha" in b:
                    ALPHA[(seq, geom, b["domain"], int(b["epochs"]))] = float(b["alpha"])
    p = ROOT / "reports/v3_round6/x_scores.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r["train_half"] == "odd":
                ALPHA.setdefault((r["seq"], r["geom"], r["domain"], int(r["epochs"])),
                                 float(r["alpha_fit"]))


def _lspace(pd_arr: np.ndarray, s: float) -> np.ndarray:
    return L(pd_arr, s)


def apply_gain(pd_tr: np.ndarray, pd_hat: np.ndarray, s: float, alpha: float) -> np.ndarray:
    """在度量域做 1 参数增益校准：L(f_α(A)) = L(A) + α·(L(f(A)) − L(A))。

    α = 1 即原样输出；α = 0 即完全退回 input。只调一个标量，不动网络。
    """
    lt, lh = _lspace(pd_tr, s), _lspace(pd_hat, s)
    return np.maximum(np.expm1(lt + alpha * (lh - lt)), 0.0) * s


def fit_gain(pd_tr: np.ndarray, pd_hat: np.ndarray, pd_ev: np.ndarray,
             sup: np.ndarray, s: float) -> tuple[float, float]:
    """最小化 X(α) 的闭式解 α* = −⟨n, Δ⟩ / ‖Δ‖²，返回 (α*, X(α*))。

    注意：这一步用了参照半 B，所以只能在**方向 1**（选型方向）上做；
    方向 2 必须沿用方向 1 定下来的 α，不得重新拟合。
    """
    # n 取 A − B（与 Δ 同一朝向），这样 α* = −⟨n,Δ⟩/‖Δ‖² 的符号是"跟噪声反着来为正"。
    # 三处内积统一用**均值**口径，否则 α 正确但 X² 会混入体素数的量纲。
    n = (_lspace(pd_tr, s) - _lspace(pd_ev, s))[sup]
    d = (_lspace(pd_hat, s) - _lspace(pd_tr, s))[sup]
    nn, dd, nd = float(np.mean(n * n)), float(np.mean(d * d)), float(np.mean(n * d))
    if dd <= 0:
        return 1.0, float(np.sqrt(nn))
    alpha = -nd / dd
    x = float(np.sqrt(max(nn - nd * nd / dd, 0.0)))
    return alpha, x


def references(seq: str) -> dict:
    """两条非学习参照：input（不去噪）与 gauss(σ) 在 σ 网格上的最优。"""
    from v3_round4_b1_prep import fwd
    from v3_round4_metrics import masked_gauss

    out = {}
    for th, eh in (("odd", "even"), ("even", "odd")):
        pd_tr, valid, s = load_half(seq, th)
        pd_ev, _, _ = load_half(seq, eh)
        shape = usable(pd_tr.shape, 2)
        sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
        pd_tr, pd_ev, valid = pd_tr[sl], pd_ev[sl], valid[sl]
        sup = bd2_support(shape, BD2_MODE) & valid
        r = {"input": x_rmse(pd_tr, pd_ev, sup, s),
             "x_in": x_rmse(pd_tr, pd_ev, sup, s)}
        rec = {}
        lg = fwd("log1p", pd_tr, s, valid)
        # σ 网格向细端补了 0.3/0.5：原 {0.4..1.0} 的最优落在边界 0.4 上，说明更温和的平滑
        # 还没探到。参照臂更强只会让"学习类臂"的结论更保守，不构成对学习臂的偏袒。
        # σ=0.2 实测 α*→2.6e4 发散：σ→0 与 α→∞ 是同一个极限，不是去噪器而是数值退化，剔除。
        # α 统一截到 [0, 5]：超过这个范围已不是"调幅度"而是在做高通外推。
        for sig in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
            pd_g = np.maximum(np.expm1(masked_gauss(lg, valid, sig)), 0.0) * s
            # 参照臂同样给增益校准：不然"每个方法拿最好的样子"对学习臂不公平
            a_g, x_g = fit_gain(pd_tr, pd_g, pd_ev, sup, s)
            a_g = min(max(a_g, 0.0), 5.0)
            x_g = x_rmse(apply_gain(pd_tr, pd_g, s, a_g), pd_ev, sup, s)
            rec[sig] = {"x": x_rmse(pd_g, pd_ev, sup, s), "x_gain": x_g, "alpha": a_g}
        best = min(rec, key=lambda k: rec[k]["x_gain"])
        r["gauss"] = {"best_sigma": best, "x": rec[best]["x"],
                      "x_gain": rec[best]["x_gain"], "alpha": rec[best]["alpha"],
                      "grid": rec}
        out[f"{th}2{eh}"] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["grid", "select", "report"], default="grid")
    ap.add_argument("--runs", default="runs/v3_round6")
    ap.add_argument("--out", default="reports/v3_round6/x_scores.json")
    ap.add_argument("--bd2-mode", choices=["pair2", "pair4"], default="pair2",
                    help="BD-2 回填映射：pair2 = 预注册主口径；pair4 = 4 对已训练映射全用（稳健性）")
    a = ap.parse_args()
    global BD2_MODE
    BD2_MODE = a.bd2_mode
    load_alpha()

    if a.what == "grid":
        # 只评**跑完**的 run（有 DONE 标记）：绝不拿半截 checkpoint 当结果
        runs = sorted(p for p in (ROOT / a.runs).iterdir()
                      if p.is_dir() and NAME_RE.match(p.name)
                      and (p / "latest.pt").exists() and (p / "DONE").exists())
        rows = [r for r in (eval_one(p) for p in runs) if r]
        for r in rows:
            xg = "   n/a  " if r["x_gain"] is None else f"{r['x_gain']:.5f}"
            rg = "  n/a " if r["x_gain"] is None else f"{r['x_gain'] / r['x_in_pair']:.4f}"
            print(f"{r['run']:36s} X={r['x_pair']:.5f}  X_in={r['x_in_pair']:.5f}  "
                  f"ratio={r['x_pair']/r['x_in_pair']:.4f}  "
                  f"| α_fit={r['alpha_fit']:+.3f} X_gain={xg}  ratio_gain={rg}  "
                  f"cov={r['coverage']:.2f}")
        out = {"rows": rows,
               "references": {sq: references(sq) for sq in ("SA64", "HSA64")}}
        (ROOT / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"wrote {ROOT/a.out}（{len(rows)} 行 + 参照）")
        return 0

    blob = json.loads((ROOT / a.out).read_text())
    rows = blob["rows"]
    if a.what == "select":
        sel = {}
        for seq in ("SA64", "HSA64"):
            sel[seq] = {}
            for geom in ("D1", "BD2"):
                cand = [r for r in rows if r["seq"] == seq and r["geom"] == geom
                        and r["train_half"] == "odd"]
                if not cand:
                    continue
                # 选型标准 = 增益校准后的 X（每个方法拿它最好的样子）；
                # 同时记下未校准的 X，便于看清校准到底贡献了多少
                best = min(cand, key=lambda r: r["x_gain_fit"])
                sel[seq][geom] = {
                    "best": {"domain": best["domain"], "epochs": best["epochs"],
                             "alpha": best["alpha_fit"],
                             "x_gain_fit": best["x_gain_fit"], "x_raw": best["x_pair"],
                             "x_in_pair": best["x_in_pair"]},
                    "grid": [{k: r[k] for k in ("domain", "epochs", "x_pair",
                                                "x_gain_fit", "alpha_fit", "x_in_pair")}
                             for r in sorted(cand, key=lambda r: r["x_gain_fit"])]}
                print(f"{seq} {geom}: best {best['domain']} e{best['epochs']}  "
                      f"α*={best['alpha_fit']:+.3f}  X_gain={best['x_gain_fit']:.5f}  "
                      f"(raw {best['x_pair']:.5f}, X_in={best['x_in_pair']:.5f}, "
                      f"ratio_gain={best['x_gain_fit']/best['x_in_pair']:.4f})")
        SEL_PATH.write_text(json.dumps(sel, indent=1, ensure_ascii=False))
        print(f"wrote {SEL_PATH}")
        return 0

    for seq in ("SA64", "HSA64"):
        print(f"\n=== {seq} ===")
        ref = blob["references"][seq]
        for k, v in ref.items():
            g = v["gauss"]
            print(f"  参照 {k}: input X={v['input']:.5f}  gauss σ*={g['best_sigma']} "
                  f"X={g['x']:.5f}（校准后 {g['x_gain']:.5f}, α={g['alpha']:+.3f}, "
                  f"ratio_gain={g['x_gain']/v['x_in']:.4f}）")
        for r in sorted([r for r in rows if r["seq"] == seq],
                        key=lambda r: (r["geom"], r["train_half"], r["x_gain_fit"])):
            xg = "  n/a  " if r["x_gain"] is None else f"{r['x_gain']:.5f}"
            rg = "  n/a " if r["x_gain"] is None else f"{r['x_gain']/r['x_in_pair']:.4f}"
            print(f"  {r['geom']:4s} {r['domain']:6s} e{r['epochs']:<3d} "
                  f"{r['train_half']}→{r['eval_half']}  X={r['x_pair']:.5f}  "
                  f"ratio={r['x_pair']/r['x_in_pair']:.4f}  | X_gain={xg} ratio_gain={rg} "
                  f"α={r['alpha_used']:+.3f}({r['alpha_src']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

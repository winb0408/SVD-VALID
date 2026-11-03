#!/usr/bin/env python3
"""Round10 评价：真实采集上**完整自监督训练**之后，A / B 两臂 vs 零训练基线 vs SN2N。

口径全部来自 `reviews/V3_PR1_round10_prereg_real_train.md`（跑之前写死），本脚本只算不改：

    X_RMSE(f) = sqrt( mean_{v∈valid} ( L(f(A))_v − L(B)_v )² ),   L(pd) = log1p(max(pd,0)/s)

在 A 半上训练、去噪 A、与噪声独立的 B 半比 ⇒ 交叉项为零 ⇒ **排序无偏**。
`X_RMSE/X_RMSE(input)`：1.0 = 没去噪，0.707 ≈ 完美去噪。

与 Round6 评价的三处**有意不同**（都记在报告的偏离节里）：
1. 支撑集用**整个 valid mask**，不取 BD-2 的 1/4 覆盖集。A/B 两臂都是整卷稠密滑窗推理
   （与 `infer.py` 同口径），本来就在全部体素上有输出，再取子集只会白白丢样本。
   代价：本轮数字**不能**与 Round6 的 `X_RMSE` 直接比大小，只能各自与自己那套参照比。
2. 高斯参照的 σ 按 **α=1 的 X_RMSE** 在方向 1 上选（Round6 是按增益校准后的 `X(α*)` 选）。
   判据 §4 用的是 α=1，所以参照也必须在同一口径下选，否则等于给参照臂挑一个在判定口径下
   不是最好的 σ。这个改动只会让参照更强、对本方法更不利。
3. 多了一条 SN2N 忠实移植臂（同一半卷、200 epoch、整帧单次前向）。

用法
----
    python -u scripts/v3_round10_eval.py --what grid     # 评所有 run → scores.json
    python -u scripts/v3_round10_eval.py --what select   # 方向 1 选配置 → selections.json
    python -u scripts/v3_round10_eval.py --what report   # 打印汇总表
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import wiener as scipy_wiener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import fwd, inv                    # noqa: E402
from v3_round4_metrics import masked_gauss                # noqa: E402

B1 = ROOT / "data/processed/v3_round6_b1"
HALF = ROOT / "data/processed/v3_round6"
RUNS = ROOT / "runs/v3_round10"
OUT = ROOT / "reports/v3_round10"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())

SEQ = ("SA64", "HSA64")
DOMAINS = ("linear", "log1p", "logeps")
SIGMAS = (0.2, 0.3, 0.4, 0.6, 0.8)
PATCH = (32, 64, 64)
OVERLAP = 0.1
NAME_RE = re.compile(r"^(SA64|HSA64)_(A|B)_(linear|log1p|logeps)_e(\d+)_(odd|even)2(odd|even)$")
OPP = {"odd": "even", "even": "odd"}


# ------------------------------------------------------------------ 基础量 ----
def L(pd: np.ndarray, s: float) -> np.ndarray:
    """冻结的 log1p 指标域 `log1p(max(pd,0)/s)`。"""
    return np.log1p(np.maximum(pd, 0.0) / s)


def load_half(seq: str, tag: str):
    d = HALF / seq
    pd = np.load(d / tag / "pd_linear.npy").astype(np.float64)
    valid = np.load(d / "valid_mask.npy").astype(bool)
    s = float(json.loads((d / "metadata.json").read_text())["s"])
    return pd, valid, s


def x_rmse(pd_hat: np.ndarray, pd_ref: np.ndarray, support: np.ndarray, s: float) -> float:
    a, b = L(pd_hat, s)[support], L(pd_ref, s)[support]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_gain(pd_tr, pd_hat, pd_ev, sup, s):
    """α* = −⟨n,Δ⟩/‖Δ‖²，返回 (α*, X(α*))。只能在方向 1 上拟合（用了参照半 B）。"""
    n = (_L(pd_tr, s) - _L(pd_ev, s))[sup]
    d = (_L(pd_hat, s) - _L(pd_tr, s))[sup]
    nn, dd, nd = float(np.mean(n * n)), float(np.mean(d * d)), float(np.mean(n * d))
    if dd <= 0:
        return 1.0, float(np.sqrt(nn))
    return -nd / dd, float(np.sqrt(max(nn - nd * nd / dd, 0.0)))


def apply_gain(pd_tr, pd_hat, s, alpha):
    """L(f_α) = L(in) + α·(L(out) − L(in))，回到线性 PD 再交给 x_rmse。"""
    a = _L(pd_tr, s) + alpha * (_L(pd_hat, s) - _L(pd_tr, s))
    return np.maximum(np.expm1(a), 0.0) * s


def overfit_block(pd_tr, pd_hat, pd_ev, sup, s):
    """κ / ‖Δ⊥‖，并做恒等式 X² = (1−κ)²X_in² + ‖Δ⊥‖² 的机器精度自检。

    **符号约定与 Round6 的 overfit_diag 一致**：κ = −⟨n,Δ⟩/‖n‖²。完美去噪（Δ = −n）时 κ=1；
    κ<0 表示修正量跟真噪声同向（在加噪）。第一版这里漏了负号，恒等式自检立刻报
    `identity_ok=False`——自检就是为这种事写的。
    """
    n = (_L(pd_tr, s) - _L(pd_ev, s))[sup]
    d = (_L(pd_hat, s) - _L(pd_tr, s))[sup]
    nn, dd, nd = float(np.mean(n * n)), float(np.mean(d * d)), float(np.mean(n * d))
    kappa = -nd / nn if nn > 0 else float("nan")
    perp2 = max(dd - kappa * kappa * nn, 0.0)
    x2 = float(np.mean((_L(pd_hat, s) - _L(pd_ev, s))[sup] ** 2))
    resid = abs(x2 - ((1 - kappa) ** 2 * nn + perp2))
    return {"kappa": kappa, "perp_norm": float(np.sqrt(perp2)),
            "noise_norm": float(np.sqrt(nn)), "corr_dn": nd / np.sqrt(nn * dd) if dd > 0 else None,
            "identity_resid": resid, "identity_ok": bool(resid < 1e-12 * max(1.0, x2))}


def contrast_block(pd_hat, pd_in, valid, s):
    """对比度类读数——用来排除「靠压平对比度把 X 压小」。全部在 log1p 指标域算。"""
    a, b = L(pd_in, s)[valid], L(pd_hat, s)[valid]
    var_in, var_out = float(np.var(a)), float(np.var(b))
    slope = float(np.dot(a - a.mean(), b - b.mean()) / (a.size * max(a.var(), 1e-30)))
    thr = np.quantile(a, 0.90)
    core = float(b[a >= thr].mean() / a[a >= thr].mean())
    dark = a <= np.quantile(a, 0.30)
    bg_std = float(b[dark].std() / a[dark].std())
    return {"slope": slope, "core_ratio": core, "bg_std_ratio": bg_std,
            "var_cut_pct": 100.0 * (1.0 - var_out / var_in) if var_in > 0 else None,
            "mean_in": float(a.mean()), "mean_out": float(b.mean())}


def _L(pd, s):
    return L(pd, s)


# ------------------------------------------------------------------ 推理 ----
def dense_infer(ckpt: Path, z: np.ndarray, device: str = "cuda") -> np.ndarray:
    """整卷稠密滑窗推理（与 `infer.py` 完全同一规则：偶尺寸零填充 + 重叠累加平均）。"""
    import torch
    from models.network import NetworkCNR

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=dev, weights_only=False)
    ta = st.get("args", {})
    model = NetworkCNR(features=ta.get("features", 16), groups=ta.get("groups", 4),
                       residual=(ta.get("arch_head", "direct") == "residual")).to(dev)
    model.load_state_dict(st["model"])
    model.eval()

    vol = z.astype(np.float32)
    pads = [(0, d % 2) for d in vol.shape]
    if any(p[1] for p in pads):
        vol = np.pad(vol, pads, mode="constant")
    acc = np.zeros_like(vol, dtype=np.float64)
    cnt = np.zeros_like(vol, dtype=np.float64)
    starts = []
    for dim, p in zip(vol.shape, PATCH):
        st_ = max(1, int(p * (1 - OVERLAP)))
        if dim <= p:
            starts.append([0])
            continue
        v = list(range(0, dim - p + 1, st_))
        if v[-1] != dim - p:
            v.append(dim - p)
        starts.append(v)
    with torch.no_grad():
        for z0 in starts[0]:
            for y0 in starts[1]:
                for x0 in starts[2]:
                    p = vol[z0:z0 + PATCH[0], y0:y0 + PATCH[1], x0:x0 + PATCH[2]]
                    x = torch.from_numpy(p)[None, None].to(dev)
                    y = model(x)[0, 0].detach().cpu().numpy()
                    acc[z0:z0 + PATCH[0], y0:y0 + PATCH[1], x0:x0 + PATCH[2]] += y
                    cnt[z0:z0 + PATCH[0], y0:y0 + PATCH[1], x0:x0 + PATCH[2]] += 1
    out = (acc / np.maximum(cnt, 1)).astype(np.float32)
    return out[:z.shape[0], :z.shape[1], :z.shape[2]]


def to_pd(domain: str, y: np.ndarray, s: float) -> np.ndarray:
    m, sd = float(NORM[domain]["mean"]), float(NORM[domain]["std"])
    return np.maximum(inv(domain, y * sd + m, s), 0.0)


# --------------------------------------------------------------- 参照臂 ----
def references(seq: str, direction: str) -> dict:
    """input / gauss(σ) / wiener3d，全部与学习臂同口径、同支撑集。"""
    th = direction.split("2")[0]
    eh = OPP[th]
    pd_tr, valid, s = load_half(seq, th)
    pd_ev, _, _ = load_half(seq, eh)
    sup = valid
    lg = fwd("log1p", pd_tr, s, valid)

    rec = {"input": {"x": x_rmse(pd_tr, pd_ev, sup, s), "x_in": x_rmse(pd_tr, pd_ev, sup, s)}}

    g = {}
    for sig in SIGMAS:
        pd_g = np.maximum(np.expm1(masked_gauss(lg, valid, sig)), 0.0) * s
        a_g, x_g = fit_gain(pd_tr, pd_g, pd_ev, sup, s)
        g[str(sig)] = {"x": x_rmse(pd_g, pd_ev, sup, s), "x_gain": x_g,
                       "alpha": min(max(a_g, 0.0), 5.0)}
    rec["gauss"] = {"grid": g, "best_by_x": min(g, key=lambda k: g[k]["x"]),
                    "best_by_x_gain": min(g, key=lambda k: g[k]["x_gain"])}

    pd_w = np.maximum(np.expm1(scipy_wiener(lg, mysize=3)), 0.0) * s
    a_w, x_w = fit_gain(pd_tr, pd_w, pd_ev, sup, s)
    rec["wiener3d"] = {"x": x_rmse(pd_w, pd_ev, sup, s), "x_gain": x_w,
                       "alpha": min(max(a_w, 0.0), 5.0)}

    # SN2N 臂：跑在同一半卷上的 200-epoch 模型，输出是 dB 体
    p_sn = RUNS / f"sn2n_{seq}_{th}"
    if (p_sn / "pd_sn2n_linear.npy").exists():
        pd_s = np.load(p_sn / "pd_sn2n_linear.npy").astype(np.float64)
        a_s, x_s = fit_gain(pd_tr, pd_s, pd_ev, sup, s)
        rec["sn2n"] = {"x": x_rmse(pd_s, pd_ev, sup, s), "x_gain": x_s,
                       "alpha": min(max(a_s, 0.0), 5.0)}
    return rec


# ------------------------------------------------------------------ 评分 ----
def eval_one(run: Path, sel: dict | None = None) -> dict | None:
    mm = NAME_RE.match(run.name)
    if not mm or not (run / "DONE").exists():
        return None
    seq, arm, dom, ep, th, eh = mm.groups()
    assert OPP[th] == eh
    z = np.load(B1 / dom / f"{seq}_{th}_x.npy")
    pd_tr, valid, s = load_half(seq, th)
    pd_ev, _, _ = load_half(seq, eh)
    y = dense_infer(run / "latest.pt", z)
    pd_hat = to_pd(dom, y, s)
    sup = valid

    alpha_fit, x_gain_fit = fit_gain(pd_tr, pd_hat, pd_ev, sup, s)
    # α 只允许在方向 1 拟合；方向 2 沿用 selections.json 里方向 1 存下的值，不重拟合。
    if th == "odd":
        alpha_used, alpha_src = alpha_fit, "fitted_on_direction1"
        x_gain = x_gain_fit
    else:
        s_arm = (sel or {}).get(seq, {}).get("arms", {}).get(arm)
        alpha_used = None if s_arm is None else s_arm["alpha"]
        alpha_src = "from_selections" if alpha_used is not None else "unavailable"
        x_gain = (None if alpha_used is None
                  else x_rmse(apply_gain(pd_tr, pd_hat, s, alpha_used), pd_ev, sup, s))
    return {
        "seq": seq, "arm": arm, "domain": dom, "epochs": int(ep), "direction": f"{th}2{eh}",
        "x": x_rmse(pd_hat, pd_ev, sup, s),
        "x_in": x_rmse(pd_tr, pd_ev, sup, s),
        "alpha_fit": alpha_fit, "x_gain_fit": x_gain_fit, "alpha_used": alpha_used,
        "alpha_src": alpha_src,
        "x_gain": x_gain,
        "overfit": overfit_block(pd_tr, pd_hat, pd_ev, sup, s),
        "contrast": contrast_block(pd_hat, pd_tr, valid, s),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["grid", "select", "report"], default="grid")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(OUT / "scores.json"))
    ap.add_argument("--sel", default=str(OUT / "selections.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.what == "grid":
        sel_path = OUT / "selections.json"
        sel = json.loads(sel_path.read_text()) if sel_path.exists() else None
        scores = {"runs": {}, "references": {}}
        for seq in SEQ:
            for d in ("odd2even", "even2odd"):
                scores["references"][f"{seq}_{d}"] = references(seq, d)
        for run in sorted(Path(args.runs).iterdir()):
            if not run.is_dir():
                continue
            r = eval_one(run, sel)
            if r:
                scores["runs"][run.name] = r
                print(f"  {run.name}: X={r['x']:.5f} X/X_in={r['x']/r['x_in']:.4f} "
                      f"α*={r['alpha_fit']:+.4f}")
        Path(args.out).write_text(json.dumps(scores, indent=1, ensure_ascii=False))
        print(f"wrote {args.out}（{len(scores['runs'])} 个 run）")
        return 0

    scores = json.loads(Path(args.out).read_text())

    if args.what == "select":
        sel = {}
        for seq in SEQ:
            ref = scores["references"][f"{seq}_odd2even"]
            sel[seq] = {"gauss": {"best_sigma": float(ref["gauss"]["best_by_x"]),
                                  "x": ref["gauss"]["grid"][ref["gauss"]["best_by_x"]]["x"]},
                        "arms": {}}
            for arm in ("A", "B"):
                cands = {k: v for k, v in scores["runs"].items()
                         if v["seq"] == seq and v["arm"] == arm and v["direction"] == "odd2even"}
                if not cands:
                    continue
                best = min(cands, key=lambda k: cands[k]["x"])
                b = cands[best]
                sel[seq]["arms"][arm] = {
                    "run": best, "domain": b["domain"], "epochs": b["epochs"],
                    "x": b["x"], "alpha": b["alpha_fit"],
                    "contrast": b["contrast"], "overfit": b["overfit"]}
                print(f"[{seq}/{arm}] 选中 {b['domain']} e{b['epochs']}  X={b['x']:.5f}")
        Path(args.sel).write_text(json.dumps(sel, indent=1, ensure_ascii=False))
        print(f"wrote {args.sel}")
        return 0

    # report
    for seq in SEQ:
        for d in ("odd2even", "even2odd"):
            r = scores["references"][f"{seq}_{d}"]
            g = r["gauss"]["grid"][r["gauss"]["best_by_x"]]
            sn = r.get("sn2n", {})
            print(f"[{seq} {d}] input X={r['input']['x']:.5f}  "
                  f"gauss(σ={r['gauss']['best_by_x']}) X={g['x']:.5f}  "
                  f"wiener X={r['wiener3d']['x']:.5f}  "
                  f"sn2n X={sn.get('x', float('nan')):.5f}")
        for arm in ("A", "B"):
            for d in ("odd2even", "even2odd"):
                rows = {k: v for k, v in scores["runs"].items()
                        if v["seq"] == seq and v["arm"] == arm and v["direction"] == d}
                for k, v in sorted(rows.items(), key=lambda kv: kv[1]["x"])[:3]:
                    print(f"    {arm} {d} {v['domain']:6s} e{v['epochs']:<3d} "
                          f"X={v['x']:.5f} ratio={v['x']/v['x_in']:.4f} "
                          f"slope={v['contrast']['slope']:.3f} core={v['contrast']['core_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

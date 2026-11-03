"""Round4-B0：真实数据三种表示域的条件统计（**不训练**）。

回答 Round4 任务书 §2 B0：
  `linear PD` / `log1p(PD/s)` / `log(PD+ε)` 三域，在**同一个**信号协变量分箱下，
  条件方差与条件残差均值（bias）是否不同？空间相关性是否改变？并附独立乘性噪声正控制。

**为什么不能直接引用 R0-v2 的结论（改动原因）**
R0-v2 的方差回归检验的是 `log(PD)`，斜率 0.033/0.104（跨箱比 1.049/1.115）；
而当前网络吃的是 `log1p(PD/s)`，且 B1 最终还要反变换回**线性 PD** 评价。
`log1p(PD/s)` 在 PD ≫ s 时接近 `log PD`、在 PD ≪ s 时接近 `PD/s`（线性），
两者不是同一件事；同方差不等于零条件均值，也不等于噪声独立。所以这里**重新测三域本身**，
不把 R0-v2 的数字平移过来。

**三个层级，必须分开写（任务书 §2 B0 要求）**
  1. 已测事实：本脚本落盘的方差/偏置/相关数值，可直接复核；
  2. 代理假设：用"固定基下的块间方差"当噪声方差代理、"块差分的空间 ACF"当噪声空间相关代理
     —— 二者都只消除**共享的静态/血管信号**，残余共用基、时间漂移、log 非线性仍在；
  3. 训练有效性：本脚本**不涉及**。"log 域方差更平" ≠ "网络在 log 域学得更好"。

**冻结的尺度（不允许各域各自自适应）**
  s   = median(pd_full[valid])            —— 每条链一个常数，取自线性 PD 域
  ε   = 1e-3 · s                          —— 固定比例，不做逐图归一
  协变量 x = log(pd_full)（线性 PD 的对数，三域共用同一组分箱边界）

用法
----
    python -u scripts/v3_round4_b0_logstats.py --dataset SA64
    python -u scripts/v3_round4_b0_logstats.py --dataset HSA64
产物
----
    reports/v3_round4/b0_logstats.json
    reports/v3_round4/fig_b0_logstats_{dataset}.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_svd import load_array, normalize_shape          # noqa: E402
from v3_phase5_real_compare import band_slice, svd_basis      # noqa: E402

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
N_BLOCKS = 4
EPS_FRAC = 1e-3
BINS = 8


def block_pd(coeff, Q, idx, shape3d):
    """固定基下的块内平均功率（与 R0 相同口径：投影算子与分块无关）。"""
    P = coeff @ Q[idx].conj().T
    return np.mean(np.abs(P) ** 2, axis=-1).reshape(shape3d).astype(np.float32)


def acf_axis(x, mask, axis, lags=6):
    out = []
    for lag in range(1, lags + 1):
        a = np.take(x, range(0, x.shape[axis] - lag), axis=axis)
        b = np.take(x, range(lag, x.shape[axis]), axis=axis)
        m = np.take(mask, range(0, mask.shape[axis] - lag), axis=axis) & \
            np.take(mask, range(lag, mask.shape[axis]), axis=axis)
        out.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 50 else float("nan"))
    return out


def cond_stats(values, x, mask, bins=BINS):
    """按共用协变量 x 分箱：Var(noise|bin) 的中位数 与 (mean_k y_k − T(整体)) 的中位数。

    `values`：(K, ...) 各块的域值；噪声 = 块间离散度；偏置 = 变换的 Jensen 间隙。
    """
    xv = x[mask]
    if xv.size < bins * 20:
        return {"bins": [], "slope": float("nan"), "n": int(xv.size)}
    qs = np.quantile(xv, np.linspace(0, 1, bins + 1))
    idx = np.digitize(xv, qs[1:-1])
    vm = values.var(axis=0, ddof=1)[mask]
    rows = []
    for i in range(bins):
        sel = idx == i
        if sel.sum() < 20:
            rows.append({"x_med": float(np.exp(np.median(xv[sel]))) if sel.any() else float("nan"),
                         "var_med": float("nan"), "gap_med": float("nan"), "n": int(sel.sum())})
            continue
        rows.append({"x_med": float(np.exp(np.median(xv[sel]))),
                     "var_med": float(np.median(vm[sel])),
                     "n": int(sel.sum())})
    xs = [r["x_med"] for r in rows]
    ys = [max(r["var_med"], 1e-30) for r in rows]
    xs = [np.log(max(v, 1e-30)) for v in xs]
    ys = [np.log(v) for v in ys]
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) == bins else float("nan")
    return {"bins": rows, "slope": slope, "n": int(xv.size),
            "slope_meaning": "x 已是对数刻度：linear 域 ≈2 表示噪声方差 ∝ 信号²；"
                             "≈0 表示方差与信号无关（对数域已稳定化）"}


def run(dataset: str, do_figure: bool) -> dict:
    t0 = time.time()
    rec: dict = {"dataset": dataset,
                 "frozen": {"n_blocks": N_BLOCKS, "eps_frac": EPS_FRAC, "bins": BINS,
                            "eps": "ε = 1e-3·s，s = median(pd_full[valid])",
                            "covariate": "x = log(pd_full)，三域共用同一组分位分箱"}}

    x4 = normalize_shape(load_array(RAW[dataset], None), 0, 0)
    z, a, b, T = x4.shape
    C = x4.reshape(-1, T).astype(np.complex64)
    del x4
    evals, V = svd_basis(C)
    lo, hi = band_slice(T)
    Q = V[:, lo:hi]
    coeff = C @ Q
    del C
    shape3d = (z, a, b)

    edges = np.linspace(0, T, N_BLOCKS + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(N_BLOCKS)]
    PD = np.stack([block_pd(coeff, Q, idx, shape3d) for idx in blocks])
    odd = block_pd(coeff, Q, np.arange(0, T, 2), shape3d)
    even = block_pd(coeff, Q, np.arange(1, T, 2), shape3d)
    pd_full = PD.mean(axis=0).astype(np.float64)
    valid = pd_full > 0

    s = float(np.median(pd_full[valid]))
    eps = EPS_FRAC * s
    q_bg = float(np.quantile(pd_full[valid], 0.20))
    bg = valid & (pd_full <= q_bg)
    rec["asset"] = {"shape_zyxt": [int(v) for v in (z, a, b, T)],
                    "band_0based": [int(lo), int(hi)], "band_modes": int(hi - lo),
                    "s_median_pd": s, "eps": eps, "q20_bg": q_bg,
                    "n_valid": int(valid.sum()), "n_bg": int(bg.sum())}

    DOMAINS = {
        "linear": lambda p: p,
        "log1p": lambda p: np.log1p(p / s),
        "logeps": lambda p: np.log(p + eps),
    }
    xcov = np.log(np.maximum(pd_full, 1e-30))

    # 实测块间相对波动（用来把正控制的噪声幅度对齐到"与真实同量级"）
    var_k = PD.var(axis=0, ddof=1)
    bg_rel_sd = float(np.median(np.sqrt(var_k[bg]) / np.maximum(pd_full[bg], 1e-30)))

    out: dict = {"domains": {}, "control_multiplicative": {}}
    for name, T_ in DOMAINS.items():
        vals = T_(PD.astype(np.float64))
        st = cond_stats(vals, xcov, bg)
        # 偏置：分箱内的 Jensen 间隙 mean_k T(PD_k) − T(mean_k PD_k)
        gap = T_(pd_full) - vals.mean(axis=0)
        xv = xcov[bg]
        qs = np.quantile(xv, np.linspace(0, 1, BINS + 1))
        for i, row in enumerate(st["bins"]):
            lo_i = qs[i]
            hi_i = qs[i + 1]
            m = bg & (xcov >= lo_i) & ((xcov < hi_i) | (i == BINS - 1))
            row["gap_med"] = float(np.median(gap[m]))
        out["domains"][name] = {
            **st,
            "gap_abs_median": float(np.median(np.abs(gap[bg]))),
            "var_med": float(np.median(vals.var(axis=0, ddof=1)[bg])),
            "odd_even_corr_bg": float(np.corrcoef(T_(odd)[bg], T_(even)[bg])[0, 1]),
            "noise_acf_blockdiff": {
                ax: acf_axis(T_(PD[0]) - T_(PD[1]), bg, i, 6)
                for i, ax in enumerate(("z", "y", "x"))
            },
        }

    # 独立乘性噪声正控制：PD_k = pd_full · exp(n_k)，n_k ~ N(0, σ²)，σ 对齐实测相对波动
    rng = np.random.default_rng(20261029)
    nz = rng.normal(0.0, bg_rel_sd, (N_BLOCKS,) + shape3d)
    PDc = pd_full[None] * np.exp(nz.astype(np.float64))
    out["control_multiplicative"]["sigma"] = bg_rel_sd
    out["control_multiplicative"]["note"] = ("独立乘性噪声：linear 域方差应 ∝ 信号²（斜率≈2），"
                                             "对数域应近似与信号无关（斜率≈0）。"
                                             "这是检验统计量本身的判别力，不是对真实数据的结论")
    for name, T_ in DOMAINS.items():
        vals = T_(PDc)
        st = cond_stats(vals, xcov, bg)
        out["control_multiplicative"][name] = {
            "slope": st["slope"], "bins": st["bins"],
            "var_med": float(np.median(vals.var(axis=0, ddof=1)[bg]))}
    del PDc, nz

    rec["bg_rel_sd"] = bg_rel_sd
    rec.update(out)
    rec["layers"] = {
        "measured_fact": "domains.* 与 control_multiplicative.* 的数值（可复算）",
        "proxy_assumption": "块间方差当噪声方差、块差分 ACF 当噪声空间相关；"
                            "只消除共享静态/血管信号，残余共用基与时间漂移未消除",
        "training_effectiveness": "NOT_MEASURED_IN_B0 —— 本脚本不训练，不能推出任何训练结论",
    }
    rec["runtime_s"] = round(time.time() - t0, 1)

    if do_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        outdir = ROOT / "reports/v3_round4"
        outdir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, key, title in ((axes[0], "domains", "real block variance"),
                               (axes[1], "control_multiplicative", "multiplicative-noise control")):
            for name, lab in (("linear", "linear PD"), ("log1p", "log1p(PD/s)"),
                              ("logeps", "log(PD+ε)")):
                d = out[key][name]
                rows = d.get("bins") if "bins" in d else None
                if rows:
                    xs = [r["x_med"] for r in rows]
                    ys = [max(r["var_med"], 1e-30) for r in rows]
                    ax.loglog(xs, ys, marker="o", ms=3,
                              label=f"{lab} (slope {d['slope']:.2f})")
            ax.set_xlabel("linear PD (bin median)")
            ax.set_ylabel("block variance")
            ax.set_title(f"{dataset}: {title}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = outdir / f"fig_b0_logstats_{dataset}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        rec["figure"] = str(p.relative_to(ROOT))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    rec = run(args.dataset, not args.no_figure)
    outdir = ROOT / "reports/v3_round4"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "b0_logstats.json"
    allrec = {}
    if p.exists():
        allrec = json.loads(p.read_text())
    allrec[args.dataset] = rec
    p.write_text(json.dumps(allrec, indent=1, ensure_ascii=False))
    print(f"wrote {p}")
    brief = {k: {d: {"slope": v[d]["slope"], "gap_abs_median": v[d]["gap_abs_median"],
                     "odd_even_corr": round(v[d]["odd_even_corr_bg"], 4)}
                 for d in ("linear", "log1p", "logeps")}
             for k, v in (("real", rec["domains"]),)}
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    print("control slopes:", {k: round(rec["control_multiplicative"][k]["slope"], 3)
                              for k in ("linear", "log1p", "logeps")})


if __name__ == "__main__":
    main()

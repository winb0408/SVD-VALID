"""V3 Phase 5 — 真实数据（SA64 / HSA64）上的「去噪前 / 单纯 SVD 滤波 / +① Wiener」对比图。

**为什么有这一步（改动原因）**：`reports/figures/v3_phase4_method_comparison.png` 是
**模拟数据上几个估计器之间**的比数，读者看不出"去噪"这件事本身（背景有没有变干净、
血管有没有被磨掉）。用户 2026-10-22 要求换成真实采集，并且必须和**单纯 SVD 滤波**
这个临床常用基线并列。所以本脚本不用模拟器，直接读 Dryad 原始 .mat。

三栏的差别**只有一个**：拿到同一批原始帧后用什么后端处理。
  1. `raw`    全秩（秩 0–T，不做任何滤波）            —— 去噪前
  2. `svd`    秩带 [0.15·T, 0.875·T)                 —— 单纯 SVD 滤波（项目标准带 60–350/400）
  3. `svd+w`  在 `svd` 的结果上再套 ① Wiener          —— 本项目方法

**定量尺子，以及为什么只能在 SVD 卷上量。** 把 400 帧拆成奇/偶两组（时间不重叠），
互为"第二张观测"，可算 n2n = MSE(pred, obs2)/MSE(输入, obs2)。但实测发现：
**全秩卷的奇/偶两半几乎完全相同**（静态软组织杂波在两组里是同一份，逐帧相关 ≈1），
分母 MSE(全秩奇, 全秩偶) 趋近 0，任何处理的比值都会炸到 10³–10⁴ 量级。
这不是算法失败，是**尺子不适用**——所以 n2n 只在 **SVD 之后**的卷上算（杂波已被
带通除掉，两半之间剩下的主要是独立噪声）。这条限制本身是真实数据的一条结论，写进
`degenerate_raw_chain` 字段存档。

**真实数据的 n2n 不与模拟器可比**：奇偶拆分把"血流逐帧去相关"也算进了噪声，天花板
不是 0.61 而是另一个未知数，所以只做**相对比较**，不给通过/不通过判定。

另外两条轴（都是尺度无关的，可跨链路比）：
  - 显影度 `p99.5/p50`：过平滑会把它压下去 → 用来检查"去噪有没有把血管磨掉"；
  - 奇/偶输出相关 `corr`：去噪越强越高，但配合显影度一起看才有意义。

预处理与模拟器一致：`log1p(PD / s_chain)`，**每条链路用自己的 s**（取该链路奇/偶
平均卷在有效体素上的中位数），但奇、偶两卷共用同一个 s。

用法：
    python -u scripts/v3_phase5_real_compare.py --dataset SA64
    python -u scripts/v3_phase5_real_compare.py --dataset HSA64
输出：
    reports/figures/v3_real_denoise_{dataset}.png
    reports/v3_phase5_real/{dataset}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_svd import load_array, normalize_shape  # noqa: E402

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
# 项目标准血带（见 data/processed/sa64_blood_remap/metadata.json）。用**比例**表示，
# 奇/偶半（T=200）才能落在同一物理位置上：60/400 = 0.15，350/400 = 0.875。
FRAC_LO, FRAC_HI = 0.15, 0.875

# R13 更正（2026-10-26，外部审查）：**比例直接取整会把官方带算少一个模式。**
# 官方 MATLAB 的 `M(:, 60:350) * U(:, 60:350)'` 是 **1-based 闭区间**，含 291 个模式；
# 对应的 0-based 半开区间是 `[59, 350)`，不是 `[60, 350)`。旧代码用
# `round(0.15*T), round(0.875*T)` 得到 `[60,350)`（290 个模式）= MATLAB 61:350，
# 整体右移一位。`scripts/official_pipeline.py::svd_blood_band(faithful=True)` 用的是
# 正确口径 `(cut1-1, cut2)`。
# 本常量默认 True → 采用官方口径；设为 False 可复现 2026-10-26 之前的旧数字。
OFFICIAL_BAND = True


def band_slice(T: int, frac_lo: float = FRAC_LO, frac_hi: float = FRAC_HI,
               official: bool | None = None):
    """把"官方 1-based 闭区间 [frac_lo*T, frac_hi*T]"换成 0-based 半开切片。

    official=True  → (round(frac_lo*T) - 1, round(frac_hi*T))，T=400 时为 (59, 350)，291 个模式
    official=False → (round(frac_lo*T),     round(frac_hi*T))，T=400 时为 (60, 350)，290 个模式
    T=200 的半卷同理：官方口径 (29, 175)，146 个模式。
    """
    if official is None:
        official = OFFICIAL_BAND
    lo = int(round(frac_lo * T))
    hi = int(round(frac_hi * T))
    return (lo - 1, hi) if official else (lo, hi)


# ---------------------------------------------------------------- SVD 功率多普勒
def svd_basis(C):
    """经济 SVD：只算 T×T 时间 Gram，不碰 15 万×400 的空间基。"""
    G = (C.conj().T @ C).astype(np.complex64)
    evals, V = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    return evals[order], V[:, order]


def band_pd(C, V, lo, hi, shape3d):
    """投影到秩带 [lo,hi) 后取平均功率 → PD 体积（线性功率，未做对数压缩）。"""
    lo, hi = int(max(0, lo)), int(min(V.shape[1], hi))
    Q = V[:, lo:hi]
    P = (C @ Q) @ Q.conj().T
    return np.mean(np.abs(P) ** 2, axis=-1).reshape(shape3d).astype(np.float32)


# ---------------------------------------------------------------- 估计器
def wiener(x, w, valid):
    """训练无关的局部方差比收缩：b = clip(1 − σ²/lv, 0, 1)，out = mu + b(x − mu)。"""
    mu = uniform_filter(x, w)
    lv = np.maximum(uniform_filter(x * x, w) - mu * mu, 0.0)
    sig2 = float(np.median(lv[valid]))
    b = np.clip(1.0 - sig2 / np.maximum(lv, 1e-9), 0.0, 1.0)
    return (mu + b * (x - mu)).astype(np.float32), b.astype(np.float32), sig2


def mse(a, b, mask):
    return float(np.mean((a[mask] - b[mask]) ** 2))


# ---------------------------------------------------------------- 结构掩膜
# 定义全部沿用 `scripts/eval_structure.py`（L22 定的唯一口径），不另发明：
#   core  参考值 >= 0.65；body 0.20–0.65；dark <= 0.05；bright_top 前 1%；
#   isolated_blob = 参考值 > 0.5 的连通块中体素数 <= BLOB_MAX_VOX 的那些。
CORE_THR, BODY_LO, DARK_THR, BLOB_THR, BLOB_MAX_VOX = 0.65, 0.20, 0.05, 0.5, 27


def to_unit(ref, valid):
    """把共享参考卷映射到 [0,1]（单调分位映射），用于定义掩膜。"""
    q0, q1 = np.percentile(ref[valid], [1.0, 99.9])
    return np.clip((ref - q0) / max(q1 - q0, 1e-12), 0.0, 1.0)


def structure_masks(unit):
    from scipy.ndimage import label

    core = unit >= CORE_THR
    body = (unit >= BODY_LO) & (unit < CORE_THR)
    dark = unit <= DARK_THR
    lab, n = label(unit > BLOB_THR)
    if n:
        sizes = np.bincount(lab.ravel())
        small = np.where(sizes <= BLOB_MAX_VOX)[0]
        small = small[small > 0]
        isolated = np.isin(lab, small)
    else:
        isolated = np.zeros_like(core)
    return {"core": core, "body": body, "dark": dark, "isolated": isolated}


def keep_ratio(out, inp, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(out[mask].mean() / max(inp[mask].mean(), 1e-12))


def setup_cjk():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(cjk)
    plt.rcParams["font.family"] = [fm.FontProperties(fname=cjk).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--w", type=int, default=7, help="① 的窗口（模拟器最优为 7）")
    ap.add_argument("--out-dir", default="reports/v3_phase5_real")
    args = ap.parse_args()

    ds = args.dataset
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    z, x, y, t = raw.shape
    shape3d = (z, x, y)
    print(f"[{ds}] raw {raw.shape} {raw.dtype}")
    C = raw.reshape(-1, t)
    evals, V = svd_basis(C)
    del raw

    lo_b, hi_b = band_slice(t)
    print(f"[{ds}] 全秩=[0,{t})  血带=[{lo_b},{hi_b})  半卷血带比例同")

    pd_full = band_pd(C, V, 0, t, shape3d)        # 去噪前
    pd_svd = band_pd(C, V, lo_b, hi_b, shape3d)   # 单纯 SVD
    del C, V

    # ---- 奇/偶两卷（带通按同一**比例**换算到 T/2）----
    raw4 = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    halves = {}
    for tag, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        Ch = raw4.reshape(-1, t)[:, sl]
        th = Ch.shape[1]
        _, Vh = svd_basis(Ch)
        halves[tag] = {
            "full": band_pd(Ch, Vh, 0, th, shape3d),
            "svd": band_pd(Ch, Vh, *band_slice(th), shape3d),
        }
        del Ch, Vh
    del raw4
    thalf = t // 2
    th_lo, th_hi = band_slice(thalf)
    print(f"[{ds}] 半卷 T={thalf}  血带=[{th_lo},{th_hi})")

    # ---- 有效体素 ----
    zz, yy, xx = np.mgrid[0:z, 0:x, 0:y]
    valid = ((zz >= 2) & (zz <= z - 3) & (yy >= 4) & (yy <= x - 5)
             & (xx >= 4) & (xx <= y - 5))
    o, e = halves["odd"], halves["even"]
    valid_svd = valid & (0.5 * (o["svd"] + e["svd"]) > 0)
    valid_full = valid & (0.5 * (o["full"] + e["full"]) > 0)
    print(f"[{ds}] 有效体素：全秩 {int(valid_full.sum())}  SVD {int(valid_svd.sum())}")

    # 每条链路各自的归一化常数（奇偶共用）
    def prep_pair(key):
        s = float(np.median(0.5 * (o[key] + e[key])[valid_svd if key == "svd" else valid_full]))
        f = lambda pd: np.log1p(np.maximum(pd, 0.0) / s).astype(np.float32)
        return s, f(o[key]), f(e[key])

    s_f, xo_f, xe_f = prep_pair("full")
    s_s, xo_s, xe_s = prep_pair("svd")
    print(f"[{ds}] 归一化常数 全秩 s={s_f:.3e}  SVD s={s_s:.3e}")

    # ---- 全帧卷（显示用）：与半卷同一 s ----
    x_full = np.log1p(np.maximum(pd_full, 0.0) / s_f).astype(np.float32)
    x_svd = np.log1p(np.maximum(pd_svd, 0.0) / s_s).astype(np.float32)
    w_full, b_full, sig2_full = wiener(x_full, args.w, valid_full)
    w_svd, b_svd, sig2_svd = wiener(x_svd, args.w, valid_svd)

    # ---- SVD 链内的定量（唯一适用的一把尺子）----
    ref_svd = mse(xo_s, xe_s, valid_svd)
    mse_raw_pair = mse(xo_f, xe_f, valid_full)

    # 结构掩膜建在**奇偶共享**的参考卷上（= 两卷均值），因此与"两卷之差"无关，不泄漏。
    ref_unit = to_unit(0.5 * (xo_s + xe_s), valid_svd)
    masks = structure_masks(ref_unit)
    print(f"[{ds}] 结构掩膜：core {int(masks['core'].sum())}  body {int(masks['body'].sum())}"
          f"  dark {int(masks['dark'].sum())}  isolated {int(masks['isolated'].sum())}")

    estimators = {
        "不处理（SVD 输出）": (xo_s, xe_s),
        "高斯平滑 σ=1": (gaussian_filter(xo_s, 1.0), gaussian_filter(xe_s, 1.0)),
        "中值滤波 3³": (median_filter(xo_s, 3), median_filter(xe_s, 3)),
        f"①Wiener w={args.w}": (wiener(xo_s, args.w, valid_svd)[0],
                                wiener(xe_s, args.w, valid_svd)[0]),
    }
    rows = {k: mse(p, q, valid_svd) / ref_svd for k, (p, q) in estimators.items()}
    corr = {k: float(np.corrcoef(p[valid_svd], q[valid_svd])[0, 1])
            for k, (p, q) in estimators.items()}
    # 结构保真：只在**奇帧输入和它的输出**之间比（同一单位，无需尺度对齐）
    keep = {}
    for k, (p, _) in estimators.items():
        keep[k] = {
            "core": keep_ratio(p, xo_s, masks["core"]),
            "body": keep_ratio(p, xo_s, masks["body"]),
            "dark": keep_ratio(p, xo_s, masks["dark"]),
            "isolated_blob": keep_ratio(p, xo_s, masks["isolated"]),
        }

    # 窗口敏感性（不算新方法，只为证明 ① 的数不是挑出来的）
    w_sens = {}
    for w in (5, 7, 9, 11):
        pw, _, _ = wiener(xo_s, w, valid_svd)
        qw, _, _ = wiener(xe_s, w, valid_svd)
        w_sens[w] = {"n2n": mse(pw, qw, valid_svd) / ref_svd,
                     "core_keep": keep_ratio(pw, xo_s, masks["core"])}

    print(f"[{ds}] 全秩奇偶 MSE = {mse_raw_pair:.4e}   SVD 奇偶 MSE = {ref_svd:.4e}"
          f"  比值 = {ref_svd / max(mse_raw_pair, 1e-30):.1f}×（→ raw 链不可用）")
    print(f"[{ds}] SVD 链内 n2n（奇→偶）与结构保真：")
    for k, v in rows.items():
        kk = keep[k]
        print(f"    {k:20s} n2n={v:.4f}  corr={corr[k]:.4f}  core保留={kk['core']:.3f}"
              f"  dark底={kk['dark']:.4f}  孤立斑点保留={kk['isolated_blob']:.3f}")
    print(f"[{ds}] ① 窗口敏感性: "
          + "  ".join(f"w={w}: n2n {v['n2n']:.3f}/core {v['core_keep']:.3f}"
                      for w, v in w_sens.items()))

    (out_dir / f"{ds}.json").write_text(json.dumps({
        "dataset": ds, "shape": [int(v) for v in shape3d], "T": int(t),
        "band_full": [0, int(t)], "band_blood": [int(lo_b), int(hi_b)],
        "band_blood_half": [th_lo, th_hi], "w": args.w,
        "norm_const": {"full": s_f, "svd": s_s},
        "sigma2": {"full": sig2_full, "svd": sig2_svd},
        "n2n_svd_chain": rows,
        "corr_odd_even": corr,
        "structure_keep_odd": keep,
        "mask_voxels": {k: int(v.sum()) for k, v in masks.items()},
        "wiener_window_sensitivity": w_sens,
        "degenerate_raw_chain": {
            "mse_full_pair": mse_raw_pair, "mse_svd_pair": ref_svd,
            "ratio": ref_svd / max(mse_raw_pair, 1e-30),
            "why": "全秩卷的奇/偶两半共享同一份静态杂波，分母趋 0，n2n 在 raw 链上无意义。",
        },
        "note": "真实数据 n2n 不做通过/不通过判定；奇偶拆分的天花板尚未校准。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 图
    # 改动原因（2026-10-22 第二次）：第一版把**单个 z 切片**按 "99.9 分位 − 40 dB" 显示，
    # 背景体素的对数值远低于下限 → 整幅几乎全黑，用户看不出"去噪前后"的差别。改为血流
    # 功率多普勒的常规呈现：沿 z 做最大值投影（MIP），显示窗取三栏合并的稳健分位，
    # 这样"杂波背景 / 散斑 / 血管"三层信息同时在图里。
    plt = setup_cjk()
    to_db = lambda v: 10.0 * np.log10(np.maximum(v, 1e-12))
    db_raw, db_svd, db_w = to_db(pd_full), to_db(pd_svd), to_db(s_s * np.expm1(w_svd))
    mip = [np.max(pd_full, axis=0), np.max(pd_svd, axis=0),
           np.max(s_s * np.expm1(w_svd), axis=0)]
    db_mip = [to_db(m) for m in mip]
    # 上限取 ②③ 两栏 99.9 分位的较大者（杂波栏 ① 会把上限抬到没意义的高度，
    # 所以它按同一窗口裁顶显示——"饱和成白色"这件事本身就说明杂波占主导）。
    m2d = valid_svd.any(axis=0)          # MIP 是 2D，掩膜也压到 2D（z=0 切片在内边距外，不能用）
    vhi = max(float(np.percentile(m[m2d], 99.9)) for m in db_mip[1:])
    vlo = max(min(float(np.percentile(m[m2d], 1.0)) for m in db_mip), vhi - 45.0)
    print(f"[{ds}] 显示窗 dB [{vlo:.1f}, {vhi:.1f}]（三栏共用，MIP={x}×{y}）")

    titles = ["① 去噪前（全秩 0–T）",
              f"② 单纯 SVD 滤波（秩 {lo_b}–{hi_b}）",
              f"③ SVD + 本项目 ①Wiener（w={args.w}）"]
    subs = ["含低秩杂波 + 热噪声，未做任何滤波",
            "杂波已除；散斑仍在，未再处理",
            f"n2n {rows[f'①Wiener w={args.w}']:.3f} / 血管核保留 "
            f"{keep[f'①Wiener w={args.w}']['core']:.2f}"]

    # 局部放大：在合并 MIP 里找血管最亮处，24×24 窗口，三栏同一位置
    iy, ix = np.unravel_index(np.argmax(db_mip[1]), db_mip[1].shape)
    r = 12
    y0, y1 = int(np.clip(iy - r, 0, x)), int(np.clip(iy + r, 24, x))
    x0, x1 = int(np.clip(ix - r, 0, y)), int(np.clip(ix + r, 24, y))

    fig = plt.figure(figsize=(16.5, 12.6))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.15, 1.15, 0.95, 1.0],
                          hspace=0.34, wspace=0.10)
    h = None
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        h = ax.imshow(db_mip[j], cmap="inferno", vmin=vlo, vmax=vhi, origin="lower")
        ax.set_title(titles[j], fontsize=13)
        ax.set_xlabel(subs[j], fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(h, ax=[fig.axes[k] for k in range(3)], fraction=0.022, pad=0.012)
    cb.set_label("血流功率 dB（三栏同一色标，沿 z 最大值投影）", fontsize=10)

    for j in range(3):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(db_mip[j][y0:y1, x0:x1], cmap="inferno", vmin=vlo, vmax=vhi,
                  origin="lower", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.635, f"同一窗口放大 24×24（y[{y0}:{y1}] x[{x0}:{x1}]）：看散斑是否变干净、"
             f"血管边缘是否变糊", ha="center", fontsize=10.5)

    zc = z // 2
    d1, d2, bm = db_svd[zc] - db_raw[zc], db_w[zc] - db_svd[zc], b_svd[zc]
    lim1 = float(np.percentile(np.abs(d1[valid_svd[zc]]), 99.0))
    lim2 = float(np.percentile(np.abs(d2[valid_svd[zc]]), 99.0))
    for j, (im, cm, lim, tt) in enumerate([
            (d1, "RdBu_r", lim1, "② 减 ①：SVD 滤掉了什么（大面积低秩杂波）"),
            (d2, "RdBu_r", lim2, "③ 减 ②：Wiener 又滤掉了什么（散斑噪声）")]):
        ax = fig.add_subplot(gs[2, j])
        hh = ax.imshow(im, cmap=cm, vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(tt, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(hh, ax=ax, fraction=0.035, pad=0.01).set_label("dB", fontsize=8)
    ax = fig.add_subplot(gs[2, 2])
    hh = ax.imshow(bm, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title("① 的逐点收缩 b（1=完全保留，0=当噪声压掉）", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(hh, ax=ax, fraction=0.035, pad=0.01)

    keys = list(rows)
    cols = ["#9aa0a6", "#b0b7bd", "#8894a0", "#2e8b57"]

    axb = fig.add_subplot(gs[3, 0])
    vals = [rows[k] for k in keys]
    bars = axb.bar(range(len(keys)), vals, color=cols, width=0.6)
    for bb, v in zip(bars, vals):
        axb.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", fontsize=11, fontweight="bold")
    axb.axhline(1.0, color="k", ls="--", lw=1)
    axb.set_ylim(0, 1.15)
    axb.set_xticks(range(len(keys)))
    axb.set_xticklabels([k.replace("（", "\n（") for k in keys], fontsize=9)
    axb.set_ylabel("n2n（越低越好；不处理 = 1.000）", fontsize=10)
    axb.set_title("奇/偶互证：还能再降多少噪", fontsize=11.5)
    axb.grid(axis="y", alpha=0.25)

    axc = fig.add_subplot(gs[3, 1])
    cv = [keep[k]["core"] for k in keys]
    bars = axc.bar(range(len(keys)), cv, color=cols, width=0.6)
    for bb, v in zip(bars, cv):
        axc.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold")
    axc.axhline(1.0, color="k", ls="--", lw=1)
    axc.set_ylim(0, 1.15)
    axc.set_xticks(range(len(keys)))
    axc.set_xticklabels([k.replace("（", "\n（") for k in keys], fontsize=9)
    axc.set_ylabel("血管核亮度保留（1.0=一点没磨）", fontsize=10)
    axc.set_title("过平滑检查", fontsize=11.5)
    axc.grid(axis="y", alpha=0.25)

    axd = fig.add_subplot(gs[3, 2])
    for k, c in zip(keys, cols):
        axd.scatter(rows[k], keep[k]["core"], s=110, color=c, zorder=3,
                    edgecolor="k", linewidth=0.6)
        axd.annotate(k.split("（")[0].strip(), (rows[k], keep[k]["core"]),
                     textcoords="offset points", xytext=(7, -3), fontsize=9)
    axd.axhspan(0.95, 1.12, xmin=0.0, xmax=0.35, color="#2e8b57", alpha=0.10)
    axd.text(0.03, 1.06, "好角落：又干净又保血管", fontsize=9, color="#2e8b57")
    axd.axhline(1.0, color="k", ls="--", lw=1)
    axd.set_xlim(0, 1.08); axd.set_ylim(0.5, 1.12)
    axd.set_xlabel("n2n（越左越干净）", fontsize=10)
    axd.set_ylabel("血管核保留（越上越保真）", fontsize=10)
    axd.set_title("两把尺子一起看才能定胜负", fontsize=11.5)
    axd.grid(alpha=0.25)

    fig.suptitle(f"{ds} 真实采集血流：去噪前 / 单纯 SVD 滤波 / SVD+①Wiener"
                 f"　（{z}×{x}×{y}，T={t}；奇偶各 {thalf} 帧互为答案）",
                 fontsize=15, y=0.975)
    out = fig_dir / f"v3_real_denoise_{ds}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[{ds}] wrote {out}")


if __name__ == "__main__":
    main()

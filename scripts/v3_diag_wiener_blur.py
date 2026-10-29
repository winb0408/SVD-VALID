"""诊断：为什么 n2n 说 Wiener 好、肉眼说 Wiener 差？（**只做归因，不产生新判据**）

## 触发原因（用户 2026-10-23 的读图反馈）

用户看图后指出：SA64 上 ③(SVD+Wiener) 比 ②(SVD) 好，但 **HSA64 上 ③ 明显比 ② 差**，
④ 介于两者之间。这与 `reports/v3_phase5_real/{ds}_network.json` 的 n2n 排序**相反**
（HSA64：②=1.0000、③=0.3830、④≈0.95；n2n 越小越好 → 指标认为 ③ 最好）。

## 本脚本要回答的唯一问题

`③` 的低 n2n 是"真的把噪声去掉了"，还是"把画面抹平了"？n2n 只统计奇/偶两次采集的
**残差不一致度**，一个常数场可以拿到完美分数，因此它**结构上无法惩罚偏置与模糊**。
所以需要另一类与 n2n 正交的量来归因（**注意：下列量不是新判据，不用于改任何结论**）：

1. `b` 系数分布：`b=clip(1-sig2/lv,0,1)`。`b≈0` 处 `out=mu`（纯局部均值=模糊），
   `b≈1` 处原样保留。`b≈0` 的比例越高，这条链越接近"抹平"。
2. `sig2` 与 `lv` 的比值：`sig2` 是 `wiener` 唯一的数据驱动尺度，
   若 `sig2` 高于大部分 `lv`，则 `b` 在全场趋 0 —— w=7 固定但实际强度由数据定。
3. 对比度：`dB(core 中位) − dB(dark 中位)`，在 MIP 上算（用户看的就是 MIP）。
   掉得多 = 血管与背景拉不开 = 用户说的"比 SVD 差"。
4. 背景绝对电平：`dark` 区的 MIP dB 中位，相对 ② 的变化。
5. 背景平坦度：`dark` 区 MIP dB 的空间标准差（越小越"糊成一片"）。

用法：
    python -u scripts/v3_diag_wiener_blur.py --dataset HSA64
    python -u scripts/v3_diag_wiener_blur.py --dataset SA64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from prepare_svd import load_array, normalize_shape                    # noqa: E402
from v3_phase5_real_compare import (                                    # noqa: E402
    RAW, FRAC_LO, FRAC_HI, band_pd, structure_masks, svd_basis, to_unit, wiener, band_slice)

NET_Z = 32


def to_db(v):
    return 10.0 * np.log10(np.maximum(v, 1e-30))


def grad_energy(m, mask2d):
    """MIP 上 mask 内的梯度能量（相邻差平方均值），用于看"有没有被抹平"。"""
    gx = np.zeros_like(m); gy = np.zeros_like(m)
    gx[:, :-1] = np.diff(m, axis=1)
    gy[:-1, :] = np.diff(m, axis=0)
    g = np.sqrt(gx ** 2 + gy ** 2)
    inner = mask2d.copy()
    inner[:, -1] = False
    inner[-1, :] = False
    return float(g[inner].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HSA64", choices=sorted(RAW))
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--w", type=int, default=7)
    ap.add_argument("--out-dir", default="reports/v3_phase5_real")
    args = ap.parse_args()

    ds = args.dataset
    raw = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    z, x, y, t = raw.shape
    shape3d = (z, x, y)
    C = raw.reshape(-1, t)
    _, V = svd_basis(C)
    lo_b, hi_b = band_slice(t)
    pd_svd = band_pd(C, V, lo_b, hi_b, shape3d)
    del C, V, raw

    raw4 = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    halves = {}
    for tag, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        Ch = raw4.reshape(-1, t)[:, sl]
        th = Ch.shape[1]
        _, Vh = svd_basis(Ch)
        halves[tag] = band_pd(Ch, Vh, *band_slice(th), shape3d)
        del Ch, Vh
    del raw4

    crop = (slice(0, NET_Z), slice(0, x), slice(0, y))
    pd_svd = pd_svd[crop]
    for tag in halves:
        halves[tag] = halves[tag][crop]
    zc, xc, yc = pd_svd.shape

    zz, yy, xx = np.mgrid[0:zc, 0:xc, 0:yc]
    valid = ((zz >= 2) & (zz <= zc - 3) & (yy >= 4) & (yy <= xc - 5)
             & (xx >= 4) & (xx <= yc - 5))
    o, e = halves["odd"], halves["even"]
    valid_svd = valid & (0.5 * (o + e) > 0)

    # 与主脚本完全同口径：s = 共享参考卷（奇偶均值）在有效体上的中位
    s = float(np.median(0.5 * (o + e)[valid_svd]))
    xo = np.log1p(np.maximum(o, 0.0) / s).astype(np.float32)
    out3, b, sig2 = wiener(xo, args.w, valid_svd)

    mu = uniform_filter(xo, args.w)
    lv = np.maximum(uniform_filter(xo * xo, args.w) - mu * mu, 0.0)

    bv = b[valid_svd]
    lvv = lv[valid_svd]
    bstat = {
        "sig2": sig2,
        "frac_b_lt_0.05": float((bv < 0.05).mean()),
        "frac_b_gt_0.95": float((bv > 0.95).mean()),
        "b_median": float(np.median(bv)),
        "lv_median": float(np.median(lvv)),
        "sig2_over_lv_median": float(sig2 / max(float(np.median(lvv)), 1e-12)),
        "frac_lv_lt_sig2": float((lvv < sig2).mean()),
    }

    # ---- MIP 上的对比度 / 电平 / 平坦度（用户实际看的就是 MIP）----
    # 2D 掩膜：规则与常数完全沿用 3D 版（CORE_THR/BODY_LO/DARK_THR/BLOB_THR），
    # 但参考卷换成"共享奇偶均值卷的 MIP"，因为用户看的是 MIP，且 MIP 参考同样不含
    # 任何被评链的信息（只用 ② 的奇偶均值），不引入泄漏。
    xe = np.log1p(np.maximum(e, 0.0) / s).astype(np.float32)
    ref_mip = np.max(0.5 * (xo + xe), axis=0)
    masks = structure_masks(to_unit(ref_mip, valid_svd.any(axis=0)))
    mip2 = to_db(np.max(o, axis=0))
    mip3 = to_db(np.max(s * np.expm1(out3), axis=0))
    m2d = valid_svd.any(axis=0)

    def stat(m, mask):
        mm = mask & m2d
        return {"n": int(mm.sum()), "median_db": float(np.median(m[mm]))}

    rows = {}
    for tag, m in (("②SVD", mip2), ("③SVD+Wiener", mip3)):
        dark = stat(m, masks["dark"])
        core = stat(m, masks["core"])
        body = stat(m, masks["body"])
        rows[tag] = {
            "dark_median_db": dark["median_db"],
            "core_median_db": core["median_db"],
            "body_median_db": body["median_db"],
            "contrast_core_minus_dark_db": core["median_db"] - dark["median_db"],
            "contrast_body_minus_dark_db": body["median_db"] - dark["median_db"],
            "dark_db_spatial_std": float(m[masks["dark"] & m2d].std()),
            "grad_energy_body": grad_energy(m, masks["body"] & m2d),
            "p50_db": float(np.percentile(m[m2d], 50)),
            "p99_9_db": float(np.percentile(m[m2d], 99.9)),
        }

    # 同窗渲染：复刻主脚本的 vlo/vhi 规则（vhi 由 ②③④ 定，这里只有 ②③）
    vhi = max(np.percentile(mip2[m2d], 99.9), np.percentile(mip3[m2d], 99.9))
    vlo = max(min(np.percentile(mip2[m2d], 1.0), np.percentile(mip3[m2d], 1.0)), vhi - 45.0)
    for tag, m in (("②SVD", mip2), ("③SVD+Wiener", mip3)):
        cm = m[m2d]
        rows[tag]["clip_hi_frac"] = float((cm > vhi).mean())
        rows[tag]["clip_lo_frac"] = float((cm < vlo).mean())

    print(f"\n=== {ds}  Wiener 归因诊断（w={args.w}，裁剪网格 {pd_svd.shape}）===")
    print("b 系数分布（决定温和/过度平滑）：")
    for k, v in bstat.items():
        print(f"  {k:22s} {v:.6g}")
    print("\nMIP 上 ② vs ③（同窗 vlo=%.2f vhi=%.2f dB）：" % (vlo, vhi))
    keys = ["dark_median_db", "core_median_db", "body_median_db",
            "contrast_core_minus_dark_db", "contrast_body_minus_dark_db",
            "dark_db_spatial_std", "grad_energy_body", "p50_db", "p99_9_db",
            "clip_hi_frac", "clip_lo_frac"]
    print(f"  {'指标':32s} {'②SVD':>12s} {'③SVD+Wiener':>14s} {'③−②':>10s}")
    for k in keys:
        a, c = rows["②SVD"][k], rows["③SVD+Wiener"][k]
        print(f"  {k:32s} {a:12.4f} {c:14.4f} {c - a:+10.4f}")

    rec = {"dataset": ds, "grid": list(pd_svd.shape), "w": args.w,
           "b_stats": bstat, "mip_contrast": rows,
           "shared_window": {"vlo": float(vlo), "vhi": float(vhi)},
           "note": "归因诊断，不产生判据、不改变任何预注册结论"}
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{ds}_wiener_diag.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"\n记盘：{p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

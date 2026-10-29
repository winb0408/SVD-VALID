"""V3 Phase 1 / Gate 1：噪声空间相关长度的多代理交叉核验。

要回答的不是"ACF 是多少"，而是**量到的这条 ACF 里，哪一部分是固定的散斑，
哪一部分是随时间缓慢漂移的杂波/生理运动**。两者的后果完全不同：
前者是 VALID 该去掉的噪声，后者去掉它就是去掉了信号。

三个代理：

P1  background ACF of the final PD image
    电池里最直接的那个——训练图里剩下的东西。但它是**固定成分 + 慢抖动**的混合。

P2  repeat-observation difference（两帧组 PD 之差）
    A/B 两组按 p=5 交错覆盖全片（V2 已测该划分的组内时间相关≈0.07）。
    两组共享的**固定散斑在 log 差里逐体素抵消**，剩下的只有随时间变化的那部分。
    所以 P2 单独量出"抖动分量"的空间相关。P1 与 P2 若能对上，说明固定成分不是主因；
    P2 远低于 P1，说明 P1 主要被固定图案/慢杂波撑着。

P3  depth-stratified P1
    PSF 随深度变，slab 之间的 ℓ 是否一致。

最后把两种 ACF 分别代回 Gate 1 的泄漏上界，给出 m 的取值。

用法：
    python -u scripts/v3_correlation_length.py --sequences SA64 HSA64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.iq_io import load_iq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_pipeline import SEQUENCE_SETTINGS, svd_blood_band  # noqa: E402
from v3_precheck_sampling_geometry import PATS, leak, off  # noqa: E402,F401

DATA_ROOT = Path("/ssd1/wjb/Data/Dyrad/data")
OFFICIAL = Path("data/processed/official")

LAGS = [d for d in range(1, 7)]
AXES = {0: "z", 1: "y", 2: "x"}


# ------------------------------------------------------------------ ACF utils
def acf_1d(x, mask, axis, lag, mu, var):
    """沿 axis 平移 lag 的掩码配对协方差，归一化到背景方差。"""
    d = [0, 0, 0]
    d[axis] = lag
    a0 = tuple(slice(max(0, -di), n - max(0, di)) for n, di in zip(x.shape, d))
    b0 = tuple(slice(max(0, di), n - max(0, -di)) for n, di in zip(x.shape, d))
    m = mask[a0] & mask[b0]
    if m.sum() < 500:
        return float("nan"), 0
    num = float(((x[a0][m] - mu) * (x[b0][m] - mu)).mean())
    return num / var, int(m.sum())


def acf_all_axes(x, mask, lags=LAGS):
    mu = float(x[mask].mean())
    var = float(((x[mask] - mu) ** 2).mean())
    out = {}
    for ax, name in AXES.items():
        out[name] = [acf_1d(x, mask, ax, d, mu, var) for d in lags]
    return out


def plateau_removed(curve, tail=2):
    """去掉远 lag 的平台（慢分量），把快速去相关那一段单独拿出来。"""
    vals = np.array([c[0] for c in curve], float)
    ok = np.isfinite(vals)
    if ok.sum() < 3:
        return np.full_like(vals, np.nan)
    plat = float(np.nanmean(vals[ok][-tail:]))
    return (vals - plat) / max(1.0 - plat, 1e-9)


def ell_from_lag1(rho):
    """Gaussian 等效相关长度：rho(d)=exp(-d^2/(2 ell^2))。"""
    if not np.isfinite(rho) or not (0.0 < rho < 1.0):
        return float("nan")
    return float(np.sqrt(-0.5 / np.log(rho)))


def lambda_from_fit(curve):
    """单调指数拟合 rho(d)=exp(-d/lambda)，用 lag 1..3 的最小二乘（对数域）。"""
    d = np.arange(1, len(curve) + 1, dtype=float)
    r = np.array([c[0] for c in curve], float)
    ok = np.isfinite(r) & (r > 1e-3) & (r < 1.0)
    if ok.sum() < 2:
        return float("nan")
    slope = np.polyfit(d[ok], np.log(r[ok]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("nan")


# --------------------------------------------------------- P2: 两组 PD 之差
def two_group_pd(seq):
    """返回 (PD_A, PD_B, valid)；A/B 按 p=5 交错覆盖全片。"""
    st = SEQUENCE_SETTINGS[seq]
    x = load_iq(DATA_ROOT / seq / f"Data_{seq}.mat", st["variable"])
    if st["sum_arrays"] and x.ndim == 5:
        x = np.squeeze(x.sum(axis=-1))
    filtered, _ = svd_blood_band(x, st["cut1"], st["cut2"], True)
    del x
    t = filtered.shape[3]
    sel = (np.arange(t) % 10) < 5
    p = np.abs(filtered) ** 2
    pa = p[..., sel].sum(axis=3)
    pb = p[..., ~sel].sum(axis=3)
    del filtered, p
    valid = np.load(OFFICIAL / seq / "valid_mask.npy").astype(bool)
    return pa, pb, valid, int(sel.sum()), int((~sel).sum())


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=["SA64", "HSA64"])
    ap.add_argument("--out", default="data/processed/v3_correlation_length.json")
    args = ap.parse_args()

    res = {}
    for seq in args.sequences:
        print(f"\n===== {seq} =====", flush=True)
        r = {"P1": {}, "P2": {}, "P3": {}}

        # ---- P1：整幅 PD 的背景 ACF
        pd = np.load(OFFICIAL / seq / "pd_linear.npy").astype(np.float64)
        valid = np.load(OFFICIAL / seq / "valid_mask.npy").astype(bool)
        lg = np.log(np.maximum(pd, 1e-12))
        bg = valid & (pd <= np.quantile(pd[valid], 0.20))
        r["P1"]["n_bg"] = int(bg.sum())
        r["P1"]["acf"] = acf_all_axes(lg, bg)

        # ---- P3：分深度
        z = pd.shape[0]
        slabs = {"z0_12": slice(0, 12), "z12_24": slice(12, 24), "z24_37": slice(24, z)}
        for name, sl in slabs.items():
            m = np.zeros_like(bg)
            m[sl] = bg[sl]
            if m.sum() < 500:
                continue
            r["P3"][name] = {"n_bg": int(m.sum()), "acf": acf_all_axes(lg, m)}

        # ---- P2：重复观测差分
        pa, pb, _, na, nb = two_group_pd(seq)
        d = np.log(np.maximum(pa, 1e-12)) - np.log(np.maximum(pb, 1e-12))
        bgd = valid & (np.maximum(pa, pb) <= np.quantile(np.maximum(pa, pb)[valid], 0.20))
        r["P2"]["n_frames"] = [na, nb]
        r["P2"]["n_bg"] = int(bgd.sum())
        r["P2"]["acf"] = acf_all_axes(d, bgd)
        r["P2"]["std_log_ratio"] = float(np.std(d[bgd]))
        del pa, pb, d

        # ---- 汇总：两条曲线 × 两种读法
        def summarise(acf):
            s = {}
            for name in ("z", "y", "x"):
                curve = acf[name]
                fast = plateau_removed(curve)
                s[name] = {
                    "rho_lag1": curve[0][0],
                    "rho_fast_lag1": float(fast[0]),
                    "ell_gauss_raw": ell_from_lag1(curve[0][0]),
                    "ell_gauss_fast": ell_from_lag1(float(fast[0])),
                    "lambda_exp_raw": lambda_from_fit(curve),
                    "plateau": float(np.nanmean([c[0] for c in curve[-2:]])),
                }
            return s

        r["summary"] = {"P1": summarise(r["P1"]["acf"]), "P2": summarise(r["P2"]["acf"])}
        for name, v in r["P3"].items():
            r["P3"][name]["summary"] = summarise(v["acf"])
        res[seq] = r

        # ---- 打印
        for tag in ("P1", "P2"):
            s = r["summary"][tag]
            print(f"-- {tag}  n_bg={r[tag]['n_bg']}")
            print(f"   {'ax':>3} {'rho(1)':>8} {'plateau':>8} {'rho_fast(1)':>11} "
                  f"{'ell_raw':>8} {'ell_fast':>9} {'lam_exp':>8}")
            for name in ("z", "y", "x"):
                v = s[name]
                print(f"   {name:>3} {v['rho_lag1']:>8.4f} {v['plateau']:>8.4f} "
                      f"{v['rho_fast_lag1']:>11.4f} {v['ell_gauss_raw']:>8.3f} "
                      f"{v['ell_gauss_fast']:>9.3f} {v['lambda_exp_raw']:>8.3f}")

        # ---- 代回 Gate 1 上界
        print("-- Gate 1 泄漏上界（各向异性，按两种读法）")
        for tag, key in (("raw", "ell_gauss_raw"), ("fast", "ell_gauss_fast")):
            L = np.array([r["summary"]["P1"][a][key] for a in ("z", "y", "x")])
            if not np.all(np.isfinite(L)):
                print(f"   {tag}: 非有限，跳过")
                continue
            for m in (1.0, 1.5, 2.0, 3.0):
                vals = [leak(p, a, y, L, m=m) for p in PATS for a in p for y in p if a != y]
                print(f"   {tag:<5} ell=({L[0]:.2f},{L[1]:.2f},{L[2]:.2f}) "
                      f"m={m:<4} min={min(vals):.3f} mean={np.mean(vals):.3f} max={max(vals):.3f}")
            r.setdefault("gate1", {})[tag] = {"ell": L.tolist()}

    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

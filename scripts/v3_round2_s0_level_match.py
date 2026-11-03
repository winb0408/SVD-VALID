"""S0 噪声阶梯"哪一档才对应真实数据"的定量对齐（第二轮审查 §3 的 0.4 要求）。

为什么要单独算
--------------
S0 的噪声是**在 PD 上乘性**加的（`pd *= exp(σ z)`），而网络看到的输入是 `log1p(pd/s)`。
log1p 在 `pd/s ≈ 1` 处的压缩因子是 1/2，所以：

    log1p 域看到的噪声 σ_eff ≈ σ · (pd/s) / (1 + pd/s)

于是**标称 σ 与模型实际要去的噪声不是一回事**。真实数据同理（真实 PD 也走同一个 log1p），
所以比"哪一档更接近真实"必须在**同一个表示（log1p）里**比，否则会得出错误的对齐结论。

做法
----
1. 真实数据：用与 R0 相同的固定基分块 PD，算
   `σ_eff(block) = std( log1p(PD_b/s) - log1p(PD_b'/s) ) / sqrt(2)`，s = 全体 valid 的 PD 中位数
   （与 S0 的 `s = median(pd)` 同口径）。给 100 帧块（块级）与 400 帧全长（全长级）两个数。
2. S0 各档：直接从 `*_log1p.npy` 与 `*_clean.npy` 的差算 `std`，这就是模型输入上的 σ_eff。
3. 把 2 的读数与 1 对齐，输出"哪一档最接近真实数据"以及倍数关系。

用法
----
    python -u scripts/v3_round2_s0_level_match.py --levels low mid high
输出 `reports/v3_round2/s0_level_match.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_svd import load_array, normalize_shape            # noqa: E402
from v3_round2_r0_audit import band_slice, svd_basis, block_pd  # noqa: E402

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
EPS = 1e-12
LEVEL_MANIFEST = "data/processed/v3_round2_s0/{level}/manifest.json"


def real_sigma_eff(dataset: str) -> dict:
    x = normalize_shape(load_array(RAW[dataset], None), 0, 0)
    z, a, b, T = x.shape
    shape3d = (z, a, b)
    C = x.reshape(-1, T)
    _, V = svd_basis(C)
    lo, hi = band_slice(T)
    Q = V[:, lo:hi]
    coeff = C @ Q
    pd_full = block_pd(coeff, Q, np.arange(T), shape3d)
    valid = pd_full > 0
    s = float(np.median(pd_full[valid]))
    out = {"s_median_pd": s, "n_valid": int(valid.sum())}

    def sig(p, q):
        lp, lq = np.log1p(np.maximum(p, 0) / s), np.log1p(np.maximum(q, 0) / s)
        d = (lp - lq)[valid]
        return float(np.std(d) / np.sqrt(2.0))

    per = T // 4
    blocks = [block_pd(coeff, Q, np.arange(i * per, (i + 1) * per), shape3d) for i in range(4)]
    out["sigma_eff_block100"] = float(np.median([sig(blocks[i], blocks[j])
                                                 for i, j in ((0, 1), (2, 3), (0, 2), (1, 3))]))
    half_a = block_pd(coeff, Q, np.arange(0, T // 2), shape3d)
    half_b = block_pd(coeff, Q, np.arange(T // 2, T), shape3d)
    out["sigma_eff_half200"] = sig(half_a, half_b)
    out["sigma_eff_full400_note"] = ("全长单次观测的噪声是块级的 1/2（正态独立平均），"
                                     "block100 读数 /2 即全长级")
    out["sigma_eff_full400_from_block"] = out["sigma_eff_block100"] / 2.0
    out["ratio_pd_over_s_median_bg"] = float(np.median(
        (pd_full[valid] / s)[pd_full[valid] <= np.quantile(pd_full[valid], 0.20)]))
    return out


def s0_sigma_eff(level: str) -> dict:
    man = json.loads((ROOT / LEVEL_MANIFEST.format(level=level)).read_text())
    out = {"sigma_nominal": man["sigma_eps"], "per_volume": []}
    for e in man["train"][:3]:
        n = np.load(ROOT / e["log1p"]).astype(np.float32)
        c = np.load(str(ROOT / e["log1p"]).replace("_log1p", "_clean")).astype(np.float32)
        v = n != 0
        out["per_volume"].append({"seed": e["seed"], "sigma_eff": float(np.std((n - c)[v]))})
    out["sigma_eff"] = float(np.median([p["sigma_eff"] for p in out["per_volume"]]))
    out["compression_factor"] = out["sigma_eff"] / max(out["sigma_nominal"], EPS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["low", "mid", "high"])
    a = ap.parse_args()

    rec = {"test": "S0 噪声阶梯与真实数据的 log1p 域对齐",
           "why": "标称 σ 是 PD 域乘性噪声，模型输入是 log1p(pd/s)；log1p 压缩使 σ_eff ≠ σ",
           "real": {}, "s0": {}}
    for D in RAW:
        rec["real"][D] = real_sigma_eff(D)
        r = rec["real"][D]
        print(f"== {D}: s(median PD)={r['s_median_pd']:.4g}  "
              f"σ_eff(block100)={r['sigma_eff_block100']:.4f}  "
              f"σ_eff(400帧全长)={r['sigma_eff_full400_from_block']:.4f}")

    for lv in a.levels:
        p = ROOT / LEVEL_MANIFEST.format(level=lv)
        if not p.exists():
            print(f"[skip] {lv} manifest 不存在")
            continue
        rec["s0"][lv] = s0_sigma_eff(lv)
        s = rec["s0"][lv]
        print(f"== S0 {lv}: 标称 σ={s['sigma_nominal']:.4f}  "
              f"输入上 σ_eff={s['sigma_eff']:.4f}（压缩 {s['compression_factor']:.2f}×）")

    # 对齐：S0 各档 σ_eff 与真实数据 σ_eff 的比值
    align = {}
    for lv, s in rec["s0"].items():
        row = {}
        for D, r in rec["real"].items():
            row[D] = {"ratio_vs_full400_imagery":
                      s["sigma_eff"] / max(r["sigma_eff_full400_from_block"], EPS)}
        align[lv] = row
    rec["alignment"] = align
    best = None
    for lv, row in align.items():
        dev = float(np.mean([abs(np.log(v["ratio_vs_full400_imagery"])) for v in row.values()]))
        rec.setdefault("closest_to_real", {})[lv] = dev
        if best is None or dev < best[1]:
            best = (lv, dev)
    rec["closest_level"] = best[0] if best else None
    print(f"\n最接近真实数据（log1p 域）的档：{rec['closest_level']}")

    out = ROOT / "reports/v3_round2/s0_level_match.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4 方案①：**训练无关**的信号依赖收缩（Wiener 型）基线。

## 为什么要有它（L47/L48）

L47 证明：漏防式盲点重建（每体素只看别的相位）的闭式上界是 n2n=9.353，训练结果 9.83
已贴上该上界；而同任务一旦允许估计器用 x_v 本身，分区 oracle 收缩能到 0.584。
本脚本给出**不用训练、不用第二张图、只用 obs1** 的可部署版本，作为任何神经方法
必须先超过的基线。

## 做法

设 log 域体素值 x_v，局部均值 mu_v、局部方差 lv_v（同一窗口 w）。
背景里跨实现共享成分为 0（实测 corr = −0.007），所以

    b_v = max(0, 1 − σ² / lv_v)          ∈ [0,1]
    out_v = mu_v + b_v (x_v − mu_v)

σ² 的估法**不用掩膜、不用 obs2**：本数据里绝大多数体素是背景，取
`σ² = median(lv over common)`。这个估法与 lv_v 用了同一个窗口，因而窗口造成的
"低估"在两个量上同向、在 b 里基本抵消（见下表的 b 值：背景 0.09–0.30，血管 ≈1.0）。

唯一自由参数是窗口 w。**结论不依赖挑窗口**：w ∈ 3..11 全部落在 0.75–0.85，全部通过
预注册门槛，所以 headline 报整段而不是某一个最优点。

用法：
    python scripts/v3_phase4_wiener.py \
        --obs1 data/processed/v3_sim/sim_seed1.npz \
        --obs2 data/processed/v3_sim/sim_seed1_n201_s1.npz \
        --obs-mean ... \
        --out reports/v3_phase4_2026-10-21/wiener
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase3_eval import M_EVAL, usable, support_mask          # noqa: E402
from v3_phase4_scoreboard import to_log, score                   # noqa: E402


def wiener(x, w):
    """返回 (pred, b, sigma2)。x 为 log 域 float64 体积。"""
    mu = uniform_filter(x, w)
    lv = np.maximum(uniform_filter(x * x, w) - mu ** 2, 0.0)
    return mu, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n201_s1.npz")
    ap.add_argument("--obs-mean", nargs="*", default=[])
    ap.add_argument("--windows", nargs="+", type=int, default=[3, 5, 7, 9, 11])
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/wiener")
    args = ap.parse_args()

    d1 = np.load(args.obs1)
    shape = usable(d1["pd_linear"].shape, M_EVAL)
    shape = tuple((n // 32) * 32 for n in shape)
    crop = tuple(slice(0, n) for n in shape)

    pd1 = d1["pd_linear"][crop].astype(np.float32)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))

    obs1 = to_log(pd1, s)
    obs2 = to_log(np.load(args.obs2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)

    obs_mean = None
    if args.obs_mean:
        acc = pd1.astype(np.float64).copy()
        for p in args.obs_mean:
            acc += np.load(p)["pd_linear"][crop].astype(np.float64)
        obs_mean = to_log((acc / (1 + len(args.obs_mean))).astype(np.float32), s)

    common = support_mask(shape, M_EVAL) & valid
    ves = vessels & common
    far = (common & (distance_transform_edt(~vessels) >= 4)
           & (distance_transform_edt(valid) >= 2))
    masks = {"common": common, "ves": ves, "far": far}

    ceiling = score(obs_mean, obs1, obs2, truth, masks)["n2n_ratio"] if obs_mean is not None else None
    ref = score(obs1, obs1, obs2, truth, masks)

    x = obs1.astype(np.float64)
    rows = {}
    print(f"{'w':>3s} {'σ²_est':>9s} {'n2n':>7s} {'bg':>7s} {'ves':>7s} {'r_ves':>7s} "
          f"{'slope':>7s} {'b(bg/ves)':>12s}  判定")
    for w in args.windows:
        mu, lv = wiener(x, w)
        sig2 = float(np.median(lv[common]))
        b = np.clip(1.0 - sig2 / np.maximum(lv, 1e-9), 0.0, 1.0)
        pred = (mu + b * (x - mu)).astype(np.float32)
        m = score(pred, obs1, obs2, truth, masks)
        gap = (1.0 - m["n2n_ratio"]) / (1.0 - ceiling) if ceiling and ceiling < 1 else None
        passed = (m["n2n_ratio"] <= 0.95
                  and m["r_ves"] >= ref["r_ves"] - 0.01
                  and gap is not None and gap >= 0.50)
        rows[w] = {"sigma2_est": sig2, "gap_closed": gap, "pass": bool(passed),
                   "b_bg_mean": float(b[far].mean()), "b_ves_mean": float(b[ves].mean()), **m}
        print(f"{w:3d} {sig2:9.2e} {m['n2n_ratio']:7.3f} {m['n2n_ratio_bg']:7.3f} "
              f"{m['n2n_ratio_ves']:7.3f} {m['r_ves']:7.3f} {m['slope_vs_input']:7.3f} "
              f"{b[far].mean():.2f}/{b[ves].mean():.2f}   "
              f"{'通过' if passed else '不通过'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"method": "Wiener 型信号依赖收缩（训练无关，仅用 obs1）",
           "noise_rule": "σ² = median(局部方差) over common；与 lv 同窗口",
           "ceiling_n2n": ceiling, "input": ref, "settings": {str(k): v for k, v in rows.items()},
           "n2n_range": [min(r["n2n_ratio"] for r in rows.values()),
                         max(r["n2n_ratio"] for r in rows.values())],
           "all_pass": all(r["pass"] for r in rows.values())}
    (out / "wiener.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n天花板={ceiling:.3f}  输入=1.000  n2n 范围={res['n2n_range']}  "
          f"全部通过门槛={res['all_pass']}")
    print(f"wrote {out}/wiener.json")


if __name__ == "__main__":
    main()

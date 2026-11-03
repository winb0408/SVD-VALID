"""SPW 的**留出验证**：参数在场景 1 上选好，原样搬到别的场景，一个都不许改。

## 协议（先写门槛与结构，再跑）

场景 1（`sim_seed1` vs `sim_seed1_n201_s1`）用来**选参**。选出的不是一个点而是一个
平台：在 `spw.json` 里相邻设定给出的 n2n 都落在 0.68–0.72，因此这里固定 6 组
（下面 `CONFIGS`，先写死），搬到：

  - 场景 1 的**另一族**实现（obs1 = n101, obs2 = n201）——检验有没有挑到
    "这一对实现的巧合"；
  - 场景 2 / 3 / 5（obs1 = sim_seedK, obs2 = sim_seedK_n101）——这些场景的**参数、
    几何、血管布局全都不同**，而且只有一对实现，所以天花板只能沿用场景 1 的
    ≈0.636（同生成器族），**gap 只能当参考、不能当通过依据**。因此留出场景的判定
    只看：n2n < 1（确实降噪）和 r_ves 不降。

对照组必须是同场景、同窗宽的 ①（bmax = 1），否则分不清"提升来自 bmax>1 还是来自
换了场景"。脚本对每个场景都同时跑 ① 和 SPW。

用法：
    python scripts/v3_phase4_spw_validate.py --out reports/v3_phase4_2026-10-21/spw_val
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
from v3_phase4_spw import local_var                              # noqa: E402

# (mu 的均匀窗半径 r, lv 的窗 w, b 的上限 bmax) —— 场景 1 平台区的 6 组，事前固定
CONFIGS = [(3, 5, 1.05), (3, 5, 1.10), (3, 3, 1.05), (4, 5, 1.05),
           (2, 5, 1.05), (3, 7, 1.05)]

SCENES = [
    ("scene1_family2", "data/processed/v3_sim/sim_seed1_n101.npz",
     "data/processed/v3_sim/sim_seed1_n201_s1.npz", 0.6359),
    ("scene2", "data/processed/v3_sim/sim_seed2.npz",
     "data/processed/v3_sim/sim_seed2_n101.npz", 0.6359),
    ("scene3", "data/processed/v3_sim/sim_seed3.npz",
     "data/processed/v3_sim/sim_seed3_n101.npz", 0.6359),
    ("scene5", "data/processed/v3_sim/sim_seed5.npz",
     "data/processed/v3_sim/sim_seed5_n101.npz", 0.6359),
]


def load_pair(p1, p2):
    d1 = np.load(p1)
    shape = tuple((n // 32) * 32 for n in usable(d1["pd_linear"].shape, M_EVAL))
    crop = tuple(slice(0, n) for n in shape)
    pd1 = d1["pd_linear"][crop].astype(np.float32)
    valid = d1["valid"][crop].astype(bool)
    vessels = d1["vessels"][crop].astype(bool)
    s = float(np.median(pd1[valid]))
    x = to_log(pd1, s).astype(np.float64)
    obs2 = to_log(np.load(p2)["pd_linear"][crop].astype(np.float32), s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)
    common = support_mask(shape, M_EVAL) & valid
    masks = {"common": common, "ves": vessels & common,
             "far": (common & (distance_transform_edt(~vessels) >= 4)
                     & (distance_transform_edt(valid) >= 2))}
    return x, obs2, truth, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21/spw_val")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = {}
    for name, p1, p2, ceil in SCENES:
        x, obs2, truth, masks = load_pair(p1, p2)
        xf = x.astype(np.float32)
        ref = score(xf, xf, obs2, truth, masks)
        print(f"\n=== {name} ===  输入 r_ves={ref['r_ves']:.4f}  "
              f"（far={masks['far'].sum()} ves={masks['ves'].sum()} 体素）")
        print(f"{'设定':>16s} {'n2n':>7s} {'bg':>7s} {'ves':>7s} {'r_ves':>7s} {'gap*':>6s}")
        rows = {}
        for r, w, bmax in CONFIGS:
            mu = uniform_filter(x, 2 * r + 1)
            _, lv = local_var(x, w)
            sig2 = float(np.median(lv[masks["common"]]))
            for tag, bm in (("①(bmax=1)", 1.0), (f"SPW(bmax={bmax:g})", bmax)):
                b = np.clip(bm * (1.0 - sig2 / np.maximum(lv, 1e-9)), 0.0, bm)
                m = score((mu + b * (x - mu)).astype(np.float32), xf, obs2, truth, masks)
                gap = (1.0 - m["n2n_ratio"]) / (1.0 - ceil)
                rows[f"r{r}_w{w}_{tag}"] = {"r": r, "w": w, "bmax": bm, "gap_ref": gap,
                                            "b_bg": float(b[masks["far"]].mean()),
                                            "b_ves": float(b[masks["ves"]].mean()), **m}
                print(f"{f'r{r} w{w} {tag}':>16s} {m['n2n_ratio']:7.4f} "
                      f"{m['n2n_ratio_bg']:7.4f} {m['n2n_ratio_ves']:7.4f} "
                      f"{m['r_ves']:7.4f} {gap:6.3f}")
        base = [v["n2n_ratio"] for k, v in rows.items() if k.endswith("①(bmax=1)")]
        spw = [v["n2n_ratio"] for k, v in rows.items() if "SPW" in k]
        res[name] = {"input": ref, "rows": rows,
                     "mean_n2n_base": float(np.mean(base)),
                     "mean_n2n_spw": float(np.mean(spw)),
                     "delta": float(np.mean(spw) - np.mean(base)),
                     "spw_all_below_input": bool(max(spw) < 1.0),
                     "spw_worst": float(max(spw)), "base_worst": float(max(base))}
        print(f"  均值：① = {res[name]['mean_n2n_base']:.4f}   SPW = "
              f"{res[name]['mean_n2n_spw']:.4f}   差 = {res[name]['delta']:+.4f}")

    deltas = [v["delta"] for v in res.values()]
    print(f"\n四个场景的 Δ（SPW − ①）：" +
          "  ".join(f"{k}={v['delta']:+.4f}" for k, v in res.items()))
    print(f"平均 Δ = {np.mean(deltas):+.4f}   最大 SPW n2n = "
          f"{max(v['spw_worst'] for v in res.values()):.4f}   "
          f"全部 < 1 = {all(v['spw_all_below_input'] for v in res.values())}")
    res["summary"] = {"mean_delta": float(np.mean(deltas)),
                      "all_scenes_improve": bool(all(d < 0 for d in deltas)),
                      "note": "ceiling 沿用场景 1 的 0.636，留出场景的 gap 仅供参考"}
    (out / "spw_validate.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"wrote {out}/spw_validate.json")


if __name__ == "__main__":
    main()

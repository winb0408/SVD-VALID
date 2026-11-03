#!/usr/bin/env python3
"""Round9 / Phase 1B 标定：**只用校准场景 e900/e901** 选出
  (a) 固定 tet 臂（arm B）的 pattern 与角色顺序；
  (b) 高斯基线的最佳 sigma。

**这一步不允许看 e902–e911 的任何数字**（`reviews/V3_PR1_round9_prereg.md` 写死）。

arm B 的选择规则（预注册，与训练结果无关、与测试数据无关）：

  对每个 pattern P（8 个上游合法 tet）与每种"两输入/两目标"的指派 (i,j | k,l)，
  在**无噪成像期望** `gt_imaging_pd` 的 log1p 域上算

      J(P, i, j, k, l) = 0.5 * MSE(v_k, v_i) + 0.5 * MSE(v_l, v_j)     （仅 valid 体素）

  这正是"网络在第 P 套几何下要把输入映射到目标"时，**目标本身离输入有多远**的度量：
  J 越小，目标越接近输入的无噪值，N2N 的监督信号越"可学"。取 argmin 作为 arm B。

  为什么这不是"用测试结果挑方法"：J 完全由几何 + 标定场景的成像真值决定，
  与网络、损失、训练过程、以及 e902–e911 的成绩都无关；而且它在任何训练开始**之前**
  就已算完并写进预注册。

高斯基线：在 log1p 域做 isotropic Gaussian（`mode="nearest"`），在**校准场景**上扫
sigma ∈ [0.1, 3.0] 步长 0.05，取 log1p 域 MSE 最小者。维纳基线不加可调参数
（`scipy.signal.wiener`, `mysize=3`, 噪声方差取校准得到的 sigma_eps）。

运行： /data2/wjb/anaconda3/envs/threePM/bin/python scripts/v3_round9_calibrate_baseline_arm_b.py
输出： reports/v3_round9/calibration.json
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sampling import TETRIS_PATTERNS, tetris_from_pattern  # noqa: E402

CAL = ROOT / "data/processed/v3_round2_s0/low/eval"
OUT = ROOT / "reports/v3_round9"
SCENES = ["e900", "e901"]


def load(scene):
    """读一个校准场景。

    体积是 37×64×64（Z 为奇数），而 tet 采样要求三个空间维都是偶数。
    这里**裁掉最后一帧**（不是补零）——补零会造出一整层假体素参与 J 的平均，
    而裁掉的 1/37 帧对"选几何"这件事没有信息量。
    部署时的推理（`infer.py`）仍按原样 pad，这里只是标定用的裁剪。
    """
    d = np.load(CAL / f"s0_low_{scene}.npz")
    s = float(d["s"])
    nz = d["valid"].shape[0] // 2 * 2
    valid = d["valid"][:nz]
    obs = np.log1p(np.maximum(d["pd_linear"][:nz], 0) / s)
    truth = np.log1p(np.maximum(d["gt_imaging_pd"][:nz], 0) / s)
    return obs, truth, valid


def views(arr, pattern):
    out = []
    for v in tetris_from_pattern(arr.astype(np.float32), pattern):
        out.append(np.asarray(v, dtype=np.float64))
    return out


def main():
    data = [load(s) for s in SCENES]

    # ---------- (a) 固定 tet 臂：argmin J ----------
    rows = []
    for p in range(len(TETRIS_PATTERNS)):
        acc = {}
        for obs, truth, valid in data:
            tv, vv = views(truth, p), views(valid.astype(np.float32), p)
            for order in itertools.permutations(range(4)):
                i, j, k, l = order
                m = (vv[i] > 0.5) & (vv[j] > 0.5) & (vv[k] > 0.5) & (vv[l] > 0.5)
                if not m.any():
                    continue
                jv = 0.5 * float(((tv[k] - tv[i])[m] ** 2).mean()) + \
                     0.5 * float(((tv[l] - tv[j])[m] ** 2).mean())
                acc.setdefault(order, []).append(jv)
        # 8 个 pattern × 24 个置换；但 (i,j) 与 (j,i) 以及目标对调后
        # 0.5·MSE(k,i)+0.5·MSE(l,j) 会变，所以 24 个都要看
        for order, vals in acc.items():
            rows.append({"pattern": p, "role_order": list(order),
                         "J": float(np.mean(vals)),
                         "J_per_scene": [float(v) for v in vals]})
    rows.sort(key=lambda r: r["J"])
    best = rows[0]
    per_pattern_best = {}
    for r in rows:
        per_pattern_best.setdefault(r["pattern"], r)
    pattern_table = [{"pattern": p,
                      "best_role_order": v["role_order"], "J": v["J"],
                      "corners": TETRIS_PATTERNS[p].tolist()}
                     for p, v in sorted(per_pattern_best.items(),
                                        key=lambda kv: kv[1]["J"])]

    # ---------- (b) 高斯基线 sigma ----------
    sigmas = np.arange(0.10, 3.001, 0.05)
    gmse, gmae = [], []
    for sg in sigmas:
        m, a = [], []
        for obs, truth, valid in data:
            sm = gaussian_filter(obs.astype(np.float64), sigma=float(sg), mode="nearest")
            m.append(float(((sm - truth)[valid] ** 2).mean()))
            a.append(float(np.abs(sm - truth)[valid].mean()))
        gmse.append(np.mean(m))
        gmae.append(np.mean(a))
    k = int(np.argmin(gmse))
    gauss_best = {"sigma": float(sigmas[k]), "log1p_mse": float(gmse[k]),
                  "log1p_mae": float(gmae[k]),
                  "mse_at_sigma_0p5": float(gmse[int(np.argmin(np.abs(sigmas - 0.5)))]),
                  "mse_at_sigma_1p0": float(gmse[int(np.argmin(np.abs(sigmas - 1.0)))])}

    # ---------- 参考：维纳（无自由参数） ----------
    wm, wa = [], []
    for obs, truth, valid in data:
        w = wiener(obs.astype(np.float64), mysize=3)
        wm.append(float(((w - truth)[valid] ** 2).mean()))
        wa.append(float(np.abs(w - truth)[valid].mean()))
    wiener_ref = {"log1p_mse": float(np.mean(wm)), "log1p_mae": float(np.mean(wa)),
                  "params": "mysize=3, noise var 取 scipy 估计（无可调参数）"}

    res = {
        "scenes_used_for_calibration": SCENES,
        "scenes_never_touched_here": "e902-e911",
        "arm_b_selection_rule": "argmin J = 0.5*MSE(truth_vk, truth_vi) + 0.5*MSE(truth_vl, truth_vj)",
        "arm_b": {"pattern": best["pattern"],
                  "corners": TETRIS_PATTERNS[best["pattern"]].tolist(),
                  "role_order": best["role_order"],
                  "J": best["J"],
                  "command": (f"--mode fixed --pattern {best['pattern']} "
                              f"--role-order {' '.join(map(str, best['role_order']))}")},
        "worst_configuration": {"pattern": rows[-1]["pattern"],
                               "role_order": rows[-1]["role_order"],
                               "J": rows[-1]["J"]},
        "J_spread_best_to_worst": {"min": rows[0]["J"], "max": rows[-1]["J"]},
        "per_pattern_best": pattern_table,
        "top10": rows[:10],
        "gaussian_baseline_on_calibration": gauss_best,
        "wiener_baseline_on_calibration": wiener_ref,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("arm B =", res["arm_b"]["command"], " J =", round(best["J"], 6))
    print("J 最好 / 最差 =", round(rows[0]["J"], 6), "/", round(rows[-1]["J"], 6),
          f"（{rows[-1]['J'] / max(rows[0]['J'], 1e-12):.2f}×）")
    print("\nper-pattern 最优：")
    for t in pattern_table:
        print(f"  pattern {t['pattern']} corners {t['corners']} "
              f"role_order {t['best_role_order']}  J={t['J']:.6f}")
    print("\nGaussian 最佳 sigma =", gauss_best)
    print("Wiener(参考) =", wiener_ref)
    print(f"\nwrote {OUT / 'calibration.json'}")


if __name__ == "__main__":
    main()

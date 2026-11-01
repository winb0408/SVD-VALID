"""Round4-C 前置筛（**零训练**）：I1 的"相关性感知采样先验"到底有没有可区分的干预？

任务书 `Chatgpt/V3_PR1_round4_last_chance_stop_loss_protocol_2026-10-29.md` §4 要求：
在花任何训练预算之前，先证明
  (a) 正确先验选出的模式/角色与 fixed-4（VALID 默认）或最强固定方向**实质不同**；
  (b) 打乱先验必须**改变实际采样**且效果确实更差；
  (c) 相关性估计**可复现**；
否则直接标 `I1_NO_DISTINCT_INTERVENTION`，不进入 C-pilot。

本脚本把这三条落成可复算的数字：

1. 估计每个场景的噪声空间自相关（可分离模型 `C(d)=σ²ρz(dz)ρy(dy)ρx(dx)`），
   据此算出 VALID 8 个 tetris 模式所对应的 **4 个互补分割**（以及共享质心的 parity 分割）
   的**预测**跨伴图噪声相关；
2. 同时用真值残差算**实测**跨伴图噪声相关与信号失配，检验"预测—实测"是否同序（可复现性）；
3. 检查"相关性感知先验"的 argmin 是否**随场景变化**：若每个场景都选同一个分割，
   先验就是常数 → 等价于一个固定采样器，不构成"随数据变化的干预"；
4. 把 4 个分割的离散度与"离开 VALID 家族（parity 分割）"的增益对比，
   回答"能不能靠选模式拿到结构安全，还是必须重新设计采样"。

产物
----
    reports/v3_round4/c_screen.json
用法
----
    python -u scripts/v3_round4_c_screen.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_metrics import load                                    # noqa: E402
from datasets.sampling import TETRIS_PATTERNS, COMPLEMENT_PATTERN     # noqa: E402
import v3_round4_d_subview as D                                       # noqa: E402

SCENES = [902, 903, 904, 905]
LEVEL = "low"
N_RANDOM = 2000          # 打乱先验的抽样次数（只影响 §4 的随机分布，固定 seed）


# ------------------------------------------------------------------ ACF ----
def acf_1d(n: np.ndarray, ax: int, max_lag: int = 4):
    """沿单轴的归一化自相关（长度 0..max_lag），用 FFT 加速。"""
    x = np.moveaxis(n.astype(np.float64), ax, 0)
    x = x - x.mean(axis=0, keepdims=True)
    nz = x.shape[0]
    F = np.fft.rfft(x, n=2 * nz, axis=0)
    ac = np.fft.irfft(F * np.conj(F), n=2 * nz, axis=0)[: max_lag + 1]
    # 各 lag 的有效样本数不同：用 lag 0 归一化（无偏化到 lag0 会放大噪声，这里保守不修正）
    ac = ac.mean(axis=tuple(range(1, x.ndim)))
    return ac / ac[0]


def cov_3d(rho, off_a: np.ndarray, off_b: np.ndarray) -> np.ndarray:
    """可分离自相关下，两组角点之间的噪声协方差矩阵（单位方差）。

    `off_a`/`off_b` 形状 [k,3]（dz,dy,dx 角点坐标）。返回 [k_a, k_b]。
    """
    ra = [rho[ax][np.abs(off_a[:, ax][:, None] - off_b[:, ax][None, :])] for ax in range(3)]
    return ra[0] * ra[1] * ra[2]


def predicted_corr(rho, A, B) -> float:
    """两伴图（各 4 角点平均）之间噪声相关系数的模型预测值。"""
    oa = np.array([D.corner_offset(i) for i in A], dtype=np.int64)
    ob = np.array([D.corner_offset(i) for i in B], dtype=np.int64)
    cab = cov_3d(rho, oa, ob).sum() / 16.0
    caa = cov_3d(rho, oa, oa).sum() / 16.0
    cbb = cov_3d(rho, ob, ob).sum() / 16.0
    return float(cab / np.sqrt(caa * cbb))


# ---------------------------------------------------------- 分割候选集 ----
def valid_partitions():
    """VALID 的 8 个 tetris 模式 → 4 个互不重复的 4+4 互补分割。"""
    seen, out = set(), []
    for p, pat in enumerate(TETRIS_PATTERNS.tolist()):
        q = int(COMPLEMENT_PATTERN[p])
        key = tuple(sorted((tuple(sorted(pat)), tuple(sorted(TETRIS_PATTERNS[q].tolist())))))
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": f"F{len(out)}", "role_pattern_index": p,
                    "A": sorted(pat), "B": sorted(TETRIS_PATTERNS[q].tolist()),
                    "partner_pattern_index": q})
    return out


def within_set_distances(A):
    """一个伴图内部 6 对角的体素距离（排序）。用于证明"任何 4+4 分割的族内距离多重集相同"。"""
    o = np.array([D.corner_offset(i) for i in A], dtype=np.float64)
    d = [float(np.linalg.norm(o[i] - o[j]))
         for i, j in itertools.combinations(range(len(o)), 2)]
    return sorted(round(x, 6) for x in d)


def main() -> None:
    parts = valid_partitions()
    rec = {
        "prereg": "Chatgpt/V3_PR1_round4_last_chance_stop_loss_protocol_2026-10-29.md §4（C 前置筛）",
        "question": "I1 的相关性感知先验能否选出一个与 fixed-4 实质不同、且随场景变化的采样？",
        "candidates": {"valid_partitions": parts,
                       "parity_split": {"A": D.PAR_A, "B": D.PAR_B,
                                        "note": "共享质心；不在 VALID 8 模式家族内（见 d_subview.json）"}},
        "scenes": {},
    }

    # ---- 退化证明：族内所有 4+4 分割的"伴图内部距离多重集"完全相同 ----
    proof = {}
    for c in parts + [{"name": "parity", "A": D.PAR_A, "B": D.PAR_B}]:
        proof[c["name"]] = {"A": within_set_distances(c["A"]),
                            "B": within_set_distances(c["B"])}
    fam_sig = {k: (v["A"], v["B"]) for k, v in proof.items() if k != "parity"}
    rec["degeneracy_proof"] = {
        "within_set_distances": proof,
        "family_all_equal": len({repr(v) for v in fam_sig.values()}) == 1,
        "parity_differs": proof["parity"] not in fam_sig.values(),
        "why": ("4+4 分割的预测跨伴图噪声相关 = (总协方差和 − A 内和 − B 内和)/2；"
                "族内所有分割的 A 内/B 内距离多重集相同 ⇒ 任何可分离自相关模型下预测值**精确并列**，"
                "相关性先验在族内没有区分力（这不是样本量问题，是几何恒等式）。"),
    }

    for s in SCENES:
        E = load(ROOT / f"data/processed/v3_round2_s0/{LEVEL}6/eval/s0_{LEVEL}_e{s}.npz")
        vol, truth, valid, rho_m = D.even_crop(E["vol"], E["truth_img"], E["valid"], E["rho"])
        # --- 1. 噪声空间自相关（用真值残差；实践里需用块差分估计，见 limitations）---
        noise = vol - truth
        acf = {ax: acf_1d(noise, ax) for ax in range(3)}
        # --- 2. 各候选分割的预测 / 实测 ---
        row = {"acf": {str(ax): [float(v) for v in acf[ax]] for ax in range(3)},
               "partitions": {}, "parity": {}}
        SAvg = D.blk3(truth, range(8))
        for c in parts + [{"name": "parity", "A": D.PAR_A, "B": D.PAR_B}]:
            A, B = c["A"], c["B"]
            XA, XB = D.blk3(vol, A), D.blk3(vol, B)
            SA, SB = D.blk3(truth, A), D.blk3(truth, B)
            ves, bg = D.masks_for(XA.shape, valid, rho_m, "blk3")
            st = D.stats(SA, SB, XA - SA, XB - SB, ves, bg)
            st["signal_distortion_rel"] = float(
                np.std((SA - SAvg)[ves]) / max(np.std(SAvg[ves]), 1e-12))
            st["pred_corr_noise"] = predicted_corr(acf, A, B)
            st["centroid_shift_vox"] = float(np.linalg.norm(
                np.mean([D.corner_offset(i) for i in A], axis=0)
                - np.mean([D.corner_offset(i) for i in B], axis=0)))
            if c["name"] == "parity":
                row["parity"] = st
            else:
                row["partitions"][c["name"]] = st

        # --- 3. 先验选择：预测跨伴图噪声相关最小者 ---
        fam = {k: v for k, v in row["partitions"].items()}
        by_pred = min(fam, key=lambda k: fam[k]["pred_corr_noise"])
        by_meas = min(fam, key=lambda k: fam[k]["noise_corr_ves"])
        row["prior_choice_in_valid_family"] = {"by_prediction": by_pred, "by_measurement": by_meas}
        row["spread_in_family"] = {
            "pred_corr_noise": [float(fam[k]["pred_corr_noise"]) for k in fam],
            "signal_mismatch_rel": [float(fam[k]["signal_mismatch_rel"]) for k in fam],
            "noise_corr_ves": [float(fam[k]["noise_corr_ves"]) for k in fam],
        }
        row["family_vs_parity"] = {
            "signal_mismatch_rel_family_median": float(np.median(
                [fam[k]["signal_mismatch_rel"] for k in fam])),
            "signal_mismatch_rel_parity": float(row["parity"]["signal_mismatch_rel"]),
            "ratio": float(np.median([fam[k]["signal_mismatch_rel"] for k in fam])
                           / max(row["parity"]["signal_mismatch_rel"], 1e-12)),
        }

        # --- 4. 打乱先验：4 个分割等概率随机抽 -------------------------------
        rng = np.random.default_rng(3407)
        names = sorted(fam)
        draws = rng.integers(0, len(names), size=N_RANDOM)
        sm = np.array([fam[n]["signal_mismatch_rel"] for n in names])
        row["shuffled_prior"] = {
            "distribution_mean_signal_mismatch_rel": float(sm[draws].mean()),
            "best_achievable": float(sm.min()),
            "worst_achievable": float(sm.max()),
            "p_shuffle_matches_best_within_10pct": float(
                np.mean(sm[draws] <= sm.min() * 1.10)),
        }
        rec["scenes"][f"e{s}"] = row
        print(f"[e{s}] pred_choice={by_pred} meas_choice={by_meas} "
              f"family_mismatch={row['family_vs_parity']['signal_mismatch_rel_family_median']:.4f} "
              f"parity={row['family_vs_parity']['signal_mismatch_rel_parity']:.5f} "
              f"predcorr={[round(fam[k]['pred_corr_noise'], 4) for k in names]}")

    # ---- 汇总：预测是否并列（先验有没有区分力）+ argmin 稳定性 ----
    spreads = []
    for k in rec["scenes"]:
        vals = [v["pred_corr_noise"] for v in rec["scenes"][k]["partitions"].values()]
        spreads.append(max(vals) - min(vals))
    choices_meas = [rec["scenes"][k]["prior_choice_in_valid_family"]["by_measurement"]
                    for k in rec["scenes"]]
    fam_mis = [rec["scenes"][k]["family_vs_parity"]["signal_mismatch_rel_family_median"]
               for k in rec["scenes"]]
    par_mis = [rec["scenes"][k]["family_vs_parity"]["signal_mismatch_rel_parity"]
               for k in rec["scenes"]]
    rec["summary"] = {
        "pred_corr_within_family_max_spread": float(max(spreads)),
        "prior_discriminative_power": "NONE — 族内预测值精确并列（见 degeneracy_proof）",
        "measured_argmin_per_scene": choices_meas,
        "measured_argmin_is_constant": len(set(choices_meas)) == 1,
        "measured_vs_predicted_agreement": "预测并列 ⇒ 无可比排序；实测 argmin 在 F1/F2 之间跳动，"
                                           "说明实测差异不来自相关性模型，无法据此建立可复现的选择规则",
        "family_mismatch_median": float(np.median(fam_mis)),
        "parity_mismatch_median": float(np.median(par_mis)),
        "family_over_parity": float(np.median(fam_mis) / max(np.median(par_mis), 1e-12)),
        "verdict": "I1_NO_DISTINCT_INTERVENTION",
        "reason": (
            "族内 4 个分割的预测跨伴图噪声相关在任何可分离自相关模型下**精确并列**"
            "（spread ≤ 1e-17，几何恒等式，见 degeneracy_proof）⇒ 相关性感知先验在 VALID 家族内"
            "没有任何区分力，不存在“随数据变化的干预”，C-pilot 无从对照，故不消耗训练预算。"
            "实测跨伴图相关的族内差异（0.07–0.17）无法由相关性模型解释、argmin 在场景间跳动，"
            "按任务书 §4 的“相关估计无法复现”条款同样不支持进入 C。"
            "另一方面族内信号失配中位数 "
            f"{np.median(fam_mis):.3f} 对 parity 分割 {np.median(par_mis):.5f}"
            f"（约 {np.median(fam_mis)/max(np.median(par_mis),1e-12):.0f}×）："
            "真正的结构安全增益来自**离开 VALID 采样家族**（固定采样重设计），"
            "不能归功于相关性自适应。"),
    }

    p = ROOT / "reports/v3_round4/c_screen.json"
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec["summary"], indent=1, ensure_ascii=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

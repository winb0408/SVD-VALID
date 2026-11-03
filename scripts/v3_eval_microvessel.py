"""R12：以微血管图像质量为主判据 —— 弱血管 recall@FPR / 连通性 / CNR。

## 为什么要新写一个判据（旧判据的证伪记录）

现有 `scripts/eval_structure.py` 的分区是**用输入自己定义的**（`input >= 0.65` 叫 core，
`input <= 0.05` 叫 dark）。它有三个问题，逐条是这次要修的：

1. **它把观测当参考**。输入 `pd_linear` 自己带着流血的晕和散斑噪声，"core" 区里既有真血管也有噪声峰，
   所以"保住了 core"也可能是保住了噪声。
2. **它按定义看不见弱血管**。弱血管在输入上本来就不亮（对比度 6 dB + 背景散斑），落在
   `0.2 <= input < 0.65` 的 body 桶里，与"血管旁的晕"混在一起，无法单独回答"弱血管被压掉了多少"。
3. **在 v1 仿真上完全退化**：v1 的真值掩膜外恒为 0，`(0,1)` 区间的体素数是 **0**，
   弱血管区是空集 → 任何"弱血管指标"在 v1 上都是 `nan`。这不是门槛问题，是**任务定义问题**
   （R06 已修，v2 的 `(0,1)` 体素占比 0.0203 / 0.0157）。

## 本判据的定义（估计量，一句话一个）

区域全部由**真值 `rho`** 划分（不碰观测，所以不会把噪声当血管）：

| 区域 | 定义 | 含义 |
|---|---|---|
| `deep` | `rho >= 0.9` | 粗血管核心 |
| `core` | `rho >= 0.5` | 血管主体 |
| `weak` | `0.1 <= rho < 0.5` | **弱血管 / 血管边缘**，本判据的主角 |
| `bg`   | `rho <= 0` | 真背景（v2 里严格无血管） |

**判据只有一类，是被标定逼出来的（见下节"被证伪的端点"）**：

| 端点 | 定义 | 已知答案 |
|---|---|---|
| `r_weak`（主） | `weak` 区 pred vs 真值的 Pearson | 真值 = 1.0；输入 = 0.79；纯低通 = 0.35–0.47 |
| `r_core` / `r_deep`（辅） | 同式，换成 `core` / `deep` | 真值 = 1.0 |
| `r_all`（辅） | all-valid Pearson | — |
| `zero` 约定 | 常数输出（方差为 0）的相关系数**记 0.0**，不记 `nan` | 常数输出不含结构信息，排序上必须落在最差端 |

**被证伪的端点（2026-10-27 已知答案标定，数字见 `reports/v3_R12/microvessel_metric.md`）**：
`recall@fpr5`、`auc(weak vs bg)`、`iou_core`、`ncomp_ratio`、`cnr_weak_db` **五个都不单调**——
无噪真值自己在 `recall@fpr5` 上只有 0.74/0.64，**低于含噪输入**的 0.83/0.73；AUC 上真值 0.965/0.958
也低于输入的 0.973/0.965，低通甚至到 0.995；真值的 `iou_core` 只有 0.05。
根因不是门槛选错，而是 **v2 真值本身没有"血管 vs 背景"的体素级可分性**：PSF σ=(3.0,2.37,3.97) 体素
vs 血管半径 ~1.1，6 dB 对比度下背景被晕填满（`truth` 背景中位数 0.31、最大 2.12，而 core 中位数 1.59，
都在 log1p 域）。所以这五个端点在本数据上**只作描述，不作判据**，JSON 里带 `monotone: false` 标记。

**不做的端点：血管宽度/直径。** PSF σ=(3.0, 2.37, 3.97) 体素 vs 血管半径 ~1.1 体素，
成像后的半高宽由 PSF 决定而不是由血管决定（R06 §… 剖面已证），所以"宽度测量"在**任何**方法上都
≈PSF 宽度：它对方法不敏感，做了只会制造假结论。这一条按 Plan §0.5 记录为**撤除端点**，
不是"没做"。`width` 字段在 JSON 里显式写 `"not_computed"` 并给出原因。

## 已知答案标定（每次改动这个文件都必须重跑）

`--calibrate` 用四个已知答案跑一遍，必须满足：
`recall(truth) > recall(input)`、`recall(zero) == 0`、`iou_core(truth) == 1.0`、
`cnr_weak_db(truth) > cnr_weak_db(oracle_lowpass)`。
不满足就说明**判据本身坏了**，不许拿去判方法。

用法
----
    python -u scripts/v3_eval_microvessel.py --calibrate            # 只跑已知答案标定
    python -u scripts/v3_eval_microvessel.py --npz data/.../v2_eval_s30.npz \
        --pred A_rand=runs/x/infer/denoised.npy ... --out reports/.../micro.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

WEAK_LO, WEAK_HI = 0.1, 0.5
MIN_COMP_VOX = 8          # 连通分量最小体素数：小于它算碎斑，不计入分量数
FPR_NOMINAL = 0.05


# ----------------------------------------------------------------- 数据与区域
def load(npz: Path) -> dict:
    """读 v2 数据集，返回真值域里的量。区域划分只看 `rho`，不看观测。"""
    d = np.load(npz)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s = float(d["s"]) if "s" in d.files else float(np.median(d["pd_linear"][valid]))
    vol = np.log1p(d["pd_linear"].astype(np.float32) / max(s, 1e-12)).astype(np.float32)
    truth = np.log1p(d["gt_flow_pd"].astype(np.float32) / max(s, 1e-12)).astype(np.float32)
    vol[~valid] = 0.0
    truth[~valid] = 0.0
    return {
        "vol": vol, "truth": truth, "valid": valid, "rho": rho,
        "deep": valid & (rho >= 0.9),
        "core": valid & (rho >= 0.5),
        "weak": valid & (rho >= WEAK_LO) & (rho < WEAK_HI),
        "bg": valid & (rho <= 0.0),
        "npz": str(npz),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """常数预测（std=0）返回 0.0 —— 它不含结构信息，排序上必须落在最差端，不能记 nan。"""
    if a.size < 3:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """rank-based AUC = P(a > b)（Mann-Whitney，处理并列按平均秩）。诊断用，非判据。"""
    if a.size == 0 or b.size == 0:
        return float("nan")
    x = np.concatenate([a, b])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < xs.size:                     # 并列取平均秩
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    ra = ranks[:a.size].sum()
    return float((ra - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def _ncomp(mask: np.ndarray) -> int:
    lab, n = label(mask)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())[1:]
    return int(np.sum(sizes >= MIN_COMP_VOX))


def metrics(pred: np.ndarray, E: dict) -> dict:
    """全部端点。`pred` 必须与 `E["vol"]` 同形状、同一 log1p 尺度。"""
    v, deep, core, weak, bg = (E["valid"], E["deep"], E["core"], E["weak"], E["bg"])
    truth = E["truth"]

    out: dict = {
        "r_deep": _pearson(pred[deep], truth[deep]),
        "r_core": _pearson(pred[core], truth[core]),
        "r_weak": _pearson(pred[weak], truth[weak]),
        "r_all": _pearson(pred[v], truth[v]),
        "n_weak": int(weak.sum()), "n_core": int(core.sum()), "n_bg": int(bg.sum()),
    }

    # --- 以下全部为**描述性诊断**，不作判据（已知答案标定证明它们不单调，见模块 docstring）
    out["diagnostics_monotone"] = False
    if bg.sum() and weak.sum():
        thr = float(np.quantile(pred[bg], 1.0 - FPR_NOMINAL))
        out["thr"] = thr
        out["recall_at_fpr5"] = float(np.mean(pred[weak] > thr))
        out["fpr_actual"] = float(np.mean(pred[bg] > thr))
        sd_b = float(np.std(pred[bg]))
        mu_w = float(np.mean(pred[weak]))
        mu_b = float(np.mean(pred[bg]))
        out["cnr_weak"] = (mu_w - mu_b) / sd_b if sd_b > 0 else float("inf")
        out["cnr_weak_db"] = 20.0 * np.log10(out["cnr_weak"]) if out["cnr_weak"] > 0 else float("-inf")
        out["bg_leak"] = mu_b / mu_w if mu_w > 0 else float("nan")
        out["auc_weak"] = _auc(pred[weak], pred[bg])          # 诊断
        out["auc_core"] = _auc(pred[core], pred[bg]) if core.sum() else float("nan")
    else:
        out.update({"thr": None, "recall_at_fpr5": float("nan"), "fpr_actual": float("nan"),
                    "cnr_weak": float("nan"), "cnr_weak_db": float("nan"), "bg_leak": float("nan"),
                    "auc_weak": float("nan"), "auc_core": float("nan")})

    # --- 连通性：前景取 pred 里最亮的 |core| 个体素（公平：所有方法同样的前景体素预算）
    k = int(core.sum())
    if k > 0:
        flat = pred[v]
        thr_top = float(np.sort(flat)[-k]) if flat.size >= k else float(flat.min())
        fg = np.zeros_like(pred, bool)
        fg[v] = pred[v] >= thr_top
        lab_t, n_t = label(core)
        out["iou_core"] = float((fg & core).sum() / max((fg | core).sum(), 1))
        out["ncomp_pred"] = _ncomp(fg)
        out["ncomp_truth"] = _ncomp(core)
        out["ncomp_ratio"] = (out["ncomp_pred"] / out["ncomp_truth"]
                              if out["ncomp_truth"] else float("nan"))
    else:
        out.update({"iou_core": float("nan"), "ncomp_pred": 0, "ncomp_truth": 0,
                    "ncomp_ratio": float("nan")})

    # --- PSNR / SSIM（审查意见 R12 点名的标准指标）
    # 解读必须与 RMSE 类同口径：R06 已证伪"RMSE 类端点可单独作判据"（v1 仿真里全零输出
    # 就能拿最好的 RMSE），PSNR 是 RMSE 的单调函数，所以同样只作**描述性**记录。
    dr = float(np.ptp(truth[v])) if v.any() else 0.0
    if dr > 0:
        mse_ = float(np.mean((pred[v] - truth[v]) ** 2))
        out["psnr_db"] = float(10.0 * np.log10(dr ** 2 / mse_)) if mse_ > 0 else float("inf")
        try:
            from skimage.metrics import structural_similarity as _ssim
            out["ssim"] = float(_ssim(pred, truth, data_range=dr, win_size=7))
        except Exception as e:                                  # noqa: BLE001
            out["ssim"] = float("nan")
            out["ssim_error"] = f"{type(e).__name__}: {e}"
        out["psnr_ssim_data_range"] = dr
        out["psnr_ssim_role"] = "descriptive_only（PSNR 是 RMSE 的单调函数，见 R06 证伪记录）"
    else:
        out.update({"psnr_db": float("nan"), "ssim": float("nan"),
                    "psnr_ssim_data_range": 0.0})

    # --- 显式声明不做的端点（Plan §0.5：撤除端点必须留记录，不是"没做"）
    out["width"] = "not_computed"
    out["width_reason"] = ("PSF σ=(3.0,2.37,3.97) voxel 远宽于血管半径 ~1.1 voxel，"
                           "成像后半高宽由 PSF 决定、对方法不敏感；测得的是 PSF 不是血管")
    return out


# ----------------------------------------------------------------- 已知答案
def known_answers(E: dict, psf_sigma) -> dict:
    lo = gaussian_filter(E["vol"], sigma=psf_sigma, mode="nearest").astype(np.float32)
    lo[~E["valid"]] = 0.0
    return {
        "truth": E["truth"],                                  # 完美答案
        "input": E["vol"],                                    # 什么都不做
        "zero": np.zeros_like(E["vol"]),                       # 最强的平凡解
        "oracle_lowpass": lo,                                  # 纯低通（已知的"糊一下"）
    }


def calibrate(npz_list: list[Path]) -> dict:
    from v3_simulator import SimCfg
    sig = SimCfg().psf_sigma
    rec = {"psf_sigma": list(map(float, sig)), "volumes": {}, "checks": {}}
    ok = True
    for npz in npz_list:
        E = load(npz)
        per = {k: metrics(v, E) for k, v in known_answers(E, sig).items()}
        rec["volumes"][npz.stem] = per
        t, i, z, o = per["truth"], per["input"], per["zero"], per["oracle_lowpass"]
        c = {
            # 判据端点（r_*）：必须严格单调 真值 > 输入 > 纯低通 > 常数
            "r_weak_truth_is_one": bool(abs(t["r_weak"] - 1.0) < 1e-6),
            "r_weak_truth_gt_input": bool(t["r_weak"] > i["r_weak"]),
            "r_weak_input_gt_lowpass": bool(i["r_weak"] > o["r_weak"]),
            "r_weak_lowpass_gt_zero": bool(o["r_weak"] > z["r_weak"]),
            "r_core_truth_is_one": bool(abs(t["r_core"] - 1.0) < 1e-6),
            "r_deep_truth_is_one": bool(abs(t["r_deep"] - 1.0) < 1e-6),
            "zero_is_constant_zero_corr": bool(z["r_weak"] == 0.0 and z["r_deep"] == 0.0),
            # 非单调诊断端点的**反向**检查：无噪真值在这里输给含噪输入，正是不许作判据的理由
            "falsified_recall_not_monotone": bool(t["recall_at_fpr5"] <= i["recall_at_fpr5"]),
            "falsified_auc_not_monotone": bool(t["auc_weak"] <= max(i["auc_weak"], o["auc_weak"])),
            "falsified_iou_truth_not_one": bool(t["iou_core"] < 0.9),
            # SSIM 在本数据上也不单调：s31 上"全零输出"的 SSIM 高于真输入（SSIM 奖励平滑）。
            # PSNR 在 s30/s31 上恰好单调（输入 > 低通 > 全零），但它只是 RMSE 的单调函数，
            # 按 R06 的证伪记录同样只作描述性记录。
            "falsified_ssim_zero_beats_input": bool(z["ssim"] > i["ssim"]),
        }
        # 单调性检查必须每个体都过；"反标定"检查只要求**至少一个体**出现退化
        # （退化本身就是我们要证明的，不能要求每个体都退化——s31 的 SSIM 退化、s30 没有）
        c["monotone_pass"] = all(v for k, v in c.items() if not k.startswith("falsified_"))
        rec["checks"][npz.stem] = c
        ok &= c["monotone_pass"]
    fals = sorted({k for c in rec["checks"].values() for k, v in c.items()
                   if k.startswith("falsified_")})
    rec["falsified_anywhere"] = {k: any(rec["checks"][v][k] for v in rec["checks"]) for k in fals}
    # 每条被撤除的端点，都必须在至少一个体上真的表现出退化；否则"撤除理由"就没有证据
    ok &= all(rec["falsified_anywhere"].values())
    rec["all_pass"] = bool(ok)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="只跑已知答案标定（无 GPU、无需网络）")
    ap.add_argument("--npz", nargs="*", default=None)
    ap.add_argument("--pred", nargs="*", default=None, help="名字=denoised.npy 的列表")
    ap.add_argument("--out", default=str(ROOT / "reports/v3_R12/microvessel.json"))
    args = ap.parse_args()

    npzs = [Path(p) for p in (args.npz or [])] or [
        ROOT / "data/processed/v3_sim_v2/eval/v2_eval_s30.npz",
        ROOT / "data/processed/v3_sim_v2/eval/v2_eval_s31.npz",
    ]

    rec = {"metric_version": "R12-v1", "weak_range": [WEAK_LO, WEAK_HI],
           "min_comp_vox": MIN_COMP_VOX, "fpr_nominal": FPR_NOMINAL}
    if args.calibrate or not args.pred:
        rec["calibration"] = calibrate(npzs)
        c = rec["calibration"]
        print(f"已知答案标定 all_pass={c['all_pass']}")
        for vol, per in c["volumes"].items():
            print(f"\n[{vol}]")
            hdr = f"{'arm':16s} {'r_weak':>8s} {'recall@fpr5':>12s} {'fpr':>7s} {'cnr_db':>8s} {'iou_core':>9s} {'ncomp':>6s}"
            print(hdr)
            for name, m in per.items():
                print(f"{name:16s} {m['r_weak']:8.4f} {m['recall_at_fpr5']:12.4f} "
                      f"{m['fpr_actual']:7.4f} {m['cnr_weak_db']:8.2f} {m['iou_core']:9.4f} "
                      f"{m['ncomp_pred']:6d}")
            print("checks:", c["checks"][vol])
        if not c["all_pass"]:
            raise SystemExit("已知答案标定未通过：判据本身有问题，不许拿去判方法")

    if args.pred:
        rec["methods"] = {}
        for spec in args.pred:
            name, path = spec.split("=", 1)
            pred = np.load(path).astype(np.float32)
            rec["methods"][name] = {}
            for npz in npzs:
                E = load(npz)
                if pred.shape != E["vol"].shape:
                    raise SystemExit(f"{name} 形状 {pred.shape} != {E['vol'].shape}")
                p = pred.copy()
                p[~E["valid"]] = 0.0
                rec["methods"][name][npz.stem] = metrics(p, E)
                m = rec["methods"][name][npz.stem]
                print(f"[{name}] {npz.stem}: r_weak={m['r_weak']:.4f} "
                      f"recall={m['recall_at_fpr5']:.4f} cnr={m['cnr_weak_db']:.2f}dB "
                      f"iou={m['iou_core']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

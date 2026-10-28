"""S0 正控制数据集：**已知 PSF + 加性零均值独立噪声**的 3D 血管 phantom。

为什么单独造一套数据（第二轮审查 §3 S0a–S0f）
------------------------------------------------
实验 A 的 v2 仿真里，残余波动是**散斑**（乘性、空间相关、与信号纠缠），失败之后无法区分
"网络/损失/评价链有问题" 与 "物理噪声太难"。S0 要的是一个**教科书式能解**的 N2N 任务：

    干净成像期望（已知）  +  两次独立零均值加性噪声  ->  (obs1, obs2)

如果连这个都学不出来，那就不是物理的问题，是实现/目标/评价的问题。反之，S0 一过，
实验 B/C 才有意义。

噪声定义（必须写清，否则会被误读成"随便加了点噪声"）
----------------------------------------------------
网络的输入域是 `log1p(PD/s)`。本脚本的噪声**加在 log 域**：

    log obs_i = log(clean_pd) + sigma * z_i ,   z_i ~ N(0,1) iid

等价于 PD 域的**乘性**扰动 `obs_i = clean_pd * exp(sigma z_i)`。选这个等价形式而不是 PD 域加性，
是因为 2026 年 SVD-SN2N 的核心论点正是"对数压缩把乘性残余变成方差与信号无关的可加扰动"
（`reports/v3_R11/sn2n_difference_matrix.md` §5），而 VALID 的训练目标也建在 log-PD 上。
`--noise-domain linear` 提供 PD 域加性的对照档，用于说明这个选择有影响。

噪声档位（S0b）
---------------
`low` 用 **v2 实测标定**：`reports/v3_r06/v2_endpoint_audit.json` 里 `q3_n2n` 的
`denominator_mse_obs1_obs2` → 单次观测的 log 域噪声 std = `sqrt(mse/2)` = **0.0385**（s30）
/ 0.0383（s31）。mid = 2x，high = 4x。**不凭空定档**。

真值分列（审查 0.5）
-------------------
    rho          几何密度（连续 0–1），几何真值
    gt_flow_pd   只含血流分量的期望（对 N2N 目标**不是**正确答案，仅参考）
    gt_imaging_pd 血流 + 静态本底的**无噪成像期望**  <- S0 主终点 r_roi 用它
    pd_linear / pd2_linear  两次独立含噪观测

用法
----
    python scripts/v3_round2_s0_dataset.py --level low --n-train 12 --n-eval 2
产物
----
    data/processed/v3_round2_s0/<level>/train/s0_<level>_s{seed}.npz (+ _log1p.npy)
    data/processed/v3_round2_s0/<level>/eval/s0_<level>_e{seed}.npz
    data/processed/v3_round2_s0/<level>/manifest.json
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

from v3_simulator import SimCfg, _fft_convolve, gauss_field, vessel_density  # noqa: E402

# 单次观测的 log 域噪声 std，来自 reports/v3_r06/v2_endpoint_audit.json（q3_n2n）。
V2_SIGMA_EPS = 0.0385
LEVEL_SIGMA = {"low": V2_SIGMA_EPS, "mid": 2 * V2_SIGMA_EPS, "high": 4 * V2_SIGMA_EPS,
               # **诊断档，不属于审查要求的 low/mid/high 噪声阶梯**（2026-10-27 19:00 加入）：
               # σ_eff ≈ 0.0066，比 low 还低 3 倍。用途是回答一个 S0 之外的问题——
               # "这条训练链在**显然可做**的任务上到底能不能去噪"。如果在这个档上仍拿不到
               # n2n < 1，那就与信噪比无关，是训练/损失/架构的问题。
               "vlow": V2_SIGMA_EPS / 3.0}

PSF = np.array(SimCfg().psf_sigma, float)          # (z,y,x)，体素
# PD = |field|^2，而 field 已被 PSF 模糊：PD 图像的等效平滑核是 PSF 的自相关，sigma 乘 sqrt(2)。
PSF_PD = PSF * np.sqrt(2.0)
FLOW_CONTRAST_DB = 8.0      # ROI(ρ≥0.5) 血流 PD 相对背景中位数，固定档（只让噪声变）


def make_clean(seed: int, shape=(37, 64, 64)):
    """返回 rho / 干净血流分量 / 干净完整成像。所有随机性由 seed 决定，噪声另开流。"""
    rng = np.random.default_rng(seed)
    # 血管半径分三档：0.5–0.8（亚体素，难）、0.8–1.2（中）、1.2–1.6（易）
    rho = np.zeros(shape, np.float32)
    for n_tubes, radius, soft in ((8, 0.75, 1.0), (8, 1.1, 1.5), (5, 1.5, 2.0)):
        rho = np.maximum(rho, vessel_density(rng, shape, n_tubes, radius, soft))
    rho = rho.astype(np.float32)

    flow_img = _fft_convolve(rho.astype(np.complex64), PSF_PD).real.astype(np.float32)
    flow_img = np.maximum(flow_img, 0.0)

    # 静态本底：低频起伏，两次观测完全共享（相当于"该保留的组织散斑平台"）
    bg = np.abs(gauss_field(rng, shape, (6.0, 8.0, 8.0))).astype(np.float32) ** 2
    bg = bg / max(float(np.median(bg)), 1e-12)

    valid = np.ones(shape, bool)
    valid[:2] = False

    # 定标：让 ρ≥0.5 的血流 PD 中位数 = 背景中位数 * 10^(contrast/10)
    core = rho >= 0.5
    target = 10.0 ** (FLOW_CONTRAST_DB / 10.0)
    a_flow = target * float(np.median(bg[valid])) / max(float(np.median(flow_img[core])), 1e-12)
    return {"rho": rho, "flow": a_flow * flow_img, "bg": bg, "valid": valid,
            "clean": a_flow * flow_img + bg}


def add_noise(clean, valid, sigma, rng, domain="log", corr_sigma=None):
    """加噪声。

    domain="log"    : log obs = log clean + sigma*z（PD 域乘性）
    domain="linear" : obs = clean + sigma*std(clean)*z（PD 域加性，对照档）
    corr_sigma      : 若给了 (sz,sy,sx)，把 z 先与一个单位方差复高斯场卷积再取实部，
                      使噪声**空间相关**（S0c 相关性阶梯），并重新标准化到单位方差。
    """
    if corr_sigma is None:
        z = rng.standard_normal(clean.shape).astype(np.float32)
    else:
        w = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)) / np.sqrt(2)
        z = _fft_convolve(w.astype(np.complex64), corr_sigma).real.astype(np.float32)
        z = z / max(float(np.std(z[valid])), 1e-12)
    z[~valid] = 0.0
    if domain == "log":
        return (clean * np.exp(sigma * z)).astype(np.float32)
    scale = float(np.std(clean[valid])) * sigma
    return (clean + scale * z).astype(np.float32)


def build_one(seed: int, level: str, domain: str, corr_sigma, with_second=True) -> dict:
    c = make_clean(seed)
    sigma = LEVEL_SIGMA[level]
    rng1 = np.random.default_rng(seed + 10_000)
    rng2 = np.random.default_rng(seed + 20_000)
    pd1 = add_noise(c["clean"], c["valid"], sigma, rng1, domain, corr_sigma)
    pd2 = (add_noise(c["clean"], c["valid"], sigma, rng2, domain, corr_sigma)
           if with_second else None)
    s = float(np.median(pd1[c["valid"]]))
    return {"pd_linear": pd1, "pd2_linear": pd2, "gt_flow_pd": c["flow"].astype(np.float32),
            "gt_imaging_pd": c["clean"].astype(np.float32), "rho": c["rho"],
            "vessels": (c["rho"] >= 0.5), "valid": c["valid"], "s": np.float32(s),
            "noise_sigma": np.float32(sigma), "level": level}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low", choices=sorted(LEVEL_SIGMA))
    ap.add_argument("--noise-domain", default="log", choices=["log", "linear"])
    ap.add_argument("--corr-sigma", nargs=3, type=float, default=None,
                    help="噪声空间相关核 (z,y,x)，体素；默认不给 = 独立噪声（S0a/S0b）")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--train-seed0", type=int, default=200)
    ap.add_argument("--eval-seed0", type=int, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.level
    if args.corr_sigma is not None:
        tag += "_corr" + "-".join(f"{v:g}" for v in args.corr_sigma)
    if args.noise_domain != "log":
        tag += "_" + args.noise_domain
    out = Path(args.out) if args.out else ROOT / "data/processed/v3_round2_s0" / tag
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)

    man = {"level": args.level, "sigma_eps": LEVEL_SIGMA[args.level],
           "noise_domain": args.noise_domain, "corr_sigma": args.corr_sigma,
           "psf_pd_sigma": PSF_PD.tolist(), "flow_contrast_db": FLOW_CONTRAST_DB,
           "sigma_source": "reports/v3_r06/v2_endpoint_audit.json q3_n2n -> sqrt(mse/2)",
           "train": [], "eval": []}
    t0 = time.time()
    for i in range(args.n_train):
        seed = args.train_seed0 + i
        d = build_one(seed, args.level, args.noise_domain, args.corr_sigma, with_second=False)
        npz = out / "train" / f"s0_{tag}_s{seed}.npz"
        np.savez_compressed(npz, **{k: d[k] for k in
                                    ("pd_linear", "gt_flow_pd", "gt_imaging_pd", "rho",
                                     "vessels", "valid", "s", "noise_sigma")})
        log1p = np.log1p(d["pd_linear"] / max(float(d["s"]), 1e-12)).astype(np.float32)
        log1p[~d["valid"]] = 0.0
        npy = out / "train" / f"s0_{tag}_s{seed}_log1p.npy"
        np.save(npy, log1p)
        man["train"].append({"seed": seed, "npz": str(npz.relative_to(ROOT)),
                             "log1p": str(npy.relative_to(ROOT)), "s": float(d["s"])})

    for i in range(args.n_eval):
        seed = args.eval_seed0 + i
        d = build_one(seed, args.level, args.noise_domain, args.corr_sigma, with_second=True)
        npz = out / "eval" / f"s0_{tag}_e{seed}.npz"
        np.savez_compressed(npz, **{k: d[k] for k in
                                    ("pd_linear", "pd2_linear", "gt_flow_pd", "gt_imaging_pd",
                                     "rho", "vessels", "valid", "s", "noise_sigma")})
        man["eval"].append({"seed": seed, "npz": str(npz.relative_to(ROOT)), "s": float(d["s"])})

    man["elapsed_s"] = round(time.time() - t0, 1)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"[{tag}] 训练 {len(man['train'])}、评估 {len(man['eval'])} -> {out}  "
          f"({man['elapsed_s']}s)  sigma={LEVEL_SIGMA[args.level]}")

    # 造完立刻自检：噪声档真的落在指定 sigma 上（否则后面所有结论都建立在一个错标定上）
    d = build_one(args.eval_seed0, args.level, args.noise_domain, args.corr_sigma, with_second=True)
    lg1 = np.log(np.maximum(d["pd_linear"], 1e-12))
    lg2 = np.log(np.maximum(d["pd2_linear"], 1e-12))
    empir = float(np.std((lg1 - lg2)[d["valid"]]) / np.sqrt(2.0))
    print(f"  自检：实测 log 域单次噪声 std = {empir:.4f}（目标 {LEVEL_SIGMA[args.level]:.4f}）")


if __name__ == "__main__":
    main()

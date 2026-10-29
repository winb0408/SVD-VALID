"""实验 A（外部审查 R01/R07 整改后的第一个必跑项）：恢复基础能力。

## 这一步要回答的唯一问题

**faithful VALID（逐块随机 Tetris、同 tet 配对）和 fixed-4 这两个真正"原版口径"的臂，
在**整卷滑窗推理**下，到底能不能超过"什么都不做"？**

Phase 3 的全部数字都不能回答这个问题，原因有两条，已在
`reports/v3_repair_phase0_baselines.md` §1 逐项核验：

1. Phase 3 的推理是 `scripts/v3_phase3_eval.py::reconstruct()`——8 个相位各前向一次、
   按格点回填（"相位回填"）。上游 VALID v1.0 的推理是 `reconstruct_dataset`：整卷前向、
   滑窗重叠加权。**两者不是同一件事**，Phase 3 测的是我们自己那套。
2. Phase 3 的基线臂 A 是 `--pairing cross`（互补 tet 配对），上游是 `--pairing within`
   （同 tet 的 2 角作输入、另外 2 角作目标）。**cross 配对上游没有。**

所以本脚本从零训两个真正 faithful 的臂，并且**只用 `infer.py`（整卷滑窗）推理**，
图注与 JSON 都写明 API。

## 评估协议（跑之前写死）

| 项目 | 设定 |
|---|---|
| 训练体 | `sim_seed0.npz` 的 `pd_linear` → `log1p(PD / median(PD[valid]))`（与 Phase 3 同一个体、同一个变换） |
| 评估体 | `sim_seed1.npz`（**另一个种子，血管几何不同**），裁到能被 2 整除的前缀 (36,64,64) |
| 共同分母 | **评估体的全部 valid 体素**（整卷推理，所有臂同一个分母，没有"覆盖 12.5%"这类折扣） |
| 参考真值 | `gt_flow_pd` 用**与输入同一个尺度** s 变换：`log1p(gt/s)`，s = median(pd_linear[valid]) |
| 网络/预算 | `NetworkCNR(features=16, groups=4)`；40 epoch × 100 step；patch 32×64×64；seed 3407/3408/3409 |
| 推理 | `infer.py --patch 32 64 64 --overlap 0.1`（子进程调用 CLI，不是内部函数） |
| 主终点 | **RMSE vs 真值（valid 内，log1p 域）**，门槛 = "低于输入自己的 RMSE" |
| 辅终点 | `r`（全 valid）、`r_ves`（血管膨胀 1 内）、`bg_std_ratio`（背景散斑标准差 / 输入，1=没压）、`sep_ratio`（对比度保留） |

**为什么不设"通过/不通过"的新门槛**：审查要求的是与既定 Gate 对照；这里唯一既定的比较是
"是否优于输入"，属于**同一分母上的直接比较**，不引入新阈值。真值来自仿真器，量纲已知。

**已知局限（必须随结果一起报）**：仿真器的血流项是 `gauss_field(psf) * vessels`，
不是"血管密度→PSF 卷积→IQ→SVD"的显式成像链（R06 未改），所以"真值"是**无噪成像真值**
的一个实现，不是几何真值；纹理与对比度水平受仿真器配置支配。

用法：
    python -u scripts/v3_expA_faithful.py --gpus 0 2
    python -u scripts/v3_expA_faithful.py --eval-only      # 只重跑评估（复用已有 ckpt）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PY = sys.executable

from v3_phase5_real_compare import wiener                          # noqa: E402

# 两条臂：都是 stride=1 + within 配对（= 上游口径），只差"每块重抽"还是"固定一个模式"
ARMS = {
    "A_rand":   ["--pairing", "within", "--mode", "random"],
    "A_fixed4": ["--pairing", "within", "--mode", "fixed", "--pattern", "4"],
}
WIENER_W = 7        # 与真实数据 ①Wiener w=7 同一个设定，便于横向对照
PATCH = [32, 64, 64]


def prep_volume(src: Path, dst: Path) -> Path:
    """仿真观测 PD → log1p(PD / median(PD[valid]))，valid 外置零。与 Phase 3 完全一致。"""
    d = np.load(src)
    pd = d["pd_linear"].astype(np.float32)
    valid = d["valid"].astype(bool)
    s = float(np.median(pd[valid]))
    x = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    x[~valid] = 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, x)
    return dst


def train(arm: str, seed: int, gpu: int, train_vol: Path, out_root: Path,
          epochs: int, steps: int, features: int) -> Path:
    out = out_root / f"{arm}_s{seed}"
    if (out / "latest.pt").exists():
        print(f"  [skip] {out} 已存在")
        return out
    cmd = [PY, "-u", "train.py", "--train-volume", str(train_vol),
           "--stride", "1", "--patch", *[str(p) for p in PATCH],
           "--epochs", str(epochs), "--steps-per-epoch", str(steps),
           "--features", str(features), "--seed", str(seed),
           "--out", str(out), *ARMS[arm]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log = out.with_suffix(".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{arm} seed {seed} 训练失败，见 {log}")
    print(f"  {arm} s{seed}: {time.time() - t0:.0f}s → {out}")
    return out


def infer(ckpt_dir: Path, vol_npy: Path, gpu: int) -> np.ndarray:
    """**只用 infer.py 的 CLI**，产出整卷重建。返回值就是它写盘的那个数组。"""
    dst = ckpt_dir / "infer"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "infer.py", "--volume", str(vol_npy),
           "--checkpoint", str(ckpt_dir / "latest.pt"),
           "--patch", *[str(p) for p in PATCH], "--overlap", "0.1",
           "--device", "cuda", "--output", str(dst)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"infer.py 失败：{ckpt_dir}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return np.load(dst / "denoised.npy").astype(np.float32)


def load_vol_and_truth(eval_sim: Path, pair_sim: Path):
    """读评估体 + 第二观测，返回 (vol, truth, valid, ves, s, vol2)。

    输入与真值**共用** `s = median(pd_linear[valid])`（R07）；第二观测也用同一个 s。
    抽成函数是为了让 `v3_expA_reconstruct_eval.py` 与主评价共用同一份加载/口径代码，
    避免两处各写一遍后悄悄漂移。
    """
    d = np.load(eval_sim)
    pd = d["pd_linear"].astype(np.float32)
    gt = d["gt_flow_pd"].astype(np.float32)
    vessels = d["vessels"].astype(bool)
    valid = d["valid"].astype(bool)
    s = float(np.median(pd[valid]))
    vol = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    vol[~valid] = 0.0
    truth = np.log1p(gt / max(s, 1e-12)).astype(np.float32)
    ves = vessel_mask_dilated(vessels, 1) & valid
    p2 = np.load(pair_sim)
    if not np.array_equal(p2["vessels"], vessels):
        raise SystemExit(f"{pair_sim} 的血管几何与 {eval_sim} 不同，不是同一场景的两次观测")
    vol2 = np.log1p(p2["pd_linear"].astype(np.float32) / max(s, 1e-12)).astype(np.float32)
    return vol, truth, valid, ves, s, vol2


def vessel_mask_dilated(vessels, it=1):
    from scipy.ndimage import binary_dilation
    return binary_dilation(vessels.astype(bool), iterations=it) if it > 0 else vessels.astype(bool)


def n2n_metrics(pred, obs1, obs2, valid, ves):
    """本项目既有的 N2N 口径（与 J1.2 的 N 项同一公式），分母是**两次独立观测**的互差。

    `sim_seed1` 与 `sim_seed1_n201_s1` 共享固定散斑（同一 static_seed 流位置）、
    只有逐次独立的散斑/杂波/热噪声/血流实现不同——正是 Phase 4 定义的"两次观测"。
    尺度必须**共用** obs1 的 s（R07 纪律），否则分子分母尺度不一致。
    """
    sel = valid
    den = float(np.mean((obs1[sel] - obs2[sel]) ** 2))
    num = float(np.mean((pred[sel] - obs2[sel]) ** 2))
    selv = sel & ves
    den_v = float(np.mean((obs1[selv] - obs2[selv]) ** 2))
    num_v = float(np.mean((pred[selv] - obs2[selv]) ** 2))
    return {"n2n_ratio": num / den if den else float("nan"),
            "n2n_ratio_ves": num_v / den_v if den_v else float("nan"),
            "n2n_den": den, "n2n_den_ves": den_v}


def metrics(pred, truth, valid, ves, ref_bg_std):
    """R06 审计的直接后果：真值**处处比观测暗得多**（背景精确为 0，血管内 0.89 vs 观测 2.20），
    所以**任何 RMSE 型指标都被"把输出缩小到 0"退化最小化**——实测全零输出的全局 RMSE
    0.109（输入 0.706）、血管内 RMSE 0.571（输入 0.979、最好的臂 0.751）。
    因此 `rmse` 与 `rmse_ves` 都**不得单独作为判据**，`zero` 基线是这条退化路线的已知答案对照。

    真正非退化的判读项只有两个：
      * `r_ves`（血管内 Pearson）——对整体缩放不变，常数输出无定义（nan）；
      * `n2n_ratio`（对第二次独立观测的误差 / 输入的同一误差）——输入恒为 1.0，
        全零输出是 266（灾难），所以它不会奖励"抹平"。
    `bg_std_ratio` 作为辅助（它同样把 0 当作最好，须与 `r_ves` 合读）。
    """
    sel = valid
    rmse = float(np.sqrt(np.mean((pred[sel] - truth[sel]) ** 2)))
    rmse_ves = float(np.sqrt(np.mean((pred[sel & ves] - truth[sel & ves]) ** 2)))
    bg_std = float(np.std(pred[sel & ~ves]))
    sep = float(np.mean(pred[sel & ves]) - np.mean(pred[sel & ~ves]))
    sep_t = float(np.mean(truth[sel & ves]) - np.mean(truth[sel & ~ves]))
    return {
        "rmse": rmse,
        "rmse_ves": rmse_ves,
        "r": float(np.corrcoef(pred[sel], truth[sel])[0, 1]),
        "r_ves": float(np.corrcoef(pred[sel & ves], truth[sel & ves])[0, 1]),
        "bg_std_ratio": bg_std / ref_bg_std if ref_bg_std else float("nan"),
        "bg_mean": float(np.mean(pred[sel & ~ves])),
        "sep_ratio": sep / sep_t if abs(sep_t) > 1e-9 else float("nan"),
        "n_eval": int(sel.sum()), "n_ves": int((sel & ves).sum()),
    }


def summarise(block):
    out = {}
    for arm, per in block.items():
        vals = list(per.values())
        out[arm] = {}
        for k in ("rmse", "rmse_ves", "r", "r_ves", "bg_std_ratio", "bg_mean", "sep_ratio",
                  "n2n_ratio", "n2n_ratio_ves"):
            v = np.array([x[k] for x in vals], float)
            out[arm][k] = {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)),
                           "max": float(np.nanmax(v))}
        out[arm]["n_seeds"] = len(vals)
    return out


def figure(out_dir, rec, imgs, vol, truth, wiener_img, ves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mip(x):
        p = np.expm1(np.maximum(x, 0))
        den = np.median(p[p > 0]) if (p > 0).any() else 1.0
        return 10 * np.log10(np.maximum(p.max(axis=0), 1e-9) / max(den, 1e-12))

    arms = sorted(rec["summary"])
    panels = [("input log1p(PD)", vol), ("truth (blood only)", truth),
              (f"① Wiener w={WIENER_W}", wiener_img)]
    for a in arms:
        seeds = sorted(rec["arms"][a], key=lambda z: rec["arms"][a][z]["n2n_ratio"])
        panels.append((f"{a} (infer.py)", imgs[(a, seeds[len(seeds) // 2])]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.05 * len(panels), 3.7))
    for ax, (name, img) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(mip(img), cmap="magma", vmin=-25, vmax=10, aspect="equal")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="dB re median")
    fig.suptitle("Experiment A — faithful VALID vs input vs Wiener "
                 "(held-out simulated volume, full-volume infer.py)", fontsize=10)
    fig.savefig(out_dir / "fig_expA_mip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    order = ["input", f"wiener_w{WIENER_W}", "zero"] + arms
    keys = [("r_ves", "Pearson r inside vessels — scale-invariant, non-degenerate"),
            ("n2n_ratio", "N2N vs 2nd observation / input  (input=1.0; 'all zeros'=266)"),
            ("bg_std_ratio", "background speckle std / input  (0 = fully suppressed)"),
            ("rmse", "global RMSE — degenerate: 'all zeros' wins, do NOT use")]
    colors = ["#999999", "#c00000", "#7f7f00"] + ["#4472c4"] * len(arms)
    for ax, (k, title) in zip(axes, keys):
        vals, err_lo, err_hi = [], [], []
        for a in order:
            if a in rec["summary"]:
                s_ = rec["summary"][a]
            else:                                  # 基线：单一实现，没有种子离散
                m = rec["baseline"][a]["mean"]
                s_ = {kk: {"mean": m[kk], "min": m[kk], "max": m[kk]} for kk in m}
            vals.append(s_[k]["mean"])
            err_lo.append(s_[k]["mean"] - s_[k]["min"])
            err_hi.append(s_[k]["max"] - s_[k]["mean"])
        ax.bar(range(len(order)), vals, yerr=[err_lo, err_hi], capsize=4, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("bars = mean over seeds, whiskers = min/max; "
                 "baselines (grey/red) have no seed spread", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_expA_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/fig_expA_mip.png  {out_dir}/fig_expA_metrics.png")


def ckpt_budget(ckpt: Path) -> dict:
    """从 checkpoint 里读出**实际**训练配置（`train.py` 会把 args 存进 latest.pt）。

    为什么需要它：`--eval-only` 时命令行上的 `--epochs/--steps` 只是"请求值"，
    忘了传就会把错误的预算写进 json（评测长预算时踩过这个坑，长跑 120×200 被记成 40×100）。
    以 checkpoint 为准，命令行值只作对照。缺 torch 或字段不全时返回 error，不阻断评测。
    """
    try:
        import torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    a = blob.get("args") or {}
    keep = ("epochs", "steps_per_epoch", "patch", "features", "pairing", "mode",
            "pattern", "stride", "seed", "weight_reg", "weight_identity")
    return {k: a[k] for k in keep if k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-sim", default="data/processed/v3_sim/sim_seed0.npz")
    ap.add_argument("--eval-sim", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--pair-sim", default="data/processed/v3_sim/sim_seed1_n201_s1.npz",
                    help="同一场景的**第二次独立观测**（共享固定散斑，static_seed=1），"
                         "用作 N2N 分母。换掉它就不是两次观测了。")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--infer-gpu", type=int, default=None,
                    help="推理用哪张卡（默认取 --gpus 的第一张）")
    ap.add_argument("--runs", default="runs/v3_expA")
    ap.add_argument("--out", default="reports/v3_expA")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.runs)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_gpu = args.gpus[0] if args.infer_gpu is None else args.infer_gpu

    prep = Path("data/processed/v3_sim/train_log1p.npy")
    prep_volume(Path(args.train_sim), prep)
    print(f"训练体 {prep}（log1p(PD/median)）")

    # ---------------------------------------------------------------- 训练
    ckpts = {}
    if not args.eval_only:
        jobs = [(a, s) for a in args.arms for s in args.seeds]
        gpu_of = {j: args.gpus[i % len(args.gpus)] for i, j in enumerate(jobs)}
        buckets: dict[int, list] = {}
        for j in jobs:
            buckets.setdefault(gpu_of[j], []).append(j)
        procs = []
        for gpu, items in buckets.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            lines = []
            for arm, seed in items:
                out = out_root / f"{arm}_s{seed}"
                if (out / "latest.pt").exists():
                    continue
                cmd = [PY, "-u", "train.py", "--train-volume", str(prep),
                       "--stride", "1", "--patch", *[str(p) for p in PATCH],
                       "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps),
                       "--features", str(args.features), "--seed", str(seed),
                       "--out", str(out), *ARMS[arm]]
                lines.append("cd %s && %s >> %s.log 2>&1" % (ROOT, " ".join(cmd), out))
            if not lines:
                continue
            script = out_root / f"runner_gpu{gpu}.sh"
            script.write_text("#!/bin/bash\nset -e\n" + "\n".join(lines) + "\n")
            script.chmod(0o755)
            procs.append((gpu, subprocess.Popen(
                ["bash", str(script)], cwd=ROOT, env=env,
                stdout=open(out_root / f"runner_gpu{gpu}.log", "w"),
                stderr=subprocess.STDOUT)))
        for gpu, p in procs:
            p.wait()
            print(f"GPU {gpu} 训练结束，返回码 {p.returncode}")
            if p.returncode:
                raise SystemExit(f"GPU {gpu} 有任务失败，见 {out_root}/runner_gpu{gpu}.log")
    for a in args.arms:
        for s in args.seeds:
            p = out_root / f"{a}_s{s}"
            if (p / "latest.pt").exists():
                ckpts[(a, s)] = p

    # ---------------------------------------------------------------- 评估体
    # 不裁体积（infer.py 自己补偶数再裁回）；加载与口径统一走 load_vol_and_truth
    vol, truth, valid, ves, s_eval, vol2 = load_vol_and_truth(
        ROOT / args.eval_sim, ROOT / args.pair_sim)
    shape = tuple(int(n) for n in vol.shape)
    ref_bg_std = float(np.std(vol[valid & ~ves]))

    vol_npy = out_dir / "eval_input_log1p.npy"
    np.save(vol_npy, vol)
    print(f"评估体 {args.eval_sim} shape={shape} valid={int(valid.sum())} "
          f"vessels={int((valid & ves).sum())} scale s={s_eval:.4g}")
    print(f"第二次观测 {args.pair_sim}（N2N 分母，共用同一 s）")

    rec = {
        "eval_sim": args.eval_sim, "train_sim": args.train_sim,
        "shape": list(shape), "valid_voxels": int(valid.sum()),
        "vessel_voxels_dilated": int((valid & ves).sum()),
        "scale_s_eval": s_eval, "bg_std_ref_input": ref_bg_std,
        "inference_api": "infer.py (full-volume sliding window, patch 32x64x64, overlap 0.1)",
        "budget_requested": {"epochs": args.epochs, "steps_per_epoch": args.steps,
                             "patch": PATCH, "features": args.features, "seeds": args.seeds},
        "arms_args": ARMS, "baseline": {}, "arms": {},
        "truth_definition": "gt_flow_pd (blood-only after band SVD [59,350)) transformed by "
                            "log1p(./s) with s = median(pd_linear[valid]) of the eval volume",
    }

    # ---------------------------------------------------------------- 基线
    w_out, _, sig2 = wiener(vol, WIENER_W, valid)
    zero_out = np.zeros_like(vol)      # R06 的退化控制：真值背景恒为 0 的直接后果
    rec["baseline"]["input"] = metrics(vol, truth, valid, ves, ref_bg_std)
    rec["baseline"][f"wiener_w{WIENER_W}"] = metrics(w_out, truth, valid, ves, ref_bg_std)
    rec["baseline"][f"wiener_w{WIENER_W}"]["sig2"] = float(sig2)
    rec["baseline"]["zero"] = metrics(zero_out, truth, valid, ves, ref_bg_std)
    rec["baseline"]["zero"]["note"] = ("已知答案的退化控制，不是候选方法。真值背景恒为 0 ⇒ "
                                       "全零输出在全局 rmse 上极低；它把 bg_std_ratio 压到 0，"
                                       "却把 rmse_ves 推到最大。任何'优于输入'只能靠 rmse_ves 判。")
    for name, arr in (("input", vol), (f"wiener_w{WIENER_W}", w_out), ("zero", zero_out)):
        rec["baseline"][name].update(n2n_metrics(arr, vol, vol2, valid, ves))
    for k, v in rec["baseline"].items():
        rec["baseline"][k] = {"mean": v, "min": v, "max": v, "n_seeds": 1}
    for name in ("input", f"wiener_w{WIENER_W}", "zero"):
        b = rec["baseline"][name]["mean"]
        print(f"{name:12s} rmse={b['rmse']:.4f}  rmse_ves={b['rmse_ves']:.4f}  "
              f"bg_std_ratio={b['bg_std_ratio']:.3f}  sep_ratio={b['sep_ratio']:.3f}  "
              f"n2n={b['n2n_ratio']:.4f}")

    # ---------------------------------------------------------------- 整卷推理 + 评分
    imgs = {}
    for (a, sd), ck in sorted(ckpts.items()):
        pred = infer(ck, vol_npy, infer_gpu)
        if pred.shape != vol.shape:
            raise RuntimeError(f"{ck}: infer.py 输出 {pred.shape} != {vol.shape}")
        m = metrics(pred, truth, valid, ves, ref_bg_std)
        m.update(n2n_metrics(pred, vol, vol2, valid, ves))
        rec["arms"].setdefault(a, {})[str(sd)] = m
        imgs[(a, str(sd))] = pred
        print(f"{a:10s} s{sd}  rmse_ves={m['rmse_ves']:.4f}  r_ves={m['r_ves']:.3f}  "
              f"bg_std_ratio={m['bg_std_ratio']:.3f}  n2n={m['n2n_ratio']:.4f}  "
              f"n2n_ves={m['n2n_ratio_ves']:.4f}  (global rmse={m['rmse']:.4f})")

    rec["summary"] = summarise(rec["arms"])
    # 预算以 checkpoint 里存的训练 args 为准（命令行值只是"请求值"，见 ckpt_budget 文档）
    rec["budget_checkpoints"] = {f"{a}_s{sd}": ckpt_budget(ck / "latest.pt")
                                 for (a, sd), ck in sorted(ckpts.items())}
    for name, b in rec["budget_checkpoints"].items():
        print(f"  [ckpt] {name}: epochs={b.get('epochs')} steps/epoch={b.get('steps_per_epoch')} "
              f"pairing={b.get('pairing')} mode={b.get('mode')} pattern={b.get('pattern')}")
    (out_dir / "expA_metrics.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    # ---------------------------------------------------------------- 结论行
    base = rec["baseline"]["input"]["mean"]
    print("\n=== 与输入的直接对照（同一 valid 分母、同一真值）===")
    print("非退化判读项：r_ves（对缩放不变）+ n2n_ratio（输入=1.0，全零=266）；"
          "rmse/rmse_ves 被'输出全零'退化最小化，不作判据。")
    for a in sorted(rec["summary"]):
        s_ = rec["summary"][a]
        rv = s_["r_ves"]["mean"]
        dr = rv - base["r_ves"]
        n2n = s_["n2n_ratio"]["mean"]
        verdict = []
        verdict.append("血管保真优于输入" if dr > 0 else "血管保真不如输入")
        verdict.append("噪声低于输入" if n2n < 1 else f"噪声高于输入 {n2n:.1f}×")
        print(f"{a:10s} Δr_ves={dr:+.3f}(r_ves={rv:.3f})  n2n={n2n:.3f}  "
              f"bg_std_ratio={s_['bg_std_ratio']['mean']:.3f}  → {' / '.join(verdict)}")
    if not args.skip_figure:
        figure(out_dir, rec, imgs, vol, truth, w_out, ves)


if __name__ == "__main__":
    main()

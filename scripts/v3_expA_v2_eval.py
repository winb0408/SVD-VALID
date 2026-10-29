"""实验 A/阶段 2 的评价：v2 前向模型的多体 + 上游预算的 checkpoint。

端点集合与门槛**先写**在 `reports/v3_expA_v2/prereg.md` §3.2/§3.3（写于训练结果之前）：

| 角色 | 端点 | 说明 |
|---|---|---|
| 主 | `n2n_ratio` | `mse(pred,obs2)/mse(obs1,obs2)`，valid，log1p；输入=1，全零=266 |
| 主 | `r_roi` | 血流足迹 ρ>0 内 pred vs 真值的 Pearson（对缩放不变） |
| 辅 | `r_core` | ρ≥0.5 内 Pearson |
| 辅 | `bg_std_ratio` | 背景（ρ=0）std / 输入背景 std |
| 对照 | `input` / `zero` / `wiener_w7` / `oracle_lowpass` | 已知答案；`zero` 用来复核"判据是否还退化" |

`rmse*` 仍然打印，但**按 R06 §2.1 的证伪记录不得单独作为判据**——它只用于机制判读。

R07 口径：输入、真值、第二观测**共用**同一个 `s = median(pd_linear[valid])`；
输入与真值都取 `log1p(·/s)`。v2 的 npz 里已经存了这个 `s`，本脚本读它，不重算。

用法
----
    python -u scripts/v3_expA_v2_eval.py --runs runs/v3_expA_v2_200k --gpu 0
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
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase5_real_compare import wiener  # noqa: E402
from v3_simulator import SimCfg                   # noqa: E402  # 只读 PSF 标定值

PY = sys.executable
PATCH = [32, 64, 64]
WIENER_W = 7


def load_eval(npz: Path) -> dict:
    d = np.load(npz)
    pd = d["pd_linear"].astype(np.float32)
    gt = d["gt_flow_pd"].astype(np.float32)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s = float(d["s"]) if "s" in d.files else float(np.median(pd[valid]))
    vol = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    truth = np.log1p(gt / max(s, 1e-12)).astype(np.float32)
    vol2 = np.log1p(d["pd2_linear"].astype(np.float32) / max(s, 1e-12)).astype(np.float32)
    for a in (vol, truth, vol2):
        a[~valid] = 0.0
    return {"vol": vol, "truth": truth, "vol2": vol2, "valid": valid, "rho": rho, "s": s,
            "roi": valid & (rho > 0.0),
            "core": valid & (rho >= 0.5),
            "bg": valid & (rho <= 0.0),
            "npz": str(npz.relative_to(ROOT)),
            # 输入本身与输入半分辨率（第二次观测）的背景 std 比，用于 bg_std_ratio 的分母
            "ref_bg_std": float(np.std(vol[valid & (rho <= 0.0)])),
            }


def metrics(pred: np.ndarray, E: dict) -> dict:
    v, roi, core, bg = E["valid"], E["roi"], E["core"], E["bg"]
    truth, o1, o2 = E["truth"], E["vol"], E["vol2"]

    def mse(a, b, m):
        return float(np.mean((a[m] - b[m]) ** 2))

    den, num = mse(o1, o2, v), mse(pred, o2, v)
    out = {
        "n2n_ratio": num / den if den else float("nan"),
        "n2n_ratio_roi": (mse(pred, o2, roi) / mse(o1, o2, roi)) if mse(o1, o2, roi) else float("nan"),
        "r_roi": float(np.corrcoef(pred[roi], truth[roi])[0, 1]) if roi.sum() > 2 else float("nan"),
        "r_core": float(np.corrcoef(pred[core], truth[core])[0, 1]) if core.sum() > 2 else float("nan"),
        "r_all": float(np.corrcoef(pred[v], truth[v])[0, 1]),
        "rmse": float(np.sqrt(mse(pred, truth, v))),
        "rmse_roi": float(np.sqrt(mse(pred, truth, roi))) if roi.any() else float("nan"),
        "bg_std_ratio": float(np.std(pred[bg])) / E["ref_bg_std"] if E["ref_bg_std"] else float("nan"),
        "bg_mean": float(np.mean(pred[bg])),
        "roi_mean": float(np.mean(pred[roi])) if roi.any() else float("nan"),
        "n_valid": int(v.sum()), "n_roi": int(roi.sum()), "n_core": int(core.sum()),
    }
    return out


def infer(ckpt_dir: Path, vol_npy: Path, gpu: int, tag: str) -> np.ndarray:
    """只用 `infer.py` 的 CLI（上游测试口径：满分辨率 patch + 重叠拼接）。"""
    dst = ROOT / "runs" / "_infer_v2" / f"{ckpt_dir.name}__{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "infer.py", "--volume", str(vol_npy),
           "--checkpoint", str(ckpt_dir / "latest.pt"),
           "--patch", *[str(p) for p in PATCH], "--overlap", "0.1",
           "--device", "cuda", "--output", str(dst)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"infer.py 失败：{ckpt_dir}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return np.load(dst / "denoised.npy").astype(np.float32)


def write_vol(x: np.ndarray, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x.astype(np.float32))
    return p


def ckpt_budget(ckpt: Path) -> dict:
    """从 checkpoint 里读**实际**训练配置（避免命令行请求值与实际值不符，见 expA §4）。"""
    try:
        import torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:                                    # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    a = blob.get("args") or {}
    keep = ("epochs", "steps_per_epoch", "patch", "features", "pairing", "mode",
            "pattern", "stride", "seed", "weight_reg", "clip_grad", "lr_schedule",
            "train_volumes", "train_volume", "arch_head", "zero_init_final",
            "arm_label")
    return {k: a[k] for k in keep if k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/v3_expA_v2_200k")
    ap.add_argument("--manifest", default="data/processed/v3_sim_v2/manifest.json")
    ap.add_argument("--arms", nargs="+", default=["A_rand", "A_fixed4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="reports/v3_expA_v2/metrics.json")
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text())
    eval_npz = [ROOT / e["npz"] for e in man["eval"]]
    print(f"评估体：{[p.name for p in eval_npz]}")
    Es = [load_eval(p) for p in eval_npz]

    rec: dict = {
        "meta": {
            "prereg": "reports/v3_expA_v2/prereg.md §3.2/§3.3（写于训练结果之前）",
            "eval_volumes": [e["npz"] for e in Es],
            "inference_api": "infer.py（满分辨率 patch + overlap 0.1 拼接；上游 test_pipeline 口径）",
            "scale_rule": "输入/真值/第二观测共用 s = median(pd_linear[valid])（R07）",
            "arms": args.arms, "seeds": args.seeds,
            "controls": ["input", "zero", "wiener_w7", "oracle_lowpass"],
        },
        "controls": {}, "arms": {},
    }

    # ---------------- 四条已知答案对照 ----------------
    sig = SimCfg().psf_sigma
    for tag, fn in [
        ("input", lambda E: E["vol"]),
        ("zero", lambda E: np.zeros_like(E["vol"])),
        ("wiener_w7", lambda E: wiener(E["vol"], WIENER_W, E["valid"])[0]),
        # oracle_lowpass 是"只按标定 PSF 糊一下"的已知答案：它不是去噪器，用来回答
        # "一个纯低通能不能拿到同样的结构相关"——若能，网络的增益就不能归因于学习。
        ("oracle_lowpass", lambda E: gaussian_filter(E["vol"], sigma=sig, mode="nearest")),
    ]:
        per = {}
        for E in Es:
            name = Path(E["npz"]).stem
            per[name] = metrics(fn(E).astype(np.float32), E)
        rec["controls"][tag] = per
        print(f"\n[{tag}] " + "  ".join(
            f"{k}: n2n={m['n2n_ratio']:.3f} r_roi={m['r_roi']:.4f} bg={m['bg_std_ratio']:.3f}"
            for k, m in per.items()))

    # ---------------- 网络臂 ----------------
    for arm in args.arms:
        for seed in args.seeds:
            ck = ROOT / args.runs / f"{arm}_s{seed}"
            if not (ck / "latest.pt").exists():
                print(f"[skip] {ck} 不存在")
                continue
            per = {}
            for E in Es:
                name = Path(E["npz"]).stem
                vp = write_vol(E["vol"], ROOT / "runs" / "_infer_v2" / "vols" / f"{name}_in.npy")
                t0 = time.time()
                pred = infer(ck, vp, args.gpu, name)
                per[name] = metrics(pred, E)
                per[name]["infer_seconds"] = round(time.time() - t0, 1)
                per[name]["out_mean"] = float(np.mean(pred[E["valid"]]))
                print(f"  [{arm} s{seed}] {name}: n2n={per[name]['n2n_ratio']:.3f} "
                      f"r_roi={per[name]['r_roi']:.4f} bg={per[name]['bg_std_ratio']:.3f} "
                      f"({per[name]['infer_seconds']}s)")
            rec["arms"][f"{arm}_s{seed}"] = {
                "budget_checkpoints": ckpt_budget(ck / "latest.pt"), "per_volume": per}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

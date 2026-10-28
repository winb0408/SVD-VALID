"""S0 正控制的评价：已知 PSF + 独立加性噪声（log 域）的 3D 血管 phantom。

与实验 A 评价脚本的**唯一**区别（也是审查 0.5 要求的）：
主终点 `r_roi` 比的对象是 **`gt_imaging_pd`（无噪完整成像期望）**，不是血流分量真值。
血流分量真值仍然算，但降级为参考量 `r_flow`。

端点（与 `reviews/V3_PR1_round2_plan.md` §3.2 一致，门槛写在该文件，本脚本不改门槛）：
    n2n_ratio   mse(pred, obs2)/mse(obs1, obs2)，主
    r_roi       ρ>0 内 pred vs gt_imaging 的 Pearson，主
    r_flow      ρ>0 内 pred vs gt_flow 的 Pearson，参考
    bg_std_ratio 背景 std / 输入背景 std，辅

用法：
    python -u scripts/v3_round2_s0_eval.py --level low --runs runs/v3_round2_s0/low
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import infer, metrics, write_vol, ckpt_budget   # noqa: E402
from v3_phase5_real_compare import wiener                            # noqa: E402
from v3_round2_s0_dataset import PSF_PD                              # noqa: E402

PATCH = [32, 64, 64]
WIENER_W = 7


def load_eval(npz: Path) -> dict:
    d = np.load(npz)
    valid = d["valid"].astype(bool)
    rho = d["rho"].astype(np.float32)
    s = float(d["s"])
    lg = lambda a: np.log1p(np.maximum(a.astype(np.float32), 0.0) / max(s, 1e-12))
    vol, vol2 = lg(d["pd_linear"]), lg(d["pd2_linear"])
    truth = lg(d["gt_imaging_pd"])
    flow = lg(d["gt_flow_pd"])
    for a in (vol, vol2, truth, flow):
        a[~valid] = 0.0
    return {"vol": vol, "vol2": vol2, "truth": truth, "flow": flow,
            "valid": valid, "rho": rho, "s": s, "npz": str(npz.relative_to(ROOT)),
            "roi": valid & (rho > 0.0), "core": valid & (rho >= 0.5),
            "bg": valid & (rho <= 0.0),
            "ref_bg_std": float(np.std(vol[valid & (rho <= 0.0)])),
            "sigma_eps": float(d["noise_sigma"])}


def _freq_wiener(E: dict) -> np.ndarray:
    """最优平稳线性去噪（oracle）：S_xx 用真值估，S_nn 用 (obs1−obs2)/2 估。

    只在评价里当"上界参照"用——它看了真值和第二次观测，不是任何可实现方法。
    意义：确认这个任务**有**收益空间。若它明显优于输入而网络远差于输入，
    说明瓶颈在训练/目标/推理链，而不在数据。
    """
    v = E["valid"].astype(np.float32)
    F = lambda a: np.fft.fftn(a.astype(np.float64))
    sxx = np.abs(F(E["truth"] * v)) ** 2
    snn = np.abs(F((E["vol"] - E["vol2"]) * v)) ** 2 / 2.0
    w = sxx / (sxx + snn + 1e-12)
    return np.real(np.fft.ifftn(F(E["vol"] * v) * w)).astype(np.float32)


def metrics_s0(pred: np.ndarray, E: dict) -> dict:
    m = metrics(pred, E)
    roi, flow = E["roi"], E["flow"]
    m["r_flow"] = float(np.corrcoef(pred[roi], flow[roi])[0, 1]) if roi.sum() > 2 else float("nan")
    m["rmse_vs_imaging"] = float(np.sqrt(np.mean((pred[E["valid"]] - E["truth"][E["valid"]]) ** 2)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--arms", nargs="+", default=["A_rand", "A_fixed4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = Path(args.runs or f"runs/v3_round2_s0/{args.level}")
    man_p = Path(args.manifest or f"data/processed/v3_round2_s0/{args.level}/manifest.json")
    man = json.loads((ROOT / man_p).read_text())
    Es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]
    print(f"[{args.level}] 评估体 {[Path(E['npz']).stem for E in Es]}")

    rec = {"meta": {"level": args.level, "manifest": str(man_p),
                    "sigma_eps": man["sigma_eps"], "noise_domain": man["noise_domain"],
                    "corr_sigma": man["corr_sigma"],
                    "truth": "gt_imaging_pd（无噪完整成像期望），主终点 r_roi 用它",
                    "prereg": "reviews/V3_PR1_round2_plan.md §3.2（写在训练启动之前）",
                    "arms": args.arms, "seeds": args.seeds},
           "controls": {}, "arms": {}}

    for tag, fn in [
        ("input", lambda E: E["vol"]),
        ("zero", lambda E: np.zeros_like(E["vol"])),
        # 上界：直接把真值当预测。n2n 的理论下限就是它，任何方法高于它就是没赚到。
        ("oracle_truth", lambda E: E["truth"]),
        ("wiener_w7", lambda E: wiener(E["vol"], WIENER_W, E["valid"])[0]),
        ("oracle_lowpass", lambda E: gaussian_filter(E["vol"], sigma=PSF_PD, mode="nearest")),
        # 最优**平稳线性**滤波的上界：频域 Wiener 用真值估 S_xx、用两次观测之差估 S_nn。
        # 它比 oracle_truth 松，但比任何"手调平滑"紧得多，是判断"这任务到底能不能做"的参照：
        # 若它明显好于输入而网络远差于输入，那问题在训练，不在数据。
        ("oracle_freq_wiener", lambda E: _freq_wiener(E)),
    ]:
        per = {Path(E["npz"]).stem: metrics_s0(fn(E).astype(np.float32), E) for E in Es}
        rec["controls"][tag] = per
        print(f"[{tag}] " + "  ".join(
            f"{k}: n2n={m['n2n_ratio']:.3f} r_roi={m['r_roi']:.4f} bg={m['bg_std_ratio']:.3f}"
            for k, m in per.items()))

    for arm in args.arms:
        for seed in args.seeds:
            ck = ROOT / runs / f"{arm}_s{seed}"
            if not (ck / "latest.pt").exists():
                print(f"[skip] {ck} 不存在")
                continue
            # **只评跑完的 run**：latest.pt 每轮都会落盘，`train.py` 只在结尾写 history.json。
            # 拿一个跑到一半的 checkpoint 去评，会得到"看起来正常"的数（2026-10-27 21:44 实际发生过：
            # 一个 12/40 的 run 被评出了 n2n 1.42/1.85，若不留神就会当成最终结果发布）。
            if not (ck / "history.json").exists():
                print(f"[skip] {ck} 尚未跑完（无 history.json），不评半成品")
                rec.setdefault("skipped_incomplete", []).append(f"{arm}_s{seed}")
                continue
            per = {}
            for E in Es:
                name = Path(E["npz"]).stem
                vp = write_vol(E["vol"], ROOT / "runs/_infer_v2" / "vols" / f"s0_{name}_in.npy")
                t0 = time.time()
                pred = infer(ck, vp, args.gpu, name).astype(np.float32)
                per[name] = metrics_s0(pred, E)
                per[name]["infer_seconds"] = round(time.time() - t0, 1)
                print(f"  [{arm} s{seed}] {name}: n2n={per[name]['n2n_ratio']:.3f} "
                      f"r_roi={per[name]['r_roi']:.4f} bg={per[name]['bg_std_ratio']:.3f}")
            rec["arms"][f"{arm}_s{seed}"] = {
                "budget_checkpoints": ckpt_budget(ck / "latest.pt"), "per_volume": per}

    out = ROOT / (args.out or f"reports/v3_round2/s0_{args.level}_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Round3 A5：**平凡基线**——不做任何学习，只对输入做平滑/中值，看它们的 n2n。

为什么必须做这一步：A4 给出的"误差地板 ≈0.02、与噪声无关"说明网络的错不在噪声残留上。
但"比不动更差"这个说法只有在**平凡方法**也不比它差时才真正有杀伤力；
反过来，如果连一个高斯模糊都能把 n2n 压到 0.8 以下，那么后续任何新架构都必须先打赢这条线，
而不是打赢 identity。

判读（**跑之前**写下）：
1. 若某档存在一个固定平滑尺度 σ*（在 e900/e901 上选定、不 per-scene 挑），使 6 个场景的
   n2n 均值 < 网络当前的 n2n 均值 ⇒ 报告"网络劣于平凡平滑"，并把它写进 B 段的对照线。
2. 若平凡平滑最好也只是 ≈1.0 ⇒ 说明"在这套判据下根本没有免费的午餐"，
   网络打不过 identity 这件事不能全算在架构头上，判据本身对结构保真度更敏感（要单独说明）。
3. per-scene oracle 的 σ 与固定 σ 的差距 = "挑 σ 的乐观量"，必须与 A2 的选择乐观一起报。

用法
----
    python -u scripts/v3_round3_a5_trivial_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import load_eval, metrics  # noqa: E402

# vlow 档的最优 σ 落在网格下边界（0.4）上，说明还要更细；加 0.2/0.3 后再跑一次。
SIGMAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
CAL = (0, 1)  # e900 / e901 用来选 σ*，其余场景只用来评


def blur(vol: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """只在 valid 内部平滑（用 masked 归一化，避免体边界把 0 拉进来）。"""
    m = valid.astype(np.float32)
    num = gaussian_filter(vol * m, sigma, mode="nearest")
    den = gaussian_filter(m, sigma, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), vol)
    out[~valid] = 0.0
    return out.astype(np.float32)


def medfilt(vol: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = median_filter(vol, size=3).astype(np.float32)
    out[~valid] = 0.0
    return out


def run_level(level: str, man_dir: str, model_n2n: list[float] | None) -> dict:
    man = json.loads((ROOT / man_dir / "manifest.json").read_text())
    Es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]
    names = [Path(E["npz"]).stem for E in Es]
    print(f"[{level}] {names}")

    per = {}
    for n, E in zip(names, Es):
        row = {"identity": metrics(E["vol"], E)["n2n_ratio"]}
        for s in SIGMAS:
            row[f"blur{s:g}"] = metrics(blur(E["vol"], E["valid"], s), E)["n2n_ratio"]
        row["median3"] = metrics(medfilt(E["vol"], E["valid"]), E)["n2n_ratio"]
        per[n] = row
        print("  " + n + " " + " ".join(f"{k}={v:.3f}" for k, v in row.items()))

    # 固定 σ*：只在标定场景上平均最小
    cal_mean = {s: float(np.mean([per[names[i]][f"blur{s:g}"] for i in CAL])) for s in SIGMAS}
    star = min(cal_mean, key=cal_mean.get)
    fixed = {n: per[n][f"blur{star:g}"] for n in names}
    oracle = {n: min(per[n][f"blur{s:g}"] for s in SIGMAS) for n in names}
    oracle_sig = {n: min(SIGMAS, key=lambda s: per[n][f"blur{s:g}"]) for n in names}

    out = {
        "level": level, "manifest": f"{man_dir}/manifest.json", "volumes": names,
        "per_scene": per,
        "cal": {"scenes": [names[i] for i in CAL],
                "cal_mean_by_sigma": {f"{s:g}": v for s, v in cal_mean.items()},
                "sigma_star": star},
        "fixed_sigma_star": {"sigma": star, "per_scene": fixed,
                             "mean": float(np.mean(list(fixed.values())))},
        "per_scene_oracle": {"per_scene": oracle, "sigma": {k: float(v) for k, v in oracle_sig.items()},
                             "mean": float(np.mean(list(oracle.values())))},
        "median3_mean": float(np.mean([per[n]["median3"] for n in names])),
        "identity_mean": float(np.mean([per[n]["identity"] for n in names])),
    }
    if model_n2n is not None:
        out["model"] = {"per_scene": dict(zip(names, model_n2n)),
                        "mean": float(np.mean(model_n2n))}
    print(f"  -> identity {out['identity_mean']:.3f} | blur σ*={star:g} "
          f"{out['fixed_sigma_star']['mean']:.3f} | per-scene oracle "
          f"{out['per_scene_oracle']['mean']:.3f} | median3 {out['median3_mean']:.3f}"
          + (f" | model {out['model']['mean']:.3f}" if model_n2n else ""))
    return out


def model_n2n_from(path: str) -> list[float]:
    r = json.loads((ROOT / path).read_text())
    out = []
    for k in sorted(r["A2"]):
        rows = r["A2"][k]["arms"]["trained"]["rows"]
        out.append(float([x for x in rows if abs(x["alpha"] - 1.0) < 1e-9][0]["n2n"]))
    return out


def main() -> int:
    rec = {"prereg": "本文件 docstring 的三条判读（写在运行之前）", "sigmas": SIGMAS}
    rec["vlow6"] = run_level("vlow", "data/processed/v3_round2_s0/vlow6",
                             model_n2n_from("reports/v3_round3/a_probes_vlow6.json"))
    rec["low6"] = run_level("low", "data/processed/v3_round2_s0/low6",
                            model_n2n_from("reports/v3_round3/a_probes_crosslevel_vlowmodel_low6.json"))
    out = ROOT / "reports/v3_round3/a5_trivial_baselines.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

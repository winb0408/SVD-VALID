"""实验 A（阶段 2）的结果图：MIP 对比面板 + 端点条形图。

只读两样东西，不训练、不重跑推理：
1. `reports/v3_expA_v2/metrics.json`（`v3_expA_v2_eval.py` 的产物，含 4 条已知答案对照）；
2. `runs/_infer_v2/{ckpt}__{volume}/denoised.npy`（评价时 `infer.py` 留下的缓存卷）。

若缓存卷不存在，脚本**不**去重新推理（避免半夜偷偷占卡），而是把该面板画成"缺失"并打印提示。

为什么图要单独写一个脚本：预注册里的证据是数字，但"结果对比图"是验收时要一眼看的东西。
图只做**呈现**，不引入任何新端点——所有数字都来自 metrics.json，不在这里重算。

用法
----
    python -u scripts/v3_expA_v2_figure.py --metrics reports/v3_expA_v2/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import load_eval  # noqa: E402


def mip_db(x: np.ndarray) -> np.ndarray:
    """log1p 域卷 → MIP 的 dB 图（以中位非零值为 0 dB）。只用于显示。"""
    p = np.expm1(np.maximum(x, 0))
    den = np.median(p[p > 0]) if (p > 0).any() else 1.0
    return 10 * np.log10(np.maximum(p.max(axis=0), 1e-9) / max(den, 1e-12))


def find_pred(arm_seed: str, vol_name: str) -> np.ndarray | None:
    p = ROOT / "runs" / "_infer_v2" / f"{arm_seed}__{vol_name}" / "denoised.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def median_seed(rec: dict, arm: str, vol: str) -> str | None:
    """按该体上的 n2n_ratio 取中位那个种子（不挑最好的，避免"用图说话"）。"""
    cells = []
    for key, r in (rec.get("arms") or {}).items():
        if key.startswith(arm + "_s") and vol in (r.get("per_volume") or {}):
            cells.append((r["per_volume"][vol]["n2n_ratio"], key))
    if not cells:
        return None
    cells.sort()
    return cells[len(cells) // 2][1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/v3_expA_v2/metrics.json")
    ap.add_argument("--manifest", default="data/processed/v3_sim_v2/manifest.json")
    ap.add_argument("--out", default="reports/v3_expA_v2")
    args = ap.parse_args()

    rec = json.loads((ROOT / args.metrics).read_text())
    man = json.loads((ROOT / args.manifest).read_text())
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = sorted({k.rsplit("_s", 1)[0] for k in (rec.get("arms") or {})})

    for e in man["eval"]:
        npz = ROOT / e["npz"]
        E = load_eval(npz)
        name = npz.stem
        # 对照 Wiener/oracle 的**图**没有缓存，这里直接用同一实现重算（纯 CPU，便宜，且不改端点）
        from scipy.ndimage import gaussian_filter
        from v3_phase5_real_compare import wiener
        from v3_simulator import SimCfg
        panels = [("input  log1p(PD)", E["vol"]), ("truth (blood only)", E["truth"]),
                  ("① Wiener w=7", wiener(E["vol"], 7, E["valid"])[0]),
                  ("oracle lowpass (PSF)", gaussian_filter(E["vol"], sigma=SimCfg().psf_sigma,
                                                           mode="nearest"))]
        missing = []
        for a in arms:
            key = median_seed(rec, a, name)
            img = find_pred(key, name) if key else None
            if img is None:
                missing.append(f"{key or a}__{name}")
                img = np.full_like(E["vol"], np.nan)
            panels.append((f"{a}\n({key}, median seed)" if key else a, img))

        fig, axes = plt.subplots(1, len(panels), figsize=(3.05 * len(panels), 3.9))
        for ax, (title, img) in zip(np.atleast_1d(axes), panels):
            im = ax.imshow(mip_db(img), cmap="magma", vmin=-25, vmax=10, aspect="equal")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=np.atleast_1d(axes), shrink=0.7, label="dB re median")
        fig.suptitle(f"Experiment A / v2 — {name}  (MIP, full-volume infer.py; "
                     f"arms shown at the median seed by n2n)", fontsize=10)
        p = out_dir / f"fig_expA_v2_mip_{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")
        if missing:
            print(f"  [cache miss] {missing}  (did NOT re-run inference; run v3_expA_v2_eval.py first)")

    # ---------------- 端点条形图（两条主端点，跨体、含种子散点） ----------------
    vol_names = [Path(e["npz"]).stem for e in man["eval"]]
    order = ["input", "wiener_w7", "oracle_lowpass"] + arms
    label = {"input": "① input", "wiener_w7": "① Wiener w=7",
             "oracle_lowpass": "oracle lowpass", "zero": "zero (degenerate ctrl)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (key, title, better) in zip(axes, [
            ("n2n_ratio", "n2n_ratio  (input=1.0; lower is better; zero≈315)",
             "lower"),
            ("r_roi", "Pearson r inside ROI ρ>0  (higher is better)", "higher")]):
        xs, means, errs, cols = [], [], [], []
        for i, a in enumerate(order):
            vals = []
            for v in vol_names:
                if a in rec["controls"]:
                    m = rec["controls"][a].get(v, {}).get(key)
                    if m is not None and m == m:
                        vals.append(float(m))
                else:
                    for k, r in (rec.get("arms") or {}).items():
                        if k.startswith(a + "_s") and v in (r.get("per_volume") or {}):
                            m = r["per_volume"][v].get(key)
                            if m is not None and m == m:
                                vals.append(float(m))
            if not vals:
                continue
            xs.append(i); means.append(float(np.mean(vals)))
            errs.append([float(np.mean(vals) - np.min(vals)), float(np.max(vals) - np.mean(vals))])
            cols.append("#999999" if a in rec["controls"] else "#4472c4")
        if xs:
            ax.bar(xs, means, yerr=np.array(errs).T, capsize=4, color=cols)
            ax.set_xticks(xs)
            ax.set_xticklabels([label.get(order[i], order[i]) for i in xs], rotation=20,
                               ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("bars = mean over (eval volume x seed); whiskers = min/max; "
                 "all numbers read from metrics.json, nothing recomputed here", fontsize=9)
    fig.tight_layout()
    p = out_dir / "fig_expA_v2_metrics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

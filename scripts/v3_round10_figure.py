#!/usr/bin/env python3
"""Round10 图：真实采集上**训练后**的对照。

两张图，用途不同，不许混读（预注册 §7）：

1. `held_out_4col_{seq}.png` —— **留出半**上的四列（输入 / 高斯 / SN2N / 本方法）。
   模型只在另一半上训练过，所以这是**与 §4 数字同源**的那张图，看的是"训练后还行不行"。

2. `deploy_5col_{seq}.png` —— **全 400 帧**、全卷训练的部署图（输入 / 高斯 / SN2N / A / B）。
   这是"这笔数据最终去噪成什么样"，**不参与任何判定**：全卷训练没有留出半，
   它在自己的训练数据上出图，必然偏乐观。

显示规则（与 L88 的补记检查同一套，不逐图自动拉伸）：
    db = 10log10(max(PD,1e-12) / refmax)，refmax = **该图输入列的 mask 内 PD 最大值**，
    同一张图的所有列共用这一个 refmax 与固定范围 [-25, 0] dB。
    因此同一张图内列与列可以直接比亮度；不同图之间不能比。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import fwd                     # noqa: E402
from v3_round4_metrics import masked_gauss           # noqa: E402
from v3_round10_eval import B1, NORM, dense_infer, load_half, to_pd   # noqa: E402

VMIN, VMAX = -25.0, 0.0
OUT = ROOT / "reports/v3_round10/figs"
RUNS = ROOT / "runs/v3_round10"
HALF = ROOT / "data/processed/v3_round6"
FULL = ROOT / "data/processed/v3_round10"


def to_db(pd: np.ndarray, refmax: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(pd, 1e-12) / refmax)


def gauss_pd(pd_in, valid, s, sigma):
    lg = fwd("log1p", pd_in, s, valid)
    return np.maximum(np.expm1(masked_gauss(lg, valid, sigma)), 0.0) * s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["held_out", "deploy", "both"], default="both")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scores = json.loads((ROOT / "reports/v3_round10/scores.json").read_text())
    sel = json.loads((ROOT / "reports/v3_round10/selections.json").read_text())

    for seq in ("SA64", "HSA64"):
        if args.which in ("held_out", "both"):
            th, eh = "odd", "even"
            pd_tr, valid, s = load_half(seq, th)
            ref = scores["references"][f"{seq}_{th}2{eh}"]
            sig = float(ref["gauss"]["best_by_x"])
            arm = min(sel[seq]["arms"], key=lambda a: sel[seq]["arms"][a]["x"])
            run = RUNS / sel[seq]["arms"][arm]["run"]
            dom = sel[seq]["arms"][arm]["domain"]
            z = np.load(B1 / dom / f"{seq}_{th}_x.npy")
            pd_ours = to_pd(dom, dense_infer(run / "latest.pt", z), s)
            pd_sn2n = np.load(RUNS / f"sn2n_{seq}_{th}/pd_sn2n_linear.npy").astype(np.float64)

            cols = [("输入（SVD 带通，另一半帧）", pd_tr),
                    (f"高斯 σ={sig}", gauss_pd(pd_tr, valid, s, sig)),
                    ("SVD-SN2N（同一半卷训练）", pd_sn2n),
                    (f"本方法 {arm}（同一半卷训练）", pd_ours)]
            refmax = float(pd_tr[valid].max())
            fig, axr = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6))
            # 逐列画：每列是自己的 PD 体，但共用同一个 refmax（所以列间亮度可比）
            for i, (t, pd) in enumerate(cols):
                db = to_db(pd, refmax)
                axr[0][i].imshow(db.max(axis=0), cmap="hot", vmin=VMIN, vmax=VMAX)
                axr[1][i].imshow(db[db.shape[0] // 2], cmap="hot", vmin=VMIN, vmax=VMAX)
                axr[0][i].set_title(t, fontsize=7)
                for r in (0, 1):
                    axr[r][i].set_xticks([]); axr[r][i].set_yticks([])
            axr[0][0].set_ylabel("MIP（沿 z）", fontsize=7)
            axr[1][0].set_ylabel("z 中位切片", fontsize=7)
            fig.suptitle(f"{seq} · 留出半（{th} 训练 → {eh} 显示）· 共用 dB 参考 = "
                         f"输入 PD max = {refmax:.3g} · [{VMIN:.0f}, {VMAX:.0f}] dB", fontsize=8)
            fig.tight_layout()
            p = OUT / f"held_out_4col_{seq}.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            print(f"wrote {p}")

        if args.which in ("deploy", "both"):
            pd_full = np.load(FULL / seq / "pd_full.npy").astype(np.float64)
            valid_f = np.load(HALF / seq / "valid_mask.npy").astype(bool)
            s = float(json.loads((HALF / seq / "metadata.json").read_text())["s"])
            sig = float(scores["references"][f"{seq}_odd2even"]["gauss"]["best_by_x"])
            cols = [("输入（SVD 带通，400 帧）", pd_full),
                    (f"高斯 σ={sig}", gauss_pd(pd_full, valid_f, s, sig)),
                    ("SVD-SN2N（全卷训练）",
                     None)]
            sn7 = ROOT / f"data/processed/v3_round7/{seq}/pd_sn2n_db.npy"
            if sn7.exists():
                dbv = np.load(sn7).astype(np.float64)
                cols[2] = (cols[2][0], pd_full[valid_f].max() * 10.0 ** (dbv / 10.0))
            for arm in ("A", "B"):
                b = sel[seq]["arms"][arm]
                z = np.load(B1 / b["domain"] / f"{seq}_full_x.npy")
                cols.append((f"本方法 {arm}（全卷训练, {b['domain']} e{b['epochs']}）",
                             to_pd(b["domain"], dense_infer(RUNS / f"deploy_{seq}_{arm}_full/latest.pt", z), s)))
            cols = [c for c in cols if c[1] is not None]
            refmax = float(pd_full[valid_f].max())
            fig, axr = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6))
            for i, (t, pd) in enumerate(cols):
                db = to_db(pd, refmax)
                axr[0][i].imshow(db.max(axis=0), cmap="hot", vmin=VMIN, vmax=VMAX)
                axr[1][i].imshow(db[db.shape[0] // 2], cmap="hot", vmin=VMIN, vmax=VMAX)
                axr[0][i].set_title(t, fontsize=7)
                for r in (0, 1):
                    axr[r][i].set_xticks([]); axr[r][i].set_yticks([])
            axr[0][0].set_ylabel("MIP（沿 z）", fontsize=7)
            axr[1][0].set_ylabel("z 中位切片", fontsize=7)
            fig.suptitle(f"{seq} · 全卷部署（400 帧，**无留出，非判定用**）· 共用 dB 参考 = "
                         f"{refmax:.3g} · [{VMIN:.0f}, {VMAX:.0f}] dB", fontsize=8)
            fig.tight_layout()
            p = OUT / f"deploy_5col_{seq}.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Round10 留出半四列图（预注册 §7 承诺的第二张图）。

与 `v3_round10_figure_4col.py`（五列部署图）的区别，必须说清：

| | 五列部署图 `final_5col_*` | 本图 `held_out_4col_*` |
|---|---|---|
| 训练数据 | **全 400 帧**（含本图每一帧） | 只有 **A 半**（200 帧） |
| 展示的半卷 | 全卷 | **A 半**（= 被评分的那一半） |
| 色标 | 每列独立 | **全图共用一个 dB 参考**（输入 PD 的 mask 内最大值） |
| 与 §4 数字的关系 | 无（只展示） | **同源**：就是 §4 里那个 `X_RMSE` 的被评对象 |
| 列 | C1..C5 五列 | C1 输入 / C2 高斯 / C3 SN2N / C4 本方法 |

因此：**要判断谁好，看本图；要看部署效果，看五列图。**
对齐的配置来自 `reports/v3_round10/selections.json` 与方向 2（even→odd）的确认 run，
σ 取方向 1 选中的那个，SN2N 取同一半卷 200 epoch 的输出——与 `v3_round10_eval.py` 的
`references()` 逐字同源，不另算一套。

产物
----
    reports/v3_round10/figs/held_out_4col_{seq}.png
    reports/v3_round10/display_clim_heldout.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import font_manager                   # noqa: E402
import numpy as np                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_round4_b1_prep import fwd                       # noqa: E402
from v3_round4_metrics import masked_gauss              # noqa: E402
from v3_round10_eval import (RUNS, dense_infer, load_half, to_pd)   # noqa: E402

OUT = ROOT / "reports/v3_round10/figs"
VOXEL = 0.15
AXIS_X = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Y = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Z = np.arange(2.025, 7.4525 + 1e-9, VOXEL)
FLOOR_DB = -60.0
CLIM_PCT = 2.0
# 方向 2 = even 半训练 → 去噪 even → 与 odd 比（预注册 §3）。图展示被评的那一半。
TH, EH = "even", "odd"


def pick_font() -> None:
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "SimHei", "Source Han Sans SC"):
        if any(x.name == f for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def to_db_shared(pd: np.ndarray, ref: float) -> np.ndarray:
    """**共用** dB 参考：分母固定为参考列的 mask 内最大值，不是各列自己的最大值。"""
    db = 10.0 * np.log10(np.maximum(pd, 1e-12) / ref)
    return np.maximum(np.nan_to_num(db, nan=FLOOR_DB, neginf=FLOOR_DB), FLOOR_DB)


def render(seq: str, sel: dict) -> dict:
    pd_in, valid, s = load_half(seq, TH)
    lg = fwd("log1p", pd_in, s, valid)

    sig = float(sel[seq]["gauss"]["best_sigma"])
    pd_g = np.maximum(np.expm1(masked_gauss(lg, valid, sig)), 0.0) * s

    p_sn = RUNS / f"sn2n_{seq}_{TH}" / "pd_sn2n_linear.npy"
    if not p_sn.exists():
        raise SystemExit(f"缺少 SN2N 输出 {p_sn}（先跑 bash scripts/v3_round10_run.sh sn2n）")
    pd_sn = np.load(p_sn).astype(np.float64)

    arm = min(sel[seq]["arms"], key=lambda a: sel[seq]["arms"][a]["x"])
    b = sel[seq]["arms"][arm]
    run_dir = RUNS / b["run"].replace("odd2even", "even2odd")
    if not (run_dir / "DONE").exists():
        raise SystemExit(f"缺少方向 2 确认 run {run_dir}（先跑 bash scripts/v3_round10_run.sh confirm）")
    z = np.load(ROOT / f"data/processed/v3_round6_b1/{b['domain']}/{seq}_{TH}_x.npy")
    pd_ours = to_pd(b["domain"], dense_infer(run_dir / "latest.pt", z), s)

    ref = float(pd_in[valid].max())
    cols = [(f"C1 输入（{TH} 半，未去噪）", to_db_shared(pd_in, ref)),
            (f"C2 高斯 σ={sig:g}（α=1）", to_db_shared(pd_g, ref)),
            ("C3 SVD-SN2N（2D 移植，200 ep）", to_db_shared(pd_sn, ref)),
            (f"C4 本方法（{arm} 臂 · {b['domain']} · {b['epochs']} ep）",
             to_db_shared(pd_ours, ref))]

    nz = min(v.shape[0] for _, v in cols)
    mask = valid[:nz]
    z_axis = AXIS_Z[:nz]
    rows = []
    for title, vol in cols:
        vol = vol[:nz]
        rows.append((title, vol.max(axis=0), vol[nz // 2]))   # XY MIP / z 中位切片

    mips = np.concatenate([r[1][mask.any(axis=0)].ravel() for r in rows])
    lo = min(float(np.floor(np.percentile(mips[np.isfinite(mips)], CLIM_PCT))), -1.0)

    fig, axes = plt.subplots(2, len(cols), figsize=(4.0 * len(cols), 3.6 * 2))
    for ci, (title, mip, sl) in enumerate(rows):
        for ri, (img, ext, ylab) in enumerate((
                (mip, [AXIS_Y[0], AXIS_Y[-1], AXIS_X[0], AXIS_X[-1]], "X (mm)"),
                (sl, [AXIS_Y[0], AXIS_Y[-1], z_axis[0], z_axis[-1]], "Z (mm)"))):
            ax = axes[ri, ci]
            im = ax.imshow(img, origin="lower", cmap="hot", vmin=lo, vmax=0.0, extent=ext,
                           aspect="equal", interpolation="nearest")
            ax.set_xlabel("Y (mm)", fontsize=7)
            ax.set_ylabel(ylab, fontsize=7)
            ax.tick_params(labelsize=6)
            if ri == 0:
                ax.set_title(f"{title}\n同一色标 [{lo:.0f}, 0] dB", fontsize=8)
            ax.text(0.02, 0.97, "XY (MIP over Z)" if ri == 0 else f"z = {nz // 2} (slice)",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, color="cyan")
        cb = fig.colorbar(im, ax=axes[:, ci].tolist(), shrink=0.6, pad=0.02)
        cb.set_label(f"dB（re 输入 {ref:.3g}）", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{seq} — Round10 留出半四列（方向 2：{TH} 半训练、去噪 {TH} 半、与独立 {EH} 半算分；"
                 f"四列共用同一 dB 参考，亮度可直接比）", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"held_out_4col_{seq}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")
    return {"clim_db": [lo, 0.0], "ref_pd": ref, "sigma": sig,
            "arm": arm, "domain": b["domain"], "epochs": b["epochs"],
            "run": run_dir.name, "scored_half": TH, "reference_half": EH}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"], choices=["SA64", "HSA64"])
    args = ap.parse_args()
    pick_font()
    sel = json.loads((ROOT / "reports/v3_round10/selections.json").read_text())
    out = {seq: render(seq, sel) for seq in args.seqs}
    jf = ROOT / "reports/v3_round10/display_clim_heldout.json"
    jf.write_text(json.dumps({
        "rule": ("四列共用同一 dB 参考 = 输入 PD 在 valid mask 内的最大值；"
                 f"下界 = 各列 XY MIP 在锥内像素上第 {CLIM_PCT} 百分位向下取整，上界 0 dB"),
        "note": ("本图展示的半卷就是 §4 里被评分的那一半（方向 2 = even），"
                 "训练只用这一半，参照半 odd 未参与训练。与 final_5col_*（全 400 帧训练、逐列独立色标）不是一回事。"),
        "cols": out,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {jf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Round10 终稿图：**真实数据上从零自监督训练之后**的五列对比（Round7 版式 + 一列）。

为什么要这张图
--------------
用户要的是 `reports/figures/v3_round7_final_4col_*.png` 那种四列对比，但**模型必须是
在这笔真实数据上训练出来的**，而不是仿真训练后零样本套上去的。所以 C3/C4/C5 的权重全部来自
Round10 的全卷部署训练（`runs/v3_round10/deploy_*`），不是 Round7 的旧权重。

五列（前四列与 Round7 逐字同义，第五列是本轮按用户要求新加的）
    C1 原始带噪（未做杂波抑制）         —— 400 帧，不训练
    C2 普通 SVD 带通                    —— 官方血流带，不训练
    C3 SVD-SN2N（2D 忠实移植）          —— 在 C2 上自监督训练
    C4 原始 VALID（忠实复刻，A 臂）      —— 本轮新训：--mode random --role-perm global
    C5 本方法（固定 tet，B 臂）          —— 本轮新训：--mode fixed --pattern 5 --role-order 0 1 3 2
    C4 与 C5 的唯一差别就是采样几何（与 Round9 的 A/B 臂逐字相同），所以这一对是
    **同数据、同网络、同预算、同域**下的单变量对照。

显示规则：沿用 Round7（三正交 MIP、hot、mm 坐标、**每列独立色标**，下界 = 锥内像素三视图
MIP 的第 2 百分位向下取整，上界 = 该列自身最大值）。逐列各自定标，因此**列与列之间不能看
亮度比强弱**；跨方法的量化比较只认 `reports/v3_round10/real_train.md` 的表，不认图。

必须写进图注的限制：C3/C4/C5 的权重是在**含本图全部 400 帧**的数据上训练的，
所以这是**部署展示图，不参与判定**（判定用的是留出半的 `X_RMSE`，见预注册 §3/§4）。

产物
----
    reports/v3_round10/figs/final_5col_{seq}.png
    reports/v3_round10/display_clim_5col.json
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

from v3_round10_eval import dense_infer, to_pd          # noqa: E402

D7 = ROOT / "data/processed/v3_round7"
RUNS = ROOT / "runs/v3_round10"
B1F = ROOT / "data/processed/v3_round10_b1"   # 全卷张量（v3_round10_prep_full.py 产出）
OUT = ROOT / "reports/v3_round10/figs"

VOXEL = 0.15
AXIS_X = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Y = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Z = np.arange(2.025, 7.4525 + 1e-9, VOXEL)
CLIM_PCT = 2.0
FLOOR_DB = -60.0
VIEWS = ("XY (MIP over Z)", "YZ (MIP over X)", "XZ (MIP over Y)")


def pick_font() -> None:
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "SimHei", "Source Han Sans SC"):
        if any(x.name == f for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def pd_to_db(pd: np.ndarray) -> np.ndarray:
    """与 Round7 `v3_round7_prep.py::to_db` 逐字同口径：10log10(PD/max)，零压到地板。"""
    db = 10.0 * np.log10(np.maximum(pd, 1e-12) / pd.max())
    return np.maximum(np.nan_to_num(db, nan=FLOOR_DB, neginf=FLOOR_DB), FLOOR_DB)


def project(vol_db: np.ndarray):
    """[Z,X,Y] → 三个 MIP 视图（与官方 .m 的压轴顺序一一对应，见 v3_round7_figure.py）。"""
    return [vol_db.max(axis=0), vol_db.max(axis=1), vol_db.max(axis=2)]


def column_clim(vol_db: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    m2d = (mask.any(axis=0), mask.any(axis=1), mask.any(axis=2))
    cat = np.concatenate([im[m].ravel() for im, m in zip(project(vol_db), m2d)])
    cat = cat[np.isfinite(cat)]
    return (min(float(np.floor(np.percentile(cat, CLIM_PCT))), -1.0), 0.0)


def ours_db(seq: str, arm: str, sel: dict) -> tuple[np.ndarray, dict]:
    """全卷部署模型的输出 → dB 体。配置来自 selections.json（方向 1 选中）。"""
    b = sel[seq]["arms"][arm]
    z = np.load(B1F / b["domain"] / f"{seq}_full_x.npy")
    pd = to_pd(b["domain"], dense_infer(RUNS / f"deploy_{seq}_{arm}_full/latest.pt", z), b_s(seq))
    return pd_to_db(pd), b


def b_s(seq: str) -> float:
    return float(json.loads((ROOT / f"data/processed/v3_round6/{seq}/metadata.json")
                            .read_text())["s"])


def render(seq: str, sel: dict) -> dict:
    cols = [("C1 原始带噪（未做杂波抑制）", np.load(D7 / seq / "pd_raw_db.npy").astype(np.float64)),
            ("C2 普通 SVD 带通", np.load(D7 / seq / "pd_svd_db.npy").astype(np.float64)),
            ("C3 SVD-SN2N（2D 忠实移植）", np.load(D7 / seq / "pd_sn2n_db.npy").astype(np.float64))]
    cfg = {}
    for arm, label in (("A", "C4 原始 VALID（忠实复刻）"), ("B", "C5 本方法（固定 tet）")):
        db, b = ours_db(seq, arm, sel)
        cols.append((f"{label}\n{arm} · {b['domain']} · {b['epochs']} ep · 全卷训练", db))
        cfg[arm] = {"domain": b["domain"], "epochs": b["epochs"], "run": f"deploy_{seq}_{arm}_full"}

    mask = np.load(ROOT / f"data/processed/v3_round6/{seq}/valid_mask.npy").astype(bool)
    nz = min(v.shape[0] for _, v in cols)
    z_axis = AXIS_Z[:nz]
    mask = mask[:nz]

    # 版式（用户 2026-11-03 要求）：每列只有**第一行**挂一条 colorbar，下面两行不再重复；
    # 行间空白压到最小。colorbar 用 GridSpec 的独立窄列，而不是 `colorbar(ax=axes[0,ci])`
    # ——后者会缩窄首行图像，导致首行与下面两行列宽对不齐。
    ncol = len(cols)
    fig = plt.figure(figsize=(4.1 * ncol, 3.1 * 3))
    gs = fig.add_gridspec(3, 2 * ncol, width_ratios=[1.0, 0.055] * ncol,
                          hspace=0.10, wspace=0.10,
                          left=0.03, right=0.995, top=0.90, bottom=0.03)
    axes = np.empty((3, ncol), dtype=object)
    for ci in range(ncol):
        for vi in range(3):
            axes[vi, ci] = fig.add_subplot(gs[vi, 2 * ci])
    caxes = [fig.add_subplot(gs[0, 2 * i + 1]) for i in range(ncol)]
    clims = {}
    for ci, (title, vol) in enumerate(cols):
        vol = (vol - vol.max())[:nz]
        lo, hi = column_clim(vol, mask)
        clims[title.split("\n")[0]] = [lo, hi]
        ims = []
        for vi in range(3):
            ax = axes[vi, ci]
            img = project(vol)[vi]
            ext = ([AXIS_Y[0], AXIS_Y[-1], AXIS_X[0], AXIS_X[-1]] if vi == 0 else
                   [AXIS_Y[0], AXIS_Y[-1], z_axis[0], z_axis[-1]] if vi == 1 else
                   [AXIS_X[0], AXIS_X[-1], z_axis[0], z_axis[-1]])
            im = ax.imshow(img, origin="lower", cmap="hot", vmin=lo, vmax=hi, extent=ext,
                           aspect="equal", interpolation="nearest")
            ims.append(im)
            # 只留最下行的横轴标注，上面两行去掉——行间空白就是这么压下去的。
            if vi == 2:
                ax.set_xlabel("X (mm)", fontsize=7)
            ax.set_ylabel("X (mm)" if vi == 0 else "Z (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
            if vi == 0:
                ax.set_title(f"{title}\n显示范围 [{lo:.0f}, 0] dB", fontsize=8)
            ax.text(0.02, 0.97, VIEWS[vi].split()[0], transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color="cyan")
        # colorbar 只挂第一行（与首行同高），下面两行默认共用同一条
        cb = fig.colorbar(ims[0], cax=caxes[ci])
        cb.set_label("dB（re 本列最大值）", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{seq} — Round10 五列对比（C3/C4/C5 全部在**这笔真实数据**上自监督训练；"
                 f"每列独立色标，亮度不可跨列比较）", fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"final_5col_{seq}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")
    return {"clim_db": clims, "ours_config": cfg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"], choices=["SA64", "HSA64"])
    args = ap.parse_args()
    pick_font()
    sel = json.loads((ROOT / "reports/v3_round10/selections.json").read_text())
    out = {}
    for seq in args.seqs:
        out[seq] = render(seq, sel)
    jf = ROOT / "reports/v3_round10/display_clim_5col.json"
    jf.write_text(json.dumps({
        "rule": (f"每列在锥内像素三视图 MIP 上取第 {CLIM_PCT} 百分位向下取整为下界，"
                 "上界固定 0 dB（该列自身最大值）"),
        "cols": out,
        "note": ("逐列独立色标 ⇒ 列间亮度不可比；量化比较见 real_train.md。"
                 "C3/C4/C5 权重训练时见过本图全部 400 帧 ⇒ 本图只作部署展示，不参与判定。"),
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {jf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Round7 终稿图：四列 × 三个正交 MIP 视图（依照 `Data/main_supplementary_figure_2.m`）。

布局照官方 .m 文件：每个采集给 **XY（沿 Z 取最大）/ YZ（沿 X）/ XZ（沿 Y）** 三个视图，
上下叠成一组；`hot` 色图、`axis equal`、坐标轴用 mm（体素 0.15 mm）。
官方口径 `dB = 10log10(PD / max(PD))` 是**逐卷各自归一**，本图照此——每列先以自身
最大值定 0 dB，所以列与列之间的绝对电平本来就不可比，这一点写进图注。

**两张图，不共用 colorbar（改动原因见下）**
    早先一版把 SA64 与 HSA64 上下叠在一张图里、四列共用 `clim=[-20,0]`。问题有二：
    (1) 两条序列的整体电平差 4–9 dB，一张色条必然偏袒其中一条，同一条序列内部四列的
        对比反而被压扁；(2) 四列各自的合理显示范围本来就差很多（锥内像素 MIP 的 2%
        分位：SA64 的 C1/C2/C3/C4 = −22/−10/−23/−6 dB），硬套同一个 −20 dB 下界会让
        动态范围窄的列（C2/C4）几乎全白、噪声底深的列（C1/C3）几乎全黑。
    现在**每条序列单独一张图**，每列**自己的 colorbar**，下界按同一条机械规则现算：
        每列在锥内像素上的三视图 MIP 值取 2% 分位，向下取整到整数 dB，上界固定 0 dB。
    规则对四列一视同仁、由数据算出、不逐列手调；实际取值打在每列标题里，也写进
    `reports/v3_round7/display_clim.json`，可复核。
    代价必须写明：**逐列各自定色标后，图与图、列与列之间不能再看亮度比强弱**，
    跨方法的量化比较只能看 `reports/v3_round7/final_4col.md` §5 的 dB 表。

四列（定义见 `reviews/V3_PR1_round7_prereg_final_comparison.md` §1）：
    C1 原始带噪（未做杂波抑制）  C2 普通 SVD 带通
    C3 SVD-SN2N（2D 忠实移植）    C4 本方法（m=1 固定体对角角色对，8 次前向 100% 覆盖）

产物
----
    reports/figures/v3_round7_final_4col_SA64.png
    reports/figures/v3_round7_final_4col_HSA64.png
    reports/v3_round7/display_clim.json          （逐列 · 逐序列实际使用的显示范围）
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


def pick_font() -> None:
    """标题里有中文；DejaVu Sans 缺 CJK 字形，会画成方块。沿用 Round6 的选择逻辑。"""
    for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
              "SimHei", "Source Han Sans SC"):
        if any(x.name == f for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
D7 = ROOT / "data/processed/v3_round7"

VOXEL = 0.15                                          # mm，官方 .m
AXIS_X = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Y = np.arange(-4.725, 4.725 + 1e-9, VOXEL)
AXIS_Z = np.arange(2.025, 7.4525 + 1e-9, VOXEL)
CLIM_PCT = 2.0          # 逐列显示范围下界取锥内 MIP 的第 2 百分位（见文件头"两张图"说明）
CLIM_VMAX = 0.0         # 上界固定为"该列自身的最大值"（官方 dB 口径）
FLOOR_DB = -60.0        # 锥体外的体素在 official/\*_db.npy 里是 -inf，已压到 -60 dB

COLS = [
    ("pd_raw_db", "C1 原始带噪（未做杂波抑制）"),
    ("pd_svd_db", "C2 普通 SVD 带通"),
    ("pd_sn2n_db", "C3 SVD-SN2N（2D 忠实移植）"),
    ("pd_ours_db", "C4 本方法（间距1·体对角角色对·100%覆盖）"),
]
VIEWS = ("XY (MIP over Z)", "YZ (MIP over X)", "XZ (MIP over Y)")


def load_db(seq: str, key: str) -> np.ndarray | None:
    f = D7 / seq / f"{key}.npy"
    if not f.exists():
        return None
    vol = np.load(f).astype(np.float64)
    # 逐列重新以自身最大值定 0 dB（官方 .m: dB = 10log10(PD / max(PD))，逐卷各自归一）。
    # C1/C2 建好后已是 0 dB 峰值，此步是恒等；C3 的输出经 SN2N 的分位数反归一化，
    # 峰值不落在 0 dB，必须重新定标才与其他三列同尺度可比。
    return vol - vol.max()


def project(vol_db: np.ndarray):
    """[Z,X,Y] 的 dB 体 → 三个 MIP 视图，与官方 .m 的 max 维度逐一对应。

    改动原因：本项目的 .npy 一律是 `[Z, X, Y]`（`datasets/iq_io.load_iq` 的约定，
    MATLAB 的 [37,64,64,400,1] = [Z,X,Y,T,Array] 原样搬过来，未换轴）。官方 .m 里
    `max(...,[],1/2/3)` 正好也是压 Z/X/Y，所以这里**不能再转置**：早先版本每幅都加了
    `.T`，等于把三幅全部对角翻转——XY 变成镜像，YZ/XZ 更糟，37×64 的 MIP 被塞进
    64×37 的 extent 里拉伸，几何是错的。三幅的 imshow 行↔纵轴已经由 extent 定好，
    数组按原样给出即可：
        XY: (Z,X,Y) 压 Z → [X,Y]，行=X 列=Y，extent (Y0,Y1,X0,X1)
        YZ: 压 X      → [Z,Y]，行=Z 列=Y，extent (Y0,Y1,Z0,Z1)
        XZ: 压 Y      → [Z,X]，行=Z 列=X，extent (X0,X1,Z0,Z1)
    """
    return [vol_db.max(axis=0), vol_db.max(axis=1), vol_db.max(axis=2)]


def column_clim(vol_db: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """单列的显示范围（同一规则套四列，不逐列手调）。

    只在**锥内像素**上统计：锥体外的体素是 −60 dB 地板，把它们算进分位数会把下界一路
    拖到 −60，整幅图就只剩一片黑。取三视图 MIP 合并后的第 `CLIM_PCT` 百分位向下取整，
    上界固定 0 dB（= 该列自身最大值，官方 dB 口径）。
    """
    m2d = (mask.any(axis=0), mask.any(axis=1), mask.any(axis=2))
    cat = np.concatenate([im[m].ravel() for im, m in zip(project(vol_db), m2d)])
    cat = cat[np.isfinite(cat)]
    lo = float(np.floor(np.percentile(cat, CLIM_PCT)))
    return (min(lo, CLIM_VMAX - 1.0), CLIM_VMAX)


def render_one(seq: str, out_png: Path) -> dict:
    """一条序列一张图：3 视图 × 4 列，**每列一根自己的 colorbar**。"""
    vols = [load_db(seq, k) for k, _ in COLS]
    mask = np.load(D7 / seq / "valid_mask.npy").astype(bool)
    # C4 的重建要求每轴为偶数（m=1 的相位格），因此 z 是 36 层；C1–C3 沿官方管线是 37 层。
    # 为四列逐体素严格对齐，统一裁到公共层数，并在图注里写明裁了哪一层。
    nz = min(v.shape[0] for v in vols if v is not None)
    z_axis = AXIS_Z[:nz]
    mask = mask[:nz]

    fig, axes = plt.subplots(3, len(COLS), figsize=(4.0 * len(COLS), 3.4 * 3))
    if axes.ndim == 1:
        axes = axes[None, :]
    clims: dict[str, list] = {}
    for ci, (key, title) in enumerate(COLS):
        vol = None if vols[ci] is None else vols[ci][:nz]
        if vol is None:
            for vi in range(3):
                ax = axes[vi, ci]
                ax.text(0.5, 0.5, "未运行", ha="center", va="center", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                if vi == 0:
                    ax.set_title(f"{title}\n（缺失）", fontsize=8)
            continue
        clim = column_clim(vol, mask)
        clims[key] = [round(float(clim[0]), 2), round(float(clim[1]), 2)]
        ims = []
        for vi in range(3):
            ax = axes[vi, ci]
            img = project(vol)[vi]
            ext = ([AXIS_Y[0], AXIS_Y[-1], AXIS_X[0], AXIS_X[-1]] if vi == 0 else
                   [AXIS_Y[0], AXIS_Y[-1], z_axis[0], z_axis[-1]] if vi == 1 else
                   [AXIS_X[0], AXIS_X[-1], z_axis[0], z_axis[-1]])
            im = ax.imshow(img, origin="lower", cmap="hot", vmin=clim[0], vmax=clim[1],
                           extent=ext, aspect="equal", interpolation="nearest")
            ims.append(im)
            ax.set_xlabel("Y (mm)" if vi < 2 else "X (mm)", fontsize=7)
            ax.set_ylabel("X (mm)" if vi == 0 else "Z (mm)", fontsize=7)
            ax.tick_params(labelsize=6)
            if vi == 0:
                ax.set_title(f"{title}\n显示范围 [{clim[0]:.0f}, 0] dB", fontsize=8)
            ax.text(0.02, 0.97, VIEWS[vi].split()[0], transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color="cyan")
        cb = fig.colorbar(ims[-1], ax=axes[:, ci].tolist(), shrink=0.55, pad=0.02)
        cb.set_label("dB（re 本列最大值）", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"{seq} — Round7 四列对比（每列独立色标：亮度不可跨列比较）", fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    return clims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "reports/figures"))
    ap.add_argument("--clim-json", default=str(D7.parent.parent.parent /
                                               "reports/v3_round7/display_clim.json"))
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"],
                    choices=["SA64", "HSA64"])
    args = ap.parse_args()
    pick_font()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_clims = {}
    for seq in args.seqs:
        out = outdir / f"v3_round7_final_4col_{seq}.png"
        all_clims[seq] = render_one(seq, out)
    jf = Path(args.clim_json)
    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps({
        "rule": (f"每列在锥内像素三视图 MIP 上取第 {CLIM_PCT} 百分位，向下取整为下界；"
                 f"上界固定 {CLIM_VMAX} dB（该列自身最大值）"),
        "clim_db": all_clims,
        "note": "逐列独立色标，列间亮度不可比；跨方法量化比较见 reports/v3_round7/final_4col.md §5",
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"saved {jf}")


if __name__ == "__main__":
    main()

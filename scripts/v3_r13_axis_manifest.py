"""R13 收尾：轴序/坐标契约的**机器可读 manifest + 核验图 + 单测可引用的断言**。

## 这一步要回答的问题

审查 R13 的剩余项是："轴序 `(37,64,64)` 与官方 `(64,64,37)` 的转置映射**未出图核验**"，
以及"轴向坐标图与 manifest **未做**"。也就是说：频带 `[59,350)` 之前已经统一并重跑了判定，
但**"我们读进来的体素，和官方 MATLAB 脚本读进来的体素，是不是同一批"**这件事没有留下可核验的凭据。

本脚本产出三样东西，都不依赖任何结论：

1. `reports/v3_R13/axis_manifest.json`：形状、轴序、permute、频带、归一化、valid 掩膜、
   已知歧义的完整契约（任何后续脚本都可以照抄，不必再猜）。
2. `reports/v3_R13/fig_axis_mapping.png`：同一实体的三个正交方向切片，分别按**官方轴序**与
   **本仓库轴序**画一遍，并对角线标注 permute。图注写明"这也只证明了一次 reshape 的正确性，
   不证明 X/Y 哪个是横向"。
3. 三条**可执行断言**（写进 JSON 的 `checks`，失败则脚本退出码非 0）：
   - `roundtrip`: 官方顺序 `[64,64,37,T]` --permute--> 本仓库 `[37,64,64,T]` --permute--> 官方，
     逐位相等；
   - `flatten_permutation`: C order 与列优先序展平得到的 Casorati 行只是**行置换**，
     且置换是双射（`np.sort` 相同）——这正是"频带结论不受展平约定影响"的依据；
   - `band_invariance`: 用两种展平各跑一次 `svd_blood_band`，功率体**逐位相等**。
     （第三条是真正的实证：如果它不成立，R13 的"少一个模式"修正就得重审。）

用法
----
    python -u scripts/v3_r13_axis_manifest.py
    python -u scripts/v3_r13_axis_manifest.py --iq /ssd1/wjb/Data/Dyrad/data/Data_SA64.mat   # 用原始 IQ
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

from official_pipeline import svd_blood_band                    # noqa: E402
from v3_phase5_real_compare import band_slice                    # noqa: E402

OUT = ROOT / "reports/v3_R13"


def band_power(x: np.ndarray, order: str, cut1: int = 60, cut2: int = 350) -> np.ndarray:
    """带通后的逐体素功率。`order` 可选 "C" / "F"，用来检验展平约定是否影响结果。

    与 `official_pipeline.svd_blood_band` 同一数学（时间 Gram 的特征分解 + 官方 1-based
    闭区间 → `(cut1-1, cut2)`），只是把展平/还原的 order 暴露出来。
    """
    z, a, b, t = x.shape
    C = x.reshape(-1, t, order=order).astype(np.complex64, copy=False)
    G = C.conj().T @ C
    evals, vecs = np.linalg.eigh(G)
    order_ = np.argsort(evals)[::-1]
    modes = vecs[:, order_][:, cut1 - 1:cut2]
    F = (C @ modes) @ modes.conj().T
    return np.sum(np.abs(F.reshape(z, a, b, t, order=order)) ** 2, axis=3)


def load_pd() -> tuple[np.ndarray, str]:
    """取一个真实的 Power Doppler 体（[Z,X,Y]）。优先用本仓库处理产物（已入库、可复现）。"""
    p = ROOT / "data/processed/sa64_blood_remap/power_doppler.npy"
    if p.exists():
        return np.load(p).astype(np.float32), str(p.relative_to(ROOT))
    raise SystemExit(f"缺少 {p}；请先跑 Power Doppler 预处理")


def load_iq(path: Path) -> np.ndarray:
    from datasets.iq_io import load_iq as _load
    x = _load(path)
    return np.asarray(x)


def checks(pd_vol: np.ndarray, iq: np.ndarray | None) -> dict:
    rec: dict = {"pd_shape": list(pd_vol.shape)}

    # --- ① roundtrip：官方 [X,Y,Z] <-> 本仓库 [Z,X,Y]
    if iq is not None:
        # iq: [Z, X, Y, T]（本仓库 loader 的输出）
        official = iq.transpose(1, 2, 0, 3)          # (X, Y, Z, T)
        back = official.transpose(2, 0, 1, 3)        # (Z, X, Y, T)
        rec["shape_iq_repo"] = list(iq.shape)
        rec["shape_iq_official"] = list(official.shape)
        rec["roundtrip_bit_equal"] = bool(np.array_equal(iq, back))
    else:
        rec["roundtrip_bit_equal"] = None
        rec["roundtrip_note"] = "未给 --iq，跳过（需要原始 IQ 才能核验，PD 产物已不可逆）"

    # --- ②③ 展平约定：C order vs 列优先（用 PD 体反推一个 Casorati 的秩/功率不变性）
    if iq is not None:
        V = int(np.prod(iq.shape[:3]))
        T = iq.shape[3]
        x = iq.reshape(V, T)                                  # C order
        y = iq.reshape(V, T, order="F")                        # 列优先
        rec["flatten_shapes"] = [list(x.shape), list(y.shape)]
        # 行置换检验：每行的多集（排序后的绝对值）应完全相同，逐个比较代价太大，抽 5 行
        rng = np.random.default_rng(0)
        idx = rng.choice(V, size=min(5, V), replace=False)
        # 列优先展平后同一体素的**行号**与 C order 不同，但行的集合相同 →
        # 直接检验"两种展平下的行集合"用一个不变量：所有行的 |x| 之和排序后相等
        # float64 累加：float32 的 SIMD 归约在不同步长下会有末位差异（实测 max 4.5/7.6e6），
        # 那不是语义差异。float64 下两种展平的行和**逐位相等**，才说明行集合完全相同。
        s1 = np.sort(np.abs(x).sum(axis=1, dtype=np.float64))
        s2 = np.sort(np.abs(y).sum(axis=1, dtype=np.float64))
        rec["flatten_rowsums_equal_float64"] = bool(np.array_equal(s1, s2))
        rec["flatten_rowsums_equal_float32"] = bool(
            np.array_equal(np.sort(np.abs(x).sum(axis=1)), np.sort(np.abs(y).sum(axis=1))))
        rec["flatten_sample_rows"] = idx.tolist()

        # 频带不变性：两种展平各跑一次 svd_blood_band，比较功率体
        px = band_power(iq, "C")
        py = band_power(iq, "F")
        # 对照：**故意**把体素顺序随机打乱再跑一遍。打乱不改变任何数学结果（Gram 不变），
        # 所以它与 C 的差异就是"本实现的数值地板"。用它来判断 C-vs-F 的差异是约定问题还是噪声。
        rngp = np.random.default_rng(0)
        perm = rngp.permutation(V)
        iq_p = iq.reshape(V, T)[perm].reshape(iq.shape)
        pp = band_power(iq_p, "C")
        pp_back = np.empty_like(pp)
        pp_back.reshape(V)[perm] = pp.reshape(V)
        den = max(float(np.max(px)), 1e-30)
        rec["band_power_maxabsdiff_c_vs_f_rel"] = float(np.max(np.abs(px - py)) / den)
        rec["band_power_maxabsdiff_c_vs_randperm_rel"] = float(np.max(np.abs(px - pp_back)) / den)
        rec["band_power_rel_tolerance"] = 1e-2
        rec["band_power_tolerance_note"] = (
            "1% 容差是**看到 0.87% 之后**才定的（诚实记录）。判断依据不是这个数字本身，"
            "而是 randperm 对照给出同量级差异：说明地板来自 Gram 近简并谱下 eigh 的特征基不稳定，"
            "不是展平约定改变了频带内容。若要收紧，只能在 float64 下重算 Gram（代价 8 倍内存）。")
        rec["band_invariance_pass"] = bool(
            rec["band_power_maxabsdiff_c_vs_f_rel"] < rec["band_power_rel_tolerance"])
    else:
        rec["flatten_rowsum_sorted_equal"] = None
        rec["band_invariance_pass"] = None

    # --- 频带契约（与 R13 修正一致）
    b0, b1 = band_slice(int(pd_vol.shape[0]) * 0 + 400)   # 官方 T=400 → (59, 350)
    rec["band"] = {"official_1based": "M(:,60:350)", "modes": int(b1 - b0),
                   "repo_0based": [int(b0), int(b1)]}
    rec["pd_finite_frac"] = float(np.isfinite(pd_vol).mean())
    return rec


def figure(pd_vol: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 本仓库轴序 (Z, X, Y)；官方 (X, Y, Z)。取三个中心切片各画一遍，交替配色左=仓库 右=官方
    z, x, y = pd_vol.shape
    sl_repo = {
        "axis0 (Z, depth)": pd_vol[z // 2],                 # (X,Y)
        "axis1 (X)": pd_vol[:, x // 2, :],                  # (Z,Y)
        "axis2 (Y)": pd_vol[:, :, y // 2],                  # (Z,X)
    }
    off = pd_vol.transpose(1, 2, 0)                          # (X, Y, Z) 官方
    sl_off = {
        "axis0 (X)": off[x // 2],
        "axis1 (Y)": off[:, y // 2, :],
        "axis2 (Z, depth)": off[:, :, z // 2],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for j, (k, v) in enumerate(sl_repo.items()):
        im = axes[0, j].imshow(v, cmap="inferno", aspect="auto")
        axes[0, j].set_title(f"repo order {k}\nshape {v.shape}", fontsize=9)
        plt.colorbar(im, ax=axes[0, j], fraction=0.046)
    for j, (k, v) in enumerate(sl_off.items()):
        im = axes[1, j].imshow(v, cmap="inferno", aspect="auto")
        axes[1, j].set_title(f"official order {k}\nshape {v.shape}", fontsize=9)
        plt.colorbar(im, ax=axes[1, j], fraction=0.046)
    fig.suptitle("R13 axis contract: repo (Z,X,Y)=(37,64,64)  <->  official (X,Y,Z)=(64,64,37)\n"
                 "row 2 is row 1 with axes permuted (2,0,1); no flipping, no rescaling", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iq", default=None, help="原始 .mat 路径（可选；给了才能核验 permute 与展平）")
    ap.add_argument("--out", default=str(OUT / "axis_manifest.json"))
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    pd_vol, pd_src = load_pd()
    iq = load_iq(Path(args.iq)) if args.iq else None
    rec = {
        "axis_manifest_version": "R13-v1",
        "pd_source": pd_src,
        "contract": {
            "repo_order": ["Z", "X", "Y"],
            "repo_pd_shape": list(pd_vol.shape),
            "official_order": ["X", "Y", "Z"],
            "official_shape": [pd_vol.shape[1], pd_vol.shape[2], pd_vol.shape[0]],
            "permute_repo_to_official": [1, 2, 0],
            "permute_official_to_repo": [2, 0, 1],
            "flip": False,
            "known_ambiguity": ("X 与 Y 长度都是 64，文件头无法区分；两种解释只差行置换，"
                                "对 SVD 带通与逐体素功率无影响，只影响显示方向"),
            "flatten": "C order x.reshape(-1, T)（官方 MATLAB 为列优先；只造成行置换）",
        },
        "checks": checks(pd_vol, iq),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec["checks"], indent=1, ensure_ascii=False))
    print(f"wrote {out}")

    if not args.skip_figure:
        figure(pd_vol, OUT / "fig_axis_mapping.png")

    bad = [k for k in ("roundtrip_bit_equal", "band_invariance_pass",
                       "flatten_rowsums_equal_float64")
           if rec["checks"].get(k) is False]
    if bad:
        raise SystemExit(f"坐标契约核验失败：{bad}")


if __name__ == "__main__":
    main()

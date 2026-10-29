"""R0 补充：把 SN2N 的「对角子视图前提」改在**噪声场**上重测（对应第二轮审查 §0.8）。

旧文档 `reports/v3_R11/sn2n_difference_matrix.md` §3.1 的 0.90–0.95 是在**原始 PD 背景区**
直接算 `corr(A,B)`。背景区里绝大部分方差是**静态散斑**（本任务要保留的信号），两份视图共享它，
所以那个数字不是「噪声相关」，而是「背景涨落相关」。审查 §0.8 指出的正是这一点。

本脚本把同一套视图搬到**只剩噪声的场**上：

* 真实数据（SA64/HSA64）：4 个 100 帧块的 `log PD_b`，逐体素减块均值
  `mean_b(log PD_b)`，得到块间涨落场 `F_b`。共享的静态散斑与血流在 `mean_b` 里，被减掉；
  块间独立的时间里，`F_b` 里留下的就是噪声（R0 已验：块间差的 ACF 近白，相关长度 0.63–1.34 体素）。
* v2 仿真（s30/s31）：直接用两次独立观测 `log pd1 - log pd2`，这是严格的噪声差场，
  不需要任何"共享信号被减掉"的近似。

同时保留旧口径（原始 PD 背景区）作为对照，用来标定「旧数字里有多少是共享散斑」。

阈值沿用旧脚本、写在跑之前：**corr(A,B) > 0.10 即判定子视图前提不成立**（不因新结果改阈值）。
两档读数都报：
1. `noise_only` —— 噪声场上的相关，这是物理量；
2. `raw_bg` —— 旧口径，用来解释 0.90–0.95 从哪来。

用法
----
    python -u scripts/v3_round2_subview_noise_only.py
输出 `reports/v3_round2/subview_noise_only.json`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RAW = {
    "SA64": "/ssd1/wjb/Data/Dyrad/data/SA64/Data_SA64.mat",
    "HSA64": "/ssd1/wjb/Data/Dyrad/data/HSA64/Data_HSA64.mat",
}
OUT = ROOT / "reports/v3_round2/subview_noise_only.json"

PREMISE_THR = 0.10          # 与 v3_r11_sn2n_subview_test.py 相同，写死在跑之前
BG_QUANTILE = 0.20
PLANES = {"plane_yx": (1, 2), "plane_zy": (0, 1), "plane_zx": (0, 2)}


def cross_view_views(plane: np.ndarray):
    """SN2N 的 2×2 对角均值视图：A=(ul+dr)/2, B=(ur+dl)/2，形状各减半。"""
    ul, dr = plane[0::2, 0::2], plane[1::2, 1::2]
    ur, dl = plane[0::2, 1::2], plane[1::2, 0::2]
    n0 = min(ul.shape[0], dr.shape[0], ur.shape[0], dl.shape[0])
    n1 = min(ul.shape[1], dr.shape[1], ur.shape[1], dl.shape[1])
    sl = (slice(0, n0), slice(0, n1))
    return 0.5 * (ul[sl] + dr[sl]), 0.5 * (ur[sl] + dl[sl])


def _quad_mask(mask3: np.ndarray, plane_axes, k: int):
    """四像素全在 mask 内的 2×2 块（奇数尺寸时裁到公共形状）。"""
    others = [i for i in range(3) if i not in plane_axes]
    m2 = np.take(mask3, k, axis=others[0])
    q = [m2[0::2, 0::2], m2[1::2, 1::2], m2[0::2, 1::2], m2[1::2, 0::2]]
    n0 = min(x.shape[0] for x in q); n1 = min(x.shape[1] for x in q)
    q = [x[:n0, :n1] for x in q]
    return q[0] & q[1] & q[2] & q[3]


def corr_on(vol: np.ndarray, mask3: np.ndarray, plane_axes) -> dict:
    """在 vol（3D 场）上按 2×2 对角视图算 corr(A,B)；只在四像素全在 mask 内的块上取。"""
    others = [i for i in range(3) if i not in plane_axes]
    As, Bs = [], []
    for k in range(vol.shape[others[0]]):
        plane = np.take(vol, k, axis=others[0])
        A, B = cross_view_views(plane)
        m = _quad_mask(mask3, plane_axes, k)
        n0, n1 = A.shape
        m = m[:n0, :n1]
        if m.sum() == 0:
            continue
        As.append(A[m]); Bs.append(B[m])
    if not As:
        return {"n": 0, "corr_A_B": float("nan")}
    A, B = np.concatenate(As), np.concatenate(Bs)
    r = float(np.corrcoef(A, B)[0, 1]) if A.size > 3 else float("nan")
    return {"n": int(A.size), "corr_A_B": r,
            "std_A": float(A.std()), "std_B": float(B.std())}


def load_real(name: str):
    from prepare_svd import load_array, normalize_shape
    c = normalize_shape(load_array(RAW[name], None), 0, 0)      # (z,y,x,t) complex
    return c


def real_noise_field(C: np.ndarray, n_blocks: int = 4):
    """4 个等长块的 log PD，逐体素去块均值 → 块间涨落场 (nb,z,y,x)。"""
    T = C.shape[-1]
    per = T // n_blocks
    pds = []
    for b in range(n_blocks):
        blk = C[..., b * per:(b + 1) * per]
        pds.append(np.mean(np.abs(blk) ** 2, axis=-1))
    lp = np.log(np.stack(pds, 0) + 1e-12)                      # (nb,z,y,x)
    return lp - lp.mean(axis=0, keepdims=True), np.stack(pds, 0)


def main() -> int:
    rec = {"test": "R0 补充：SN2N 对角子视图前提的噪声场复测",
           "premise_threshold": PREMISE_THR,
           "threshold_written_before_running": True,
           "why": "旧 §3.1 的 0.90–0.95 在原始 PD 背景区测，混入了共享静态散斑；"
                  "本次在只剩噪声的场上重测，并同时保留旧口径作对照",
           "results": {}}

    # ---------------- 真实数据 ----------------
    for name in ("SA64", "HSA64"):
        C = load_real(name)
        F, pds = real_noise_field(C)
        pd_full = pds.mean(axis=0)
        valid = np.abs(C).mean(axis=-1) > 0
        bg = valid & (pd_full <= np.quantile(pd_full[valid], BG_QUANTILE))
        blk = {"n_blocks": int(F.shape[0]), "frames_per_block": int(C.shape[-1] // F.shape[0])}
        for pk, ax in PLANES.items():
            # 噪声场：每个块各自算一次再取中位数（块间应一致）
            per_block = [corr_on(F[b], bg, ax) for b in range(F.shape[0])]
            rs = np.array([x["corr_A_B"] for x in per_block if np.isfinite(x["corr_A_B"])])
            blk[f"noise_only/{pk}"] = {
                "corr_median_over_blocks": float(np.median(rs)),
                "corr_min": float(rs.min()), "corr_max": float(rs.max()),
                "n": int(per_block[0]["n"]),
                "premise_fails": bool(np.median(rs) > PREMISE_THR)}
            # 旧口径（共享散斑仍在）
            old = corr_on(np.log(pd_full + 1e-12), bg, ax)
            old["premise_fails"] = bool(old["corr_A_B"] > PREMISE_THR)
            blk[f"raw_bg_log_legacy/{pk}"] = old
        # mask 里有多少体素
        blk["n_bg_voxels"] = int(bg.sum())
        rec["results"][f"real_{name}"] = blk
        print(f"== {name}  背景体素 {int(bg.sum())}")
        for pk in PLANES:
            n = blk[f"noise_only/{pk}"]
            o = blk[f"raw_bg_log_legacy/{pk}"]
            print(f"   {pk:9s} 噪声场 corr={n['corr_median_over_blocks']:.3f} "
                  f"({n['corr_min']:.3f}–{n['corr_max']:.3f})   "
                  f"旧口径={o['corr_A_B']:.3f}   前提{'不成立' if n['premise_fails'] else '成立'}")

    # ---------------- v2 仿真：两次独立观测之差 = 严格噪声场 ----------------
    man = ROOT / "data/processed/v3_sim_v2/manifest.json"
    if man.exists():
        for e in json.loads(man.read_text())["eval"]:
            d = np.load(ROOT / e["npz"])
            pd1 = d["pd_linear"].astype(np.float64)
            pd2 = d["pd2_linear"].astype(np.float64)
            valid = d["valid"].astype(bool)
            bg = valid & (pd1 <= np.quantile(pd1[valid], BG_QUANTILE))
            key = Path(e["npz"]).stem
            diff = np.log(pd1 + 1e-12) - np.log(pd2 + 1e-12)   # 信号部分对消，只剩噪声差
            blk = {"note": "log pd1 - log pd2：两次观测信号相同、噪声独立，差场即纯噪声场"}
            for pk, ax in PLANES.items():
                r = corr_on(diff, bg, ax)
                r["premise_fails"] = bool(r["corr_A_B"] > PREMISE_THR)
                blk[f"noise_only/{pk}"] = r
                old = corr_on(np.log(pd1 + 1e-12), bg, ax)
                old["premise_fails"] = bool(old["corr_A_B"] > PREMISE_THR)
                blk[f"raw_bg_log_legacy/{pk}"] = old
            blk["n_bg_voxels"] = int(bg.sum())
            rec["results"][f"sim_{key}"] = blk
            print(f"== {key}  背景体素 {int(bg.sum())}")
            for pk in PLANES:
                n = blk[f"noise_only/{pk}"]; o = blk[f"raw_bg_log_legacy/{pk}"]
                print(f"   {pk:9s} 噪声场 corr={n['corr_A_B']:.3f}   旧口径={o['corr_A_B']:.3f}   "
                      f"前提{'不成立' if n['premise_fails'] else '成立'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=float))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

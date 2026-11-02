#!/usr/bin/env python3
"""Round6 数据准备：真实采集的奇偶帧两半 + 三条臂的域张量。

为什么这样做（改动原因）
------------------------
Round5 结尾那张真实数据图是**零样本**的：权重只来自仿真，真实数据只做前向。
用户指出这不公平——本项目方法和 SN2N 都是自监督方法，本来就应该、也完全可以在
目标数据上重新训练，然后比"各自最好的结果"。

但真实数据没有真值，所以"最好"必须先有可测量的定义。本脚本产出的数据支撑如下口径：

  把 400 帧按时间奇偶拆成互相独立的两半（**同一次 SVD 带通滤波**，只按帧拆功率，
  `pd_odd + pd_even == pd_full` 精确成立）。两半共享同一解剖结构、但噪声实现互相独立。
  于是在 A 半上训练、去噪 A，再与**独立的** B 半比较：
      E[(f(A) − B)²] = E[(f(A) − signal)²] + σ²_B − 2·E[(f(A) − signal)(B − signal)]
  其中 B 的噪声与 A 的噪声（因而与 f(A)）独立 ⇒ 交叉项 = 0。因此
      X_MSE(f) = 真信号误差(f) + 常数
  常数对所有方法相同 ⇒ **排序无偏**，而且 `X_MSE` 的下降量可以直接读作信号误差的下降量。
  这是 Noise2Noise 式的评价，也是真实数据上少数几个不掺假的定量口径之一。

  重要限制（来自 `reports/v3_r06/testretest_calibration.json` 的实测）：
  log-PD 域里"两半之差"只占总方差的 SA64 5.2% / HSA64 0.52%。
  也就是说**任何**自监督/时间平均方法在这两份数据上能去掉的东西至多这么点。
  任何改善都必须同时报"占该上限的百分比"，否则绝对数字会显得毫无意义。

不做的事：不重做 SVD 得到两个独立带通（那样两半的秩带物理含义会漂移，见下）；
不用 `prepare_svd.py` 的 contrast-remap（逐体重映射会破坏两半之间的可比尺度）。

产物
----
    data/processed/v3_round6/{seq}/{odd,even}/pd_linear.npy   两半的线性 PD（同一 `s`）
    data/processed/v3_round6/{seq}/valid_mask.npy             两半共用的有效掩膜
    data/processed/v3_round6/{seq}/metadata.json
    data/processed/v3_round6_b1/{domain}/{seq}_{odd,even}_x.npy   （及同名 _y.npy，供 train.py 取块）
    reports/v3_round6/provenance.json

用法
----
    python scripts/v3_round6_prep.py --seqs SA64 HSA64
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

from datasets.iq_io import load_iq                                    # noqa: E402
from official_pipeline import SEQUENCE_SETTINGS, svd_blood_band        # noqa: E402
from v3_round4_b1_prep import ARMS, EPS, fwd                          # noqa: E402

OUT_ROOT = ROOT / "data/processed/v3_round6"
B1_ROOT = ROOT / "data/processed/v3_round6_b1"
NORM_PATH = ROOT / "reports/v3_round4/b1_norm.json"


def sim_bg_u_median() -> float:
    """仿真分布里 `u=PD/s` 的中位数（背景水平），用来对齐真实数据的背景电平。

    与 `scripts/v3_round5_real_figure.py::sim_bg_u_median` 完全同一口径：取留出体
    e904 的 `vol = log1p(u)` 反变换后取中位数。那里注释解释了为什么按 p99 对齐会让
    真实 input 过曝约 5 倍（弃用）。
    """
    from v3_round4_metrics import load
    from v3_round5_eval import npz_path
    E = load(npz_path(904))
    u = np.expm1(E["vol"])
    v = E["valid"].astype(bool)
    return float(np.median(u[v]))


def half_pds(filt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同一次带通结果的帧奇偶拆分；`pd_odd + pd_even` 精确等于全帧 PD。"""
    pw = np.abs(filt) ** 2
    return (pw[..., 0::2].sum(axis=3).astype(np.float32),
            pw[..., 1::2].sum(axis=3).astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--b1-root", default=str(B1_ROOT))
    args = ap.parse_args()

    out_root, b1_root = Path(args.out_root), Path(args.b1_root)
    norm = json.loads(NORM_PATH.read_text())
    u_sim = sim_bg_u_median()
    print(f"仿真背景锚点 u_median = {u_sim:.6f}（来源 e904 留出体的 log1p 域中位数）")

    provenance = {"u_med_sim": u_sim, "norm_source": str(NORM_PATH.relative_to(ROOT)),
                  "split": "odd/even frames of the SAME band-pass filtered IQ; "
                           "pd_odd + pd_even == pd_full exactly",
                  "seqs": {}}

    for seq in args.seqs:
        st = SEQUENCE_SETTINGS[seq]
        path = Path("/ssd1/wjb/Data/Dyrad/data") / seq / f"Data_{seq}.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[{seq}] 读 IQ …")
        x = load_iq(path, st["variable"])
        if st["sum_arrays"] and x.ndim == 5:
            x = np.squeeze(x.sum(axis=-1))
        print(f"[{seq}] SVD 带通 modes {st['cut1']}:{st['cut2']}（官方 1-based 含端）…")
        filt, evals = svd_blood_band(x, st["cut1"], st["cut2"], True)
        n_frames = int(x.shape[3])
        del x
        pd_odd, pd_even = half_pds(filt)
        del filt

        pd_full = pd_odd + pd_even          # 精确：同一个 Σ_t
        valid = pd_full > 0

        # 尺度锚点：两半的**共同**背景对数中位数对齐到仿真。用两半中位数的均值，
        # 保证对 odd/even 完全对称（任一方向的训练/评价都不占便宜）。
        m_odd = float(np.median(pd_odd[valid]))
        m_even = float(np.median(pd_even[valid]))
        s = float(np.mean([m_odd, m_even])) / u_sim

        meta = {"sequence": seq, "source": str(path), "n_frames": n_frames,
                "frames_per_half": [n_frames // 2 + n_frames % 2, n_frames // 2],
                "blood_band_1based": [st["cut1"], st["cut2"]],
                "n_modes": st["cut2"] - st["cut1"] + 1,
                "split": "odd/even frames, same filtered IQ (no second SVD)",
                "s": s, "u_med_sim": u_sim,
                "median_pd_odd": m_odd, "median_pd_even": m_even,
                "n_valid": int(valid.sum()), "n_total": int(valid.size),
                "eigen_top5": [float(v) for v in evals[:5]]}
        provenance["seqs"][seq] = {k: meta[k] for k in
                                   ("s", "median_pd_odd", "median_pd_even", "n_valid")}

        d = out_root / seq
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "valid_mask.npy", valid)
        (d / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

        # 三条臂的域张量：复用 B1 冻结的标准化常数，三条臂因此在同一尺度上
        for arm in ARMS:
            od = b1_root / arm
            od.mkdir(parents=True, exist_ok=True)
            m, sd = float(norm[arm]["mean"]), float(norm[arm]["std"])
            for tag, pd in (("odd", pd_odd), ("even", pd_even)):
                half_dir = d / tag
                half_dir.mkdir(parents=True, exist_ok=True)
                np.save(half_dir / "pd_linear.npy", pd)
                z = ((fwd(arm, pd, s, valid) - m) / sd).astype(np.float32)
                # cross 配对下 train.py 只用 _x；_y 写同一份，保持目录约定完整
                np.save(od / f"{seq}_{tag}_x.npy", z)
                np.save(od / f"{seq}_{tag}_y.npy", z)

        for tag, pd in (("odd", pd_odd), ("even", pd_even)):
            u = pd[valid].astype(np.float64) / s
            print(f"[{seq}/{tag}] u_median={np.median(u):.3f} u_p99={np.percentile(u,99):.2f} "
                  f"valid={int(valid.sum())}")
        print(f"[{seq}] s={s:.6g} 两半中位 {m_odd:.6g} / {m_even:.6g}")

    (ROOT / "reports/v3_round6").mkdir(parents=True, exist_ok=True)
    p = ROOT / "reports/v3_round6/provenance.json"
    p.write_text(json.dumps(provenance, indent=1, ensure_ascii=False))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

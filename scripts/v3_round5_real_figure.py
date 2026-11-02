"""Round5 定性图（**真实采集**）：把三条采样臂放到 SA64 / HSA64 上做零样本对比。

为什么需要这张图
----------------
`reports/figures/v3_round5_sampling_direct.png` 下排那张切片来自
`data/processed/v3_round2_s0/low6`——那是**仿真**散斑数据（manifest：`flow_contrast_db 8.0`、
`sigma_eps 0.0385`、`rho` 均值 0.0098），结构本来就弱，"洗不洗差别不大"是数据决定的，
不是方法结论。要看直观差异必须换成真实采集（用户 2026-11-05 的要求）。

口径（必须诚实标注的三件事）
----------------------------
1. **零样本**：网络只在仿真 low6 上训练过，这里直接前向到真实数据，没有在真实数据上训练
   （真实数据只有一卷、两个噪声实现奇/偶，训了就记住，没有无偏评分集——见
   `scripts/v3_phase5_real_network.py`）。
2. **无真值**：真实数据的"干净 PD"不存在，所以本图**只做定性**，不进任何判据。
3. **尺度对齐是启发式**：输入是 `data/processed/official/<DS>/pd_linear.npy`（默认官方
   血流频带 60–350 模、400 帧求和，**已还原成线性功率**，不是 `log_dynamic_range_db=60`
   的显示压缩量）。仿真里 `u = PD/s` 的背景水平中位数≈1，所以这里令 `s` 使
   `median(log u)` 与仿真一致——**背景水平对齐**，而不是拿某根亮血管的 p99 对齐
   （那样整体会亮 5 倍以上，网络看到的输入和训练时不是同一个分布）。
   三条臂走同一套变换，因此**臂间对比是公平的**；绝对质量未标定。

BD-2 的 25% 覆盖
----------------
`reconstruct_full` 只学了 `1→6`、`4→3` 两条角色映射，m=2 下 8 个相位只有 {3,6} 有输出，
所以单卷部署只恢复 25% 体素（这是固定角色采样的**部署代价**，不是实现缺陷）。
未恢复体素渲染成灰色，与 `v3_round5_sampling_direct.png` 同一口径。

用法
----
    python -u scripts/v3_round5_real_figure.py --gpu 2
产物
----
    reports/figures/v3_round5_real_sa64_hsa64.png
    reports/v3_round5/real_zeroshot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from v3_round5_eval import ARMS, SCATTER, npz_path, reconstruct_full, usable  # noqa: E402
from v3_round4_metrics import load, masked_gauss                               # noqa: E402
from v3_round4_b1_prep import fwd, inv                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["SA64", "HSA64"]
GAUSS_SIGMA = 0.6          # 与 A5 冻结的 σ*（low 档）一致

SERIES = [
    ("input", "input（未去噪）", "#7a7a7a"),
    ("gauss", "gauss 平滑 σ*=0.6", "#1f77b4"),
    ("D1",    "D1 · 对角配对（间距 1）", "#2ca02c"),
    ("BDlin", "BD-2 · linear", "#ff9f0a"),
    ("BDlog", "BD-2 · logeps", "#d62728"),
]


def official(ds: str) -> tuple[np.ndarray, np.ndarray]:
    d = ROOT / f"data/processed/official/{ds}"
    pd = np.load(d / "pd_linear.npy").astype(np.float64)
    valid = np.load(d / "valid_mask.npy").astype(bool)
    return pd, valid


def sim_bg_u_median() -> float:
    """仿真训练分布里 `u = PD/s` 的中位数（背景水平），用作真实数据的尺度锚点。"""
    E = load(npz_path(904))
    u = np.expm1(E["vol"])                 # vol = log1p(u) → u
    v = E["valid"].astype(bool)
    return float(np.median(u[v]))


def run_dataset(ds: str, norm: dict, gpu: int, device) -> dict:
    import torch
    from models.network import NetworkCNR

    pd, valid = official(ds)
    shape = usable(pd.shape, 2)

    # 尺度锚点：令 u 的中位数与仿真一致（背景水平对齐）
    u_med_sim = sim_bg_u_median()
    s = float(np.median(pd[:shape[0], :shape[1], :shape[2]][valid[:shape[0], :shape[1], :shape[2]]])
              / u_med_sim)

    pd_c = pd[:shape[0], :shape[1], :shape[2]]
    v_c = valid[:shape[0], :shape[1], :shape[2]]

    out = {"s": s, "shape": list(shape),
           "u_median": float(np.median(pd_c[v_c]) / s),
           "u_p99": float(np.percentile(pd_c[v_c] / s, 99.0)),
           "arms": {}}

    # input / gauss（gauss 与 A5 同口径：在冻结的 log1p 输入域上做归一化卷积）
    lg_in = fwd("log1p", pd_c, s, v_c)
    out["arms"]["input"] = {"pd": pd_c, "filled": np.ones(shape, bool)}
    out["arms"]["gauss"] = {"pd": np.maximum(np.expm1(masked_gauss(lg_in, v_c, GAUSS_SIGMA)), 0.0) * s,
                            "filled": np.ones(shape, bool)}

    for arm, key, scat in (("S_log", "D1", None), ("O_lin", "BDlin", SCATTER["O_lin"]),
                           ("O_log", "BDlog", SCATTER["O_log"])):
        spec = ARMS[arm]
        dom, m = spec["domain"], spec["stride"]
        ck = ROOT / f"runs/v3_round5/{arm}_s3407/latest.pt"
        if not ck.exists():
            raise FileNotFoundError(f"缺少 checkpoint：{ck}（Round5 不重训，请先确认 runs/ 目录）")
        state = torch.load(ck, map_location=device)
        targs = state.get("args", {})
        model = NetworkCNR(features=targs.get("features", 16), groups=targs.get("groups", 4),
                           residual=(targs.get("arch_head", "direct") == "residual")).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        x = fwd(dom, pd_c, s, v_c)
        z = ((x - norm[dom]["mean"]) / norm[dom]["std"]).astype(np.float32)
        y, filled = reconstruct_full(model, z, m, device, scatter_to=scat)
        xh = y * norm[dom]["std"] + norm[dom]["mean"]
        pd_hat = np.maximum(inv(dom, xh, s), 0.0)
        out["arms"][key] = {"pd": pd_hat, "filled": filled}
        out["arms"][key]["coverage"] = float(filled.mean())

    out["valid"] = v_c
    return out


def to_db(pd: np.ndarray, ref: float, valid: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """线性 PD → dB（相对 `ref`），按 `valid & filled` 取 MIP（axis 0 = z），其余置 NaN。"""
    m = np.where(valid & filled, pd, np.nan)
    mip = np.nanmax(np.where(np.isfinite(m), m, -np.inf), axis=0)
    mip[~np.isfinite(mip)] = np.nan
    db = 10.0 * np.log10(np.maximum(mip, 1e-30) / ref)
    db[db < -60] = np.nan
    return db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--window", type=float, nargs=2, default=[-30.0, -3.0],
                    help="显示动态范围（dB，相对 input 的 99.9 分位）")
    ap.add_argument("--out-png", default="reports/figures/v3_round5_real_sa64_hsa64.png")
    ap.add_argument("--out-json", default="reports/v3_round5/real_zeroshot.json")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib import font_manager

    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                 "Source Han Sans CN", "SimHei", "DejaVu Sans"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    norm = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    res = {}
    for ds in DATASETS:
        print(f"[real] {ds} …", flush=True)
        res[ds] = run_dataset(ds, norm, args.gpu, device)
        print(f"  s={res[ds]['s']:.3e}  u_median={res[ds]['u_median']:.3f} "
              f"u_p99={res[ds]['u_p99']:.2f}  "
              + "  ".join(f"{k}:{v.get('coverage', 1.0):.2f}" for k, v in res[ds]["arms"].items()),
              flush=True)

    # ---- 画图 ----
    nrow, ncol = len(DATASETS), len(SERIES)
    fig = plt.figure(figsize=(4.0 * ncol + 0.9, 4.1 * nrow + 1.5))
    gs = fig.add_gridspec(nrow, ncol + 1, wspace=0.06, hspace=0.16,
                          left=0.035, right=0.925, top=0.86, bottom=0.065,
                          width_ratios=[1] * ncol + [0.05])

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#3d3d3d")
    lo, hi = args.window

    for r, ds in enumerate(DATASETS):
        R = res[ds]
        valid = R["valid"]
        # 参考电平：input 的 99.9 分位（所有臂共用，保证臂间可比）
        ref = float(np.percentile(R["arms"]["input"]["pd"][valid], 99.9))
        for c, (key, label, _color) in enumerate(SERIES):
            ax = fig.add_subplot(gs[r, c])
            a = R["arms"][key]
            db = to_db(a["pd"], ref, valid, a["filled"])
            # 灰格显示：未恢复体素在 MIP 后若整条 (y,x) 无覆盖，则补灰提示（仅视觉，不改数值）
            im = ax.imshow(db, vmin=lo, vmax=hi, cmap=cmap, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=12, pad=8)
            if c == 0:
                ax.set_ylabel(f"{ds}\n深度 z（MIP）", fontsize=11)
            cov = a.get("coverage", 1.0)
            ax.text(0.02, 0.03, f"覆盖 {cov*100:.0f}%", transform=ax.transAxes,
                    fontsize=9, color="w",
                    bbox=dict(fc="k", alpha=0.55, ec="none", pad=2))

    # 存 MIP（dB）供事后核对可读性，不必重跑 GPU
    np.savez_compressed(ROOT / "reports/v3_round5/real_mips_db.npz",
                        **{f"{ds}_{k}": to_db(R["arms"][k]["pd"],
                                             float(np.percentile(R["arms"]["input"]["pd"][R["valid"]], 99.9)),
                                             R["valid"], R["arms"][k]["filled"])
                           for ds, R in res.items() for k, _l, _c in SERIES})

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label=f"线性 PD（dB，相对 input 的 99.9 分位；窗口 {lo:.0f}…{hi:.0f} dB）")

    fig.text(0.5, 0.955, "Round5 定性对照：两条采样几何在**真实采集**上的零样本表现",
             ha="center", fontsize=16)
    fig.text(0.5, 0.915,
             "零样本（仿真 low6 训练的权重 → 真实采集），真实数据无真值，**仅定性、不进判据**；"
             "尺度 s 按「背景 log 中位数」对齐到仿真训练分布",
             ha="center", fontsize=10.5, color="#b00020")
    fig.text(0.5, 0.888,
             "MIP 沿 z 投影（axis 0），显示窗口以 input 的 99.9 分位为 0 dB 参考、所有臂共用；"
             "灰 = 该臂未恢复的体素（BD-2 单卷只覆盖 25%，部署需 8× 推理，见文末脚注）",
             ha="center", fontsize=10)
    fig.text(0.5, 0.022,
             "注：BD-2 的 25% 覆盖来自预注册的角色映射（只学过 1→6、4→3 两条），"
             "是固定角色采样的部署代价，不是实现缺陷；D1（间距 1）覆盖率 100%。",
             ha="center", fontsize=9.5, color="#444")
    fig.savefig(ROOT / args.out_png, dpi=150)
    print(f"saved {ROOT / args.out_png}")

    # ---- JSON（去掉大数组） ----
    dump = {}
    for ds, R in res.items():
        dump[ds] = {k: v for k, v in R.items() if k not in ("valid",)}
        dump[ds]["arms"] = {k: {"coverage": float(v.get("coverage", 1.0))} for k, v in R["arms"].items()}
    (ROOT / args.out_json).write_text(json.dumps(
        {"note": "零样本定性图，无真值，不进判据", "window_db": args.window, "datasets": dump},
        ensure_ascii=False, indent=2))
    print(f"saved {ROOT / args.out_json}")


if __name__ == "__main__":
    main()

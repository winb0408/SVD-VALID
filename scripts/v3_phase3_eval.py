"""V3 Phase 3 / 探针评估：把四条臂的重建结果和仿真真值对起来，并出图。

评估协议（先写死，再看结果）：
  * **评估体** = 与训练不同种子的仿真体（血管几何不同），只用它的 `pd_linear` 和真值
    `gt_flow_pd`；形状裁到能被 2m 整除的前缀（37 -> 36），所有臂用同一个裁剪。
  * **推理** = 8 个相位各前向一次，再按各自格点回填（这就是方案里"用 m^3 个平移格点
    补全体积"的做法，m=1 时退化成 8 个相位全覆盖）。四条臂一律走同一段推理代码，
    包括 m=1 的臂——否则比较的是推理方式而不是采样几何。
  * **共同分母**：所有臂都只在 `m=2 的 8 相位支撑集`（体积的 12.5%）上算指标，
    这样 m=1 的臂不会因为"看到了更多体素"而占便宜。这个支撑集之外的重建一律不评分。
  * **三个数**：
      r      —— 在支撑集上、与真值 log1p 的皮尔逊相关（整体保真）
      r_ves  —— 只算血管体素（膨胀 1 后的血管掩膜）内的相关：能不能把血管恢复出来
      bg     —— 背景体素的平均 log1p 电平。**这个数在 2026-10-21 被判定不可用**，原因写死在
                下面"bg 为什么换成 bg_std"一段里，保留列只为和已有报告对得上。
      bg_std —— 背景体素 log1p 的标准差；再除以输入自己的 bg_std 得到 bg_std_ratio
                （输入 = 1.0 表示一点没压，真值 = 0.0 表示全压掉）。这个才是"残余散斑"。
      sep    —— (血管均值 - 背景均值) / 真值的同一量：对比度保留比例
  * **bg 为什么换成 bg_std（2026-10-21）**：bg 是背景的**均值**，而归一化用的除数是
    `median(PD[valid])`——血管稀疏，这个中位数本身就落在背景散斑的水平上，于是输入体
    的背景 log1p ≈ log1p(1) = 0.693。任何"保直流"的去噪器都不会改变背景均值，所以 bg
    必然卡在 0.693 附近：实测五条臂（含输入本身）全是 0.69–0.70，真值是 0，但这不是
    "没压散斑"的证据，而是这个统计量对"压散斑"根本不敏感——去噪降的是**方差**。
    因此新增 bg_std / bg_std_ratio，已知答案校准：输入 bg_std_ratio = 1.0，真值 = 0.0。
    bg 列不再用于任何结论。
  * 不设"通过/不通过"的门槛：探针要看的是臂间差与它的种子间波动哪个大，
    所以每臂 3 个种子，报告均值与极差。

用法：
    python -u scripts/v3_phase3_eval.py
输出 `reports/v3_phase3_2026-10-20/probe_metrics.json` 与图。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from datasets.sampling import strided_phase                        # noqa: E402
from models.network import NetworkCNR                             # noqa: E402

M_EVAL = 2          # 共同分母用的支撑集始终是 m=2 的 8 相位


def usable(shape, m):
    step = 2 * m
    return tuple((n // step) * step for n in shape)


def phase_slices(shape, m, index, shift=(0, 0, 0)):
    dz, dy, dx = index // 4, (index // 2) % 2, index % 2
    return tuple(slice(m * o + s, n, 2 * m)
                 for n, o, s in zip(shape, (dz, dy, dx), shift))


def support_mask(shape, m):
    mask = np.zeros(shape, bool)
    for i in range(8):
        sl = phase_slices(shape, m, i)
        mask[sl] = True
    return mask


def reconstruct(model, vol, stride, device, scatter_to=None):
    """8 个相位各前向一次，回填到各自格点；返回体积与"哪些体素被填了"。

    2026-10-20 补充 `scatter_to`：固定角色臂只学过"相位 A -> 相位 B"这一条映射，
    把它的输出拿去填 **所有** 8 个相位，等于问它一堆没学过的问题（`fold-all` 口径），
    会系统性低估固定角色臂。`scatter_to` 给出每个输入相位对应的目标相位，才是这些臂
    训练时的真实用法（`own-pairing` 口径）。随机角色臂没有固定对应，只能走 fold-all。
    """
    out = np.zeros(vol.shape, np.float32)
    filled = np.zeros(vol.shape, bool)
    with torch.no_grad():
        for i in range(8):
            target = i if scatter_to is None else scatter_to.get(i)
            if target is None:
                continue
            view = strided_phase(torch.from_numpy(vol), i, stride)
            x = view[None, None].to(device)
            y = model(x)[0, 0].detach().cpu().numpy()
            sl = phase_slices(vol.shape, stride, target)
            out[sl] = y
            filled[sl] = True
    return out, filled


def vessel_mask_dilated(vessels, it=1):
    v = vessels.astype(bool)
    if it <= 0:
        return v
    from scipy.ndimage import binary_dilation
    return binary_dilation(v, iterations=it)


def metrics(pred, truth, common, valid, ves, ref_bg_std=None):
    sel = common & valid
    bg_std = float(np.std(pred[sel & ~ves]))
    m = {
        "r": float(np.corrcoef(pred[sel], truth[sel])[0, 1]),
        "bg": float(np.mean(pred[sel & ~ves])),
        "bg_std": bg_std,
        "bg_std_ratio": bg_std / ref_bg_std if ref_bg_std else float("nan"),
        "r_ves": float(np.corrcoef(pred[sel & ves], truth[sel & ves])[0, 1])
        if (sel & ves).sum() > 10 else float("nan"),
        "n_eval": int(sel.sum()),
    }
    sep = float(np.mean(pred[sel & ves]) - np.mean(pred[sel & ~ves]))
    sep_t = float(np.mean(truth[sel & ves]) - np.mean(truth[sel & ~ves]))
    m["sep"] = sep
    m["sep_ratio"] = sep / sep_t if abs(sep_t) > 1e-9 else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sim", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--runs", default="runs/v3_probe")
    ap.add_argument("--out", default="reports/v3_phase3_2026-10-20")
    ap.add_argument("--shape-div", type=int, default=0,
                    help="把评估体每个轴再裁到该数的整数倍（0=不裁）。CRN 里 AvgPool3d(8) "
                         "对不能被 8 整除的视图会 floor 掉尾部切片，而 m=1 的视图在 36 深时是 18 层，"
                         "所以 32（=16m·2 的最小公倍数）能让所有臂的视图尺寸都回到训练时的整数划分。")
    ap.add_argument("--device", default="cuda:0",
                    help="在 CUDA_VISIBLE_DEVICES 之内选卡；默认 0")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d = np.load(args.eval_sim)
    shape = usable(d["pd_linear"].shape, M_EVAL)
    if args.shape_div:
        shape = tuple((n // args.shape_div) * args.shape_div for n in shape)
    crop = tuple(slice(0, n) for n in shape)
    pd = d["pd_linear"][crop].astype(np.float32)
    gt = d["gt_flow_pd"][crop].astype(np.float32)
    vessels = d["vessels"][crop].astype(bool)
    valid = d["valid"][crop].astype(bool)
    s = float(np.median(pd[valid]))
    vol = np.log1p(pd / max(s, 1e-12)).astype(np.float32)
    vol[~valid] = 0.0
    truth = np.log1p(gt / max(s, 1e-12)).astype(np.float32)
    common = support_mask(shape, M_EVAL) & valid
    ves = vessel_mask_dilated(vessels, 1) & valid
    # bg_std 的参照：input 自己的背景方差（ratio=1 表示没压，truth 的 ratio≈0）
    ref_bg_std = float(np.std(vol[common & ~ves]))
    ref_truth_std = float(np.std(truth[common & ~ves]))
    print(f"bg_std 参照：input={ref_bg_std:.4f}  truth={ref_truth_std:.4f}")

    runs = sorted(Path(args.runs).glob("*/latest.pt"))
    if not runs:
        raise SystemExit(f"{args.runs} 下没有 latest.pt")
    rec = {"eval_sim": args.eval_sim, "shape": list(shape), "common_voxels": int(common.sum()),
           "vessel_voxels_in_common": int((common & ves).sum()),
           "bg_std_ref": {"input": ref_bg_std, "truth": ref_truth_std},
           "arms": {}, "arms_own": {}}

    # own-pairing 口径的分母：**按每条臂自己的 stride 和它学过的两个目标相位**取格点。
    # 2026-10-21 修正：原来写死成 {3,6}@m=2，对 m=1 的诊断臂 E 直接是空集——因为
    # stride=m 时相位偏移是 {0, m}，m=2 的格点全落在偶宇称子格上，m=1 的相位 6 落在
    # 奇宇称子格上，两者不相交，于是 E 的 own-pairing 全是 NaN（不是模型塌了）。
    def own_support_for(stride, targets):
        sup = np.zeros(shape, bool)
        for i in targets:
            sup[phase_slices(shape, stride, i)] = True
        return sup & valid

    # common 口径的分母永远是 {3,6}@m=2（所有臂同一个分母，m=1 的臂也一样）
    common36 = own_support_for(M_EVAL, (3, 6))
    rec["common36_voxels"] = int(common36.sum())
    rec["common36_vessels"] = int((common36 & ves).sum())

    imgs = {}
    imgs_own = {}
    for ck in runs:
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = NetworkCNR(features=a["features"], groups=a["groups"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        arm = ck.parent.name.rsplit("_s", 1)[0]
        seed = ck.parent.name.rsplit("_s", 1)[1]

        pred, filled = reconstruct(model, vol, a["stride"], device)
        m = metrics(pred, truth, common, valid, ves, ref_bg_std)
        m.update({"stride": a["stride"], "mode": a["mode"],
                  "slots": a.get("slots"), "filled": int(filled.sum()),
                  "steps": a["epochs"] * a["steps_per_epoch"]})
        rec["arms"].setdefault(arm, {})[seed] = m
        imgs[(arm, seed)] = pred

        # 第三条：所有臂都走 fold-all，但只在 {3,6} 格点上算——分母与 own-pairing 相同，
        # 于是"m=1 vs m=2 覆盖率"这一问可以在同一个分母上读，不受 fold-all 偏置影响。
        rec.setdefault("arms_common", {}).setdefault(arm, {})[seed] = metrics(
            pred, truth, common36, valid, ves, ref_bg_std)

        row = (f"{arm} s{seed}  fold-all: r={m['r']:.3f} r_ves={m['r_ves']:.3f} "
               f"bg_std={m['bg_std_ratio']:.3f}(mean={m['bg']:.3f})")
        if a.get("slots"):
            sxa, sxb, sya, syb = a["slots"]
            own_support = own_support_for(a["stride"], (sya, syb))
            if "own_support_voxels" not in rec:
                rec["own_support_voxels"] = int(own_support.sum())
                rec["own_support_vessels"] = int((own_support & ves).sum())
            pred2, _ = reconstruct(model, vol, a["stride"], device,
                                   scatter_to={sxa: sya, sxb: syb})
            m2 = metrics(pred2, truth, own_support, valid, ves, ref_bg_std)
            rec["arms_own"].setdefault(arm, {})[seed] = m2
            imgs_own[(arm, seed)] = pred2
            row += (f" | own: r={m2['r']:.3f} r_ves={m2['r_ves']:.3f} "
                    f"bg_std={m2['bg_std_ratio']:.3f}")
        print(row)

    def summarise(block):
        out = {}
        for arm, per in block.items():
            vals = list(per.values())
            out[arm] = {k: {"mean": float(np.mean([v[k] for v in vals])),
                            "min": float(np.nanmin([v[k] for v in vals])),
                            "max": float(np.nanmax([v[k] for v in vals]))}
                        for k in ("r", "r_ves", "bg", "bg_std", "bg_std_ratio",
                                  "sep_ratio")}
            out[arm]["n_seeds"] = len(vals)
        return out

    rec["summary"] = summarise(rec["arms"])
    rec["summary_common"] = summarise(rec["arms_common"])
    rec["summary_own"] = summarise(rec["arms_own"])
    (out_dir / "probe_metrics.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    for tag, summ in (("fold-all（全部体素，对固定角色臂偏低）", rec["summary"]),
                      ("common（所有臂同分母：{3,6} 格点，fold-all）", rec["summary_common"]),
                      ("own-pairing（固定角色臂只填它学过的目标相位）", rec["summary_own"])):
        if not summ:
            continue
        print(f"\n{tag}\n臂                 r (mean [min,max])         r_ves                    bg_std_ratio")
        for arm in sorted(summ):
            s_ = summ[arm]
            print(f"{arm:<18} {s_['r']['mean']:.3f} [{s_['r']['min']:.3f},{s_['r']['max']:.3f}]  "
                  f"{s_['r_ves']['mean']:.3f} [{s_['r_ves']['min']:.3f},{s_['r_ves']['max']:.3f}]  "
                  f"{s_['bg_std_ratio']['mean']:.3f}   (bg均值={s_['bg']['mean']:.3f})")

    make_figures(out_dir, rec, imgs, vol, truth, common, vessels, s)


def make_figures(out_dir, rec, imgs, vol, truth, common, vessels, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted(rec["arms"])
    show = {}
    for arm in arms:
        seeds = sorted(rec["arms"][arm], key=lambda z: rec["arms"][arm][z]["r"])
        show[arm] = imgs[(arm, seeds[len(seeds) // 2])]      # 取中位种子，避免挑最好

    def mip(x):
        p = np.expm1(np.maximum(x, 0))
        return 10 * np.log10(np.maximum(p.max(axis=0), 1e-9) / max(np.median(p[p > 0]), 1e-12))

    panels = [("input log1p(PD)", vol), ("truth (blood only)", truth)] + \
             [(a, show[a]) for a in arms]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6))
    for ax, (name, img) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(mip(img), cmap="magma", vmin=-25, vmax=10, aspect="equal")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="dB re median")
    fig.suptitle("Phase-3 probe: max-intensity projection (held-out simulated volume)", fontsize=10)
    fig.savefig(out_dir / "fig_probe_mip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    keys = [("r", "Pearson r vs truth (support)"),
            ("r_ves", "Pearson r inside vessels"),
            ("bg_std_ratio", "background speckle std / input  (0 = fully suppressed)")]
    for ax, (k, title) in zip(axes, keys):
        means = [rec["summary"][a][k]["mean"] for a in arms]
        lo = [rec["summary"][a][k]["mean"] - rec["summary"][a][k]["min"] for a in arms]
        hi = [rec["summary"][a][k]["max"] - rec["summary"][a][k]["mean"] for a in arms]
        ax.bar(range(len(arms)), means, yerr=[lo, hi], capsize=4, color="#4472c4")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{len(rec['arms'][arms[0]])} seeds per arm; bars = mean, whiskers = min/max",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir}/fig_probe_mip.png  {out_dir}/fig_probe_metrics.png")


if __name__ == "__main__":
    main()

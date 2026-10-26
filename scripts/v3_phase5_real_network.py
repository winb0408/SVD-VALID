"""Phase 5 / 真实数据上的**第四条链**：SVD 输出 → 本项目 ② 网络（零样本前向）。

## 为什么必须有这一条（用户 2026-10-22 的追问）

用户看完 `v3_real_denoise_{SA64,HSA64}.png` 之后提了两点，第二点是决定性的：
"这次比较的仅仅只是在 SVD 基础上加了 Wiener 滤波吗？似乎还没有对比本项目真正要实现的
方法？"——**成立**。那张图的三条链里，①全秩、②SVD、③SVD+Wiener 全是**训练无关**的
基线/沿用；本项目真正要实现的 VALID 式**正交自采样 + 跨槽位重建网络**（`v3_phase3_probe.py`
的 A–E 臂）一条都没上图。

## 为什么是零样本（**不在真实数据上训练**）

三条硬理由，任何一条都足以否掉"在真实数据上训练再评分"：

1. 真实数据只有一个体。L48 §5.6.2 已经证过单场景训练会**记忆**（训练目标比值 0.063，
   低于物理下限 0.615），那个数不是进步，是把这一例的噪声背下来了。
2. 真实数据只有**两份**噪声实现（奇/偶）。训练用奇→偶，评分又用奇/偶，等于拿被评分的
   那批噪声训练；没有第三份实现可以做无偏评分。
3. 真实数据 n2n 的天花板尚未校准（L50(a)）。天花板不明时，任何"调着看到过线"都是自欺。

零样本把三个坑一次绕开：它测的命题变成"**仿真器训练出来的东西能不能迁移到真实采集**"，
这正好是项目外部效度的核心（T2-3），而且是一个可以在跑之前就写下预期的干净命题。

## 预注册（**跑之前写死，不得事后改**）

判据：n2n 与结构保真全部在同一条 SVD 链、同一套掩膜、同一个裁剪网格上取。

- 有效线：`n2n <= n2n(①Wiener w=7)` **且** `core 保留 >= 0.95`；
- 稳定线：**4 个 seed 全部**满足有效线，才可称"在真实数据上有效"；
- 任一条不满足 → 记为负结果，**不得改阈值/换频带/挑 seed 重跑**。

跑之前的预期（作者押注，先记下来防止事后叙事）：**④ 很可能输**。依据是仿真器上
② 跨场景均值 0.831 已差于 ① 的 0.754，且 **血管区 n2n 1.31–1.81（比输入还差）**；
再叠一层零样本域偏移（仿真散斑统计 vs 真实采集）。预估 ④ 的 n2n 落在 0.3–1.0、
core 保留低于 0.95。

## 网格对齐（一个必须说清的比较口径）

`②` 的 checkpoint 是在 `patch=[32,64,64]` 上训练的，而真实体是 `(37,64,64)`。为让 ④ 与
③ **同网格**比较，本脚本把 z 裁到 `[0,32)`，并在**裁剪后的网格上重算** ①②③ 的全部指标
（`masks` 也从裁剪后的共享参考卷重算）。因此本文件里的 ①②③ 数字与 L50 的
`{ds}.json` **会有微小差异**（那版用满 37 层），这是口径差，不是结果变差；两个数都保留。

## 未做的事（不在这条链的范围内）

- 不在**全秩链**上评 ④：那条链的奇/偶 MSE 分母塌了（SA64 241.8×、HSA64 5180.4×），
  没有可用的尺子可打（L50(a)）。
- 不生成新的真实数据频带：半卷血带直接用 `FRAC_LO/FRAC_HI` 的比例换算（30–175 / 200），
  与 ③ 完全一致，**不复用** `data/processed/sa64_{odd,even}_remap/`（那是 10-09 用旧频带
  秩 10–80 / 200 算的，与当前口径不一致）。

用法：
    python -u scripts/v3_phase5_real_network.py --dataset SA64
    python -u scripts/v3_phase5_real_network.py --dataset HSA64
输出：
    reports/figures/v3_real_network_{dataset}.png
    reports/v3_phase5_real/{dataset}_network.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from prepare_svd import load_array, normalize_shape          # noqa: E402
from v3_phase5_real_compare import (                          # noqa: E402
    RAW, FRAC_LO, FRAC_HI, band_pd, keep_ratio, mse, setup_cjk,
    structure_masks, svd_basis, to_unit, wiener,
)

NET_Z = 32          # ② 的 patch=[32,64,64]，z 只能留 32 层
CKPT_GLOB = "runs/v3_n2n_multi/res_s*/latest.pt"


def net_estimator(ckpt_paths, shape3d, device):
    """返回 [(name, fn)]：每个 checkpoint 一个整卷前向的估计器（②=残差+中心可见）。"""
    import torch
    from models.network import NetworkCNR

    out = []
    for ck in ckpt_paths:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        a = blob["args"]
        if a.get("infer") != "full":
            raise SystemExit(f"{ck} 不是整卷前向的 ② checkpoint（infer={a.get('infer')}）")
        model = NetworkCNR(features=a["features"], groups=a["groups"]).to(device)
        model.load_state_dict(blob["model"])
        model.eval()

        def fn(x, model=model, a=a):
            with torch.no_grad():
                xin = torch.as_tensor(np.ascontiguousarray(x), device=device)[None, None].float()
                h = model(xin)
                pred = (xin + h if a.get("residual") else h)[0, 0].cpu().numpy()
            return pred.astype(np.float32)

        name = f"④SVD+②网络 {ck.parent.name}"
        out.append((name, fn, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--w", type=int, default=7, help="① 的窗口（与 L50 一致）")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB)
    ap.add_argument("--out-dir", default="reports/v3_phase5_real")
    args = ap.parse_args()

    import torch
    device = torch.device("cpu")   # 单卷单次前向，32×64×64，CPU 足够；不占 GPU

    ds = args.dataset
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(ROOT.glob(args.ckpt_glob))
    if not ckpts:
        raise SystemExit(f"没找到 checkpoint：{args.ckpt_glob}")
    print(f"[{ds}] 零样本 checkpoint：{[p.parent.name for p in ckpts]}")

    # ---------------- 数据：全秩 / SVD / 奇偶两卷 ----------------
    raw = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    z, x, y, t = raw.shape
    shape3d = (z, x, y)
    print(f"[{ds}] raw {raw.shape} {raw.dtype}")
    C = raw.reshape(-1, t)
    _, V = svd_basis(C)
    lo_b, hi_b = int(round(FRAC_LO * t)), int(round(FRAC_HI * t))
    pd_full = band_pd(C, V, 0, t, shape3d)
    pd_svd = band_pd(C, V, lo_b, hi_b, shape3d)
    del C, V, raw

    raw4 = normalize_shape(load_array(RAW[ds]), args.emitting_array, args.frames)
    halves = {}
    for tag, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        Ch = raw4.reshape(-1, t)[:, sl]
        th = Ch.shape[1]
        _, Vh = svd_basis(Ch)
        halves[tag] = {
            "full": band_pd(Ch, Vh, 0, th, shape3d),
            "svd": band_pd(Ch, Vh, round(FRAC_LO * th), round(FRAC_HI * th), shape3d),
        }
        del Ch, Vh
    del raw4
    th_lo, th_hi = int(round(FRAC_LO * (t // 2))), int(round(FRAC_HI * (t // 2)))
    print(f"[{ds}] 全秩=[0,{t}) 血带=[{lo_b},{hi_b})  半卷 T={t//2} 血带=[{th_lo},{th_hi})")

    # ---------------- 裁剪到网络网格 (32,64,64) ----------------
    crop = (slice(0, NET_Z), slice(0, x), slice(0, y))
    for tag in halves:
        for k in halves[tag]:
            halves[tag][k] = halves[tag][k][crop]
    pd_full, pd_svd = pd_full[crop], pd_svd[crop]
    zc_, xc_, yc_ = pd_svd.shape
    print(f"[{ds}] 网络网格 {pd_svd.shape}（z 裁掉 {z - zc_} 层）")

    # ---------------- 有效体素与结构掩膜（全部在裁剪网格上重算）----------------
    zz, yy, xx = np.mgrid[0:zc_, 0:xc_, 0:yc_]
    valid = ((zz >= 2) & (zz <= zc_ - 3) & (yy >= 4) & (yy <= xc_ - 5)
             & (xx >= 4) & (xx <= yc_ - 5))
    o, e = halves["odd"], halves["even"]
    valid_svd = valid & (0.5 * (o["svd"] + e["svd"]) > 0)
    valid_full = valid & (0.5 * (o["full"] + e["full"]) > 0)

    def prep_pair(key, vv):
        s = float(np.median(0.5 * (o[key] + e[key])[vv]))
        f = lambda pd: np.log1p(np.maximum(pd, 0.0) / s).astype(np.float32)
        return s, f(o[key]), f(e[key])

    s_s, xo, xe = prep_pair("svd", valid_svd)
    s_f, xo_f, xe_f = prep_pair("full", valid_full)
    print(f"[{ds}] 归一化常数 全秩 s={s_f:.3e}  SVD s={s_s:.3e}")
    print(f"[{ds}] 有效体素 全秩 {int(valid_full.sum())}  SVD {int(valid_svd.sum())}")

    ref_unit = to_unit(0.5 * (xo + xe), valid_svd)
    masks = structure_masks(ref_unit)
    print(f"[{ds}] 掩膜 core {int(masks['core'].sum())} body {int(masks['body'].sum())}"
          f" dark {int(masks['dark'].sum())} isolated {int(masks['isolated'].sum())}")

    # ---------------- 估计器（每条都对 odd/even 各跑一遍，再看奇偶一致性）----------------
    nets = net_estimator(ckpts, pd_svd.shape, device)
    est = {
        "不处理（SVD 输出）": (xo, xe),
        f"①Wiener w={args.w}": (wiener(xo, args.w, valid_svd)[0],
                                wiener(xe, args.w, valid_svd)[0]),
    }
    for name, fn, _a in nets:
        est[name] = (fn(xo), fn(xe))

    ref = mse(xo, xe, valid_svd)
    rows, keep, corr, pred_odd = {}, {}, {}, {}
    for k, (p, q) in est.items():
        rows[k] = mse(p, q, valid_svd) / ref
        corr[k] = float(np.corrcoef(p[valid_svd], q[valid_svd])[0, 1])
        keep[k] = {m: keep_ratio(p, xo, masks[m])
                   for m in ("core", "body", "dark", "isolated")}
        pred_odd[k] = p

    base_name = f"①Wiener w={args.w}"
    net_names = [n for n, _f, _a in nets]

    # ---------------- 预注册判定（阈值写死，见 docstring）----------------
    THR_CORE = 0.95
    verdict = {}
    for n in net_names:
        ok_n2n = rows[n] <= rows[base_name]
        ok_core = keep[n]["core"] >= THR_CORE
        verdict[n] = {"n2n_ok": bool(ok_n2n), "core_ok": bool(ok_core),
                      "pass": bool(ok_n2n and ok_core)}
    n_pass = sum(v["pass"] for v in verdict.values())
    stable = n_pass == len(net_names)

    print(f"\n[{ds}] 网格 {pd_svd.shape}  参照 n2n({base_name}) = {rows[base_name]:.4f}")
    print(f"{'估计器':28s} {'n2n':>8s} {'core':>6s} {'body':>6s} {'dark':>6s}"
          f" {'isol':>6s} {'corr':>7s}  判定")
    for k in rows:
        v = verdict.get(k)
        if v is None:
            tag = ""
        elif v["pass"]:
            tag = "通过"
        else:
            tag = "不通过: " + ",".join(t for t, b in (("n2n", v["n2n_ok"]),
                                                       ("core", v["core_ok"])) if not b)
        print(f"{k:28s} {rows[k]:8.4f} {keep[k]['core']:6.3f} {keep[k]['body']:6.3f}"
              f" {keep[k]['dark']:6.3f} {keep[k]['isolated']:6.3f} {corr[k]:7.4f}  {tag}")
    print(f"[{ds}] 预注册结论：{n_pass}/{len(net_names)} 个 seed 通过；"
          f"{'达到稳定线' if stable else '未达到稳定线（判负）'}")

    # ---------------- 记盘 ----------------
    best = min(net_names, key=lambda n: rows[n])
    rec = {
        "dataset": ds, "grid": list(pd_svd.shape), "T": t,
        "band_full": [0, t], "band_blood": [lo_b, hi_b],
        "band_blood_half": [th_lo, th_hi],
        "norm_const": {"full": s_f, "svd": s_s},
        "checkpoints": [str(p.relative_to(ROOT)) for p in ckpts],
        "protocol": "零样本：checkpoint 只在仿真场景 seed2-4 训练、seed5 早停，未在真实数据上训练",
        "prereg": {
            "rule": "n2n <= n2n(①Wiener) 且 core>=0.95；4 个 seed 全满足才可称有效",
            "core_thr": THR_CORE,
            "prediction": ("④ 很可能输：预估 n2n 0.3–1.0、core < 0.95。依据：仿真上 ② 均值 "
                           "0.831 差于 ① 0.754，血管区 n2n 1.31–1.81，再叠零样本域偏移。"),
            "written_before_run": True,
        },
        "n2n_svd_chain": rows, "corr_odd_even": corr, "structure_keep_odd": keep,
        "mask_voxels": {k: int(v.sum()) for k, v in masks.items()},
        "valid_voxels": {"full": int(valid_full.sum()), "svd": int(valid_svd.sum())},
        "verdict": verdict, "seeds_passed": n_pass, "stable": bool(stable),
        "best_net": best,
        "note": ("真实数据 n2n 不做绝对通过/不通过判定（天花板未校准）；上表的通过线是"
                 "**相对 ①** 的预注册门槛，只用于回答'④ 能不能赢 ①'。"),
    }
    (out_dir / f"{ds}_network.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1))

    # ---------------- 图：四栏 MIP + 放大 + 柱状 ----------------
    plt = setup_cjk()
    from scipy.ndimage import zoom

    def to_db(v):
        return 10.0 * np.log10(np.maximum(v, 1e-30))

    pw_o = wiener(xo, args.w, valid_svd)[0]

    def unlog(v, s):
        return s * np.expm1(v)

    panels = [("① 去噪前（全秩 0–T）", pd_full, valid_full),
              (f"② 单纯 SVD（秩 {lo_b}–{hi_b}）", pd_svd, valid_svd),
              (f"③ ②+①Wiener（w={args.w}）", unlog(pw_o, s_s), valid_svd),
              (f"④ ②+本项目 ② 网络（零样本，{best.split()[-1]}）",
               unlog(pred_odd[best], s_s), valid_svd)]

    mips = [to_db(np.max(p, axis=0)) for _t, p, _v in panels]
    m2d = [v.any(axis=0) for _t, _p, v in panels]
    vhi = max(float(np.percentile(m[m2], 99.9)) for m, m2 in zip(mips[1:], m2d[1:]))
    vlo = max(min(float(np.percentile(m[m2], 1.0)) for m, m2 in zip(mips, m2d)),
              vhi - 45.0)

    fig = plt.figure(figsize=(22, 15))
    for i, ((title, _p, _v), m, m2) in enumerate(zip(panels, mips, m2d)):
        ax = fig.add_subplot(3, 4, i + 1)
        im = ax.imshow(m, vmin=vlo, vmax=vhi, cmap="inferno", interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[:4], fraction=0.02, pad=0.01, label="dB（四栏同窗）")

    # 第二排：同一角落的原始像素（与 L50 图同一切片口径）
    zmid = pd_svd.shape[0] // 2
    y0, y1, x0, x1 = 20, 44, 20, 44
    unit_ref = to_unit(0.5 * (xo + xe), valid_svd)
    yy2, xx2 = np.unravel_index(
        int(np.argmax(np.where(unit_ref[zmid] > 0.0, unit_ref[zmid], -1))), unit_ref[zmid].shape)
    y0 = int(np.clip(yy2 - 12, 0, xc_ - 24)); x0 = int(np.clip(xx2 - 12, 0, yc_ - 24))
    y1, x1 = y0 + 24, x0 + 24
    for j, (lbl, vol) in enumerate((("②", pd_svd), ("③", unlog(pw_o, s_s)),
                                    ("④", unlog(pred_odd[best], s_s)))):
        ax = fig.add_subplot(3, 4, 5 + j)
        seg = to_db(vol[zmid, y0:y1, x0:x1])
        ax.imshow(seg, vmin=vlo, vmax=vhi, cmap="inferno", interpolation="nearest")
        ax.set_title(f"{lbl} 放大 z={zmid} [{y0}:{y1},{x0}:{x1}]", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 4, 8)
    d = to_db(unlog(pred_odd[best], s_s))[zmid] - to_db(unlog(pw_o, s_s))[zmid]
    lim = float(np.percentile(np.abs(d), 99)) or 1.0
    im2 = ax.imshow(d, vmin=-lim, vmax=lim, cmap="coolwarm", interpolation="nearest")
    ax.set_title("④ − ③（dB，正=④ 更亮）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)

    # 第三排：n2n 与 core 保留
    names = list(rows)
    ax = fig.add_subplot(3, 2, 5)
    ax.bar(range(len(names)), [rows[n] for n in names],
           color=["#888"] + ["#4c72b0"] + ["#c44e52"] * len(net_names))
    ax.axhline(rows[base_name], ls="--", c="k", lw=1,
               label=f"预注册线 = ① {rows[base_name]:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8, rotation=20)
    ax.set_ylabel("n2n（越低越好）", fontsize=10); ax.legend(fontsize=9)
    ax.set_title("噪声：④ vs ①（同链同掩膜）", fontsize=11)

    ax = fig.add_subplot(3, 2, 6)
    ax.bar(range(len(names)), [keep[n]["core"] for n in names],
           color=["#888"] + ["#4c72b0"] + ["#c44e52"] * len(net_names))
    ax.axhline(THR_CORE, ls="--", c="k", lw=1, label=f"预注册线 = {THR_CORE}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8, rotation=20)
    ax.set_ylabel("血管核保留（越接近 1 越好）", fontsize=10); ax.legend(fontsize=9)
    ax.set_title("结构：④ 有没有把血管糊掉", fontsize=11)

    fig.suptitle(f"{ds}：第四条链 = SVD + 本项目 ② 网络（零样本，未在真实数据上训练）"
                 f"　通过 {n_pass}/{len(net_names)} 个 seed", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outp = fig_dir / f"v3_real_network_{ds}.png"
    fig.savefig(outp, dpi=110)
    print(f"[{ds}] 图 → {outp}")
    print(f"[{ds}] 数值 → {out_dir / f'{ds}_network.json'}")


if __name__ == "__main__":
    main()

"""判据 J1.1 的**已知答案标定**（§0.5 第 2 条）。

## 为什么是"剂量-响应"而不是"给一堆估计器排序"

J1.0 的第一版标定试过给 7 个估计器排序，**失败**（见 `v3_judge.py` 的版本变更记录）：
幻影里共享静态杂波占了误差的绝大部分，各估计器相对真值的总误差差异只有 ~0.1 dB，
落在真值自身散斑的噪声地板里，排序不可信。更关键的是"总 MSE"本身被背景主导，
看不出结构破坏——这正是项目当初为什么要另设 core/body/dark 分区指标。

所以改成**受控剂量扫描**：每道门对应一类**只由它负责**的缺陷，逐剂量看三件事——
(1) 该门是否单调响应；(2) 另外两道门是否（正确地）不响应；(3) 真值误差是否朝预期方向走。
这直接检验"这道门有没有在管它该管的事"，不依赖估计器之间的细微排序。

三类缺陷（都与实际观察到的失败方式对应）：
- **各向同性模糊**（对应"抹平换分"，③ 在 HSA64 的病）：应被 **S** 抓住，**N** 会反向。
- **背景均匀抬亮**（对应"往背景添东西"，④ 的病）：应被 **B** 抓住，**N** 对零方差偏移
  在数学上**完全无反应**（`p−q` 里常数抵消），这是 N 的结构性盲区。
- **叠加零均值噪声**（对应"没去噪"）：应被 **N** 抓住，**B** 对零均值不敏感。

## 预先写下的判定（**标定协议 v2**）

> ⚠️ 协议从 v1 改到 v2，改动只有"用什么统计量衡量敏感性"，改因与两次失败运行的原始数字
> 全部记在 `reports/v3_judge/JUDGE_v1.md` §2。v1 用 Spearman 秩相关，两个问题：(i) 剂量
> 构造成恒定时（抬亮扫描里 n2n 精确恒为 1.0000）秩相关无定义；(ii) 单调但幅度仅 0.07 dB
> 的交叉响应（注噪扫描里 B_db）会得到 ρ=−1.00，把"数值上完全不敏感"读成"完美反向"。
> 敏感性应当按**扫描首末的响应幅度**衡量，这是 v2 的唯一改动。

- **C1（证伪旧判据）**：模糊扫描中 `N_rel` 至少减半（n2n 变"好"），同时**真值核心区误差
  至少 +3 dB**（真实损伤加剧）→ **n2n 作为验收判据被证伪**。
- **C2**：模糊扫描中 `S_rel` 至少掉 0.20，且 σ=3 时 S 门不通过。
- **C3**：抬亮扫描中 `B_db` 至少升 2 dB 且 c 最大时 B 门不通过；同时 `|ΔN_rel| < 1%`
  （N 对抬亮不敏感——常数在 `p−q` 里数学上抵消）。
- **C4**：注噪扫描中 `N_rel` 至少翻倍且 α 最大时 N 门不通过；同时 `|ΔB_db| < 0.5 dB`。
- **C5（合取检查）**：温和收缩(w=5)与项目原设置(w=7) 三道门全过；模糊/抬亮/注噪各自
  至少一道门不过；"不处理"因 N=1 不过。
- 任一不满足 → 判据不成立，如实记为失败，**不得改门槛让它过**。

用法：
    python -u scripts/v3_judge_calibration.py
输出：
    reports/v3_judge/calibration.json
    reports/figures/v3_judge_calibration.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from v3_phase5_real_compare import (                      # noqa: E402
    FRAC_LO, FRAC_HI, band_pd, setup_cjk, structure_masks, svd_basis, to_unit, wiener,
)
from v3_judge import (                                    # noqa: E402
    JUDGE_VERSION, THR_B_DB, THR_N_REL, THR_S_REL, THR_TEXT, judge, mse,
)
from v3_simulator import SimCfg, gauss_field, simulate      # noqa: E402

NET_Z = 32


def _wiener(x, w, valid, sig_mult):
    """与 `v3_phase5_real_compare.wiener` 逐位一致的公式，外加 sig2 的倍率。"""
    mu = uniform_filter(x, w)
    lv = np.maximum(uniform_filter(x * x, w) - mu * mu, 0.0)
    sig2 = float(np.median(lv[valid])) * sig_mult
    b = np.clip(1.0 - sig2 / np.maximum(lv, 1e-9), 0.0, 1.0)
    return (mu + b * (x - mu)).astype(np.float32)


def flow_realization(vessels, cfg, seed):
    """与 `simulate` 内血流分支逐位一致的独立实现（同一血管，不同散斑）。"""
    rng = np.random.default_rng(seed)
    flow = np.empty(tuple(cfg.shape) + (cfg.T,), np.complex64)
    for t in range(cfg.T):
        flow[..., t] = gauss_field(rng, tuple(cfg.shape), tuple(cfg.psf_sigma)) * vessels
    if vessels.any():
        v_rms = np.sqrt(np.mean(np.abs(flow[..., 0][vessels]) ** 2))
        flow *= (cfg.flow_amp * 0.6) / max(float(v_rms), 1e-12)
    return flow.astype(np.complex64) * 1e3


def build_phantom(scene_seed, noise_seed, truth_reps):
    cfg = SimCfg(shape=(32, 64, 64), T=400, seed=scene_seed,
                 noise_seed=noise_seed, static_seed=None)
    sim = simulate(cfg)
    X, flow, vessels = sim["X"], sim["flow"], sim["vessels"]
    z, x, y, t = X.shape
    shape3d = (z, x, y)
    print(f"[calib] 幻影 {X.shape}  血管体素 {int(vessels.sum())}")

    halves, Vs = {}, {}
    for tag, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        Ch = X.reshape(-1, t)[:, sl]
        th = Ch.shape[1]
        _, Vh = svd_basis(Ch)
        halves[tag] = band_pd(Ch, Vh, round(FRAC_LO * th), round(FRAC_HI * th), shape3d)
        Vs[tag] = Vh
        del Ch
    th = t // 2
    th_lo, th_hi = int(round(FRAC_LO * th)), int(round(FRAC_HI * th))
    print(f"[calib] 半卷 T={th} 血带=[{th_lo},{th_hi})")

    truth = {}
    for tag in halves:
        sl = slice(0, None, 2) if tag == "odd" else slice(1, None, 2)
        acc = np.zeros(shape3d, np.float64)
        for k in range(truth_reps):
            fk = (flow if k == 0 else flow_realization(vessels, cfg, 9100 + k))
            acc += band_pd(fk.reshape(-1, t)[:, sl], Vs[tag], th_lo, th_hi, shape3d)
        truth[tag] = (acc / truth_reps).astype(np.float32)
        del acc

    crop = (slice(0, NET_Z), slice(0, x), slice(0, y))
    for d in (halves, truth):
        for k in d:
            d[k] = d[k][crop]
    zc, xc, yc = halves["odd"].shape

    zz, yy, xx = np.mgrid[0:zc, 0:xc, 0:yc]
    valid = ((zz >= 2) & (zz <= zc - 3) & (yy >= 4) & (yy <= xc - 5)
             & (xx >= 4) & (xx <= yc - 5))
    o, e = halves["odd"], halves["even"]
    valid = valid & (0.5 * (o + e) > 0)

    s = float(np.median(0.5 * (o + e)[valid]))
    f = lambda pd: np.log1p(np.maximum(pd, 0.0) / s).astype(np.float32)   # noqa: E731
    xo, xe = f(o), f(e)
    truth_log = 0.5 * (f(truth["odd"]) + f(truth["even"]))
    masks = structure_masks(to_unit(0.5 * (xo + xe), valid))
    print(f"[calib] 有效体素 {int(valid.sum())}  core {int(masks['core'].sum())} "
          f"body {int(masks['body'].sum())} dark {int(masks['dark'].sum())}")
    return xo, xe, truth_log, masks, valid, dict(shape=list(X.shape), band=[th_lo, th_hi])


def score(p, q, xo, xe, truth_log, masks, valid):
    """三路判据 + 三个分区的真值误差（真值已知，故可算）。"""
    j = judge(p, q, xo, xe, masks, valid)
    for tag, mk in (("all", valid), ("core", masks["core"] & valid),
                    ("dark", masks["dark"] & valid)):
        j[f"true_{tag}_rel"] = (mse(p, truth_log, mk)
                                / max(mse(xo, truth_log, mk), 1e-30))
        j[f"true_{tag}_db"] = float(10.0 * np.log10(max(j[f"true_{tag}_rel"], 1e-30)))
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-seed", type=int, default=3)
    ap.add_argument("--noise-seed", type=int, default=303)
    ap.add_argument("--truth-reps", type=int, default=3)
    ap.add_argument("--out-dir", default="reports/v3_judge")
    args = ap.parse_args()

    xo, xe, truth_log, masks, valid, meta = build_phantom(
        args.scene_seed, args.noise_seed, args.truth_reps)

    # 自检：本地 wiener 副本在 sig_mult=1 时必须与项目原实现逐位一致
    assert np.array_equal(wiener(xo, 7, valid)[0], _wiener(xo, 7, valid, 1.0)), \
        "本地 wiener 副本与项目实现不一致"

    rng = np.random.default_rng(7777)

    # ---------------- 三组剂量扫描 ----------------
    sweeps = {}
    sigmas = [0.0, 0.5, 1.0, 2.0, 3.0]
    sweeps["blur"] = {
        "dose": sigmas, "unit": "高斯 σ（体素）",
        "rows": [score(gaussian_filter(xo, s), gaussian_filter(xe, s),
                       xo, xe, truth_log, masks, valid) if s > 0 else
                 score(xo, xe, xo, xe, truth_log, masks, valid) for s in sigmas],
    }
    offs = [0.0, 0.05, 0.15, 0.30, 0.60]
    sweeps["offset"] = {
        "dose": offs, "unit": "均匀抬亮 c（log1p 单位）",
        "rows": [score(xo + c, xe + c, xo, xe, truth_log, masks, valid) for c in offs],
    }
    alphas = [0.0, 0.3, 0.6, 1.0, 1.5]
    # 必须给奇/偶**各自独立**的噪声：加同一张噪声会在 p−q 里精确抵消，
    # 测的就不是"注噪"而是"抬亮"（v1 的 bug，已修）。
    def mk_noise():
        z = gaussian_filter(rng.standard_normal(xo.shape), 1.0).astype(np.float32)
        return z / max(float(z.std()), 1e-12)
    noise_o, noise_e = mk_noise(), mk_noise()
    sweeps["noise"] = {
        "dose": alphas, "unit": "叠加零均值噪声 α",
        "rows": [score(xo + a * noise_o * 0.1, xe + a * noise_e * 0.1,
                       xo, xe, truth_log, masks, valid) for a in alphas],
    }

    # ---------------- 合取检查用的估计器阶梯 ----------------
    ladder = {
        "不处理(SVD 输出)": (xo, xe),
        "温和收缩 w=5": (_wiener(xo, 5, valid, 1.0), _wiener(xe, 5, valid, 1.0)),
        "项目原设置 w=7": (_wiener(xo, 7, valid, 1.0), _wiener(xe, 7, valid, 1.0)),
        "过度平滑 w=13": (_wiener(xo, 13, valid, 1.0), _wiener(xe, 13, valid, 1.0)),
        "阈值虚高 ×4": (_wiener(xo, 7, valid, 4.0), _wiener(xe, 7, valid, 4.0)),
        "高斯模糊 σ=2": (gaussian_filter(xo, 2.0), gaussian_filter(xe, 2.0)),
        "抬亮 c=0.3": (xo + 0.3, xe + 0.3),
        "注噪 α=1.5": (xo + 1.5 * noise_o * 0.1, xe + 1.5 * noise_e * 0.1),
    }
    lad = {k: score(p, q, xo, xe, truth_log, masks, valid)
           for k, (p, q) in ladder.items()}

    # ---------------- 判定 ----------------
    # ---------------- 判定（标定协议 v2，见 docstring 的变更说明）----------------
    def span(key, field):
        """扫描首末两档的差值（响应幅度），比秩相关更贴"敏不敏感"这个问题。"""
        r = sweeps[key]["rows"]
        return float(r[-1][field] - r[0][field])

    checks = {
        "C1_n2n_falsified": bool(
            span("blur", "n_rel") <= -0.5 * sweeps["blur"]["rows"][0]["n_rel"]
            and span("blur", "true_core_db") >= 3.0),
        "C2_S_tracks_blur": bool(
            span("blur", "s_rel") <= -0.20 and not sweeps["blur"]["rows"][-1]["ok_s"]),
        "C3_B_tracks_offset": bool(
            span("offset", "b_db") >= 2.0
            and abs(span("offset", "n_rel")) < 0.01
            and not sweeps["offset"]["rows"][-1]["ok_b"]),
        "C4_N_tracks_noise": bool(
            sweeps["noise"]["rows"][-1]["n_rel"] >= 2.0 * sweeps["noise"]["rows"][0]["n_rel"]
            and abs(span("noise", "b_db")) < 0.5
            and not sweeps["noise"]["rows"][-1]["ok_n"]),
        "C5_ladder_conjunction": bool(
            lad["温和收缩 w=5"]["pass"] and lad["项目原设置 w=7"]["pass"]
            and not lad["不处理(SVD 输出)"]["pass"]
            and not lad["高斯模糊 σ=2"]["pass"]
            and not lad["抬亮 c=0.3"]["pass"]
            and not lad["注噪 α=1.5"]["pass"]),
    }
    spans = {f"{k}/{fld}": span(k, fld) for k in sweeps
             for fld in ("n_rel", "b_db", "s_rel", "true_all_db",
                         "true_core_db", "true_dark_db")}
    j1_ok = all(checks.values())

    # ---------------- 打印 ----------------
    for k, sw in sweeps.items():
        print(f"\n=== 扫描 {k}（{sw['unit']}）===")
        print(f"  {'剂量':>6s} {'N_rel':>8s} {'B_db':>8s} {'S_rel':>8s} "
              f"{'真值 core dB':>12s} {'真值 dark dB':>12s}  过门")
        for d, r in zip(sw["dose"], sw["rows"]):
            gates = "".join(t for t, o in (("N", r["ok_n"]), ("B", r["ok_b"]),
                                           ("S", r["ok_s"])) if o) or "-"
            print(f"  {d:6.2f} {r['n_rel']:8.4f} {r['b_db']:8.2f} {r['s_rel']:8.4f} "
                  f"{r['true_core_db']:12.3f} {r['true_dark_db']:12.3f}  {gates}")
    print(f"\n=== 合取检查（估计器阶梯）===")
    print(f"  {'估计器':20s} {'N_rel':>8s} {'B_db':>8s} {'S_rel':>8s} "
          f"{'真值 all dB':>12s}  未过")
    for k, r in lad.items():
        bad = ",".join(t for t, o in (("N", r["ok_n"]), ("B", r["ok_b"]),
                                      ("S", r["ok_s"])) if not o) or "全过"
        print(f"  {k:20s} {r['n_rel']:8.4f} {r['b_db']:8.2f} {r['s_rel']:8.4f} "
              f"{r['true_all_db']:12.3f}  {bad}")
    print(f"\n=== 预注册检查 ===")
    for k, v in checks.items():
        print(f"  {k:26s} {'通过' if v else '**不通过**'}")
    print("  响应幅度: " + "  ".join(f"{k}={v:+.3f}" for k, v in spans.items()))
    print(f"\n[calib] {JUDGE_VERSION} 标定{'通过' if j1_ok else '**失败**'}")

    # ---------------- 记盘 + 图 ----------------
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"judge_version": JUDGE_VERSION, "thresholds": THR_TEXT,
           "thr": {"n_rel": THR_N_REL, "b_db": THR_B_DB, "s_rel": THR_S_REL},
           "phantom": dict(meta, scene_seed=args.scene_seed,
                           noise_seed=args.noise_seed, truth_reps=args.truth_reps,
                           valid_voxels=int(valid.sum()),
                           mask_voxels={k: int(v.sum()) for k, v in masks.items()}),
           "sweeps": {k: {"dose": v["dose"], "unit": v["unit"], "rows": v["rows"]}
                      for k, v in sweeps.items()},
           "ladder": lad, "spans": spans, "checks": checks,
           "J1_calibration_passed": j1_ok,
           "prereg": {"rule": "见脚本 docstring 的 C1–C5", "written_before_run": True}}
    (out_dir / "calibration.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    plt = setup_cjk()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    panels = [("blur", "s_db", "S_rel（本判据）", 10.0 * np.log10(THR_S_REL),
               "true_core_db", "真值核心区误差"),
              ("offset", "b_db", "B_db（本判据）", THR_B_DB,
               "true_dark_db", "真值暗区误差"),
              ("noise", "b_db", "B_db（本判据）", THR_B_DB,
               "true_all_db", "真值总误差")]
    titles = {"blur": "各向同性模糊（对应抹平换分）",
              "offset": "背景均匀抬亮（对应 ④ 的病）",
              "noise": "叠加零均值噪声（对应没去噪）"}
    for ax, (key, gfield, glab, gthr, tr, tr_lab) in zip(axes, panels):
        sw = sweeps[key]
        x = sw["dose"]
        to_db = lambda v: 10.0 * np.log10(np.maximum(v, 1e-30))   # noqa: E731
        ax.plot(x, [to_db(r["n_rel"]) for r in sw["rows"]], "o-", color="#c0392b",
                label="N_db（n2n，每档都更「好」）", lw=2)
        ax.plot(x, [r[gfield] for r in sw["rows"]], "s-", color="#2471a3",
                label=glab, lw=2)
        ax.plot(x, [r[tr] for r in sw["rows"]], "k--", alpha=0.55, lw=1.6,
                label=f"{tr_lab}（有真值）")
        ax.axhline(10.0 * np.log10(THR_N_REL), color="#c0392b", ls=":", lw=1.1,
                   label="N 门槛 −3.0 dB")
        ax.axhline(gthr, color="#2471a3", ls=":", lw=1.1, label=f"门槛 {gthr:+.2f} dB")
        ax.set_xlabel(sw["unit"], fontsize=11)
        ax.set_ylabel("dB（越低越好）", fontsize=11)
        ax.set_title(titles[key], fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"判据 {JUDGE_VERSION} 已知答案标定：三道门各管一类缺陷；"
                 f"n2n 对模糊是反的、对抬亮恒不变（{'通过' if j1_ok else '失败'}）",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figp = ROOT / "reports/figures/v3_judge_calibration.png"
    fig.savefig(figp, dpi=140)
    print(f"[calib] 图 {figp.relative_to(ROOT)}")
    print(f"[calib] 记盘 {out_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()

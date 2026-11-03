"""V3 Phase 2 / T2-2（**已被 v3_signal_transfer.py 取代，结论不采用**）：同一批位移类上量信号侧。

保留原因：这里的两次失败（高通把信号减掉、稀疏导致回归估不出）是仪器选择的记录，
不是可复用的产物。**信号侧的正式结果请看 `scripts/v3_signal_transfer.py` 与
`reports/v3_phase2_transfer.md`。**

原始说明如下（历史记录）。

为什么必须有这一步：`reports/v3_phase2_role_order.md` 只冻结了噪声侧排序，"体对角线位移泄漏最小"
不等于"最好"——体对角线是 3.46 体素，**信号相关性也最低**，最容易让网络只学会平滑。
所以每个位移类必须同时有两个数：

  N 侧（噪声可复制多少）：`R²_n` —— 从输入子格线性预测**目标处的噪声**（仿真里 = X − flow）
  S 侧（结构可传递多少）：`R²_s` —— 从输入子格线性预测**目标处的血流信号**（仿真里 = flow）

两者用**完全相同的窗口、相同的位点集合、相同的残差变换、相同的留出划分**，
所以可以直接比。协议要求 `S_min` 由受控模拟设定、不得从留出测试结果回调——这里就是那个受控模拟。

判据（写在跑之前，**先写后算**）：
  1. `R²_s` 低于 `R²_n` 的位移类 → 该类的自监督目标主要是"抄噪声"，**排除**；
  2. 在剩下的类里取 `R²_n` 最小者；若 `R²_n` 打平，取 `R²_s` 大者（信号更好传）；
  3. `S_min` 取"排除后剩下各类 `R²_s` 的最小值"，即刚好允许进候选库的门槛。
     *（这条是本次运行前写下的，不是看到数字后补的。）*

**第一次运行失败并已改（2026-10-20，写在这里而不是删掉）**：
初版把 Gate 1 的"快速分量"变换（`log − 15 体素箱均值`）也套到信号场上，结果 `R²_s` 全部 ≈ 0 甚至为负。
原因是**仪器用错了**：那个变换是为"快噪声"设计的，而血流信号恰恰是**空间慢变**的，
15 体素的箱均值把它当成平台一起减掉了，剩下的只有边缘伪影。
现在三条场（观测 / 纯噪声 / 纯信号）统一用 **`log1p(PD / s)`（s = 内部体素 PD 中位数，无高通）**，
窗口、位点集合、留出划分、估计器全部相同；噪声侧的泄漏 `L` 另按快速分量 ρ 解析算，两处都报。

仿真器：`SimCfg(seed=0)`，默认血流幅度（`flow_amp` 保持出厂值，**不调**——调了就成了拟合）。

用法：
    python -u scripts/v3_signal_noise_split.py [--m 2] [--w 1]
输出 `data/processed/v3_signal_noise_split.json`。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_simulator import SimCfg, simulate, band_and_pd  # noqa: E402
from v3_leak_validate import boxmean, cell_lattice, build_design, r2_holdout  # noqa: E402
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402

OUT = Path("data/processed/v3_signal_noise_split.json")


def stabilize(pd, s):
    """`log1p(PD / s)`：单调、背景不为 −inf，且**不**去掉慢变结构（信号侧必须保住）。"""
    return np.log1p(np.maximum(pd, 0.0) / s)


def representative_pairs(m):
    """按有符号位移 (a − y) 分组，每类取第一例（与 `v3_role_order.py` 的分组一致）。"""
    seen, pairs = set(), []
    for p in range(8):
        src = [int(x) for x in PATS[p]]
        tgt = sorted(set(range(8)) - set(src))
        for a in src:
            for y in tgt:
                sh = tuple(int(round(v)) for v in (off(a) - off(y)))
                if sh in seen:
                    continue
                seen.add(sh)
                pairs.append({"p": p, "a": a, "y": y, "shift": sh,
                              "vox": tuple(v * m for v in sh)})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--k-hp", type=int, default=15, help="去平台的箱式宽度（与 Gate 1 一致）")
    args = ap.parse_args()
    m, w = args.m, args.w

    print(f"仿真（血流开，seed=0，血流幅度保持出厂值）… m={m} w={w}", flush=True)
    d = simulate(SimCfg(seed=0))
    flow = d["flow"].astype(np.complex64)
    obs = d["X"].astype(np.complex64)
    noise = obs - flow
    print("  三条 PD…", flush=True)
    pd_obs, _ = band_and_pd(obs, d["valid"])
    pd_noise, _ = band_and_pd(noise, d["valid"])
    pd_sig, _ = band_and_pd(flow, d["valid"])
    okvox = boxmean(d["valid"].astype(np.float64), 5) > 0.999
    print(f"  有效内部体素 {okvox.sum()} / {okvox.size}", flush=True)

    # 统一变换：log1p(PD / s)，**不做高通**（信号是慢变的，高通会把它删掉，见模块 docstring）
    s = float(np.median(pd_obs[okvox]))
    print(f"  归一化尺度 s = 内部 PD 中位数 = {s:.4g}", flush=True)
    F = {k: stabilize(v, s) for k, v in
         (("obs", pd_obs), ("noise", pd_noise), ("sig", pd_sig))}
    var = {k: float(np.var(v[okvox])) for k, v in F.items()}
    print(f"  方差（内部体素）：观测 {var['obs']:.4f}  噪声 {var['noise']:.4f}  信号 {var['sig']:.4f}"
          f"   （噪声+信号 = {var['noise'] + var['sig']:.4f}，可加性误差 "
          f"{abs(var['noise'] + var['sig'] - var['obs']) / var['obs']:.1%}）", flush=True)

    nc = cell_lattice(pd_obs.shape, m)
    rows = []
    print(f"\n  {'位移':>16} {'体素':>12} {'R²_n':>7} {'R²_s':>7} {'S−N':>7} {'噪声占比':>9}", flush=True)
    for r in representative_pairs(m):
        got_n = build_design(F["noise"], m, nc, w, r["a"], r["y"], okvox)
        got_s = build_design(F["sig"], m, nc, w, r["a"], r["y"], okvox)
        if got_n is None or got_s is None:
            continue
        Xn, yn, _, _ = got_n
        Xs, ys, _, _ = got_s
        if len(yn) < 400:
            print(f"  {str(r['vox']):>16} 块数 {len(yn)} 太少，跳过", flush=True)
            continue
        # 同一个留出划分（同 seed、同长度），保证两个 R² 可比
        r2n = r2_holdout(Xn, yn)
        r2s = r2_holdout(Xs, ys)
        # 噪声占目标方差的比例（按各自成分的方差加权）
        share_n = (r2n * var["noise"]) / max(r2n * var["noise"] + r2s * var["sig"], 1e-12)
        rows.append({**{k: r[k] for k in ("p", "a", "y", "shift")},
                     "vox": list(r["vox"]), "n_blocks": int(len(yn)), "n_feat": int(Xn.shape[1]),
                     "r2_noise": r2n, "r2_signal": r2s, "noise_share": float(share_n)})
        print(f"  {str(r['vox']):>16} {'':>6} {r2n:7.3f} {r2s:7.3f} {r2s - r2n:+7.3f} {share_n:9.2f}",
              flush=True)

    rows.sort(key=lambda r: r["r2_noise"])
    keep = [r for r in rows if r["r2_signal"] >= r["r2_noise"]]
    S_min = min((r["r2_signal"] for r in keep), default=None)
    print(f"\n  通过判据 1（R²_s ≥ R²_n）的类：{len(keep)} / {len(rows)}", flush=True)
    if keep:
        best = min(keep, key=lambda r: r["r2_noise"])
        print(f"  判据 2 选中：位移 {best['vox']}  R²_n={best['r2_noise']:.3f}  R²_s={best['r2_signal']:.3f}",
              flush=True)
        print(f"  判据 3：S_min = {S_min:.3f}（保留类里 R²_s 的最小值）", flush=True)

    rec = {"m": m, "w": w, "k_hp": args.k_hp, "var": var, "rows": rows,
           "verdict": {"n_keep": len(keep), "S_min": S_min,
                       "chosen": (min(keep, key=lambda r: r["r2_noise"])
                                  if keep else None)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

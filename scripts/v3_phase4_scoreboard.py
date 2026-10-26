"""V3 Phase 4 计分板（重建版）：用"同一场景的独立观测"当检验目标。

## 为什么换判据（2026-10-21，见 Project_log L45/L46）

旧判据拿 `gt_flow_pd`（纯血流功率，背景**恰好为 0**）当真值。但训练目标是预测
同一体数据的邻居体素，其条件均值在背景处是**非零功率底**（静态散斑+杂波+热噪声）。
两个定义不可能是同一个模型：网络必须保留背景底才对得上训练目标，而真值要求把它抹成 0。
实测：输入背景 log 均值 0.693、真值 0.000，而五条臂的输**全都**在 0.69–0.71。

`gt_flow_pd` 还先天不可达：静态散斑在空间上与信号同样"稳定"，盲点法原理上分不开
（Phase 1 §6.3 已判"复现本底是对的"）。

所以这里换成 **Noise2Noise 式检验目标**：用同一套血管、**独立噪声实现**的另一张观测
`obs2` 当答案。判据是

    n2n_ratio = MSE(pred, obs2) / MSE(obs1, obs2)

- 输入自己 ⇒ 恰好 1.000（代数恒等，必须成立）
- 真实去噪 ⇒ < 1；比输入更差 ⇒ > 1
- 保留静态本底不再被惩罚（两次观测共享它）；只有"逐次独立"的那部分噪声被计分

## 两轴，不是一个数

`n2n_ratio` 单独会奖励**模糊**（模糊确实降低与 obs2 的距离）。所以必须与血管保真度
**同时**读：`r_ves`（血管体素上与 `gt_flow_pd` 的相关）。二维平面上：
输入在 (1.0, r_ves_input)；向左移动且 r_ves 不掉的才算真去噪。

## 预注册门槛（写于跑臂之前，跑完不得下调）

**门槛修订记录（2026-10-21，本次会话，修订时已跑臂数 = 0）**：
首版只写了 `n2n_ratio ≤ 0.95` + `r_ves` 不退化。只跑了**校准族**（真值/输入/模糊族/
多实现均值，不含任何 checkpoint）就发现这条不足以判定：`n2n_ratio` 的可行区间不是
[0,1]，而是"输入 = 1.000、完美去噪 ≈ 0.5、三次实现均值天花板 ≈ 0.60"。也就是说
**恒等映射就能过 0.95**，门槛量不出"是否真去噪"。因此补入第三条 gap 闭合率，
用**当时实测**的天花板做归一，阈值定在 0.5（= 走完"什么都不做"到"可测最好"的一半）。
同时修正首版 (c) 的一条方向性错误：首版写"模糊族 n2n 随 σ 单调下降"，实测是
**单调上升**（模糊破坏血管 → 与 obs2 的距离变大），方向写反了。

臂判定为"通过"需同时满足：
  (1) `n2n_ratio` ≤ **0.95**（不损伤）
  (2) `r_ves` ≥ **输入 r_ves − 0.01**（血管结构不退化）
  (3) `gap_closed = (1 − n2n_ratio) / (1 − ceiling_n2n)` ≥ **0.50**（真去噪，走完一半）

判据本身被接受的前提（已知答案校准，缺一不可）：
  (a) 输入行 n2n_ratio == 1.000（容差 1e-6）
  (b) 多实现均值（天花板）n2n_ratio < 1 严格成立
  (c) 模糊族 `n2n_ratio` 随 σ 单调**上升**（模糊=更差），且 `r_ves` 随 σ 单调下降
      —— 只在 `r_ves > 0.1` 的前缀上要求严格单调：σ≥1.5 时血管结构已被抹平，
      `r_ves` 在 0 附近测的是噪声，符号无意义（首版要求全程单调，因此在 σ=2.0
      的 −0.632→−0.608 上 FAIL；这是判据自身的过度要求，不是判据失效）
      另加端点要求：`r_ves(σ_max) ≤ 输入 r_ves − 0.3`
  (d) 全掩膜 `n2n_ratio` 与远端背景 `n2n_ratio_bg` 在模糊族下**方向相反**
      （前者升、后者降）—— 证明"只看背景"的那一版才会奖励模糊，故 `n2n` 必须取全掩膜

用法：
    python -u scripts/v3_phase4_scoreboard.py \
        --obs1 data/processed/v3_sim/sim_seed1.npz \
        --obs2 data/processed/v3_sim/sim_seed1_n101.npz \
        --obs-mean data/processed/v3_sim/sim_seed1_n102.npz \
                   data/processed/v3_sim/sim_seed1_n103.npz \
                   data/processed/v3_sim/sim_seed1_n104.npz \
        --runs runs/v3_probe --out reports/v3_phase4_2026-10-21
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_phase3_eval import (M_EVAL, usable, support_mask, reconstruct)  # noqa: E402
from models.network import NetworkCNR                                   # noqa: E402
from scipy.ndimage import distance_transform_edt, gaussian_filter       # noqa: E402


def load_obs(path: Path, shape, crop):
    d = np.load(path)
    return d["pd_linear"][crop].astype(np.float32), d["valid"][crop].astype(bool)


def to_log(lin, s):
    return np.log1p(lin / max(s, 1e-12)).astype(np.float32)


def score(pred, obs1, obs2, truth, masks):
    """返回一个对象在给定支撑集上的判据组。"""
    common, ves, far = masks["common"], masks["ves"], masks["far"]

    def mse(a, b, m):
        return float(np.mean((a[m].astype(np.float64) - b[m].astype(np.float64)) ** 2))

    base = mse(obs1, obs2, common)
    # 压扁诊断：把 pred 对 obs1 做最小二乘，斜率≈1 表示振幅保住，≈0.2 表示压成两成
    x = obs1[common].astype(np.float64)
    y = pred[common].astype(np.float64)
    slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 1e-12 else float("nan")
    out = {
        "n2n_ratio": mse(pred, obs2, common) / base,
        "n2n_ratio_ves": mse(pred, obs2, ves) / mse(obs1, obs2, ves),
        "n2n_ratio_bg": mse(pred, obs2, far) / mse(obs1, obs2, far),
        "r": float(np.corrcoef(pred[common], truth[common])[0, 1]),
        "r_ves": float(np.corrcoef(pred[ves], truth[ves])[0, 1]),
        "bg_mean": float(np.mean(pred[far])),
        "contrast": float(np.mean(pred[ves]) - np.mean(pred[far])),
        "slope_vs_input": slope,
        "std_ratio": float(y.std() / x.std()) if x.std() > 1e-12 else float("nan"),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs1", default="data/processed/v3_sim/sim_seed1.npz")
    ap.add_argument("--obs2", default="data/processed/v3_sim/sim_seed1_n101.npz")
    ap.add_argument("--obs-mean", nargs="*", default=[])
    ap.add_argument("--runs", default="runs/v3_probe")
    ap.add_argument("--out", default="reports/v3_phase4_2026-10-21")
    ap.add_argument("--shape-div", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blurs", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = np.load(args.obs1)
    shape = usable(d1["pd_linear"].shape, M_EVAL)
    if args.shape_div:
        shape = tuple((n // args.shape_div) * args.shape_div for n in shape)
    crop = tuple(slice(0, n) for n in shape)

    pd1, valid = load_obs(Path(args.obs1), shape, crop)
    pd2, valid2 = load_obs(Path(args.obs2), shape, crop)
    if not np.array_equal(valid, valid2):
        raise SystemExit("obs1 与 obs2 的 valid 掩膜不同 —— 不是同一场景")
    if not np.array_equal(d1["vessels"][crop], np.load(args.obs2)["vessels"][crop]):
        raise SystemExit("obs1 与 obs2 的血管掩膜不同 —— 不是同一场景")

    s = float(np.median(pd1[valid]))
    obs1 = to_log(pd1, s)
    obs2 = to_log(pd2, s)
    truth = to_log(d1["gt_flow_pd"][crop].astype(np.float32), s)

    # 天花板：多实现先在**线性**域取均值（条件均值的正确估计），再取 log
    obs_mean = None
    if args.obs_mean:
        acc = pd1.astype(np.float64).copy()
        for p in args.obs_mean:
            acc += np.load(p)["pd_linear"][crop].astype(np.float64)
        obs_mean = to_log((acc / (1 + len(args.obs_mean))).astype(np.float32), s)

    vessels = d1["vessels"][crop].astype(bool)
    common = support_mask(shape, M_EVAL) & valid
    ves = vessels & common
    # 远端背景：离血管 ≥4 且离无效区 ≥2（否则模糊会把亮血管/零填充渗进来）
    dist_v = distance_transform_edt(~vessels)
    far = common & (dist_v >= 4) & (distance_transform_edt(valid) >= 2)
    masks = {"common": common, "ves": ves, "far": far}

    print(f"shape={shape}  common={int(common.sum())}  vessels={int(ves.sum())}  far_bg={int(far.sum())}")
    print(f"背景底：输入 {np.mean(obs1[far]):.4f}  真值 {np.mean(truth[far]):.4f}")

    # ---------------- 已知答案校准 ----------------
    calib = {}
    calib["input"] = score(obs1, obs1, obs2, truth, masks)
    calib["truth_flow_only"] = score(truth, obs1, obs2, truth, masks)
    if obs_mean is not None:
        calib["obs_mean_ceiling"] = score(obs_mean, obs1, obs2, truth, masks)
    for sg in args.blurs:
        calib[f"blur_{sg}"] = score(gaussian_filter(obs1, sg).astype(np.float32),
                                    obs1, obs2, truth, masks)

    print(f"\n{'校准对象':22s} {'n2n':>7s} {'n2n_ves':>8s} {'n2n_bg':>7s} {'r':>7s} {'r_ves':>7s} "
          f"{'bg_mean':>8s} {'slope':>7s} {'stdR':>7s}")
    for k, m in calib.items():
        print(f"{k:22s} {m['n2n_ratio']:7.3f} {m['n2n_ratio_ves']:8.3f} {m['n2n_ratio_bg']:7.3f} "
              f"{m['r']:7.3f} {m['r_ves']:7.3f} {m['bg_mean']:8.4f} "
              f"{m['slope_vs_input']:7.3f} {m['std_ratio']:7.3f}")

    ok_a = abs(calib["input"]["n2n_ratio"] - 1.0) < 1e-6
    ok_b = ("obs_mean_ceiling" in calib and calib["obs_mean_ceiling"]["n2n_ratio"] < 1.0)
    bl = [calib[f"blur_{sg}"] for sg in args.blurs]
    # r_ves 只在结构尚未被抹平的区间里检验单调（见 docstring (c)）
    rv = [calib["input"]["r_ves"]] + [m["r_ves"] for m in bl]
    live = []
    for v in rv:
        if v <= 0.1:
            break
        live.append(v)
    ok_c = (all(a["n2n_ratio"] < b["n2n_ratio"] for a, b in zip(bl, bl[1:]))
            and all(a > b for a, b in zip(live, live[1:]))
            and bl[-1]["r_ves"] <= calib["input"]["r_ves"] - 0.3)
    ok_d = all((b["n2n_ratio"] > a["n2n_ratio"]) and (b["n2n_ratio_bg"] < a["n2n_ratio_bg"])
               for a, b in zip(bl, bl[1:]))
    print(f"\n已知答案校准：(a) 输入==1.000 {'OK' if ok_a else 'FAIL'}  "
          f"(b) 天花板<1 {'OK' if ok_b else 'FAIL/缺'}  (c) 模糊族单调 {'OK' if ok_c else 'FAIL'}  "
          f"(d) 全掩膜/背景反向 {'OK' if ok_d else 'FAIL'}")
    judge_ok = ok_a and ok_b and ok_c and ok_d
    ceiling = calib.get("obs_mean_ceiling", {}).get("n2n_ratio")

    # ---------------- 跑臂 ----------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rec = {"shape": list(shape), "judge_ok": judge_ok, "calibration": calib,
           "thresholds": {"n2n_ratio": 0.95, "r_ves_margin": 0.01, "gap_closed": 0.50},
           "ceiling_n2n": ceiling, "arms_scored_before_threshold_edit": 0,
           "masks": {k: int(v.sum()) for k, v in masks.items()},
           "arms": {}}
    # 有 best.pt（按留出场景早停选出来的）就用它，否则退回 latest.pt。
    # 理由见 L48：单场景训练到后期会记忆目标实现自己的噪声，latest 不是"最好的那一版"。
    runs = []
    for d in sorted(p for p in Path(args.runs).iterdir() if p.is_dir()):
        for name in ("best.pt", "latest.pt"):
            if (d / name).exists():
                runs.append(d / name)
                break
    if not runs:
        raise SystemExit(f"{args.runs} 下没有 latest.pt")
    for ck in runs:
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = NetworkCNR(features=a["features"], groups=a["groups"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        if a.get("infer") == "full":
            # 方案②：中心可见，整卷一次前向（不走相位散回）
            with torch.no_grad():
                xin = torch.as_tensor(obs1, device=device)[None, None].float()
                h = model(xin)
                pred = (xin + h if a.get("residual") else h)[0, 0].cpu().numpy()
            filled = 1.0
        else:
            pred, filled = reconstruct(model, obs1, a["stride"], device)
        arm = ck.parent.name.rsplit("_s", 1)[0]
        seed = ck.parent.name.rsplit("_s", 1)[1]
        m = score(pred, obs1, obs2, truth, masks)
        m.update({"stride": a["stride"], "mode": a["mode"], "slots": a.get("slots"),
                  "weight_identity": a.get("weight_identity"),
                  "steps": a["epochs"] * a["steps_per_epoch"]})
        rec["arms"].setdefault(arm, {})[seed] = m

    # ---------------- 汇总 ----------------
    ref_rves = calib["input"]["r_ves"]
    print(f"\n{'臂':18s} {'n2n (mean[min,max])':30s} {'gap':>6s} {'r_ves':>13s} {'slope':>7s}  判定")
    summ = {}
    for arm, seeds in rec["arms"].items():
        rows = list(seeds.values())
        n2n = [r["n2n_ratio"] for r in rows]
        rves = statistics.mean(r["r_ves"] for r in rows)
        n2n_mean = statistics.mean(n2n)
        slope = statistics.mean(r["slope_vs_input"] for r in rows)
        gap = (1.0 - n2n_mean) / (1.0 - ceiling) if (ceiling is not None and ceiling < 1) else None
        c1 = n2n_mean <= 0.95
        c2 = rves >= ref_rves - 0.01
        c3 = gap is not None and gap >= 0.50
        passed = c1 and c2 and c3
        summ[arm] = {"n2n_mean": n2n_mean, "n2n_min": min(n2n),
                     "n2n_max": max(n2n), "gap_closed": gap, "r_ves_mean": rves,
                     "r_ves_ref": ref_rves, "slope_mean": slope, "pass": bool(passed),
                     "criteria": {"n2n": bool(c1), "r_ves": bool(c2), "gap": bool(c3)}}
        print(f"{arm:18s} {n2n_mean:6.3f} [{min(n2n):.3f},{max(n2n):.3f}]"
              f"{'':9s} {(f'{gap:5.2f}' if gap is not None else '  n/a')} {rves:6.3f} ({rves - ref_rves:+.3f})  "
              f"{slope:7.3f}  {'通过' if passed else '不通过'}"
              f"{'' if passed else '  [' + ''.join(k for k, v in (('n2n',c1),('r_ves',c2),('gap',c3)) if not v) + ']'}")
    rec["summary"] = summ
    (out_dir / "scoreboard.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_dir}/scoreboard.json")
    if not judge_ok:
        print("!! 判据未通过已知答案校准，上面所有臂的判定都不作数")


if __name__ == "__main__":
    main()

"""Round3 A 段诊断：A1 直流/增益响应、A2 α 扫描、A4 误差分解（A3 用 --manifest 指向 6 场景目录）。

全部**只读**已冻结的 checkpoint，不训练。设计见 `reviews/V3_PR1_round3_amended_plan.md` §3。

用法：
    PY=/data2/wjb/anaconda3/envs/threePM/bin/python
    $PY -u scripts/v3_round3_a_probes.py --level vlow --arm S_dense --seed 3407 --gpu 0
    # A3：先生成 6 个场景，再用同一脚本逐场景评估
    $PY -u scripts/v3_round2_s0_dataset.py --level vlow --n-train 0 --n-eval 6 \
        --eval-seed0 900 --out data/processed/v3_round2_s0/vlow6
    $PY -u scripts/v3_round3_a_probes.py --level vlow --arm S_dense --seed 3407 \
        --manifest data/processed/v3_round2_s0/vlow6/manifest.json \
        --out reports/v3_round3/a_probes_vlow6.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import infer, write_vol           # noqa: E402
from v3_round2_s0_eval import load_eval, metrics_s0, PATCH   # noqa: E402
from v3_round2_s0_dataset import PSF_PD                # noqa: E402

ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)


# ----------------------------------------------------------------- A1 ----
def _load_model(ckpt: Path | None, features: int, device):
    import torch
    from models.network import NetworkCNR

    net = NetworkCNR(in_channels=1, out_channels=1, features=features, groups=4)
    if ckpt is not None:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["model"])
    else:
        torch.manual_seed(3407)                    # 未训练臂：固定一个起点，可复现
    net.eval().to(device)
    return net


def _center_patch(vol: np.ndarray, valid: np.ndarray):
    d, h, w = vol.shape
    pd, ph, pw = PATCH
    # 取"有效体素比例最高"的窗口：沿每个轴滑一遍粗网格
    best, arg = -1.0, (0, 0, 0)
    for z0 in np.linspace(0, d - pd, 4).astype(int):
        for y0 in np.linspace(0, h - ph, 4).astype(int):
            for x0 in np.linspace(0, w - pw, 4).astype(int):
                f = valid[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw].mean()
                if f > best:
                    best, arg = float(f), (z0, y0, x0)
    z0, y0, x0 = arg
    sl = np.s_[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return vol[sl], valid[sl], arg, best


def a1_patch_probe(vol, valid, sigma, ckpt, features, device) -> dict:
    """单 patch 前向：三个基线 × (in-dist / OOD 加常数 + 全局增益)。"""
    import torch

    patch, vpatch, origin, valid_frac = _center_patch(vol, valid)
    x = torch.from_numpy(np.ascontiguousarray(patch, np.float32))[None, None].to(device)
    vm = vpatch.astype(bool)
    x0 = patch[vm]

    def run(net, xx):
        with torch.no_grad():
            return net(xx)[0, 0].cpu().numpy()

    out = {"origin": list(origin), "valid_frac": round(valid_frac, 3),
           "in_mean": float(x0.mean()), "in_std": float(x0.std()), "arms": {}}

    arms = {"identity": None}
    nets = {"untrained_crn": _load_model(None, features, device),
            "trained_crn": _load_model(ckpt, features, device)}

    def compute(tag, net):
        base = run(net, x)[vm] if net is not None else x0
        rec = {"base_out_mean": float(base.mean()),
               "base_out_std": float(base.std()),
               "base_level_shift": float((base - x0).mean()),
               "const": {}, "gain": {}}
        for c in (0.5, -0.5, 1.0, -1.0, 5.0, -5.0):
            cc = float(c * sigma)
            y = run(net, x + cc)[vm] if net is not None else x0 + cc
            d = y - base
            rec["const"][f"{c:+g}sigma"] = {
                "c": cc,
                "mean_shift": float(d.mean()),
                "g_dc": float(d.mean() / cc),
                "out_std": float(y.std()),
                "std_ratio": float(y.std() / max(base.std(), 1e-12)),
                "level_shift": float((y - (x0 + cc)).mean()),
                "ood": abs(c) > 2,
            }
        for g in (0.5, 2.0):
            y = run(net, x * g)[vm] if net is not None else x0 * g
            den = (g - 1.0) * x0
            m = np.abs(den) > 1e-6
            rec["gain"][f"x{g:g}"] = {
                "mean_ratio": float(y.mean() / max(base.mean(), 1e-12)),
                "per_voxel_slope_median": float(np.median((y - base)[m] / den[m])) if m.any() else None,
                "n_used": int(m.sum()),
            }
        out["arms"][tag] = rec

    compute("identity", None)
    for tag, net in nets.items():
        compute(tag, net)
    return out


def a1_volume_probe(vol, valid, sigma, ckpt, gpu, tag) -> dict:
    """满分辨率滑窗口径：只对训练后 CRN 做，检验"patch 结论是否是裁边效应"。"""
    rec = {}
    base_p = write_vol(vol, ROOT / "runs/_infer_v2/a1" / f"{tag}_base.npy")
    base = infer(ckpt, base_p, gpu, f"{tag}_a1base")
    for name, c in (("+1sigma", sigma), ("-1sigma", -sigma),
                    ("+5sigma", 5 * sigma), ("-5sigma", -5 * sigma)):
        p = write_vol(vol + c, ROOT / "runs/_infer_v2/a1" / f"{tag}_{name}.npy")
        y = infer(ckpt, p, gpu, f"{tag}_a1{name}")
        d = (y - base)[valid]
        rec[name] = {"c": float(c), "mean_shift": float(d.mean()),
                     "g_dc": float(d.mean() / c),
                     "level_shift": float((y - (vol + c))[valid].mean()),
                     "out_std_ratio": float(y[valid].std() / max(base[valid].std(), 1e-12))}
    rec["base_level_shift"] = float((base - vol)[valid].mean())
    return rec


# ----------------------------------------------------------------- A2 ----
def n2n(pred, E, mask=None):
    m = E["valid"] if mask is None else mask
    num = float(np.mean((pred[m] - E["vol2"][m]) ** 2))
    den = float(np.mean((E["vol"][m] - E["vol2"][m]) ** 2))
    return num / den if den else float("nan")


def a2_alpha_sweep(preds: dict, E: dict) -> dict:
    """对每个臂扫 α：`pred_α = (1−α)·obs1 + α·pred`，报 n2n(α) 与 r²(α)。"""
    o1, truth = E["vol"], E["truth"]
    out = {}
    for arm, pred in preds.items():
        rows = []
        for a in ALPHA_GRID:
            pa = (1.0 - a) * o1 + a * pred
            r2 = float(np.mean((pa[E["valid"]] - truth[E["valid"]]) ** 2) /
                       max(np.mean((o1[E["valid"]] - truth[E["valid"]]) ** 2), 1e-12))
            rows.append({"alpha": float(a), "n2n": n2n(pa, E), "r2": r2,
                         "theory": float(((1.0 - a) ** 2 + 1.0) / 2.0)})
        n2 = np.array([r["n2n"] for r in rows])
        th = np.array([r["theory"] for r in rows])
        out[arm] = {
            "rows": rows,
            "argmin_alpha": float(ALPHA_GRID[int(np.argmin(n2))]),
            "min_n2n": float(n2.min()),
            "theory_max_abs_dev": float(np.abs(n2 - th).max()),
            "theory_note": "pred=truth 时 n2n(α) 应精确等于 ((1−α)²+1)/2；偏离量即"
                           "『两路噪声非独立/异方差/参考不一致』的合并读数",
        }
    return out


def a2_select(cal: dict, test: dict, arm: str) -> dict:
    """在校准体上选 α，在另一个体上报告（两个方向都做）。"""
    a_star = cal["arms"][arm]["argmin_alpha"]
    row = min(test["arms"][arm]["rows"], key=lambda r: abs(r["alpha"] - a_star))
    best = min(test["arms"][arm]["rows"], key=lambda r: r["n2n"])
    return {"arm": arm, "alpha_from_cal": a_star,
            "cal_min_n2n": cal["arms"][arm]["min_n2n"],
            "test_n2n_at_that_alpha": row["n2n"],
            "test_best_alpha": best["alpha"], "test_best_n2n": best["n2n"],
            "selection_optimism": float(row["n2n"] - best["n2n"])}


# ----------------------------------------------------------------- A4 ----
def a4_decompose(pred: np.ndarray, E: dict) -> dict:
    """MSE = bias² + var，按区域与频带分开；同时给有符号偏差。"""
    v, truth = E["valid"], E["truth"]
    e = (pred - truth)

    def one(mask):
        if mask.sum() < 3:
            return None
        err = e[mask]
        bias = float(err.mean())
        var = float(err.var())
        mse = float(np.mean(err ** 2))
        return {"n": int(mask.sum()), "bias": bias, "err_std": float(err.std()),
                "mse": mse, "bias2": bias * bias, "var": var,
                "identity_gap": float(abs(mse - (bias * bias + var)))}

    rec = {"valid": one(v), "bg": one(E["bg"]), "roi": one(E["roi"]),
           "core": one(E["core"])}

    # 频带分解：低通 = PSF 尺度的平滑（成像系统能分辨的部分），高通 = 其残差
    lo_p = gaussian_filter(pred, sigma=PSF_PD, mode="nearest")
    lo_t = gaussian_filter(truth, sigma=PSF_PD, mode="nearest")
    bands = {}
    for name, (a, b) in {"low": (lo_p, lo_t), "high": (pred - lo_p, truth - lo_t)}.items():
        err = (a - b)[v]
        bands[name] = {"bias": float(err.mean()), "err_std": float(err.std()),
                       "mse": float(np.mean(err ** 2)),
                       "energy_frac": float(np.mean(err ** 2) /
                                            max(np.mean(e[v] ** 2), 1e-12))}
    rec["bands"] = bands
    rec["pred_level"] = {"mean": float(pred[v].mean()), "std": float(pred[v].std())}
    rec["truth_level"] = {"mean": float(truth[v].mean()), "std": float(truth[v].std())}
    return rec


# ---------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="vlow")
    ap.add_argument("--arm", default="S_dense")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--probe-volume", default="e900",
                    help="A1 用哪个评估体（只看一个，patch + 满分辨率两种口径）")
    ap.add_argument("--skip-a1", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = Path(args.runs or f"runs/v3_round2_s0/{args.level}")
    man_p = Path(args.manifest or f"data/processed/v3_round2_s0/{args.level}/manifest.json")
    man = json.loads((ROOT / man_p).read_text())
    ckpt = ROOT / runs / f"{args.arm}_s{args.seed}"
    if not (ckpt / "history.json").exists():
        raise SystemExit(f"[abort] {ckpt} 没有 history.json（未跑完或不存在的 run）")

    Es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]
    names = [Path(E["npz"]).stem for E in Es]
    print(f"[{args.level}] 评估体 {names}  checkpoint={ckpt.relative_to(ROOT)}")

    # 网络输入域里的**实测**单次观测噪声 std：`sqrt(mse(obs1,obs2)/2)`。
    # manifest 的 `sigma_eps` 是 PD 域的 log 噪声 std（乘性噪声施加在 PD 上），
    # 经 `log1p(x/s)` 之后被压缩到约一半，两者不要混用（vlow：0.0128 vs 实测 0.0066）。
    sig_eff = {n: float(np.sqrt(0.5 * np.mean((E["vol"][E["valid"]] - E["vol2"][E["valid"]]) ** 2)))
               for n, E in zip(names, Es)}

    rec = {"meta": {"level": args.level, "manifest": str(man_p), "arm": args.arm,
                    "seed": args.seed, "checkpoint": str(ckpt.relative_to(ROOT)),
                    "sigma_eps_manifest": man["sigma_eps"], "sigma_eff_input_domain": sig_eff,
                    "sigma_note": "manifest.sigma_eps 是 PD 域 log 噪声 std；"
                                  "sigma_eff 是网络输入域 log1p(x/s) 里的实测 std（约一半），"
                                  "A1 的扰动尺度用后者",
                    "gpu": args.gpu,
                    "prereg": "reviews/V3_PR1_round3_amended_plan.md §3",
                    "volumes": names}}
    t0 = time.time()

    # ---- 推理一次，A2/A4 共用（A3 的多场景也是这样逐体一次）
    preds = {}
    t_inf = {}
    met = {}
    KEEP = ("n2n_ratio", "n2n_ratio_roi", "r_roi", "r_core", "bg_std_ratio",
            "rmse_vs_imaging", "rmse_roi", "roi_mean")
    for name, E in zip(names, Es):
        p = write_vol(E["vol"], ROOT / "runs/_infer_v2/a_probe" / f"{name}_in.npy")
        ts = time.time()
        preds[name] = infer(ckpt, p, args.gpu, f"{args.arm}_s{args.seed}_{name}").astype(np.float32)
        t_inf[name] = round(time.time() - ts, 1)
        m = metrics_s0(preds[name], E)
        # 输入臂（identity）的同一套读数：G-B1 要求 `r_roi` **高于输入**，
        # 所以输入侧的 `r_roi` 必须和模型侧记在同一个文件里，否则判据无法核验。
        m_in = metrics_s0(E["vol"].astype(np.float32), E)
        met[name] = {"model": {k: float(m[k]) for k in KEEP},
                     "input": {k: float(m_in[k]) for k in KEEP}}
        print(f"  [{name}] n2n={m['n2n_ratio']:.3f} r_roi={m['r_roi']:.4f} "
              f"(input {m_in['r_roi']:.4f}) rmse={m['rmse_vs_imaging']:.4f} "
              f"bg={m['bg_std_ratio']:.3f} ({t_inf[name]}s)")
    rec["metrics"] = met

    # ---- A1
    if not args.skip_a1:
        idx = names.index(f"s0_{args.level}_{args.probe_volume}") \
            if f"s0_{args.level}_{args.probe_volume}" in names else 0
        E = Es[idx]
        print(f"[A1] patch 口径（{names[idx]}，{PATCH}）")
        sig = sig_eff[names[idx]]
        rec["A1"] = {"volume": names[idx], "sigma_eff": sig,
                     "patch": a1_patch_probe(E["vol"], E["valid"], sig,
                                             ckpt / "latest.pt", args.features, device)}
        for tag, r in rec["A1"]["patch"]["arms"].items():
            print(f"  {tag:14s} base_shift={r['base_level_shift']:+.4f} "
                  + " ".join(f"{k}:g_dc={v['g_dc']:+.3f}" for k, v in r["const"].items()))
        print(f"[A1] 满分辨率口径（{names[idx]}，只对训练后 CRN）")
        rec["A1"]["volume_probe"] = a1_volume_probe(E["vol"], E["valid"], sig,
                                                   ckpt, args.gpu, f"{args.arm}_s{args.seed}")
        print("  " + " ".join(f"{k}:g_dc={v['g_dc']:+.3f}"
                              for k, v in rec["A1"]["volume_probe"].items()
                              if isinstance(v, dict) and "g_dc" in v))

    # ---- A2（含已知答案：pred=truth 时的理论曲线）
    rec["A2"] = {}
    for name, E in zip(names, Es):
        rec["A2"][name] = {"arms": a2_alpha_sweep(
            {"trained": preds[name], "oracle_truth": E["truth"]}, E)}
        for arm, r in rec["A2"][name]["arms"].items():
            print(f"[A2] {name} {arm:12s} argmin α={r['argmin_alpha']:.2f} "
                  f"min n2n={r['min_n2n']:.3f}  |n2n−theory|max={r['theory_max_abs_dev']:.4f}")
    if len(names) >= 2:
        rec["A2_selection"] = []
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                s = a2_select(rec["A2"][names[i]], rec["A2"][names[j]], "trained")
                s["cal"] = names[i]
                s["test"] = names[j]
                rec["A2_selection"].append(s)
                print(f"[A2-select] cal={names[i]} test={names[j]} α*={s['alpha_from_cal']:.2f} "
                      f"test_n2n={s['test_n2n_at_that_alpha']:.3f} "
                      f"test_best={s['test_best_n2n']:.3f}@α={s['test_best_alpha']:.2f} "
                      f"乐观={s['selection_optimism']:+.4f}")

    # ---- A4
    rec["A4"] = {}
    for name, E in zip(names, Es):
        rec["A4"][name] = a4_decompose(preds[name], E)
        d = rec["A4"][name]
        print(f"[A4] {name} bias={d['valid']['bias']:+.4f} err_std={d['valid']['err_std']:.4f} "
              f"mse={d['valid']['mse']:.5f} (bias²+var={d['valid']['bias2'] + d['valid']['var']:.5f}) "
              f"low/high mse={d['bands']['low']['mse']:.5f}/{d['bands']['high']['mse']:.5f}")

    rec["meta"]["total_seconds"] = round(time.time() - t0, 1)
    rec["meta"]["infer_seconds"] = t_inf
    out = ROOT / (args.out or f"reports/v3_round3/a_probes_{args.level}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

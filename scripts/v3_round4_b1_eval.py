"""Round4-B1 pilot 评估：三域臂 → 反变换回线性 PD → 统一域指标（**只推理，不训练**）。

判据与选参在 `reviews/V3_PR1_round4_prereg.md` §3.2–3.3 与
`reviews/V3_PR1_round4_prereg_b1_amendment.md` 里写死：

* 选参只看 e900/e901，主结果只报 e902–e905；
* 统一**反变换回线性 PD**后与 `gt_imaging_pd` 比（另列线性域相对误差）；
* 与 `gauss(σ*)`（A5 冻结 σ*）在各轴上对照。

反变换链：`y_std → x = y_std·std + mean → PD = inv_arm(x, s) → log1p(PD/s)`，
最后一步是为了与 P0 的冻结指标域对齐（`truth_img = log1p(gt_imaging_pd/s)`）。

用法
----
    python -u scripts/v3_round4_b1_eval.py --gpu 1
产物
----
    reports/v3_round4/b1_pilot.json
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

from v3_round4_metrics import load, masked_gauss       # noqa: E402
from v3_round4_metrics_v3 import metrics_v3            # noqa: E402
from v3_round4_b1_prep import fwd, inv                 # noqa: E402
from v3_expA_v2_eval import infer, write_vol, ckpt_budget   # noqa: E402

ARMS = ("linear", "log1p", "logeps")
CAL = [900, 901]
HELD = [902, 903, 904, 905]
SIGMA_STAR = 0.6
LEVEL = "low"


def npz_path(seed: int) -> Path:
    return ROOT / f"data/processed/v3_round2_s0/{LEVEL}6/eval/s0_{LEVEL}_e{seed}.npz"


def pred_log1p(arm: str, ckpt: Path, E: dict, norm: dict, gpu: int, tag: str) -> np.ndarray:
    """整卷推理 → 反变换回线性 PD → 回到冻结的 log1p 指标域。"""
    cache = ROOT / "runs/_infer_v2/b1" / f"{tag}_pred_log1p.npy"
    if cache.exists():
        return np.load(cache).astype(np.float32)
    s = float(E["s"])
    valid = E["valid"]
    x = fwd(arm, np.expm1(E["vol"]) * s, s, valid)     # E["vol"] 是 log1p；先还原 PD 再用同一 fwd
    z = (x - norm[arm]["mean"]) / norm[arm]["std"]
    p = write_vol(z, ROOT / "runs/_infer_v2/b1" / f"{tag}_in.npy")
    y = infer(ckpt, p, gpu, tag)
    xhat = y * norm[arm]["std"] + norm[arm]["mean"]
    pd = inv(arm, xhat, s)
    lg = np.log1p(np.maximum(pd, 0.0) / max(s, 1e-12)).astype(np.float32)
    lg[~valid] = 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lg)
    return lg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--arm-suffix", default="",
                    help="目录后缀：预注册允许的『最优臂 15→40 epoch』续训写在 "
                         "runs/v3_round4/b1_<arm>_s3407<suffix>，用本参数区分，避免覆盖 pilot 结果。")
    ap.add_argument("--out-json", default="reports/v3_round4/b1_pilot.json")
    args = ap.parse_args()

    norm = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())
    out: dict = {"prereg": ["reviews/V3_PR1_round4_prereg.md §3.2-3.3",
                            "reviews/V3_PR1_round4_prereg_b1_amendment.md"],
                 "cal": CAL, "held": HELD, "sigma_gauss_frozen": SIGMA_STAR,
                 "norm": {a: norm[a] for a in ARMS}, "arms": {}}

    E_all = {s: load(npz_path(s)) for s in CAL + HELD}
    base = {}
    for s, E in E_all.items():
        base[f"e{s}"] = {"input": metrics_v3(E["vol"], E),
                         "gauss": metrics_v3(masked_gauss(E["vol"], E["valid"], SIGMA_STAR), E)}

    for arm in args.arms:
        ck = ROOT / f"runs/v3_round4/b1_{arm}_s3407{args.arm_suffix}"
        if not (ck / "latest.pt").exists():
            out["arms"][arm] = {"status": "NOT_RUN", "reason": "checkpoint 不存在"}
            print(f"[{arm}] NOT_RUN")
            continue
        rec = {"checkpoint": str(ck.relative_to(ROOT)), "budget": ckpt_budget(ck / "latest.pt"),
               "cal": {}, "held": {}}
        for split, scenes in (("cal", CAL), ("held", HELD)):
            for s in scenes:
                E = E_all[s]
                pr = pred_log1p(arm, ck, E, norm, args.gpu,
                                f"b1_{arm}{args.arm_suffix}_e{s}")
                m = metrics_v3(pr, E)
                # 线性域相对误差（对 PSF 匹配真值，按 s 归一）
                s_scale = float(E["s"])
                pd_pred = np.expm1(pr) * s_scale
                pd_true = np.expm1(E["truth_img"]) * s_scale
                v = E["valid"]
                m["rmse_lin_rel"] = float(np.sqrt(np.mean((pd_pred[v] - pd_true[v]) ** 2)) / s_scale)
                m["rmse_lin_rel_vs_input"] = float(
                    np.sqrt(np.mean(((np.expm1(E["vol"]) - np.expm1(E["truth_img"]))[v] * s_scale) ** 2))
                    / s_scale)
                rec[split][f"e{s}"] = m
                print(f"[{arm} {split} e{s}] rmse_img={m['overall_rmse']:.4f} "
                      f"bias_ves={m['bias_ves']:+.4f} var_bg={m['var_bg']:.2e} n2n={m['n2n']:.3f}")
        out["arms"][arm] = rec

    # ---- B Gate（预注册 §3.3，只看 held 四体的逐场景配对）----
    gate = {"rule": "相对 log1p 臂 rmse_img 改善≥3/4 体；且不劣于 gauss 4/4；"
                    "且 recall_at_fpr5 不比 gauss 差 >0.02（4/4）", "arms": {}}
    log1p_rec = out["arms"].get("log1p", {}).get("held", {})
    if not log1p_rec:
        # 续训评估只跑单臂：没有 log1p 对照就不能按预注册规则判 Gate，直接跳过，避免写出"clear_positive_signal=False"被误读成失败。
        gate["note"] = ("本次只评估续训臂（无 log1p 三臂对照），不重跑 B Gate；续训臂的数值与 pilot 表并列报告。")
    for arm in (args.arms if log1p_rec else []):
        rec = out["arms"].get(arm, {})
        if "held" not in rec:
            gate["arms"][arm] = {"status": rec.get("status", "NOT_RUN")}
            continue
        r_imp = r_gauss = r_rec = 0
        for s in HELD:
            m = rec["held"][f"e{s}"]
            g = base[f"e{s}"]["gauss"]
            if log1p_rec and m["overall_rmse"] <= log1p_rec[f"e{s}"]["overall_rmse"]:
                r_imp += 1
            if m["overall_rmse"] <= g["overall_rmse"]:
                r_gauss += 1
            if m["ret_p10"] >= g["ret_p10"] - 0.02:
                r_rec += 1
        ok = (r_imp >= 3 and r_gauss == 4 and r_rec == 4)
        gate["arms"][arm] = {"n_rmse_better_than_log1p": r_imp,
                             "n_rmse_not_worse_than_gauss": r_gauss,
                             "n_ret_p10_within_0.02_of_gauss": r_rec,
                             "clear_positive_signal": bool(ok)}
    gate["B_NO_GAIN"] = not any(v.get("clear_positive_signal") for v in gate["arms"].values())
    out["baselines"] = base
    out["B_gate"] = gate

    p = ROOT / args.out_json
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(gate, indent=1, ensure_ascii=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()

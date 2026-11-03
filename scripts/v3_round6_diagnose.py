#!/usr/bin/env python3
"""Round6 机制诊断：把 `X_RMSE` 拆成"沿噪声方向的增益"与"正交杂质"两部分。

为什么需要这个
--------------
原始输出 α=1 时所有学习类臂都比不去噪更差，而训练损失是平滑下降的（不是数值崩溃）。
"更差"有两种完全不同的成因，必须分开：
  ① 方向错（学到的是别的东西）；
  ② 方向对但幅度失控（一边按正确方向削噪，一边注入等量无关扰动）。

把修正量 `Δ = L(f(A)) − L(A)` 对真噪声 `n = L(A) − L(B)` 做一元最小二乘：

    Δ = −κ·n + Δ⊥        （Δ⊥ ⊥ n，κ = 沿噪声方向的增益）
    X² = (1 − κ)²·X_in² + ‖Δ⊥‖²

（`κ = 1, Δ⊥ = 0` 时 `X = 0`，即完美预测 B。脚本把它当恒等式自检，残差应为机器精度。）

`κ` 大而 `‖Δ⊥‖²` 也大 ⇒ ②；`κ ≈ 0` ⇒ ①。最优增益 `α* = κ·X_in²/‖Δ‖²`，
把修正量缩到 `α*` 倍后 `X²` 变成 `X_in² − ⟨n,Δ⟩²/‖Δ‖²`。
本脚本把这三个量对每个已完成的 run 算出来，供报告引用，避免"看着像过拟合"这种说法。

用法
----
    python scripts/v3_round6_diagnose.py                 # 方向 1（选型方向）
    python scripts/v3_round6_diagnose.py --half even     # 方向 2（无偏确认）
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

from v3_round4_b1_prep import inv                                  # noqa: E402
from v3_round5_eval import usable                                # noqa: E402
from v3_round6_eval import (NAME_RE, NORM_PATH, bd2_support,     # noqa: E402
                            denoise, L, load_half)


def diag_one(run: Path) -> dict | None:
    mm = NAME_RE.match(run.name)
    if not mm:
        return None
    seq, geom, domain, epochs, th, eh = mm.groups()
    pd_tr, valid, s = load_half(seq, th)
    pd_ev, _, _ = load_half(seq, eh)
    shape = usable(pd_tr.shape, 2)
    sl = (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
    pd_tr, pd_ev, valid = pd_tr[sl], pd_ev[sl], valid[sl]
    sup = bd2_support(shape) & valid

    z = np.load(ROOT / f"data/processed/v3_round6_b1/{domain}/{seq}_{th}_x.npy")
    y, _ = denoise(run / "latest.pt", domain, z, geom)
    norm = json.loads(NORM_PATH.read_text())[domain]
    pd_hat = np.maximum(inv(domain, y * norm["std"] + norm["mean"], s), 0.0)

    n = (L(pd_tr, s) - L(pd_ev, s))[sup]          # 真噪声（A − B）
    d = (L(pd_hat, s) - L(pd_tr, s))[sup]         # 修正量
    nn, dd, nd = float(np.mean(n * n)), float(np.mean(d * d)), float(np.mean(n * d))
    kappa = -nd / nn                               # 沿噪声方向的增益
    x_in, x = float(np.sqrt(nn)), float(np.sqrt(nn + 2 * nd + dd))
    alpha = -nd / dd if dd > 0 else 1.0
    x_at_alpha = float(np.sqrt(max(nn - nd * nd / dd, 0.0))) if dd > 0 else x_in
    orth = max(dd - nd * nd / nn, 0.0)            # ‖Δ⊥‖²
    return {"run": run.name, "seq": seq, "geom": geom, "domain": domain,
            "epochs": int(epochs), "train_half": th,
            "kappa": kappa, "corr": nd / float(np.sqrt(nn * dd)) if dd > 0 else 0.0,
            "rms_delta": float(np.sqrt(dd)), "rms_noise": x_in,
            "delta_over_noise": float(np.sqrt(dd / nn)),
            "orth_over_noise": float(np.sqrt(orth / nn)),
            "alpha_star": alpha, "x_in": x_in, "x": x,
            "x_at_alpha_star": x_at_alpha,
            # 恒等式自检：(1−κ)²·X_in² + ‖Δ⊥‖² 应精确等于 X²
            "identity_residual": float((1 - kappa) ** 2 * nn + orth - (nn + 2 * nd + dd))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["odd", "even"], default="odd")
    ap.add_argument("--runs", default="runs/v3_round6")
    ap.add_argument("--out", default="reports/v3_round6/overfit_diag.json")
    a = ap.parse_args()

    runs = sorted(p for p in (ROOT / a.runs).iterdir()
                  if p.is_dir() and NAME_RE.match(p.name)
                  and p.name.endswith("odd2even" if a.half == "odd" else "even2odd")
                  and (p / "latest.pt").exists() and (p / "DONE").exists())
    rows = [r for r in (diag_one(p) for p in runs) if r]
    for r in rows:
        print(f"{r['run']:38s} κ={r['kappa']:+.3f} corr={r['corr']:+.3f} "
              f"‖Δ‖/‖n‖={r['delta_over_noise']:6.2f} α*={r['alpha_star']:+.3f} "
              f"X/X_in={r['x']/r['x_in']:6.3f} →(α*) {r['x_at_alpha_star']/r['x_in']:5.3f}")
    out = ROOT / a.out
    out.write_text(json.dumps({"train_half": a.half, "rows": rows},
                              indent=1, ensure_ascii=False))
    print(f"wrote {out}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Round3 预注册 §1.3 的算术桥：n2n_ratio = (1 + r²) / 2，r = rmse_model / rmse_input。

目的不是新判据，而是把两个已测的数（n2n_ratio 与 rmse_vs_imaging）的关系写死，
好让"训练损失 → n2n 等价"这一步换算有依据（见 reviews/V3_PR1_round2_next_plan.md §1.3）。

只读 `reports/v3_round2/` 下的存档 metrics JSON，不重跑任何训练/评价。
判据：最大偏差 ≤ 0.05 则恒等式可用；超过则说明某份 JSON 的两列不是同一口径，必须查。

用法：
    python -u scripts/v3_round3_bridge_check.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "v3_round2"

DEFAULT_FILES = [
    "s0_low_metrics.json",
    "s0_low_metrics_dense.json",
    "s0_low60_metrics.json",
    "s0_vlow_metrics.json",
    "s0_mid_metrics.json",
]


def cells(path: Path):
    """逐个 (臂 × 体积) 产出 n2n_ratio、r、恒等式预言值。"""
    d = json.loads(path.read_text())
    inp = {vol: m for vol, m in d["controls"]["input"].items() if vol != "budget_checkpoints"}
    for arm, v in d["arms"].items():
        for vol, m in v.get("per_volume", {}).items():
            if vol == "budget_checkpoints" or vol not in inp:
                continue
            r = m["rmse_vs_imaging"] / inp[vol]["rmse_vs_imaging"]
            yield arm, vol, float(m["n2n_ratio"]), float(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="恒等式可用的最大偏差（写死在预注册里，勿事后放宽）")
    args = ap.parse_args()

    rows, worst = [], (0.0, None)
    for name in args.files:
        p = SRC / name
        if not p.exists():
            print(f"MISSING {p}")
            continue
        for arm, vol, n2n, r in cells(p):
            pred = (1.0 + r * r) / 2.0
            err = pred - n2n
            rows.append((name, arm, vol, n2n, r, pred, err))
            if abs(err) > worst[0]:
                worst = (abs(err), (name, arm, vol, n2n, r, pred, err))

    print(f"cells={len(rows)}  max|pred-n2n|={worst[0]:.4f}")
    print("worst:", worst[1])
    gate = worst[0] <= args.tol
    print(f"G-BRIDGE: {'PASS' if gate else 'FAIL'} (tol={args.tol})")

    for name, arm, vol, n2n, r, pred, err in sorted(rows, key=lambda t: -abs(t[6]))[:5]:
        print(f"  {name:28s} {arm:16s} {vol:14s} n2n={n2n:.4f} r={r:.3f} pred={pred:.4f} err={err:+.4f}")

    # 训练损失 → n2n 等价的换算表（§1.3 的用法）：loss/σ² = r²  ⇒  n2n_eq = (1 + loss)/2
    print("\nloss/σ²  ->  r_train  ->  n2n 等价")
    for loss in (0.11, 0.16, 1.15):
        print(f"  {loss:5.2f}  ->  {math.sqrt(loss):5.3f}  ->  {(1 + loss) / 2:5.3f}")

    out = ROOT / "reports" / "v3_round3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bridge_check.json").write_text(json.dumps({
        "cells": len(rows), "tol": args.tol, "max_abs_err": worst[0],
        "worst": worst[1], "gate": "PASS" if gate else "FAIL",
        "files": args.files,
        "loss_to_n2n": {str(l): {"r_train": math.sqrt(l), "n2n_eq": (1 + l) / 2}
                        for l in (0.11, 0.16, 1.15)},
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {out / 'bridge_check.json'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())

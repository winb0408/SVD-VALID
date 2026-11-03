"""方案②的「记忆 vs 泛化」诊断：同一个网络，换不同目标实现算训练损失比。

## 为什么要这个

② 的损失是 `MSE(pred_B, B) / MSE(A, B)`，其**物理下限**由多实现均值给出 = 0.615
（见 phase4_report §5.6.1）。若某个 checkpoint 在**训练目标**上算出的比值远低于 0.615，
它不是"学到了更好的去噪器"，而是记住了那一次实现自己的噪声。

判据：同一个 pred，对训练目标 B_train 与从未见过的 B_held 各算一遍比值。

- 两者接近（都在 0.6–0.9）→ 真学到可预测成分，泛化成立
- 对 B_train ≪ 0.615 而对 B_held 明显更大 → 记忆，训练损失不可作为进步证据

用法：
    CUDA_VISIBLE_DEVICES=2 python -u scripts/v3_phase4_memorize_check.py \
        --run runs/v3_n2n_long/res_s3407 --out reports/.../mem_check.json
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

from models.network import NetworkCNR                                    # noqa: E402
from v3_phase4_train_n2n import prep                                    # noqa: E402

SIM = ROOT / "data/processed/v3_sim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="含 latest.pt 的目录")
    ap.add_argument("--input", default=str(SIM / "sim_seed1.npz"))
    ap.add_argument("--targets", nargs="+", default=["sim_seed1_n101", "sim_seed1_n102",
                                                    "sim_seed1_n103", "sim_seed1_n201_s1"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ck = torch.load(Path(args.run) / "latest.pt", map_location="cpu", weights_only=False)
    cargs = ck["args"]
    model = NetworkCNR(features=cargs["features"], groups=cargs["groups"])
    model.load_state_dict(ck["model"])
    model.eval().to(args.device)

    A, V = prep(Path(args.input))
    v = V > 0.5
    with torch.no_grad():
        xin = torch.as_tensor(A, device=args.device)[None, None].float()
        h = model(xin)
        pred = (xin + h if cargs.get("residual") else h)[0, 0].cpu().numpy()
    ident_self = float(np.mean((A[v] - pred[v]) ** 2))   # 参考：pred 动了多少

    print(f"checkpoint epoch={ck['epoch']}  residual={cargs.get('residual')}  "
          f"MSE(pred, A)={ident_self:.6f}")
    print(f"{'target':<22}{'MSE(pred,B)':>14}{'MSE(A,B)':>12}{'ratio':>9}")
    rows = []
    for t in args.targets:
        B, _ = prep(SIM / f"{t}.npz")
        num = float(np.mean((pred[v] - B[v]) ** 2))
        den = float(np.mean((A[v] - B[v]) ** 2))
        rows.append({"target": t, "mse_pred": num, "mse_ab": den, "ratio": num / den})
        print(f"{t:<22}{num:>14.6f}{den:>12.6f}{num / den:>9.4f}")
    print("\n下限（多实现均值给的可预测上界）= 0.615；"
          "训练目标上明显低于它 = 记住噪声，不代表泛化。")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"run": str(args.run), "epoch": ck["epoch"],
             "residual": cargs.get("residual"), "rows": rows,
             "floor_from_mean": 0.615}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

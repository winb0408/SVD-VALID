#!/usr/bin/env python3
"""Round7 C4（本方法）的 100% 覆盖重建：8 相位的固定体对角角色对，8 次前向铺满。

为什么需要 4 个模型
------------------
`train.py --pairing cross` 的输入必须是同一 tet 的 2 个角点、目标必须是互补 tet 的角点，
所以**一次训练只能接受 2 个输入相位**。要让 8 个相位都能当输入，就需要 4 次训练
（槽位表见 `reviews/V3_PR1_round7_prereg_final_comparison.md` §2）。

拼装规则：对每个输入相位 p，用覆盖它的模型前向，把输出写到 `p XOR 7` 的格点——
这是该模型**真正训练过的那条映射**（体对角互补对），不是外推规则。
m=1 时 8 个相位就是 2×2×2 块的全部 8 个体素，因此 8 次前向恰好铺满，**覆盖率 100%、零外推**。

产物
----
    data/processed/v3_round7/{seq}/pd_ours_db.npy      C4 的 dB 体（100% 覆盖）
    data/processed/v3_round7/{seq}/pd_ours_linear.npy  C4 的线性 PD
    reports/v3_round7/recon.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.network import NetworkCNR                              # noqa: E402
from v3_round4_b1_prep import inv                                  # noqa: E402
from v3_round5_eval import phase_slices, usable                    # noqa: E402

D7 = ROOT / "data/processed/v3_round7"
NORM = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())

# pair id -> slots (xa, xb, ya, yb)
PAIRS = {"P0": (1, 4, 6, 3), "P1": (0, 2, 7, 5),
         "P2": (3, 6, 4, 1), "P3": (5, 7, 2, 0)}


def load_model(path: Path, device):
    ck = torch.load(path / "latest.pt", map_location=device)
    a = ck.get("args", {})
    m = NetworkCNR(features=a.get("features", 16), groups=a.get("groups", 4),
                   residual=(a.get("arch_head", "direct") == "residual")).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def group_models(seq: str, device):
    """返回 {输入相位: (模型, 目标相位)}。"""
    table = {}
    for pid, slots in PAIRS.items():
        run = ROOT / f"runs/v3_round7/{seq}_{pid}"
        if not (run / "latest.pt").exists():
            raise FileNotFoundError(f"{run}/latest.pt 不存在（训练未完成）")
        model = load_model(run, device)
        (xa, xb, ya, yb) = slots
        table[xa] = (model, ya)
        table[xb] = (model, yb)
        assert ya == xa ^ 7 and yb == xb ^ 7, (pid, slots)
    assert sorted(table) == list(range(8)), sorted(table)
    return table


@torch.no_grad()
def reconstruct(table, z: np.ndarray, device):
    shape = usable(z.shape, 1)
    vol = z[:shape[0], :shape[1], :shape[2]]
    out = np.zeros(shape, np.float32)
    filled = np.zeros(shape, bool)
    for p in range(8):
        model, tgt = table[p]
        sub = vol[phase_slices(shape, 1, p)]
        x = torch.from_numpy(np.ascontiguousarray(sub))[None, None].to(device)
        y = model(x)[0, 0].cpu().numpy()
        if y.shape != sub.shape:
            raise RuntimeError(f"相位 {p}: 输出 {y.shape} != 输入 {sub.shape}")
        sl = phase_slices(shape, 1, tgt)
        if filled[sl].any():
            raise RuntimeError(f"相位 {p}→{tgt} 与已有覆盖重叠")
        out[sl] = y
        filled[sl] = True
    if not filled.all():
        raise RuntimeError(f"未铺满：覆盖率 {filled.mean():.4f}")
    return out, filled


def main():
    # --seqs 只为在训练尚未全部结束时先单独跑完一个序列做通路校验；
    # 正式出图仍是两个序列一起重建（recon.json 会按序列分键写入）。
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"],
                    choices=["SA64", "HSA64"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}
    if (ROOT / "reports/v3_round7/recon.json").exists():
        report = json.loads((ROOT / "reports/v3_round7/recon.json").read_text())
    for seq in args.seqs:
        z = np.load(ROOT / f"data/processed/v3_round7_b1/logeps/{seq}_full_x.npy")
        s = float(json.loads((ROOT / f"data/processed/v3_round6/{seq}/metadata.json")
                             .read_text())["s"])
        table = group_models(seq, device)
        y, filled = reconstruct(table, z, device)

        xhat = y * NORM["logeps"]["std"] + NORM["logeps"]["mean"]
        pd = inv("logeps", xhat, s)
        pd = np.maximum(pd, 0.0)
        db = 10.0 * np.log10(np.maximum(pd, 1e-12) / max(pd.max(), 1e-12))

        np.save(D7 / seq / "pd_ours_linear.npy", pd.astype(np.float32))
        np.save(D7 / seq / "pd_ours_db.npy",
                np.maximum(np.nan_to_num(db, neginf=-60.0), -60.0).astype(np.float32))
        report[seq] = {"coverage": float(filled.mean()), "shape": list(filled.shape),
                       "fwd_passes": 8, "models": sorted(PAIRS),
                       "pd_max": float(pd.max()), "pd_median": float(np.median(pd))}
        print(f"[{seq}] coverage={filled.mean():.4f} pd_max={pd.max():.4g}")

    (ROOT / "reports/v3_round7").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/v3_round7/recon.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

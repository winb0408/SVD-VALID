#!/usr/bin/env python3
"""Round9 / Phase 1A：**逐行对照上游 VALID v1.0** 的可复算证据。

本脚本不做训练，只做三件事，全部落盘成 JSON，供
`reports/v3_round9/implementation_fidelity.md` 逐项引用：

  1. 采样：上游 `generate_mask_pair` 的 rank 元组是否全块一致（= 一个全局置换）、
     每块四角是否属于上游 `idx_pair`；我们的 `TETRIS_PATTERNS` 是否与之逐元素相同。
  2. 网络：CRN 参数张量数 / 参数量 / 形状序列，权重互传后前向差（应恰为 0.0）。
  3. 损失：在**同一批张量**上，上游 `Total_loss` 表达式（含 `DWT3D(out1)` 写两遍的
     typo）与本仓库 `loss_terms()` 的数值对照；以及 DWT 八个频带的对应关系
     （哪几个只差一个符号）。

运行： /data2/wjb/anaconda3/envs/threePM/bin/python scripts/v3_round9_upstream_audit.py
输出： reports/v3_round9/upstream_audit.json
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sampling import TETRIS_PATTERNS  # noqa: E402
from models.dwt3d import VALID_REG_BANDS, dwt3d_haar  # noqa: E402
from models.network import HessianConstraintLoss3D, NetworkCNR  # noqa: E402
import train as train_mod  # noqa: E402

UP = Path("/tmp/valid_upstream")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "reports/v3_round9"


def load_up(rel, name):
    spec = importlib.util.spec_from_file_location(name, UP / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def audit_sampling(res):
    up = load_up("datasets/sampling.py", "_up_samp")
    idx_pair = torch.tensor([
        [0, 1, 2, 4], [0, 1, 3, 5], [1, 2, 3, 7], [0, 2, 3, 6],
        [0, 4, 5, 6], [1, 4, 5, 7], [2, 4, 6, 7], [3, 5, 6, 7],
    ], dtype=torch.int64)
    res["tetris_pattern_equals_upstream_idx_pair"] = bool(
        torch.equal(TETRIS_PATTERNS.to(torch.int64), idx_pair))

    # 每个 tet 是否恰好含体对角对的两个端点 —— 上游"互补 tet"配对的前提
    diag = [(0, 7), (1, 6), (2, 5), (3, 4)]
    res["no_tet_contains_a_body_diagonal_pair"] = bool(all(
        not any(a in set(p.tolist()) and b in set(p.tolist()) for a, b in diag)
        for p in idx_pair))

    trials, rank_sets, pat_counts = 5, [], []
    for t in range(trials):
        random.seed(t)
        torch.manual_seed(t)
        masks = up.generate_mask_pair(torch.zeros(1, 4, 4, 4, device=DEV))
        rows = []
        for b in range(8):
            corners = []
            for k in range(4):
                hit = torch.nonzero(masks[k][8 * b:8 * b + 8].cpu()).flatten().tolist()
                if len(hit) != 1:
                    raise AssertionError("上游某个 mask 在某块上不是单角")
                corners.append(hit[0])
            order = sorted(corners)
            rows.append((tuple(corners), tuple(order.index(c) for c in order and corners)))
        rank_sets.append(len({r for _, r in rows}))
        pat_counts.append(len({tuple(sorted(c)) for c, _ in rows}))
        legal = all(tuple(sorted(c)) in {tuple(sorted(p.tolist())) for p in idx_pair}
                    for c, _ in rows)
        res.setdefault("upstream_every_block_is_a_legal_tet", []).append(bool(legal))
        res.setdefault("upstream_rank_tuple_per_trial", []).append(
            [list(r) for _, r in rows])

    res["upstream_distinct_rank_tuples_per_patch_over_5_trials"] = rank_sets
    res["upstream_distinct_tets_per_patch_over_5_trials"] = pat_counts
    res["upstream_one_global_permutation_per_patch"] = bool(all(n == 1 for n in rank_sets))


def audit_network(res):
    up = load_up("models/model_CRN.py", "_up_crn")
    ours = NetworkCNR(features=16, groups=4)
    theirs = up.CRN(in_channels=1, out_channels=1, n_features=16, n_groups=4)
    po, pt = list(ours.parameters()), list(theirs.parameters())
    res["crn_param_tensor_count_ours"] = len(po)
    res["crn_param_tensor_count_upstream"] = len(pt)
    res["crn_total_params"] = sum(p.numel() for p in po)
    res["crn_shape_sequences_identical"] = bool(
        [tuple(a.shape) for a in po] == [tuple(b.shape) for b in pt])
    with torch.no_grad():
        for a, b in zip(po, pt):
            a.copy_(b)
    ours.to(DEV).eval(), theirs.to(DEV).eval()
    x = torch.randn(1, 1, 8, 16, 16, device=DEV)
    with torch.no_grad():
        diff = float((ours(x.to(DEV)) - theirs(x)).abs().max())
    res["crn_forward_max_abs_diff_after_weight_transfer"] = diff
    res["crn_bit_identical_forward"] = bool(diff == 0.0)
    # 属性名不同 ⇒ state_dict 键不同，权重不能直接跨仓库 load
    res["crn_state_dict_key_sets_identical"] = bool(
        set(ours.state_dict()) == set(theirs.state_dict()))


def audit_dwt(res):
    up = load_up("datasets/sampling.py", "_up_samp")
    x = torch.randn(1, 1, 8, 16, 16, device=DEV)
    up_bands = up.DWT_3D("haar")(x)
    our_bands = dwt3d_haar(x)
    names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
    table = []
    for i in range(8):
        best, best_d, best_sign = None, float("inf"), None
        for n, o in our_bands.items():
            d = float((up_bands[i] - o).abs().max())
            if d < best_d:
                best, best_d, best_sign = n, d, +1
            d = float((up_bands[i] + o).abs().max())
            if d < best_d:
                best, best_d, best_sign = n, d, -1
        table.append({"upstream_band": names[i], "our_band": best,
                      "sign": best_sign, "max_abs_diff": best_d})
    res["dwt_band_correspondence"] = table
    res["dwt_bands_flipped_in_sign"] = [
        t["our_band"] for t in table if t["sign"] == -1]
    res["dwt_reg_bands_sign_flipped"] = [
        t["our_band"] for t in table
        if t["sign"] == -1 and t["our_band"] in VALID_REG_BANDS]


def audit_loss(res):
    torch.manual_seed(7)
    o1, o2 = torch.randn(1, 1, 8, 16, 16, device=DEV), torch.randn(1, 1, 8, 16, 16, device=DEV)
    t3, t4 = torch.randn_like(o1), torch.randn_like(o2)

    hess = HessianConstraintLoss3D()
    ours = train_mod.loss_terms(o1, o2, t3, t4, hess, 1e-4, 1.0)

    up = load_up("datasets/sampling.py", "_up_samp2")
    dwt = up.DWT_3D("haar")
    mse = torch.nn.MSELoss()
    up_neighbor = 0.5 * mse(o1, t3) + 0.5 * mse(o2, t4)
    up_idt = mse(o1, o2)
    # 上游 train_pipeline.py:749-750 —— 注意两行都是 noisy_output_1（typo）
    # 上游 train_pipeline.py:753-757 选的是 **0,1,2,4** 号频带（= LLL,LLH,LHL,HLL），
    # 不是前四个；3 号是 LHH。
    REG_IDX = (0, 1, 2, 4)
    d1 = dwt(o1)
    d2 = dwt(o1)
    up_reg = hess(torch.cat([d1[i] for i in REG_IDX] + [d2[i] for i in REG_IDX], dim=1))
    up_total = up_neighbor + 1e-4 * up_reg + 1.0 * up_idt
    # 同一个上游实现、但把 d2 换成 output_2 的真实 DWT（即把 typo 修掉）：
    # 与我们的对称版本比较，用来证明 reg 的差异**只来自 typo**，不来自符号约定。
    d2_fixed = dwt(o2)
    up_reg_no_typo = hess(torch.cat([d1[i] for i in REG_IDX]
                                    + [d2_fixed[i] for i in REG_IDX], dim=1))

    res["loss_on_identical_tensors"] = {
        "ours_neighbor": float(ours["neighbor"]),
        "ours_identity": float(ours["identity"]),
        "ours_reg_symmetrized": float(ours["reg"]),
        "ours_total": float(ours["total"]),
        "upstream_neighbor": float(up_neighbor),
        "upstream_identity": float(up_idt),
        "upstream_reg_with_double_output1_typo": float(up_reg),
        "upstream_reg_typo_fixed": float(up_reg_no_typo),
        "reg_abs_diff_vs_typo_fixed_upstream": abs(float(ours["reg"]) - float(up_reg_no_typo)),
        "upstream_total": float(up_total),
        "neighbor_max_abs_diff": abs(float(ours["neighbor"]) - float(up_neighbor)),
        "identity_max_abs_diff": abs(float(ours["identity"]) - float(up_idt)),
        "reg_abs_diff_vs_typo_version": abs(float(ours["reg"]) - float(up_reg)),
    }


def audit_lr(res):
    """上游两套 schedule 哪一套真的生效 —— 用数值复算，不靠读一遍代码。"""
    import math
    from types import SimpleNamespace
    gen = load_up("utils/general.py", "_up_general")
    E, lr0 = 100, 1e-4

    def upstream_effective(epoch):
        """train_pipeline.py:741 每 batch 直接赋值的那条（poly）。"""
        opt = SimpleNamespace(param_groups=[{"lr": 0.0}])
        gen.adjust_learning_rate(opt, epoch, E, lr0)
        return opt.param_groups[0]["lr"]

    def upstream_lambdalr(epoch):
        """train_pipeline.py:686 的 LambdaLR，只在 epoch 末被 step 一次。"""
        return lr0 * min((epoch + 1) / (E * 0.1 + 1e-8),
                         0.5 * (math.cos(epoch / E * math.pi) + 1))

    def ours_poly(epoch):
        return lr0 * max(1.0 - epoch / E, 0.0) ** 0.9

    rows = []
    for e in (0, 1, 5, 10, 50, 90, 99):
        rows.append({"epoch": e,
                     "upstream_effective_batch_assign": upstream_effective(e),
                     "upstream_lambdalr_only": upstream_lambdalr(e),
                     "ours_poly": ours_poly(e)})
    res["lr_schedule"] = {
        "rows": rows,
        "ours_poly_equals_upstream_effective": all(
            abs(r["ours_poly"] - r["upstream_effective_batch_assign"]) <=
            max(5e-9, 1e-6 * abs(r["ours_poly"])) for r in rows),
        "note": "上游 741 行每个 batch 的 adjust_learning_rate(poly, power=0.9) 覆盖 686 行的 "
                "LambdaLR(warmup+cosine)，故**实际生效的是 poly**；差异仅上游有 round(·,8)。",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"upstream_path": str(UP), "device": DEV,
           "torch": torch.__version__, "numpy": np.__version__}
    if not UP.exists():
        res["error"] = "upstream 不存在"
    else:
        audit_sampling(res)
        audit_network(res)
        audit_dwt(res)
        audit_loss(res)
        audit_lr(res)
    (OUT / "upstream_audit.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("dwt_band_correspondence",)}, indent=1,
                     ensure_ascii=False))
    print("\nDWT band table:")
    for t in res.get("dwt_band_correspondence", []):
        print(f"  up {t['upstream_band']:3s} = {'+' if t['sign'] > 0 else '-'}"
              f"ours {t['our_band']:3s}   max|Δ|={t['max_abs_diff']:.2e}")
    print(f"\nwrote {OUT / 'upstream_audit.json'}")


if __name__ == "__main__":
    main()
